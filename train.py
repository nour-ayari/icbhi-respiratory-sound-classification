"""
train.py — Entraînement 2 phases avec SAM + CustomAST
=======================================================

PHASE 1 — WARMUP  (backbone frozen, classifier only)
  - Objectif : faire converger la tête de classification
  - Optimiseur : AdamW simple (pas SAM, backbone frozen = gradient trop limité)
  - LR élevé sur classifier : 1e-3
  - Durée : WARMUP_EPOCHS (défaut 5)

PHASE 2 — FINE-TUNE  (n dernières couches + classifier)
  - Objectif : adapter les features AudioSet aux sons pulmonaires
  - Optimiseur : SAM(AdamW) avec rho=0.05
  - LR faible sur backbone : 5e-6 | LR classifier : 1e-4
  - Durée : FINETUNE_EPOCHS (défaut 20)

RESUME :
  --resume checkpoints/best_model.pth   → reprend là où on s'est arrêté

CONVERGENCE CHECK :
  - Logs clairs epoch par epoch
  - Early stopping par patience séparée par phase
  - Sauvegarde best + last checkpoint avec métadonnées complètes
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.dataset import get_datasets
from src.model import CustomAST
from src.sam import SAM

# ─────────────────────────────────────────────
# CONFIG ENTRAÎNEMENT
# ─────────────────────────────────────────────
WARMUP_EPOCHS   = 5      # Phase 1 : classifier seulement
FINETUNE_EPOCHS = 25     # Phase 2 : fine-tune backbone (partiel) + SAM
WARMUP_LR       = 1e-3   # LR phase 1
FINETUNE_LR_BB  = 5e-6   # LR backbone phase 2
FINETUNE_LR_CLS = 1e-4   # LR classifier phase 2
WEIGHT_DECAY    = 1e-2
SAM_RHO         = 0.01   
N_UNFREEZE      = 4      # Nombre de couches encoder à dégeler en phase 2
PATIENCE_P1     = 4      # Early stopping phase 1
PATIENCE_P2     = 8      # Early stopping phase 2
MIN_SP          = 0.20   # Spécificité minimale pour sauvegarder

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────
def icbhi_score(labels, preds):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    se = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp = cm[0, 0] / (np.sum(cm[0, :]) + 1e-8)
    return float(se), float(sp), float((se + sp) / 2), cm


def print_cm(cm, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(12) + "".join([f"{n:>10}" for n in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:<12}" + "".join([f"{v:>10d}" for v in row]))


def build_class_weights(counts, device, mode="inv"):
    """
    mode='inv'     : w = 1/count  (fort contraste, utile si collapse)
    mode='invsqrt' : w = 1/sqrt(count)  (plus doux)
    """
    counts_t = torch.tensor(counts, dtype=torch.float32)
    if mode == "inv":
        w = 1.0 / counts_t
    else:
        w = 1.0 / torch.sqrt(counts_t)
    w = w / w.sum() * len(counts)   # normaliser pour que la somme = num_classes
    print(f"Class weights ({mode}): "
          + " | ".join([f"{LABEL_NAMES[i]}={w[i]:.3f}" for i in range(len(counts))]))
    return w.to(device)


# ─────────────────────────────────────────────
# EVAL LOOP
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mel, _, label in loader:
            mel = mel.to(device, non_blocking=True)
            logits = model(mel)
            # label est un int dans le test loader
            target = torch.tensor(label, dtype=torch.long).to(device) \
                     if not isinstance(label, torch.Tensor) \
                     else label.to(device)

            loss = criterion(logits, target)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(
                label.tolist() if isinstance(label, torch.Tensor) else list(label)
            )

    acc = accuracy_score(all_labels, all_preds)
    se, sp, score, cm = icbhi_score(all_labels, all_preds)
    avg_loss = total_loss / max(len(loader), 1)

    pred_counts = np.bincount(all_preds, minlength=NUM_CLASSES)
    true_counts = np.bincount(all_labels, minlength=NUM_CLASSES)

    return {
        "loss": avg_loss, "acc": acc, "se": se, "sp": sp, "score": score,
        "cm": cm, "all_labels": all_labels, "all_preds": all_preds,
        "pred_counts": pred_counts, "true_counts": true_counts,
    }


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, epoch, phase, metrics, is_best=False):
    ckpt = {
        "epoch":        epoch,
        "phase":        phase,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict()
                        if not isinstance(optimizer, SAM)
                        else optimizer.base_optimizer.state_dict(),
        "score":        metrics["score"],
        "se":           metrics["se"],
        "sp":           metrics["sp"],
        "acc":          metrics["acc"],
    }
    torch.save(ckpt, path)
    tag = "★ BEST" if is_best else "last"
    print(f"  [{tag}] Sauvegardé → {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        try:
            if isinstance(optimizer, SAM):
                optimizer.base_optimizer.load_state_dict(ckpt["optim_state"])
            else:
                optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("  Warning: état optimizer incompatible, ignoré.")
    print(f"  Checkpoint chargé : epoch {ckpt['epoch']} | phase {ckpt['phase']} "
          f"| score {ckpt['score']:.4f}")
    return ckpt


# ─────────────────────────────────────────────
# PHASE 1 — WARMUP
# ─────────────────────────────────────────────
def phase1_warmup(model, train_loader, test_loader, criterion, device,
                  ckpt_dir, res_dir, start_epoch=1):
    """
    Entraîne seulement la tête de classification (backbone frozen).
    Utilise AdamW simple (SAM est inutile ici, presque rien à perturber).
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 1 : WARMUP CLASSIFIER ({WARMUP_EPOCHS} epochs max)")
    print(f"{'='*60}")

    p = model.get_trainable_params()
    print(f"  Params entraînables : {p['trainable_M']:.2f}M / {p['total_M']:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=WARMUP_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=WARMUP_EPOCHS, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    history, best_score, best_epoch, no_improve = [], 0.0, 0, 0

    print(f"\n{'Ep':>4} {'Phase':>8} {'TrLoss':>8} {'EvLoss':>8} "
          f"{'Acc':>7} {'Se':>7} {'Sp':>7} {'Score':>8}")
    print("-" * 70)

    for epoch in range(start_epoch, WARMUP_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i, (mel, target, _) in enumerate(train_loader, 1):
            mel    = mel.to(device, non_blocking=True)
            target = torch.argmax(
                target.to(device, non_blocking=True), dim=1
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(mel)
                loss   = criterion(logits, target)

            if torch.isnan(loss):
                print(f"  NaN batch {i} — skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()

        metrics = evaluate(model, test_loader, criterion, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"{epoch:>4} {'warmup':>8} {avg_loss:>8.4f} {metrics['loss']:>8.4f} "
              f"{metrics['acc']:>7.4f} {metrics['se']:>7.4f} "
              f"{metrics['sp']:>7.4f} {metrics['score']:>8.4f}", end="")

        # Diagnostic collapse
        pred_dist = dict(zip(LABEL_NAMES, metrics["pred_counts"].tolist()))
        print(f"  preds={pred_dist}", end="")

        improved = metrics["score"] > best_score 
        if improved:
            best_score, best_epoch, no_improve = metrics["score"], epoch, 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_p1.pth"),
                model, optimizer, epoch, "warmup", metrics, is_best=True
            )
            print("  ← best", end="")
        else:
            no_improve += 1

        save_checkpoint(
            os.path.join(ckpt_dir, "last_p1.pth"),
            model, optimizer, epoch, "warmup", metrics
        )
        print()

        # Rapport détaillé
        if epoch % 2 == 0 or epoch == WARMUP_EPOCHS:
            print(classification_report(
                metrics["all_labels"], metrics["all_preds"],
                target_names=LABEL_NAMES, digits=3, zero_division=0
            ))
            print_cm(metrics["cm"], LABEL_NAMES)
            print(f"  True : {dict(zip(LABEL_NAMES, metrics['true_counts'].tolist()))}")
            print(f"  Pred : {pred_dist}")

        history.append({
            "epoch": epoch, "phase": "warmup",
            "train_loss": avg_loss, **{k: v for k, v in metrics.items()
                                       if k not in ("cm", "all_labels", "all_preds",
                                                    "pred_counts", "true_counts")}
        })

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        if no_improve >= PATIENCE_P1:
            print(f"\n  Early stopping phase 1 à epoch {epoch}")
            break

    print(f"\n  Meilleur score phase 1 : {best_score:.4f} (epoch {best_epoch})")
    return history, best_score


# ─────────────────────────────────────────────
# PHASE 2 — FINE-TUNE avec SAM
# ─────────────────────────────────────────────
def phase2_finetune(model, train_loader, test_loader, criterion, device,
                    ckpt_dir, res_dir, start_epoch=1):
    """
    Dégèle les N dernières couches encoder + classifier.
    Utilise AdamW + AMP. LR différencié : backbone << classifier.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 : FINE-TUNE AdamW ({FINETUNE_EPOCHS} epochs max)")
    print(f"  Dégel des {N_UNFREEZE} dernières couches encoder")
    print(f"{'='*60}")

    model.unfreeze_last_layers(N_UNFREEZE)
    p = model.get_trainable_params()
    print(f"  Params entraînables : {p['trainable_M']:.2f}M / {p['total_M']:.2f}M")

    param_groups = [
        {
            "params": [p for p in model.ast.parameters() if p.requires_grad],
            "lr": FINETUNE_LR_BB,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": model.classifier.parameters(),
            "lr": FINETUNE_LR_CLS,
            "weight_decay": WEIGHT_DECAY,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    history, best_score, best_epoch, no_improve = [], 0.0, 0, 0

    print(f"\n{'Ep':>4} {'Phase':>8} {'TrLoss':>8} {'EvLoss':>8} "
          f"{'Acc':>7} {'Se':>7} {'Sp':>7} {'Score':>8}")
    print("-" * 70)

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for mel, target, _ in train_loader:
            mel    = mel.to(device, non_blocking=True)
            target = torch.argmax(
                target.to(device, non_blocking=True), dim=1
            )

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(mel)
                loss   = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        metrics = evaluate(model, test_loader, criterion, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"{epoch:>4} {'finetune':>8} {avg_loss:>8.4f} {metrics['loss']:>8.4f} "
              f"{metrics['acc']:>7.4f} {metrics['se']:>7.4f} "
              f"{metrics['sp']:>7.4f} {metrics['score']:>8.4f}", end="")

        pred_dist = dict(zip(LABEL_NAMES, metrics["pred_counts"].tolist()))
        print(f"  preds={pred_dist}", end="")

        improved = (metrics["score"] > best_score and metrics["sp"] > MIN_SP)
        if improved:
            best_score, best_epoch, no_improve = metrics["score"], epoch, 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pth"),
                model, optimizer, epoch, "finetune", metrics, is_best=True
            )
            print("  ← best", end="")
        else:
            no_improve += 1

        save_checkpoint(
            os.path.join(ckpt_dir, "last_p2.pth"),
            model, optimizer, epoch, "finetune", metrics
        )
        print()

        if epoch % 5 == 0 or epoch == 1:
            print(classification_report(
                metrics["all_labels"], metrics["all_preds"],
                target_names=LABEL_NAMES, digits=3, zero_division=0
            ))
            print_cm(metrics["cm"], LABEL_NAMES)
            print(f"  True : {dict(zip(LABEL_NAMES, metrics['true_counts'].tolist()))}")
            print(f"  Pred : {pred_dist}")

        history.append({
            "epoch": epoch, "phase": "finetune",
            "train_loss": avg_loss, **{k: v for k, v in metrics.items()
                                       if k not in ("cm", "all_labels", "all_preds",
                                                    "pred_counts", "true_counts")}
        })

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        if no_improve >= PATIENCE_P2:
            print(f"\n  Early stopping phase 2 à epoch {epoch}")
            break

    print(f"\n  Meilleur score phase 2 : {best_score:.4f} (epoch {best_epoch})")
    return history, best_score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Chemin vers un checkpoint pour reprendre")
    parser.add_argument("--phase",     type=int,  default=0,
                        help="0=les deux phases | 1=warmup only | 2=finetune only")
    parser.add_argument("--warmup_ep", type=int,  default=WARMUP_EPOCHS)
    parser.add_argument("--fine_ep",   type=int,  default=FINETUNE_EPOCHS)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR,  exist_ok=True)

    # ── Data ────────────────────────────────────────────────────
    train_dataset, test_dataset = get_datasets()

    labels_arr = train_dataset.L
    counts     = np.bincount(labels_arr, minlength=NUM_CLASSES)
    print("Train class counts:", dict(zip(LABEL_NAMES, counts.tolist())))

    # WeightedRandomSampler avec inv count (pas sqrt) → meilleure correction
    sample_weights = [1.0 / counts[l] for l in labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    # ── Modèle ──────────────────────────────────────────────────
    model = CustomAST(
        num_classes=NUM_CLASSES,
        dropout=0.2,         # 0.4 trop agressif avec backbone frozen
        freeze_ast=True,     # commence frozen, phase 2 va dégeler
    ).to(DEVICE)

    p = model.get_trainable_params()
    print(f"Params total : {p['total_M']:.2f}M | Entraînables : {p['trainable_M']:.2f}M")

    # ── Loss avec class weights INV (pas sqrt) ──────────────────
    # inv count = contraste fort, indispensable quand le modèle collapse
    class_weights = build_class_weights(counts.tolist(), DEVICE, mode="inv")
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Resume ──────────────────────────────────────────────────
    start_phase = 1
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"\nReprise depuis : {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        start_phase = 2 if ckpt.get("phase") == "warmup" else 2
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"  → Reprise phase {start_phase}, epoch {start_epoch}")

    # ── Lancement des phases ─────────────────────────────────────
    all_history = []

    run_p1 = args.phase in (0, 1) and start_phase <= 1
    run_p2 = args.phase in (0, 2)

    if run_p1:
        h1, score1 = phase1_warmup(
            model, train_loader, test_loader, criterion, DEVICE,
            CKPT_DIR, RES_DIR, start_epoch=start_epoch,
        )
        all_history.extend(h1)
        start_epoch = 1  # réinitialise pour phase 2

    if run_p2:
        h2, score2 = phase2_finetune(
            model, train_loader, test_loader, criterion, DEVICE,
            CKPT_DIR, RES_DIR, start_epoch=start_epoch,
        )
        all_history.extend(h2)

    # ── Sauvegarde historique ────────────────────────────────────
    pd.DataFrame(all_history).to_csv(
        os.path.join(RES_DIR, "training_history.csv"), index=False
    )
    print("\nHistorique sauvegardé → training_history.csv")


if __name__ == "__main__":
    main()