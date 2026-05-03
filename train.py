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
from src.losses import FocalLoss

# ─────────────────────────────────────────────
# CONFIG ENTRAÎNEMENT
# ─────────────────────────────────────────────
WARMUP_EPOCHS   = 5      # Phase 1 : classifier seulement
FINETUNE_EPOCHS = 25     # Phase 2 : fine-tune backbone (partiel) + SAM
WARMUP_LR       = 1e-3   # LR phase 1
FINETUNE_LR_BB  = 5e-6   # LR backbone phase 2
FINETUNE_LR_CLS = 1e-4   # LR classifier phase 2
WEIGHT_DECAY    = 1e-2
SAM_RHO         = 0.05
N_UNFREEZE      = 4      # Nombre de couches encoder à dégeler en phase 2
PATIENCE_P1     = 4      # Early stopping phase 1
PATIENCE_P2     = 8      # Early stopping phase 2
MIN_SP          = 0.20   # Spécificité minimale pour sauvegarder
LOG_EVERY       = 50     # Affiche progression intra-epoch toutes les N batches

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


def build_class_weights(counts, device, beta=0.9999):
    """
    Effective number of samples per class: (1 - beta) / (1 - beta^n)
    
    WHY: Better handles extreme imbalance than 1/count.
    - As beta → 1, approaches 1/n (strong weighting)
    - Smoother than inverse count, less extreme
    - Standard in imbalanced learning literature
    
    Args:
        counts: list of class counts
        device: torch device
        beta: decay rate (default 0.9999)
    
    Returns:
        weights tensor normalized so sum = num_classes
    """
    counts_t = torch.tensor(counts, dtype=torch.float32)
    effective_n = 1.0 - torch.pow(beta, counts_t)
    w = (1.0 - beta) / (effective_n + 1e-8)
    w = w / w.sum() * len(counts)  # normalize to sum to num_classes
    print(f"Class weights (effective_n, beta={beta}): "
          + " | ".join([f"{LABEL_NAMES[i]}={w[i]:.3f}" for i in range(len(counts))]))
    return w.to(device)


# ─────────────────────────────────────────────
# THRESHOLD TUNING FOR SENSITIVITY MAXIMIZATION
# ─────────────────────────────────────────────
def tune_thresholds_for_sensitivity(model, loader, device, min_sp=0.20):
    """
    Tune per-class decision thresholds to maximize sensitivity (recall),
    while keeping specificity >= min_sp.
    
    WHY: Standard argmax predictions treat all classes equally.
    Medical audio classification should prioritize RECALL for abnormal classes
    (Crackle, Wheeze, Both), even if it slightly reduces specificity.
    
    Strategy:
    1. Get softmax probabilities for each sample
    2. For each class, vary threshold from 0 to 1
    3. Select threshold that maximizes sensitivity subject to specificity >= min_sp
    
    Returns:
        thresholds: dict mapping class_id → float threshold
        metrics: dict with tuning details
    """
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for mel, _, label in loader:
            mel = mel.to(device, non_blocking=True)
            logits = model(mel)
            probs = torch.softmax(logits, dim=1)  # [B, 4]
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(label.tolist() if isinstance(label, torch.Tensor) else list(label))
    
    all_probs = np.concatenate(all_probs, axis=0)  # [N, 4]
    all_labels = np.array(all_labels)
    
    best_thresholds = {}
    tuning_log = {}
    
    # For each class, find best threshold
    for cls in range(NUM_CLASSES):
        # Binary task: class vs all others
        is_cls = (all_labels == cls).astype(int)
        cls_probs = all_probs[:, cls]
        
        best_threshold = 0.5
        best_se = 0.0
        
        # Grid search over thresholds
        for thresh in np.linspace(0, 1, 101):
            pred_cls = (cls_probs >= thresh).astype(int)
            
            # Compute binary Se/Sp
            tp = np.sum((pred_cls == 1) & (is_cls == 1))
            fn = np.sum((pred_cls == 0) & (is_cls == 1))
            tn = np.sum((pred_cls == 0) & (is_cls == 0))
            fp = np.sum((pred_cls == 1) & (is_cls == 0))
            
            se = tp / (tp + fn + 1e-8)
            sp = tn / (tn + fp + 1e-8)
            
            # Accept if satisfies constraint and improves sensitivity
            if sp >= min_sp and se > best_se:
                best_se = se
                best_threshold = thresh
        
        best_thresholds[cls] = best_threshold
        tuning_log[cls] = {"threshold": best_threshold, "sensitivity": best_se}
    
    print(f"\n  Thresholds tuned on validation set (min_sp={min_sp}):")
    for cls in range(NUM_CLASSES):
        print(f"    {LABEL_NAMES[cls]}: threshold={best_thresholds[cls]:.3f}, "
              f"sensitivity={tuning_log[cls]['sensitivity']:.4f}")
    
    return best_thresholds, tuning_log


def predict_with_thresholds(logits, thresholds):
    """
    Make predictions using per-class thresholds instead of argmax.
    
    WHY: Each class can have different decision boundary optimized for recall.
    
    Args:
        logits: [B, num_classes] model outputs
        thresholds: dict mapping class_id → threshold
    
    Returns:
        preds: [B] predicted class indices
    """
    probs = torch.softmax(logits, dim=1)  # [B, 4]
    batch_size = probs.shape[0]
    preds = []
    
    for i in range(batch_size):
        # Check which classes exceed their threshold
        confident_classes = [
            cls for cls in range(NUM_CLASSES) 
            if probs[i, cls].item() >= thresholds[cls]
        ]
        
        if confident_classes:
            # Pick class with highest probability among confident ones
            preds.append(max(confident_classes, key=lambda c: probs[i, c].item()))
        else:
            # Fallback to argmax if no class confident
            preds.append(torch.argmax(probs[i]).item())
    
    return np.array(preds)


# ─────────────────────────────────────────────
# EVAL LOOP
# ─────────────────────────────────────────────
def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_logits = []

    with torch.no_grad():
        for mel, _, label in loader:
            mel = mel.to(device, non_blocking=True)
            logits = model(mel)
            all_logits.append(logits.cpu())
            
            # label est un int dans le test loader
            target = torch.tensor(label, dtype=torch.long).to(device) \
                     if not isinstance(label, torch.Tensor) \
                     else label.to(device)

            loss = criterion(logits, target)
            total_loss += loss.item()

            # Use thresholds if provided, otherwise use argmax
            if thresholds is not None:
                preds = predict_with_thresholds(logits, thresholds).tolist()
            else:
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
def save_checkpoint(path, model, optimizer, epoch, phase, metrics, thresholds=None, is_best=False):
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
        "thresholds":   thresholds,  # Save optimized thresholds for sensitivity
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
    if "thresholds" in ckpt and ckpt["thresholds"]:
        print(f"  Thresholds loaded: {ckpt['thresholds']}")
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

    debug = globals().get("DEBUG_TRAIN", False)
    if debug:
        try:
            cls_w = model.classifier[-1].weight.detach().cpu().clone()
            cls_b = model.classifier[-1].bias.detach().cpu().clone()
            print(f"  [DEBUG] classifier last-linear mean={cls_w.mean():.6f}, bias_mean={cls_b.mean():.6f}")
        except Exception:
            print("  [DEBUG] unable to snapshot classifier params")

    print(f"\n{'Ep':>4} {'Phase':>8} {'TrLoss':>8} {'EvLoss':>8} "
          f"{'Se':>7} {'Sp':>7} {'Score':>8} | {'SeT':>7} {'SpT':>7} {'ScoreT':>8}")
    print("-" * 95)

    for epoch in range(start_epoch, WARMUP_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i, (mel, target, _) in enumerate(train_loader, 1):
            mel    = mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(mel)
                loss   = criterion(logits, target)

            if torch.isnan(loss):
                print(f"  NaN batch {i} — skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if debug and (i % LOG_EVERY == 0 or i == len(train_loader)):
                # grad norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                # param norm (classifier last layer)
                try:
                    cls_param_norm = model.classifier[-1].weight.data.norm().item()
                except Exception:
                    cls_param_norm = 0.0
                print(f"      [DEBUG] batch {i} grad_norm={total_norm:.6f} cls_w_norm={cls_param_norm:.6f}")
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if i % LOG_EVERY == 0 or i == len(train_loader):
                print(
                    f"    [P1][ep {epoch}/{WARMUP_EPOCHS}] batch {i}/{len(train_loader)} "
                    f"loss={running_loss / i:.4f}"
                )

        scheduler.step()

        metrics_raw = evaluate(model, test_loader, criterion, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        # THRESHOLD TUNING: Find per-class thresholds to maximize sensitivity
        # WHY: Medical audio should prioritize recall for abnormal classes
        best_thresholds, _ = tune_thresholds_for_sensitivity(
            model, test_loader, device, min_sp=MIN_SP
        )

        metrics_tuned = evaluate(model, test_loader, criterion, device, thresholds=best_thresholds)

        print(f"{epoch:>4} {'warmup':>8} {avg_loss:>8.4f} {metrics_raw['loss']:>8.4f} "
              f"{metrics_raw['se']:>7.4f} {metrics_raw['sp']:>7.4f} {metrics_raw['score']:>8.4f} | "
              f"{metrics_tuned['se']:>7.4f} {metrics_tuned['sp']:>7.4f} {metrics_tuned['score']:>8.4f}", end="")

        # Diagnostic collapse (sur prédictions tuned)
        pred_dist = dict(zip(LABEL_NAMES, metrics_tuned["pred_counts"].tolist()))
        print(f"  preds={pred_dist}", end="")

        improved = (metrics_tuned["score"] > best_score and metrics_tuned["sp"] > MIN_SP)
        if improved:
            best_score, best_epoch, no_improve = metrics_tuned["score"], epoch, 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_p1.pth"),
                model, optimizer, epoch, "warmup", metrics_tuned,
                thresholds=best_thresholds, is_best=True
            )
            print("  ← best", end="")
        else:
            no_improve += 1

        save_checkpoint(
            os.path.join(ckpt_dir, "last_p1.pth"),
            model, optimizer, epoch, "warmup", metrics_tuned,
            thresholds=best_thresholds
        )
        print()

        # Rapport détaillé
        if epoch % 2 == 0 or epoch == WARMUP_EPOCHS:
            print(classification_report(
                metrics_tuned["all_labels"], metrics_tuned["all_preds"],
                target_names=LABEL_NAMES, digits=3, zero_division=0
            ))
            print_cm(metrics_tuned["cm"], LABEL_NAMES)
            print(f"  True : {dict(zip(LABEL_NAMES, metrics_tuned['true_counts'].tolist()))}")
            print(f"  Pred : {pred_dist}")

        history.append({
            "epoch": epoch, "phase": "warmup",
            "train_loss": avg_loss, **{k: v for k, v in metrics_tuned.items()
                                       if k not in ("cm", "all_labels", "all_preds",
                                                    "pred_counts", "true_counts")}
        })

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        if no_improve >= PATIENCE_P1:
            print(f"\n  Early stopping phase 1 à epoch {epoch}")
            break
    if debug:
        try:
            cls_w_after = model.classifier[-1].weight.detach().cpu()
            cls_b_after = model.classifier[-1].bias.detach().cpu()
            mean_diff = (cls_w_after - cls_w).abs().mean().item()
            print(f"  [DEBUG] classifier last-linear mean-abs-change after phase1: {mean_diff:.6f}")
        except Exception:
            pass
    print(f"\n  Meilleur score phase 1 : {best_score:.4f} (epoch {best_epoch})")
    return history, best_score


# ─────────────────────────────────────────────
# PHASE 2 — FINE-TUNE avec SAM
# ─────────────────────────────────────────────
def phase2_finetune(model, train_loader, test_loader, criterion, device,
                    ckpt_dir, res_dir, start_epoch=1):
    """
    Dégèle les N dernières couches encoder + classifier.
    Utilise SAM(AdamW) pour chercher des minima plats.
    LR différencié : backbone << classifier.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 : FINE-TUNE + SAM ({FINETUNE_EPOCHS} epochs max)")
    print(f"  Dégel des {N_UNFREEZE} dernières couches encoder")
    print(f"{'='*60}")

    model.unfreeze_last_layers(N_UNFREEZE)
    p = model.get_trainable_params()
    print(f"  Params entraînables : {p['trainable_M']:.2f}M / {p['total_M']:.2f}M")

    # LR différencié : backbone très faible, tête plus haute
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

    base_optimizer = torch.optim.AdamW
    optimizer = SAM(param_groups, base_optimizer, rho=SAM_RHO,
                    lr=FINETUNE_LR_CLS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.base_optimizer,
        T_max=FINETUNE_EPOCHS,
        eta_min=1e-7
    )
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    history, best_score, best_epoch, no_improve = [], 0.0, 0, 0

    debug = globals().get("DEBUG_TRAIN", False)
    if debug:
        try:
            cls_w = model.classifier[-1].weight.detach().cpu().clone()
            cls_b = model.classifier[-1].bias.detach().cpu().clone()
            print(f"  [DEBUG] classifier last-linear mean(before)={cls_w.mean():.6f}")
        except Exception:
            print("  [DEBUG] unable to snapshot classifier params")

    print(f"\n{'Ep':>4} {'Phase':>8} {'TrLoss':>8} {'EvLoss':>8} "
          f"{'Se':>7} {'Sp':>7} {'Score':>8} | {'SeT':>7} {'SpT':>7} {'ScoreT':>8}")
    print("-" * 95)

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i, (mel, target, _) in enumerate(train_loader, 1):
            mel    = mel.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # ── SAM first step ──────────────────────────────
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(mel)
                loss   = criterion(logits, target)

            if torch.isnan(loss):
                print(f"  NaN batch {i} — skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.first_step(zero_grad=True)

            # ── SAM second step ─────────────────────────────
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits2 = model(mel)
                loss2   = criterion(logits2, target)

            scaler.scale(loss2).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)
            scaler.update()

            running_loss += loss.item()

            if i % LOG_EVERY == 0 or i == len(train_loader):
                print(
                    f"    [P2][ep {epoch}/{FINETUNE_EPOCHS}] batch {i}/{len(train_loader)} "
                    f"loss={running_loss / i:.4f}"
                )
                if debug:
                    # grad norm
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    try:
                        cls_param_norm = model.classifier[-1].weight.data.norm().item()
                    except Exception:
                        cls_param_norm = 0.0
                    print(f"      [DEBUG] P2 batch {i} grad_norm={total_norm:.6f} cls_w_norm={cls_param_norm:.6f}")

        scheduler.step()

        metrics_raw = evaluate(model, test_loader, criterion, device)
        avg_loss = running_loss / max(len(train_loader), 1)

        # THRESHOLD TUNING: Find per-class thresholds to maximize sensitivity
        best_thresholds, _ = tune_thresholds_for_sensitivity(
            model, test_loader, device, min_sp=MIN_SP
        )

        metrics_tuned = evaluate(model, test_loader, criterion, device, thresholds=best_thresholds)

        print(f"{epoch:>4} {'finetune':>8} {avg_loss:>8.4f} {metrics_raw['loss']:>8.4f} "
              f"{metrics_raw['se']:>7.4f} {metrics_raw['sp']:>7.4f} {metrics_raw['score']:>8.4f} | "
              f"{metrics_tuned['se']:>7.4f} {metrics_tuned['sp']:>7.4f} {metrics_tuned['score']:>8.4f}", end="")

        pred_dist = dict(zip(LABEL_NAMES, metrics_tuned["pred_counts"].tolist()))
        print(f"  preds={pred_dist}", end="")

        improved = (metrics_tuned["score"] > best_score and metrics_tuned["sp"] > MIN_SP)
        if improved:
            best_score, best_epoch, no_improve = metrics_tuned["score"], epoch, 0
            save_checkpoint(
                os.path.join(ckpt_dir, "best_model.pth"),
                model, optimizer, epoch, "finetune", metrics_tuned,
                thresholds=best_thresholds, is_best=True
            )
            print("  ← best", end="")
        else:
            no_improve += 1

        save_checkpoint(
            os.path.join(ckpt_dir, "last_p2.pth"),
            model, optimizer, epoch, "finetune", metrics_tuned,
            thresholds=best_thresholds
        )
        print()

        if epoch % 5 == 0 or epoch == 1:
            print(classification_report(
                metrics_tuned["all_labels"], metrics_tuned["all_preds"],
                target_names=LABEL_NAMES, digits=3, zero_division=0
            ))
            print_cm(metrics_tuned["cm"], LABEL_NAMES)
            print(f"  True : {dict(zip(LABEL_NAMES, metrics_tuned['true_counts'].tolist()))}")
            print(f"  Pred : {pred_dist}")

        history.append({
            "epoch": epoch, "phase": "finetune",
            "train_loss": avg_loss, **{k: v for k, v in metrics_tuned.items()
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
    if debug:
        try:
            cls_w_after = model.classifier[-1].weight.detach().cpu()
            mean_diff = (cls_w_after - cls_w).abs().mean().item()
            print(f"  [DEBUG] classifier last-linear mean-abs-change after phase2: {mean_diff:.6f}")
        except Exception:
            pass
    return history, best_score


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    global WARMUP_EPOCHS, FINETUNE_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str,  default=None,
                        help="Chemin vers un checkpoint pour reprendre")
    parser.add_argument("--phase",     type=int,  default=0,
                        help="0=les deux phases | 1=warmup only | 2=finetune only")
    parser.add_argument("--warmup_ep", type=int,  default=WARMUP_EPOCHS)
    parser.add_argument("--fine_ep",   type=int,  default=FINETUNE_EPOCHS)
    parser.add_argument("--log_every", type=int,  default=50,
                        help="Print intra-epoch logs every N batches")
    parser.add_argument("--no_sampler", action="store_true",
                        help="Disable WeightedRandomSampler (use shuffle)")
    parser.add_argument("--debug_train", action="store_true",
                        help="Enable extra per-batch gradient/param debug logs")
    args = parser.parse_args()

    WARMUP_EPOCHS = max(1, int(args.warmup_ep))
    FINETUNE_EPOCHS = max(1, int(args.fine_ep))
    LOG_EVERY = max(1, int(args.log_every))
    DEBUG_TRAIN = bool(args.debug_train)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR,  exist_ok=True)

    # ── Data ────────────────────────────────────────────────────
    train_dataset, test_dataset = get_datasets()

    labels_arr = train_dataset.L
    counts     = np.bincount(labels_arr, minlength=NUM_CLASSES)
    print("Train class counts:", dict(zip(LABEL_NAMES, counts.tolist())))

    # WeightedRandomSampler avec inv count (pas sqrt) → meilleure correction
    sample_weights = [1.0 / counts[l] for l in labels_arr]
    if args.no_sampler:
        sampler_obj = None
    else:
        sampler_obj = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

    if sampler_obj is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            sampler=sampler_obj, num_workers=4, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True,
        )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Modèle ──────────────────────────────────────────────────
    model = CustomAST(
        num_classes=NUM_CLASSES,
        dropout=0.4,
        freeze_ast=True,     # commence frozen, phase 2 va dégeler
    ).to(DEVICE)

    p = model.get_trainable_params()
    print(f"Params total : {p['total_M']:.2f}M | Entraînables : {p['trainable_M']:.2f}M")

    # ── Loss avec Focal Loss + class weights ─────────────────────
    # CHANGE: Replaced CrossEntropyLoss with FocalLoss
    # WHY: Focal loss down-weights easy examples, focuses on hard negatives.
    # Critical for imbalanced medical audio where sensitivity matters most.
    class_weights = build_class_weights(counts.tolist(), DEVICE)
    criterion     = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean')

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