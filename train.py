import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.dataset import get_datasets
from src.model import CNNBiGRU

torch.manual_seed(SEED)
np.random.seed(SEED)


def icbhi_score(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    se = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp = cm[0, 0] / (np.sum(cm[0, :]) + 1e-8)
    return float(se), float(sp), float((se + sp) / 2), cm


def print_confusion_matrix(cm, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(12) + "".join([f"{name:>10}" for name in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<12}" + "".join([f"{v:>10d}" for v in row])
        print(row_str)


def per_class_metrics(all_labels, all_preds, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        })
    return rows


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    train_dataset, test_dataset = get_datasets()

    labels_arr = train_dataset.L
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    print("Train class counts:", dict(zip(LABEL_NAMES, counts.tolist())))

    # sampler modéré
    sample_weights = [1.0 / np.sqrt(counts[l]) for l in labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = CNNBiGRU().to(DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params total : {total/1e6:.2f}M | Entraînables : {trainp/1e6:.2f}M")

    class_counts = torch.tensor([3642., 1864., 886., 506.])
    class_weights = 1.0 / torch.sqrt(class_counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
   

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    history = []
    best_score = 0.0
    best_epoch = 0
    no_improve = 0
    PATIENCE = 12

    print(f"\nEntraînement {DEVICE} — {EPOCHS} epochs")
    print(f"{'Ep':>4} {'Loss':>8} {'Acc':>8} {'Se':>8} {'Sp':>8} {'Score':>8}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        # ---------------- TRAIN ----------------
        model.train()
        running_loss = 0.0

        for i, (mel, target, _) in enumerate(train_loader, 1):
            mel = mel.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            target = torch.argmax(target, dim=1)  # one-hot -> labels

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', enabled=(DEVICE == 'cuda')):
                logits = model(mel)
                loss = criterion(logits, target)

            if torch.isnan(loss):
                print(f"NaN batch {i} — skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if i == 1 or i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}", flush=True)

        scheduler.step()
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # ---------------- EVAL ----------------
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for mel, _, label in test_loader:
                mel = mel.to(DEVICE, non_blocking=True)
                logits = model(mel)
                preds = torch.argmax(logits, dim=1).cpu().tolist()

                all_preds.extend(preds)
                all_labels.extend(label.tolist())

        acc = accuracy_score(all_labels, all_preds)
        se, sp, score, cm = icbhi_score(all_labels, all_preds)
        avg_loss = running_loss / max(len(train_loader), 1)

        class_rows = per_class_metrics(all_labels, all_preds, LABEL_NAMES)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": acc,
            "Se": se,
            "Sp": sp,
            "Score": score
        }

        for c in class_rows:
            cname = c["class"]
            row[f"{cname}_precision"] = c["precision"]
            row[f"{cname}_recall"] = c["recall"]
            row[f"{cname}_f1"] = c["f1"]
            row[f"{cname}_support"] = c["support"]

        history.append(row)

        print(f"{epoch:>4} {avg_loss:>8.4f} {acc:>8.4f} {se:>8.4f} {sp:>8.4f} {score:>8.4f}", end="")

        improved = (score > best_score and sp > 0.25)

        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "score": score,
                "se": se,
                "sp": sp,
                "acc": acc
            }, os.path.join(CKPT_DIR, "best_model.pth"))
            print("  ← best", end="")
        else:
            no_improve += 1

        print()

        # affichage détaillé toutes les 2 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(classification_report(
                all_labels,
                all_preds,
                target_names=LABEL_NAMES,
                digits=3,
                zero_division=0
            ))
            print_confusion_matrix(cm, LABEL_NAMES)

            # distribution des prédictions
            pred_counts = np.bincount(all_preds, minlength=NUM_CLASSES)
            true_counts = np.bincount(all_labels, minlength=NUM_CLASSES)

            print("\nTrue counts :", dict(zip(LABEL_NAMES, true_counts.tolist())))
            print("Pred counts :", dict(zip(LABEL_NAMES, pred_counts.tolist())))

            print("\nRappels importants :")
            print("- recall faible pour une classe => le modèle rate cette classe")
            print("- precision faible pour une classe => beaucoup de faux positifs pour cette classe")
            print("- si une colonne de la matrice domine => modèle biaisé vers cette classe")
            print("- si une ligne va vers une autre classe => confusion systématique")

        # sauvegarder la confusion matrix brute de chaque epoch
        cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
        cm_df.to_csv(os.path.join(RES_DIR, f"confusion_matrix_epoch_{epoch}.csv"))

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(RES_DIR, "training_history_detailed.csv"),
        index=False
    )

    print(f"\nMeilleur epoch : {best_epoch}")
    print(f"Meilleur Score : {best_score:.4f}")


if __name__ == "__main__":
    main()