import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.dataset import get_datasets
from src.model import CustomAST

def icbhi_score(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    se = np.sum(cm[1:, 1:]) / (np.sum(cm[1:, :]) + 1e-8)
    sp = cm[0, 0]           / (np.sum(cm[0, :])  + 1e-8)
    return float(se), float(sp), float((se + sp) / 2)

def plot_confusion_matrix(all_labels, all_preds, save_path):
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.title('Confusion Matrix — HybridCNNAST')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Sauvegardée → {save_path}")

def plot_training_curves(history_path, save_path):
    import pandas as pd
    df = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], color='#E24B4A')
    axes[0].set_title('Loss par epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    # Se / Sp / Score
    axes[1].plot(df['epoch'], df['se'],    label='Sensibilité (Se)', color='#1D9E75')
    axes[1].plot(df['epoch'], df['sp'],    label='Spécificité (Sp)', color='#378ADD')
    axes[1].plot(df['epoch'], df['score'], label='Score ICBHI',       color='#D4537E', linewidth=2)
    axes[1].axhline(y=0.6831, color='gray', linestyle='--', label='Référence article (68.31%)')
    axes[1].set_title('Métriques par epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1. Charger données test
    _, test_dataset = get_datasets()
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    # 2. Charger modèle — utilise best_model si dispo, sinon last_p2
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CKPT_DIR, "last_p2.pth")
        print("best_model.pth introuvable → utilise last_p2.pth")
    model = CustomAST(num_classes=NUM_CLASSES, dropout=0.2, freeze_ast=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Modèle chargé — epoch {ckpt['epoch']} | Score {ckpt['score']:.4f}")

    # 3. Inférence
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, target, label in test_loader:
            logits = model(mel.to(DEVICE))
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label.tolist())

    # 4. Métriques
    se, sp, score = icbhi_score(all_labels, all_preds)
    print(f"\nRésultats finaux :")
    print(f"  Sensibilité (Se) : {se:.4f}  ({se*100:.2f}%)")
    print(f"  Spécificité (Sp) : {sp:.4f}  ({sp*100:.2f}%)")
    print(f"  Score ICBHI      : {score:.4f}  ({score*100:.2f}%)")
    print(f"\n  Article référence : Se=68.31% | Score=68.10%")
    print(f"  Gain Se  : {(se - 0.6831)*100:+.2f}%")
    print(f"  Gain Score : {(score - 0.6810)*100:+.2f}%")

    print("\nClassification Report :")
    print(classification_report(all_labels, all_preds,
                                target_names=LABEL_NAMES, digits=4))

    # 5. Figures
    plot_confusion_matrix(
        all_labels, all_preds,
        os.path.join(FIG_DIR, "confusion_matrix.png")
    )
    history_path = os.path.join(RES_DIR, "training_history.csv")
    if os.path.exists(history_path):
        plot_training_curves(
            history_path,
            os.path.join(FIG_DIR, "training_curves.png")
        )

if __name__ == "__main__":
    main()