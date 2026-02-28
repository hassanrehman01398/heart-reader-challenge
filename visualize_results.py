"""
Generate diagnostic plots:
  1. Per-class AUC ROC curves
  2. Confusion matrices (one per class)
  3. Training curves (from TensorBoard logs)

Usage:
    python visualize_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from config import CFG
from src.data.dataset import PTBXLDataset, SUPERCLASSES
from src.models.resnet1d import build_model


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def get_predictions(model, loader, device):
    all_proba, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals.to(device))
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_proba), np.concatenate(all_labels)


def plot_roc(y_true, y_proba, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, color) in enumerate(zip(SUPERCLASSES, COLORS)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves – Test Set")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"ROC curves → {path}")


def plot_confusion(y_true, y_proba, threshold, out_dir):
    y_pred = (y_proba >= threshold).astype(int)
    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(4 * len(SUPERCLASSES), 4))
    for i, (cls, ax) in enumerate(zip(SUPERCLASSES, axes)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(cls)
    fig.tight_layout()
    path = out_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrices → {path}")


def main():
    out_dir = Path(CFG.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt = torch.load("./checkpoints/best_model.pt", map_location=device)
    threshold = ckpt.get("threshold", CFG.threshold)

    model = build_model(num_classes=CFG.num_classes, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = PTBXLDataset(CFG.ptbxl_root, split="test",
                            sampling_rate=CFG.sampling_rate, normalize=True)
    loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    print("Running inference on test set …")
    y_proba, y_true = get_predictions(model, loader, device)

    plot_roc(y_true, y_proba, out_dir)
    plot_confusion(y_true, y_proba, threshold, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
