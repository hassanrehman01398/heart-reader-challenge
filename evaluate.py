"""
Evaluate the best checkpoint on the held-out test set (fold 10).

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt --ptbxl_root ./data/ptbxl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from src.data.dataset import PTBXLDataset, SUPERCLASSES
from src.models.resnet1d import build_model
from src.utils.metrics import compute_metrics, print_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="./checkpoints/best_model.pt")
    p.add_argument("--ptbxl_root",   default=CFG.ptbxl_root)
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--threshold",    type=float, default=None,
                   help="Override threshold from checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    threshold = args.threshold or ckpt.get("threshold", CFG.threshold)
    print(f"Checkpoint from epoch {ckpt['epoch']}, val macro F1: {ckpt['val_f1']:.4f}")
    print(f"Using threshold: {threshold:.2f}")

    model = build_model(num_classes=CFG.num_classes, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_ds = PTBXLDataset(args.ptbxl_root, split="test",
                            sampling_rate=CFG.sampling_rate,
                            normalize=CFG.normalize)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=CFG.num_workers if hasattr(CFG, "num_workers") else 4,
                              pin_memory=True)
    print(f"Test samples: {len(test_ds):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_proba, all_labels = [], []
    with torch.no_grad():
        for signals, labels in test_loader:
            logits = model(signals.to(device))
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    all_proba  = np.concatenate(all_proba)
    all_labels = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_labels, all_proba, threshold=threshold)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Macro F1-Score : {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC      : {metrics['macro_auc']:.4f}")
    print("\nPer-class breakdown:")
    for i, cls in enumerate(SUPERCLASSES):
        print(f"  {cls:<6}  F1={metrics['per_class_f1'][i]:.4f}  AUC={metrics['per_class_auc'][i]:.4f}")

    print("\nClassification Report:")
    print_report(all_labels, all_proba, SUPERCLASSES, threshold)

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = Path(CFG.results_dir) / "test_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"threshold": threshold, **metrics}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
