"""
Training entry-point.

Usage:
    python train.py
    python train.py --ptbxl_root /path/to/ptbxl --epochs 80 --batch_size 32
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import CFG
from src.data.dataset import PTBXLDataset, SUPERCLASSES, get_class_weights
from src.models.resnet1d import build_model
from src.utils.trainer import train_one_epoch, evaluate, EarlyStopping
from src.utils.metrics import find_best_threshold, print_report


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ptbxl_root", type=str, default=CFG.ptbxl_root)
    p.add_argument("--epochs", type=int, default=CFG.epochs)
    p.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p.add_argument("--lr", type=float, default=CFG.learning_rate)
    p.add_argument("--dropout", type=float, default=CFG.dropout)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(CFG.seed)

    # ── Paths ─────────────────────────────────────────────────────────────────
    ckpt_dir = Path(CFG.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(CFG.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    use_amp = CFG.use_amp and device.type == "cuda"

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = PTBXLDataset(args.ptbxl_root, split="train",
                             sampling_rate=CFG.sampling_rate,
                             normalize=CFG.normalize,
                             augment=CFG.augment_train)
    val_ds   = PTBXLDataset(args.ptbxl_root, split="val",
                             sampling_rate=CFG.sampling_rate,
                             normalize=CFG.normalize)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=CFG.num_workers,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=CFG.num_workers,
                               pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=CFG.num_classes, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Loss (weighted BCE for class imbalance) ────────────────────────────────
    pos_weights = torch.tensor(get_class_weights(train_ds)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=CFG.weight_decay)
    if CFG.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    early_stop = EarlyStopping(patience=CFG.patience)
    writer = SummaryWriter(log_dir=CFG.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_ckpt = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_f1  = val_metrics["macro_f1"]
        val_auc = val_metrics["macro_auc"]

        if CFG.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"  macro_f1={val_f1:.4f}  macro_auc={val_auc:.4f}"
              f"  lr={current_lr:.2e}")

        writer.add_scalars("loss",   {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("metric", {"macro_f1": val_f1, "macro_auc": val_auc}, epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_auc": val_auc,
            }, best_ckpt)
            print(f"  ✓ New best checkpoint saved (macro_f1={val_f1:.4f})")

        if early_stop(val_f1):
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    writer.close()
    print(f"\nTraining complete. Best val macro F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {best_ckpt}")

    # ── Post-training: find optimal threshold on val set ──────────────────────
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    all_proba, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for signals, labels in val_loader:
            logits = model(signals.to(device))
            all_proba.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())

    all_proba  = np.concatenate(all_proba)
    all_labels = np.concatenate(all_labels)
    best_thresh = find_best_threshold(all_labels, all_proba)
    print(f"\nOptimal threshold (val): {best_thresh:.2f}")

    # Save threshold alongside checkpoint
    torch.save({**checkpoint, "threshold": best_thresh}, best_ckpt)
    print(f"Threshold saved to checkpoint.")


if __name__ == "__main__":
    main()
