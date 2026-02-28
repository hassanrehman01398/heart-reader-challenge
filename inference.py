"""
Single-recording inference demo.

Usage:
    python inference.py --record_path data/ptbxl/records100/00000/00001_lr
"""

import argparse

import numpy as np
import torch
import wfdb

from config import CFG
from src.data.dataset import SUPERCLASSES
from src.models.resnet1d import build_model


def load_and_preprocess(record_path: str, normalize: bool = True) -> torch.Tensor:
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal  # (T, 12)
    sig = np.nan_to_num(sig, nan=0.0).T.astype(np.float32)  # (12, T)
    if normalize:
        mean = sig.mean(axis=1, keepdims=True)
        std  = sig.std(axis=1,  keepdims=True) + 1e-8
        sig  = (sig - mean) / std
    return torch.tensor(sig).unsqueeze(0)  # (1, 12, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record_path", required=True)
    p.add_argument("--checkpoint",  default="./checkpoints/best_model.pt")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    thresh = ckpt.get("threshold", CFG.threshold)

    model = build_model(num_classes=CFG.num_classes, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    signal = load_and_preprocess(args.record_path).to(device)

    with torch.no_grad():
        logits = model(signal)
        proba  = torch.sigmoid(logits).squeeze().numpy()

    print("\n── ECG Diagnostic Prediction ──")
    for cls, p in zip(SUPERCLASSES, proba):
        flag = "✓" if p >= thresh else " "
        print(f"  [{flag}] {cls:<6}  {p:.3f}")
    print(f"\nThreshold: {thresh:.2f}")


if __name__ == "__main__":
    main()
