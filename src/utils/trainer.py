"""Training and validation loop utilities."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import compute_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for signals, labels in tqdm(loader, desc="  train", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(signals)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(signals)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_proba, all_labels = [], []

    for signals, labels in tqdm(loader, desc="  eval ", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(signals)

        proba = torch.sigmoid(logits).cpu().numpy()
        all_proba.append(proba)
        all_labels.append(labels.cpu().numpy())

    all_proba = np.concatenate(all_proba)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_proba)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
