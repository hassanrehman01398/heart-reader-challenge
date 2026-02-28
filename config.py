"""Central configuration — edit this file to tune hyperparameters."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    ptbxl_root: str = "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"

    # ── Data ──────────────────────────────────────────────────────────────────
    sampling_rate: int = 100         # 100 Hz → 1000 samples per recording
    normalize: bool = True
    augment_train: bool = True

    # ── Model ─────────────────────────────────────────────────────────────────
    base_filters: int = 64
    dropout: float = 0.3
    num_classes: int = 5

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"     # "cosine" | "plateau"
    patience: int = 15               # early stopping patience
    use_amp: bool = True             # mixed-precision (GPU only)

    # ── Evaluation ────────────────────────────────────────────────────────────
    threshold: float = 0.5           # refined after val search

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42


CFG = Config()
