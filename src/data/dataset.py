"""
PTB-XL Dataset loader.

Loads 12-lead ECG signals at 100 Hz (1000 samples per recording) and maps
diagnostic SCP codes to the 5 superclasses required by the challenge.

Fold assignment (from ptbxl_database.csv column `strat_fold`):
  Train : folds 1-8
  Val   : fold  9
  Test  : fold  10
"""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset

# ── Superclass mapping ────────────────────────────────────────────────────────
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SC_INDEX = {sc: i for i, sc in enumerate(SUPERCLASSES)}


def load_scp_map(ptbxl_root: str) -> dict[str, str]:
    """Return {scp_code -> superclass} using scp_statements.csv."""
    path = Path(ptbxl_root) / "scp_statements.csv"
    df = pd.read_csv(path, index_col=0)
    mapping = {}
    for code, row in df.iterrows():
        sc = str(row.get("diagnostic_class", "")).strip()
        if sc in SC_INDEX:
            mapping[str(code)] = sc
    return mapping


def load_metadata(ptbxl_root: str) -> pd.DataFrame:
    """Load ptbxl_database.csv with parsed scp_codes column."""
    path = Path(ptbxl_root) / "ptbxl_database.csv"
    df = pd.read_csv(path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def build_label_vector(scp_codes: dict, scp_map: dict[str, str]) -> np.ndarray:
    """Convert a scp_codes dict to a 5-dim binary label vector."""
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in scp_codes:
        sc = scp_map.get(code)
        if sc is not None:
            vec[SC_INDEX[sc]] = 1.0
    return vec


class PTBXLDataset(Dataset):
    """
    Parameters
    ----------
    ptbxl_root : str
        Root directory of the PTB-XL dataset (contains ptbxl_database.csv).
    split : str
        One of 'train', 'val', 'test'.
    sampling_rate : int
        100 (default) → uses the records/lr/ folder.
    normalize : bool
        Z-score normalise each recording.
    augment : bool
        Apply random noise + scaling augmentation (train split only).
    """

    def __init__(
        self,
        ptbxl_root: str,
        split: str = "train",
        sampling_rate: int = 100,
        normalize: bool = True,
        augment: bool = False,
    ):
        self.root = Path(ptbxl_root)
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        self.augment = augment

        self.scp_map = load_scp_map(ptbxl_root)
        meta = load_metadata(ptbxl_root)

        if split == "train":
            self.df = meta[meta["strat_fold"] <= 8].copy()
        elif split == "val":
            self.df = meta[meta["strat_fold"] == 9].copy()
        elif split == "test":
            self.df = meta[meta["strat_fold"] == 10].copy()
        else:
            raise ValueError(f"Unknown split '{split}'")

        # Pre-compute label matrix
        self.labels = np.stack(
            [build_label_vector(row["scp_codes"], self.scp_map) for _, row in self.df.iterrows()]
        )

        # Choose filename column depending on sampling rate
        self.fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_signal(self, fname: str) -> np.ndarray:
        """Load a WFDB record. Returns array of shape (12, T)."""
        record_path = str(self.root / fname)
        record = wfdb.rdrecord(record_path)
        sig = record.p_signal  # (T, 12)
        sig = np.nan_to_num(sig, nan=0.0).T  # (12, T)
        return sig.astype(np.float32)

    def _normalise(self, sig: np.ndarray) -> np.ndarray:
        """Z-score per lead."""
        mean = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-8
        return (sig - mean) / std

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        """Random Gaussian noise + amplitude scaling."""
        if np.random.rand() < 0.5:
            sig = sig + np.random.randn(*sig.shape).astype(np.float32) * 0.01
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            sig = sig * scale
        return sig

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sig = self._load_signal(row[self.fname_col])
        if self.normalize:
            sig = self._normalise(sig)
        if self.augment:
            sig = self._augment(sig)
        label = self.labels[idx]
        return sig, label


def get_class_weights(dataset: PTBXLDataset) -> np.ndarray:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.
    pos_weight[i] = (N - n_pos_i) / n_pos_i
    """
    labels = dataset.labels
    n_pos = labels.sum(axis=0).clip(min=1)
    n_neg = len(labels) - n_pos
    return (n_neg / n_pos).astype(np.float32)
