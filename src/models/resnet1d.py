"""
ResNet1D – 1-D Residual Network for 12-lead ECG multi-label classification.

Architecture (inspired by Ribeiro et al. 2020 & He et al. 2016):
  - Stem conv
  - 4 × ResBlock with increasing channels and downsampling
  - Global Average Pooling
  - Dropout + FC head → 5 sigmoid outputs

Input  : (B, 12, 1000)   12 leads × 1000 time steps @ 100 Hz
Output : (B, 5)           probability for each diagnostic superclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Basic residual block with two 1-D convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)

        # Skip connection
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ResNet1D(nn.Module):
    """
    Parameters
    ----------
    in_channels  : number of ECG leads (12)
    num_classes  : number of output classes (5 superclasses)
    base_filters : channels after stem (doubled at each stage)
    dropout      : dropout rate in residual blocks and head
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages: [64→128→256→256→512]
        self.layer1 = ResBlock1D(base_filters,      base_filters * 2, stride=2, dropout=dropout)
        self.layer2 = ResBlock1D(base_filters * 2,  base_filters * 4, stride=2, dropout=dropout)
        self.layer3 = ResBlock1D(base_filters * 4,  base_filters * 4, stride=2, dropout=dropout)
        self.layer4 = ResBlock1D(base_filters * 4,  base_filters * 8, stride=2, dropout=dropout)

        # Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.head(x)


def build_model(num_classes: int = 5, dropout: float = 0.3) -> ResNet1D:
    return ResNet1D(in_channels=12, num_classes=num_classes, base_filters=64, dropout=dropout)
