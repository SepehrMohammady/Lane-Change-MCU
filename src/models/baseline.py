"""Baseline model: small depthwise-separable 1D CNN.

Design goals: MCU-friendly operator set (Conv1d, BatchNorm, ReLU, global
average pooling, Linear — all mapped by ST Edge AI), small footprint,
fast to train. This is the reference point the NAS search must beat on
the accuracy/footprint trade-off; the external SOTA numbers to beat are
in paper/NOTES.md.

Input:  (batch, 31 channels, 50 timesteps)
Output: 3 logits (classification) or 1 value (regression)
"""
from __future__ import annotations

import torch
from torch import nn


class DSConvBlock(nn.Module):
    """Depthwise-separable Conv1d -> BN -> ReLU."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel, stride=stride,
                      padding=kernel // 2, groups=c_in, bias=False),
            nn.Conv1d(c_in, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineDSCNN(nn.Module):
    def __init__(self, n_features: int = 31, n_outputs: int = 3,
                 widths: tuple[int, ...] = (32, 48, 64), dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, widths[0], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            blocks.append(DSConvBlock(c_in, c_out, kernel=5, stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
