"""Convolutional EEG encoders.

- :class:`EEGNet` is the compact CNN the organisers use as their reference
  baseline (Lawhern et al., 2018), reimplemented here so that the submission
  stays a single torch-only file.
- :class:`ConvTokenizer` turns an epoch into a short sequence of latent
  tokens. It is shared by the CNN baseline and by the transformer models, so
  that comparing heads later changes exactly one component.
- :class:`ConvEncoder` is the CNN + contrastive baseline: tokenizer, then a
  flatten-and-project head.

This module is inlined into ``submission.py``: it may import only the standard
library and torch.
"""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """EEGNet-v4 with a linear read-out into the embedding space."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        f1: int = 8,
        depth: int = 2,
        f2: int = 16,
        kernel_length: int = 60,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth,
                      bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )  # fmt: skip
        with torch.no_grad():
            n_features = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_times))).numel()
        self.readout = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(x.unsqueeze(1)))
        return self.readout(x.flatten(start_dim=1))


class ConvTokenizer(nn.Module):
    """Temporal then spatial convolution: ``(B, C, T)`` -> ``(B, L, d_model)`` tokens.

    A temporal filter bank is applied to every channel, a spatial convolution
    mixes the channels, and average pooling along time yields ``L`` tokens,
    each summarising ``pool_kernel`` samples (about 125 ms at the defaults).
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        d_model: int = 128,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, temporal_kernel),
                                  padding=(0, temporal_kernel // 2))  # fmt: skip
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_chans, 1), bias=False)
        self.norm = nn.BatchNorm2d(n_filters)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Conv2d(n_filters, d_model, 1)
        with torch.no_grad():
            self.n_tokens = self._features(torch.zeros(1, n_chans, n_times)).shape[1]
        self.d_model = d_model

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x.unsqueeze(1))
        x = self.act(self.norm(self.spatial(x)))
        x = self.project(self.dropout(self.pool(x)))
        return x.squeeze(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._features(x)


class ProjectionHead(nn.Module):
    """Linear projection followed by a residual MLP block and layer norm."""

    def __init__(self, in_features: int, n_outputs: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, n_outputs)
        self.residual = nn.Sequential(nn.GELU(), nn.Linear(n_outputs, n_outputs),
                                      nn.Dropout(dropout))  # fmt: skip
        self.norm = nn.LayerNorm(n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.norm(x + self.residual(x))


class ConvEncoder(nn.Module):
    """CNN + contrastive baseline: conv tokens, flattened, projected to the target space."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        d_model: int = 128,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        dropout: float = 0.25,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.tokenizer = ConvTokenizer(n_chans, n_times, d_model, n_filters, temporal_kernel,
                                       pool_kernel, pool_stride, dropout)  # fmt: skip
        self.head = ProjectionHead(self.tokenizer.n_tokens * d_model, n_outputs, head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tokenizer(x).flatten(start_dim=1))
