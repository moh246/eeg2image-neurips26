"""Matching input channels to the montage a model was trained on, by name.

The platform hands the model ``meta["ch_names"]`` in whatever order the
evaluation data uses. A model must never assume that order matches training,
so every packaged model routes its input through :class:`ChannelAligner`.

This module is inlined into ``submission.py``: it may import only the standard
library and torch.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def canonical_channel_name(name: str) -> str:
    """Normalise a channel label for matching: ``"EEG Fp1-REF"`` -> ``"fp1"``."""
    text = str(name).strip()
    if text.upper().startswith("EEG "):
        text = text[4:]
    return text.split("-", 1)[0].strip().lower()


class ChannelAligner(nn.Module):
    """Reorder ``(B, C_source, T)`` input into a model's expected channel order.

    Channels are matched by canonical name. A channel the model expects but
    the input lacks is filled with zeros, which is the post-scaling mean, and
    is listed in :attr:`missing` so callers can decide whether that is
    acceptable. Extra input channels are dropped.
    """

    def __init__(self, source_names: Sequence[str], target_names: Sequence[str]) -> None:
        super().__init__()
        source = [canonical_channel_name(name) for name in source_names]
        if len(set(source)) != len(source):
            raise ValueError("input channel names are not unique after normalisation")
        lookup = {name: position for position, name in enumerate(source)}
        index = [lookup.get(canonical_channel_name(name), -1) for name in target_names]
        self.missing: tuple[str, ...] = tuple(
            str(name) for name, position in zip(target_names, index, strict=True) if position < 0
        )
        self.n_source = len(source)
        self.identity = index == list(range(len(source)))
        self.register_buffer("index", torch.tensor(index, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_source:
            raise ValueError(f"expected {self.n_source} input channels, got {x.shape[1]}")
        if self.identity:
            return x
        present = self.index >= 0
        out = x[:, self.index.clamp(min=0), :]
        if not bool(present.all()):
            out = out * present.to(out.dtype).view(1, -1, 1)
        return out
