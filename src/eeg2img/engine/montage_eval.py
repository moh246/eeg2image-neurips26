"""Score a packaged model on inputs that have only some of its training electrodes.

This is the montage-mismatch test available without a second dataset: the
model is rebuilt through the platform code path (``load_packaged_model``) for
an input that carries a subset of the channels, exactly as it would be for a
recording from a sparser cap. A model with a fixed channel layout zero-fills
the missing electrodes; a montage-agnostic model drops their tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

from eeg2img.data.source import ArraySource
from eeg2img.engine.packages import predict_package, score


def channel_subset(n_chans: int, keep: int, seed: int) -> list[int]:
    """A seeded random choice of ``keep`` channel indices, in montage order."""
    if not 0 < keep <= n_chans:
        raise ValueError(f"keep must be in 1..{n_chans}, got {keep}")
    generator = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(n_chans, generator=generator)[:keep].tolist())


def score_with_channels(
    package_dir: str | Path,
    source: ArraySource,
    indices: torch.Tensor,
    channels: Sequence[int],
    batch_size: int = 1024,
    k: int = 5,
) -> dict[str, float | int]:
    """Competition scores when the input holds only ``channels`` of the montage."""
    predictions = predict_package(package_dir, source, indices, channels, batch_size)
    return score(predictions, source, indices, k=k)


def evaluate_channel_subsets(
    package_dir: str | Path,
    source: ArraySource,
    indices: torch.Tensor,
    keep_counts: Sequence[int],
    draws: int = 3,
    seed: int = 0,
    batch_size: int = 1024,
    metric: str = "top5_acc_subject_agg",
) -> list[dict[str, Any]]:
    """``metric`` for each number of kept channels, over ``draws`` random subsets.

    Keeping every channel has only one subset, so it is scored once.
    """
    rows = []
    for keep in keep_counts:
        n_draws = 1 if keep == source.n_chans else draws
        subsets = [channel_subset(source.n_chans, keep, seed + draw) for draw in range(n_draws)]
        values = [
            float(score_with_channels(package_dir, source, indices, subset, batch_size)[metric])
            for subset in subsets
        ]
        rows.append(
            {
                "n_channels": keep,
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "values": values,
                "subsets": [[source.ch_names[i] for i in subset] for subset in subsets],
            }
        )
    return rows
