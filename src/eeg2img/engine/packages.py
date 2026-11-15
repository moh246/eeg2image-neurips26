"""Predict and score with packaged models (the artefacts a run leaves behind).

Everything here goes through ``load_packaged_model``, the platform code path,
so what is scored is the packaged artefact and not an in-memory model.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from eeg2img.data.source import ArraySource
from eeg2img.metrics import score_predictions_indexed
from eeg2img.submission.runtime import load_packaged_model


def predict_package(
    package_dir: str | Path,
    source: ArraySource,
    indices: torch.Tensor,
    channels: Sequence[int] | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Predictions ``(N, D)`` of the package for event ``indices``, on the CPU.

    ``channels`` restricts the input to those montage channels, as if the
    recording came from a sparser cap; the package is then built for exactly
    that channel list.
    """
    kept = list(range(source.n_chans)) if channels is None else list(channels)
    names = [source.ch_names[i] for i in kept]
    model = load_packaged_model(
        {
            "submission_dir": str(package_dir),
            "device": str(source.device),
            "ch_names": names,
            "n_chans": len(names),
            "n_times": source.n_times,
            "n_outputs": source.embed_dim,
        }
    )
    select = torch.as_tensor(kept, device=source.device)
    outputs = []
    for start in range(0, len(indices), batch_size):
        batch = source.batch(indices[start : start + batch_size])
        if channels is not None:
            batch = batch.index_select(1, select)
        outputs.append(model.predict(batch).float().cpu())
    return torch.cat(outputs) if outputs else torch.zeros((0, source.embed_dim))


def score(
    predictions: torch.Tensor, source: ArraySource, indices: torch.Tensor, k: int = 5
) -> dict[str, float | int]:
    """Competition scores of ``predictions`` for event ``indices``."""
    return score_predictions_indexed(
        predictions.numpy(),
        source.image[indices].numpy(),
        source.embeddings.cpu().numpy(),
        source.subject[indices].numpy(),
        k=k,
    )


def evaluate_ensemble(
    package_dirs: Sequence[str | Path],
    source: ArraySource,
    indices: torch.Tensor,
    batch_size: int = 1024,
    metric: str = "top5_acc_subject_agg",
) -> dict[str, Any]:
    """Score each package and the mean of their predictions.

    Each packaged model returns unit-norm embeddings, so the mean gives every
    member the same weight; the scorer then averages over repeats as usual.
    """
    if not package_dirs:
        raise ValueError("an ensemble needs at least one package")
    total = torch.zeros((len(indices), source.embed_dim))
    members = []
    for package_dir in package_dirs:
        predictions = predict_package(package_dir, source, indices, batch_size=batch_size)
        members.append(float(score(predictions, source, indices)[metric]))
        total += predictions
    ensemble = score(total / len(package_dirs), source, indices)
    return {"metric": metric, "members": members, "ensemble": float(ensemble[metric]),
            "n_candidates": int(ensemble["n_candidates"]),
            "chance_top5": float(ensemble["chance_top5"])}  # fmt: skip
