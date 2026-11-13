"""Batched inference and scoring with the competition metric."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eeg2img.data.epochs import EpochSet
from eeg2img.metrics import score_predictions_indexed


@torch.inference_mode()
def predict(
    model: nn.Module, eeg: np.ndarray, batch_size: int = 256, device: str | torch.device = "cpu"
) -> np.ndarray:
    """Run ``model`` over ``(N, C, T)`` epochs and return ``(N, D)`` float32 predictions."""
    model = model.to(device).eval()
    outputs = []
    for start in range(0, len(eeg), batch_size):
        batch = torch.as_tensor(eeg[start : start + batch_size], dtype=torch.float32)
        outputs.append(model(batch.to(device)).float().cpu().numpy())
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs)


def evaluate(
    model: nn.Module,
    epochs: EpochSet,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    k: int = 5,
) -> dict[str, float | int]:
    """Score ``model`` on ``epochs`` with the sealed-phase retrieval metric.

    The gallery is every image that occurs in ``epochs``, as on the platform.
    """
    y_pred = predict(model, epochs.eeg, batch_size=batch_size, device=device)
    return score_predictions_indexed(
        y_pred, epochs.image_index, epochs.embeddings, epochs.subject, k=k
    )
