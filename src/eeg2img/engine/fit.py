"""Closed-form fitting of the reference floors.

Gradient-trained models go through :mod:`eeg2img.engine.train` instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from eeg2img.data.epochs import EpochSet
from eeg2img.models.floor import MeanEmbedding, RidgeProbe

#: Registry names of the models fitted here rather than by gradient descent.
CLOSED_FORM_MODELS = frozenset({"mean", "ridge"})


def fit_mean(model: MeanEmbedding, train: EpochSet) -> dict[str, float]:
    """Set the constant prediction to the mean training target."""
    mean = torch.as_tensor(train.targets.mean(axis=0), dtype=torch.float32)
    with torch.no_grad():
        model.mean.copy_(mean)
    return {}


def fit_ridge(
    model: RidgeProbe, train: EpochSet, alpha: float, batch_size: int = 4096
) -> dict[str, float]:
    """Solve ridge regression in closed form and load it into ``model``.

    The normal equations are accumulated in float64 over batches, so memory is
    bounded by the ``(C*T)^2`` Gram matrix rather than by the dataset size.
    ``alpha`` is the L2 penalty on the weights; the intercept is not penalised.
    """
    n_features = train.n_chans * train.n_times
    n = len(train)
    if n == 0:
        raise ValueError("cannot fit ridge on an empty training set")

    x_mean = np.zeros(n_features)
    for start in range(0, n, batch_size):
        x_mean += train.eeg[start : start + batch_size].reshape(-1, n_features).sum(axis=0)
    x_mean /= n
    y_mean = train.targets.mean(axis=0, dtype=np.float64)

    gram = torch.zeros((n_features, n_features), dtype=torch.float64)
    cross = torch.zeros((n_features, train.embed_dim), dtype=torch.float64)
    for start in range(0, n, batch_size):
        stop = start + batch_size
        x = torch.as_tensor(train.eeg[start:stop].reshape(-1, n_features), dtype=torch.float64)
        x = x - torch.as_tensor(x_mean)
        y = torch.as_tensor(train.embeddings[train.image_index[start:stop]], dtype=torch.float64)
        y = y - torch.as_tensor(y_mean)
        gram += x.T @ x
        cross += x.T @ y

    gram.diagonal().add_(alpha)
    weight = torch.linalg.solve(gram, cross)  # (F, D)
    bias = torch.as_tensor(y_mean) - torch.as_tensor(x_mean) @ weight
    with torch.no_grad():
        model.linear.weight.copy_(weight.T.float())
        model.linear.bias.copy_(bias.float())
    return {"ridge_alpha": float(alpha)}


def fit_model(model: nn.Module, train: EpochSet, fit_config: Mapping[str, Any]) -> dict[str, float]:
    """Fit ``model`` on ``train`` according to ``fit_config``; return fit statistics."""
    if isinstance(model, MeanEmbedding):
        return fit_mean(model, train)
    if isinstance(model, RidgeProbe):
        return fit_ridge(model, train, alpha=float(fit_config["alpha"]))
    raise NotImplementedError(f"no fitting routine for {type(model).__name__}")
