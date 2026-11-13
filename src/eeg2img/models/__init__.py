"""Model registry.

Every model maps ``(B, C, T)`` EEG to ``(B, n_outputs)`` embeddings and is
constructed as ``Model(n_chans=..., n_times=..., n_outputs=..., **params)``.

This module is inlined into ``submission.py``: it may import only the standard
library, torch and other inlined ``eeg2img`` modules.
"""

from __future__ import annotations

from torch import nn

from eeg2img.models.floor import MeanEmbedding, RidgeProbe

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mean": MeanEmbedding,
    "ridge": RidgeProbe,
}


def build_model(name: str, n_chans: int, n_times: int, n_outputs: int, **params) -> nn.Module:
    """Instantiate the registered model ``name``."""
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model {name!r}; registered models: {known}")
    return MODEL_REGISTRY[name](n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, **params)
