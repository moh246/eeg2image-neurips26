"""Model registry.

Every model maps ``(B, C, T)`` EEG to ``(B, n_outputs)`` embeddings and is
constructed as ``Model(n_chans=..., n_times=..., n_outputs=..., **params)``.
Models that use electrode positions also take ``ch_names``.

This module is inlined into ``submission.py``: it may import only the standard
library, torch and other inlined ``eeg2img`` modules.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from torch import nn

from eeg2img.models.conv import ConvEncoder, EEGNet
from eeg2img.models.floor import MeanEmbedding, RidgeProbe
from eeg2img.models.transformer import ChannelTransformer, ConvTransformer

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "channel_transformer": ChannelTransformer,
    "conv": ConvEncoder,
    "conv_transformer": ConvTransformer,
    "eegnet": EEGNet,
    "mean": MeanEmbedding,
    "ridge": RidgeProbe,
}


def build_model(
    name: str,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    ch_names: Sequence[str] | None = None,
    **params,
) -> nn.Module:
    """Instantiate the registered model ``name``.

    ``ch_names`` is forwarded only to models whose constructor accepts it.
    """
    if name not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model {name!r}; registered models: {known}")
    model_class = MODEL_REGISTRY[name]
    if "ch_names" in inspect.signature(model_class.__init__).parameters:
        if ch_names is None:
            raise ValueError(f"model {name!r} needs channel names")
        params["ch_names"] = list(ch_names)
    return model_class(n_chans=n_chans, n_times=n_times, n_outputs=n_outputs, **params)
