"""Reference floors: a constant predictor and a linear ridge probe.

They carry no architectural ideas. They exist to exercise the pipeline end to
end and to anchor every later result: a model that cannot beat the ridge probe
is not using the EEG signal better than a linear map.

This module is inlined into ``submission.py``: it may import only the standard
library and torch.
"""

from __future__ import annotations

import torch
from torch import nn


class MeanEmbedding(nn.Module):
    """Predict the same embedding for every epoch (retrieval chance level)."""

    def __init__(self, n_chans: int, n_times: int, n_outputs: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean.to(x.dtype).expand(x.shape[0], -1)


class RidgeProbe(nn.Module):
    """Linear map from the flattened ``(C, T)`` epoch to the embedding space."""

    def __init__(self, n_chans: int, n_times: int, n_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_chans * n_times, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(start_dim=1))
