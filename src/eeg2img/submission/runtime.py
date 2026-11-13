"""Inference wrapper that turns a trained model into a platform-ready one.

The platform builds the model from a ``meta`` dict and then calls
``predict(X)`` with ``(B, C, T)`` batches whose size, order and participant
mix are unspecified. :class:`PackagedModel` owns everything between that call
and the network: channel matching by name, time-axis matching, and output
normalisation. It keeps no state across calls.

This module is inlined into ``submission.py``: it may import only the standard
library, torch and other inlined ``eeg2img`` modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from eeg2img.channels import ChannelAligner
from eeg2img.models import build_model

PACKAGE_CONFIG = "config.json"
PACKAGE_WEIGHTS = "weights.pt"


class PackagedModel(nn.Module):
    """A trained network plus the input/output adaptation the contract needs."""

    def __init__(
        self,
        network: nn.Module,
        aligner: ChannelAligner,
        n_times: int,
        normalize_output: bool,
    ) -> None:
        super().__init__()
        self.network = network
        self.aligner = aligner
        self.n_times = n_times
        self.normalize_output = normalize_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aligner(x.to(torch.float32))
        if x.shape[-1] != self.n_times:
            x = F.interpolate(x, size=self.n_times, mode="linear", align_corners=False)
        out = self.network(x)
        if self.normalize_output:
            # The scorer averages raw predictions over repeats before normalising,
            # so unit-norm outputs give every repeat the same weight.
            out = F.normalize(out, dim=-1)
        return out.to(torch.float32)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` EEG on any device -> ``(B, D)`` float32 embeddings."""
        self.eval()
        return self(x)


def load_packaged_model(meta: dict) -> PackagedModel:
    """Rebuild a packaged model from ``meta["submission_dir"]`` for inference."""
    submission_dir = Path(meta["submission_dir"])
    package = json.loads((submission_dir / PACKAGE_CONFIG).read_text(encoding="utf-8"))
    device = meta.get("device", "cpu")

    n_outputs = int(package["n_outputs"])
    if "n_outputs" in meta and int(meta["n_outputs"]) != n_outputs:
        raise ValueError(
            f"model predicts {n_outputs}-d embeddings but the task expects {meta['n_outputs']}"
        )

    trained_names = list(package["ch_names"])
    input_names = meta.get("ch_names")
    if input_names is None:
        if int(meta.get("n_chans", len(trained_names))) != len(trained_names):
            raise ValueError("input has no channel names and a different channel count")
        input_names = trained_names
    aligner = ChannelAligner(list(input_names), trained_names)

    network = build_model(
        package["model"]["name"],
        n_chans=len(trained_names),
        n_times=int(package["n_times"]),
        n_outputs=n_outputs,
        **package["model"].get("params", {}),
    )
    state = torch.load(submission_dir / PACKAGE_WEIGHTS, map_location=device, weights_only=True)
    network.load_state_dict(state)

    model = PackagedModel(
        network,
        aligner,
        n_times=int(package["n_times"]),
        normalize_output=bool(package.get("normalize_output", True)),
    )
    return model.to(device).eval()
