"""Transformer EEG encoders.

Two models, each isolating one question:

- :class:`ConvTransformer` asks whether a sequence transformer over EEG latent
  tokens beats a plain CNN head. It reuses the exact tokenizer of
  ``ConvEncoder`` and replaces only the flatten-and-project head with a
  transformer, so the two differ in one component. ``attention="causal"``
  makes it read the tokens left to right and predict from the latest one, as
  a language-model-style decoder would; ``"bidirectional"`` attends freely.
- :class:`ChannelTransformer` asks whether a montage-agnostic encoder can use
  data recorded on other electrode layouts. Every (channel, time-patch) is a
  token that carries an embedding of its electrode's 3-D position, so no
  weight depends on the number or order of channels and the same checkpoint
  runs on a 63-channel or a 32-channel montage.

This module is inlined into ``submission.py``: it may import only the standard
library, torch and other inlined ``eeg2img`` modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from eeg2img.channels import canonical_channel_name
from eeg2img.models.conv import ConvTokenizer, ProjectionHead
from eeg2img.montage import ELECTRODE_POSITIONS


def _encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, activation="gelu",
        batch_first=True, norm_first=True,
    )  # fmt: skip
    return nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d_model),
                                 enable_nested_tensor=False)  # fmt: skip


class ConvTransformer(nn.Module):
    """Conv tokens -> transformer over the token sequence -> projection."""

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
        n_layers: int = 4,
        n_heads: int = 4,
        attention: str = "bidirectional",
        attn_dropout: float = 0.1,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if attention not in ("bidirectional", "causal"):
            raise ValueError(f"attention must be 'bidirectional' or 'causal', got {attention!r}")
        self.causal = attention == "causal"
        self.tokenizer = ConvTokenizer(n_chans, n_times, d_model, n_filters, temporal_kernel,
                                       pool_kernel, pool_stride, dropout)  # fmt: skip
        n_tokens = self.tokenizer.n_tokens
        self.time_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        self.encoder = _encoder(d_model, n_heads, n_layers, attn_dropout)
        mask = torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.head = ProjectionHead(d_model, n_outputs, head_dropout)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Contextualised tokens ``(B, L, d_model)``."""
        tokens = self.tokenizer(x) + self.time_embedding
        return self.encoder(tokens, mask=self.causal_mask if self.causal else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(x)
        # Causal: the latest token has seen the whole epoch. Bidirectional: average.
        pooled = tokens[:, -1] if self.causal else tokens.mean(dim=1)
        return self.head(pooled)


class ElectrodeEmbedding(nn.Module):
    """Embed electrodes from their 3-D scalp positions with Fourier features."""

    def __init__(self, d_model: int, n_frequencies: int = 8) -> None:
        super().__init__()
        frequencies = math.pi * 2.0 ** torch.arange(n_frequencies, dtype=torch.float32) / 4.0
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.project = nn.Sequential(
            nn.Linear(3 * 2 * n_frequencies, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """``(C, 3)`` positions -> ``(C, d_model)`` embeddings."""
        angles = positions[:, :, None] * self.frequencies
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(start_dim=1)
        return self.project(features)


def electrode_positions(ch_names: Sequence[str]) -> torch.Tensor:
    """``(C, 3)`` standard positions of ``ch_names``; raises on an unknown electrode."""
    keys = [canonical_channel_name(name) for name in ch_names]
    unknown = [str(name) for name, key in zip(ch_names, keys, strict=True)
               if key not in ELECTRODE_POSITIONS]  # fmt: skip
    if unknown:
        raise KeyError(f"no standard position for electrodes: {', '.join(unknown)}")
    return torch.tensor([ELECTRODE_POSITIONS[key] for key in keys], dtype=torch.float32)


class ChannelTransformer(nn.Module):
    """Montage-agnostic transformer over (electrode, time-patch) tokens.

    One temporal convolution, shared by all channels, cuts each channel into
    ``n_times // patch_size`` patches. Each patch token is the sum of its
    signal embedding, its electrode's position embedding and a time embedding.
    Because channels enter only through their positions, :meth:`set_channels`
    can retarget a trained model to any set of standard electrodes.
    """

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        ch_names: Sequence[str],
        d_model: int = 128,
        patch_size: int = 15,
        n_layers: int = 4,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        token_dropout: float = 0.0,
        head_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if len(ch_names) != n_chans:
            raise ValueError(f"{len(ch_names)} channel names for {n_chans} channels")
        self.patch = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.n_patches = n_times // patch_size
        self.electrode = ElectrodeEmbedding(d_model)
        self.time_embedding = nn.Parameter(torch.zeros(1, 1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        self.token_dropout = token_dropout
        self.encoder = _encoder(d_model, n_heads, n_layers, attn_dropout)
        self.head = ProjectionHead(d_model, n_outputs, head_dropout)
        self.register_buffer("positions", electrode_positions(ch_names), persistent=False)

    def set_channels(self, ch_names: Sequence[str]) -> None:
        """Retarget the model to input channels ``ch_names`` (any standard electrodes)."""
        self.positions = electrode_positions(ch_names).to(self.positions.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_chans, _ = x.shape
        if n_chans != self.positions.shape[0]:
            raise ValueError(f"expected {self.positions.shape[0]} channels, got {n_chans}")
        tokens = self.patch(x.reshape(batch * n_chans, 1, -1)).transpose(1, 2)
        tokens = tokens.reshape(batch, n_chans, self.n_patches, -1)
        tokens = tokens + self.electrode(self.positions)[None, :, None, :] + self.time_embedding
        tokens = tokens.flatten(start_dim=1, end_dim=2)  # (B, C * P, d)
        if self.training and self.token_dropout > 0:
            # Drop whole tokens, which also trains the model to cope with missing electrodes.
            n_keep = max(1, round(tokens.shape[1] * (1.0 - self.token_dropout)))
            keep = torch.rand(batch, tokens.shape[1], device=x.device).argsort(dim=1)[:, :n_keep]
            tokens = tokens.gather(1, keep[:, :, None].expand(-1, -1, tokens.shape[2]))
        return self.head(self.encoder(tokens).mean(dim=1))
