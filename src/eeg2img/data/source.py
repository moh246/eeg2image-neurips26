"""A uniform, device-side view of a dataset for training and evaluation.

A *source* exposes per-event metadata as tensors and produces model-input
batches for arbitrary event indices directly on the training device:

- :class:`ArraySource` wraps an in-memory :class:`EpochSet` (synthetic data);
- :class:`StoreSource` wraps an :class:`EpochStore`. It holds the whole
  float16 signal on the device (3.6 GB for Alljoined-1.6M) and cuts, baselines
  and clips epochs there with vectorised indexing. This avoids DataLoader
  workers entirely, which matters on Colab's two CPU cores.

Both give identical epochs to the NumPy reference path; the tests check it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg2img.data.epochs import EpochSet
from eeg2img.data.store import EpochStore


class ArraySource:
    """Source over an in-memory :class:`EpochSet`."""

    def __init__(self, epochs: EpochSet, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._eeg = torch.as_tensor(epochs.eeg)
        self.image = torch.as_tensor(epochs.image_index)
        self.subject = torch.as_tensor(epochs.subject)
        self.is_test = torch.zeros(len(epochs), dtype=torch.bool)
        self.embeddings = torch.as_tensor(epochs.embeddings).to(self.device)
        #: Integer key of each image, used to hold out images in consecutive groups.
        self.image_key = torch.arange(len(epochs.embeddings))
        self.ch_names = tuple(epochs.ch_names)
        self.sfreq = float(epochs.sfreq)
        self.n_times = epochs.n_times

    def __len__(self) -> int:
        return len(self._eeg)

    @property
    def n_chans(self) -> int:
        return len(self.ch_names)

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` float32 model inputs for event ``indices``, on the device."""
        return self._eeg[indices.cpu()].to(self.device)

    def epochset(self, indices: torch.Tensor | np.ndarray) -> EpochSet:
        """Materialise ``indices`` as an in-memory :class:`EpochSet`."""
        indices = torch.as_tensor(indices, dtype=torch.long).cpu()
        return EpochSet(
            eeg=self.batch(indices).cpu().numpy(),
            image_index=self.image[indices].numpy(),
            subject=self.subject[indices].numpy(),
            embeddings=self.embeddings.cpu().numpy(),
            ch_names=self.ch_names,
            sfreq=self.sfreq,
        )


class StoreSource(ArraySource):
    """Source over an :class:`EpochStore`, cutting epochs on the device."""

    def __init__(
        self,
        store: EpochStore,
        embeddings: np.ndarray,
        device: str | torch.device = "cpu",
        signal_device: str | torch.device | None = None,
    ) -> None:
        if len(embeddings) != len(store.images):
            raise ValueError(f"{len(embeddings)} embeddings for {len(store.images)} images")
        self.device = torch.device(device)
        # The signal may live on another device than the model, e.g. host memory
        # when the GPU is small.
        self.signal_device = torch.device(signal_device) if signal_device else self.device
        self._signal = torch.from_numpy(np.ascontiguousarray(store.eeg)).to(self.signal_device)
        self._plan = torch.as_tensor(store.plan, dtype=torch.long).to(self.signal_device)
        self.image = torch.as_tensor(store.image)
        self.subject = torch.as_tensor(store.subject)
        self.is_test = torch.as_tensor(store.is_test)
        self.embeddings = torch.as_tensor(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        self.ch_names = store.ch_names
        self.sfreq = store.sfreq
        self.n_times = store.n_times
        self.clamp = store.clamp
        self.images = list(store.images)
        # Alljoined stimuli are named by a zero-based numeric id ("00013.jpg"); grouping must
        # use that id, not the position in the list, which shifts when images are missing.
        stems = [Path(name).stem for name in self.images]
        numeric = all(stem.isdigit() for stem in stems)
        self.image_key = torch.tensor(
            [int(stem) for stem in stems] if numeric else list(range(len(stems)))
        )
        self._arange = torch.arange(self.n_times, device=self.signal_device)

    def __len__(self) -> int:
        return len(self._plan)

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        plan = self._plan[indices.to(self.signal_device)]
        src_start, src_len, dst_offset, base_start, base_len = plan.unbind(dim=1)
        last = self._signal.shape[0] - 1

        # Window samples: output position p reads recording sample src_start + p - dst_offset.
        position = self._arange[None, :] - dst_offset[:, None]
        valid = (position >= 0) & (position < src_len[:, None])
        window = self._signal[(src_start[:, None] + position).clamp(0, last)].float()  # (B, T, C)

        # Baseline: mean of base_len samples from base_start (24, or 23 for tie cases).
        n_base = int(base_len.max().item()) if len(base_len) else 0
        if n_base > 0:
            offset = self._arange[None, :n_base]
            base_valid = offset < base_len[:, None]
            base = self._signal[(base_start[:, None] + offset).clamp(0, last)].float()
            base = (base * base_valid[:, :, None]).sum(dim=1) / base_len.clamp(min=1)[:, None]
            window = window - base[:, None, :]

        window = window.clamp(-self.clamp, self.clamp) * valid[:, :, None]
        return window.transpose(1, 2).contiguous().to(self.device)


def build_source(data_config: Mapping[str, Any], device: str | torch.device = "cpu") -> ArraySource:
    """Build the source named by ``data.name``."""
    name = data_config["name"]
    if name == "synthetic":
        from eeg2img.data.synthetic import make_synthetic

        return ArraySource(make_synthetic(**dict(data_config.get("params") or {})), device)
    if name == "alljoined16m":
        store = EpochStore(data_config["store"])
        embeddings = np.load(Path(data_config["embeddings"]))
        return StoreSource(store, embeddings, device, data_config.get("signal_device"))
    raise KeyError(f"unknown dataset {name!r}")


def _choose(values: torch.Tensor, fraction: float, seed: int) -> torch.Tensor:
    """A seeded random ``fraction`` of the distinct ``values`` (at least one)."""
    unique = torch.unique(values)
    generator = torch.Generator().manual_seed(seed)
    count = max(1, round(fraction * len(unique)))
    return unique[torch.randperm(len(unique), generator=generator)[:count]]


def make_splits(source: ArraySource, split_config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Split events so every validation set is new in both subject and image.

    Whole subjects (``holdout_subjects``) and whole image groups are held out,
    mirroring the sealed shift. Events held out on exactly one axis are used by
    neither side. Returns event-index tensors:

    - ``train``: training subjects viewing training images;
    - ``val``: held-out subjects viewing held-out *training-pool* images. This
      is the model-selection set: few repeats and a large gallery, like the
      sealed data;
    - ``val_test`` (only if the data marks test images): held-out subjects
      viewing the dataset's own test images, which have many repeats.

    ``images_per_group`` holds out images in groups of consecutive stimulus ids,
    so that the exemplars of one object concept never straddle the split
    (THINGS-EEG2 orders its training images concept by concept, ten per concept).
    """
    holdout_subjects = torch.as_tensor(list(split_config["holdout_subjects"]), dtype=torch.long)
    subject_out = torch.isin(source.subject, holdout_subjects)
    if not subject_out.any() or subject_out.all():
        raise ValueError("holdout_subjects must select some, but not all, subjects")

    pool = ~source.is_test
    group_size = int(split_config.get("images_per_group", 1))
    group = torch.div(source.image_key[source.image], group_size, rounding_mode="floor")
    held_groups = _choose(
        group[pool], float(split_config["holdout_image_fraction"]), int(split_config["seed"])
    )
    image_out = torch.isin(group, held_groups) & pool

    train = ~subject_out & pool & ~image_out
    if bool(split_config.get("train_on_test_images", False)):
        train |= ~subject_out & source.is_test
    splits = {
        "train": torch.nonzero(train).squeeze(1),
        "val": torch.nonzero(subject_out & image_out).squeeze(1),
    }
    if source.is_test.any():
        splits["val_test"] = torch.nonzero(subject_out & source.is_test).squeeze(1)
    for name, indices in splits.items():
        if len(indices) == 0:
            raise ValueError(f"split {name!r} is empty")
    return splits
