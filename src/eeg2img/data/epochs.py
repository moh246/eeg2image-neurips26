"""In-memory container for task-formatted EEG epochs and their image targets."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class EpochSet:
    """EEG epochs with the image each one was recorded for.

    Attributes
    ----------
    eeg : ``(N, C, T)`` float32, already preprocessed to the model-input contract.
    image_index : ``(N,)`` int64 row of ``embeddings`` shown on each epoch.
    subject : ``(N,)`` int64 subject id of each epoch.
    embeddings : ``(M, D)`` float32 table of frozen image embeddings.
    ch_names : the ``C`` channel names, in the order of the EEG channel axis.
    sfreq : sampling rate of the time axis in Hz.
    """

    eeg: np.ndarray
    image_index: np.ndarray
    subject: np.ndarray
    embeddings: np.ndarray
    ch_names: tuple[str, ...]
    sfreq: float

    def __len__(self) -> int:
        return len(self.eeg)

    @property
    def n_chans(self) -> int:
        return self.eeg.shape[1]

    @property
    def n_times(self) -> int:
        return self.eeg.shape[2]

    @property
    def embed_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def targets(self) -> np.ndarray:
        """``(N, D)`` embedding of the image shown on each epoch."""
        return self.embeddings[self.image_index]

    def validate(self) -> EpochSet:
        """Check shapes, dtypes and finiteness; return ``self`` for chaining.

        Raises ``ValueError`` naming the first violated invariant, so a bad
        preprocessing step fails at load time instead of as NaN losses later.
        """
        if self.eeg.ndim != 3:
            raise ValueError(f"eeg must be (N, C, T), got shape {self.eeg.shape}")
        if self.eeg.dtype != np.float32:
            raise ValueError(f"eeg must be float32, got {self.eeg.dtype}")
        if self.embeddings.ndim != 2 or self.embeddings.dtype != np.float32:
            raise ValueError("embeddings must be a float32 (M, D) table")
        n = len(self.eeg)
        for name in ("image_index", "subject"):
            array = getattr(self, name)
            if array.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {array.shape}")
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{name} must be an integer array, got {array.dtype}")
        if len(self.ch_names) != self.n_chans:
            raise ValueError(f"{len(self.ch_names)} channel names for {self.n_chans} channels")
        if len(set(self.ch_names)) != len(self.ch_names):
            raise ValueError("channel names must be unique")
        if n and (self.image_index.min() < 0 or self.image_index.max() >= len(self.embeddings)):
            raise ValueError("image_index points outside the embedding table")
        if not np.isfinite(self.eeg).all():
            raise ValueError("eeg contains NaN or infinite values")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values")
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")
        return self

    def subset(self, mask: np.ndarray) -> EpochSet:
        """Epochs selected by a boolean mask or index array; the table is shared."""
        return replace(
            self,
            eeg=self.eeg[mask],
            image_index=self.image_index[mask],
            subject=self.subject[mask],
        )

    def split(
        self, holdout_subjects: set[int] | frozenset[int], holdout_images: set[int] | frozenset[int]
    ) -> tuple[EpochSet, EpochSet]:
        """Split into train and a test set that is new in both subject and image.

        This mirrors the sealed shift: test epochs come only from held-out
        subjects viewing held-out images. Epochs that are held out on exactly
        one axis are dropped from both sides so nothing leaks.
        """
        subject_out = np.isin(self.subject, list(holdout_subjects))
        image_out = np.isin(self.image_index, list(holdout_images))
        return self.subset(~subject_out & ~image_out), self.subset(subject_out & image_out)
