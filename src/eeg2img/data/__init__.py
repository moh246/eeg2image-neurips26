"""Data containers, loaders and preprocessing."""

from eeg2img.data.epochs import EpochSet
from eeg2img.data.synthetic import make_synthetic

__all__ = ["EpochSet", "make_synthetic"]
