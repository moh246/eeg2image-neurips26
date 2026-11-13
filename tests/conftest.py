"""Shared fixtures: small synthetic datasets that keep the suite fast."""

from __future__ import annotations

import pytest

from eeg2img.data.synthetic import make_synthetic


@pytest.fixture(scope="session")
def small_epochs():
    """6 subjects x 80 images x 3 repeats, 8 channels x 30 samples, 16-d targets.

    Noisy enough that single-trial retrieval is far from saturated.
    """
    return make_synthetic(
        n_subjects=6, n_images=80, n_repeats=3, n_chans=8, n_times=30, embed_dim=16, noise=6.0,
        seed=0,
    )  # fmt: skip
