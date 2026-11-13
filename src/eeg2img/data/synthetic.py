"""Synthetic EEG/image data that follows the task contract.

Used by the test suite and the CI integration test, so the whole
train -> evaluate -> package loop runs in seconds with no download. The data is
learnable by construction: each image has a low-dimensional latent that
linearly drives both its embedding and its EEG template, so a model that
recovers the latent from EEG generalises to unseen images and subjects.
Scores on it say nothing about real EEG.
"""

from __future__ import annotations

import numpy as np

from eeg2img import contract
from eeg2img.data.epochs import EpochSet


def make_synthetic(
    n_subjects: int = 4,
    n_images: int = 40,
    n_repeats: int = 4,
    n_chans: int = contract.N_CHANS,
    n_times: int = contract.N_TIMES,
    embed_dim: int = 64,
    latent_dim: int = 8,
    noise: float = 1.0,
    seed: int = 0,
) -> EpochSet:
    """Generate ``n_subjects * n_images * n_repeats`` contract-shaped epochs.

    Parameters
    ----------
    noise : standard deviation of the per-sample EEG noise relative to a
        unit-variance evoked template; larger means a harder problem.
    seed : seeds a private generator, so equal arguments give identical data.
    """
    if n_chans > len(contract.EMOTIV_32):
        raise ValueError(f"at most {len(contract.EMOTIV_32)} synthetic channels are supported")
    rng = np.random.default_rng(seed)

    latents = rng.standard_normal((n_images, latent_dim))
    embeddings = latents @ rng.standard_normal((latent_dim, embed_dim))
    embeddings += 0.05 * rng.standard_normal(embeddings.shape)

    # Evoked template: the latent sets channel-by-time weights, smoothed along time.
    mixing = rng.standard_normal((latent_dim, n_chans, n_times))
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    mixing = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), -1, mixing)
    templates = np.einsum("ml,lct->mct", latents, mixing)
    templates /= templates.std()

    subject = np.repeat(np.arange(n_subjects), n_images * n_repeats)
    image_index = np.tile(np.repeat(np.arange(n_images), n_repeats), n_subjects)
    gain = rng.uniform(0.7, 1.3, size=(n_subjects, n_chans, 1))
    eeg = gain[subject] * templates[image_index]
    eeg += noise * rng.standard_normal(eeg.shape)
    eeg = np.clip(eeg, -contract.CLAMP, contract.CLAMP)

    order = rng.permutation(len(eeg))
    return EpochSet(
        eeg=eeg[order].astype(np.float32),
        image_index=image_index[order].astype(np.int64),
        subject=subject[order].astype(np.int64),
        embeddings=embeddings.astype(np.float32),
        ch_names=contract.EMOTIV_32[:n_chans],
        sfreq=contract.SFREQ,
    ).validate()
