"""Data container invariants and the synthetic generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from eeg2img import contract
from eeg2img.data.synthetic import make_synthetic
from eeg2img.pipeline import make_split


def test_synthetic_matches_the_input_contract():
    epochs = make_synthetic(n_subjects=2, n_images=5, n_repeats=2, embed_dim=contract.EMBED_DIM)
    assert epochs.eeg.shape == (20, contract.N_CHANS, contract.N_TIMES)
    assert epochs.eeg.dtype == np.float32
    assert epochs.embeddings.shape == (5, contract.EMBED_DIM)
    assert epochs.embeddings.dtype == np.float32
    assert epochs.image_index.dtype == np.int64 and epochs.subject.dtype == np.int64
    assert epochs.ch_names == contract.EMOTIV_32
    assert epochs.sfreq == contract.SFREQ
    assert np.isfinite(epochs.eeg).all()
    assert np.abs(epochs.eeg).max() <= contract.CLAMP


def test_synthetic_is_deterministic_in_its_seed():
    first = make_synthetic(n_subjects=2, n_images=4, n_chans=4, n_times=10, seed=7)
    again = make_synthetic(n_subjects=2, n_images=4, n_chans=4, n_times=10, seed=7)
    other = make_synthetic(n_subjects=2, n_images=4, n_chans=4, n_times=10, seed=8)
    np.testing.assert_array_equal(first.eeg, again.eeg)
    np.testing.assert_array_equal(first.image_index, again.image_index)
    assert not np.array_equal(first.eeg, other.eeg)


def test_targets_index_the_embedding_table(small_epochs):
    np.testing.assert_array_equal(
        small_epochs.targets[3], small_epochs.embeddings[small_epochs.image_index[3]]
    )


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("eeg", lambda e: e.eeg.astype(np.float64), "float32"),
        ("eeg", lambda e: e.eeg[:, :, 0], r"\(N, C, T\)"),
        ("subject", lambda e: e.subject[:-1], "subject must have shape"),
        ("image_index", lambda e: e.image_index.astype(np.float32), "integer"),
        ("image_index", lambda e: e.image_index + 10_000, "outside the embedding table"),
        ("ch_names", lambda e: e.ch_names[:-1], "channel names"),
        ("ch_names", lambda e: (e.ch_names[0],) * len(e.ch_names), "unique"),
    ],
)
def test_validate_rejects_malformed_data(small_epochs, field, bad, message):
    with pytest.raises(ValueError, match=message):
        replace(small_epochs, **{field: bad(small_epochs)}).validate()


@pytest.mark.parametrize("poison", [np.nan, np.inf])
def test_validate_rejects_non_finite_eeg(small_epochs, poison):
    eeg = small_epochs.eeg.copy()
    eeg[0, 0, 0] = poison
    with pytest.raises(ValueError, match="NaN or infinite"):
        replace(small_epochs, eeg=eeg).validate()


def test_split_holds_out_subjects_and_images_without_leakage(small_epochs):
    train, test = make_split(
        small_epochs, {"holdout_subjects": [4, 5], "holdout_image_fraction": 0.25, "seed": 33}
    )
    assert set(np.unique(test.subject)) == {4, 5}
    assert not set(np.unique(train.subject)) & {4, 5}
    assert not set(np.unique(train.image_index)) & set(np.unique(test.image_index))
    assert len(np.unique(test.image_index)) == 20
    # Epochs held out on only one axis are dropped, so the two sets do not cover everything.
    assert len(train) + len(test) < len(small_epochs)
    train.validate()
    test.validate()


def test_split_is_reproducible(small_epochs):
    config = {"holdout_subjects": [0], "holdout_image_fraction": 0.5, "seed": 1}
    first = make_split(small_epochs, config)[1]
    again = make_split(small_epochs, config)[1]
    np.testing.assert_array_equal(first.image_index, again.image_index)


def test_group_coherence_detects_concept_groups():
    from eeg2img.data.checks import group_coherence

    rng = np.random.default_rng(0)
    concepts = rng.standard_normal((30, 16))
    embeddings = np.repeat(concepts, 10, axis=0) + 0.3 * rng.standard_normal((300, 16))
    keys = np.arange(300)
    shuffle = rng.permutation(300)  # row order must not matter

    right = group_coherence(embeddings[shuffle], keys[shuffle], group_size=10)
    assert right["within_group_similarity"] > right["across_boundary_similarity"] + 0.5
    assert right["clean_boundary_fraction"] > 0.9
    assert right["n_boundary_pairs"] == 29 and right["n_within_pairs"] == 270

    wrong = group_coherence(embeddings, keys, group_size=7)
    assert wrong["clean_boundary_fraction"] < 0.5
    with pytest.raises(ValueError, match="neighbouring"):
        group_coherence(embeddings[:5], keys[:5], group_size=10)
