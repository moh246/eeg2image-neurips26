"""Device-side batching and the subject/image splits."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mne")

from eeg2img.data.preprocess import PreprocessConfig
from eeg2img.data.source import ArraySource, StoreSource, make_splits
from eeg2img.data.store import EpochStore, ProcessedRecording, build_store

CH = ("Cz", "Oz", "Pz")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    recordings = {}
    for subject in range(1, 7):
        rng = np.random.default_rng(subject)
        continuous = (3.0 * rng.standard_normal((3, 6000))).astype(np.float32)
        continuous[1, 1500] = 400.0  # artefact, to exercise clipping
        onsets = np.sort(rng.uniform(1.0, 48.0, size=120))
        onsets[5] = 3.0625  # half-sample tie -> padded epoch
        onsets[-1] = 49.9  # runs past the end -> shifted window
        onsets = np.sort(onsets)
        image = rng.integers(0, 60, size=120)
        recordings[f"s{subject}"] = ProcessedRecording(
            f"s{subject}", subject, 1, 1, continuous, onsets,
            [f"{i:03d}.jpg" for i in image], [i >= 50 for i in image], CH,
        )  # fmt: skip
    path = build_store(tmp_path_factory.mktemp("store"), list(recordings), recordings.get,
                       PreprocessConfig())  # fmt: skip
    return EpochStore(path)


@pytest.fixture(scope="module")
def source(store):
    embeddings = np.random.default_rng(0).standard_normal((len(store.images), 12))
    return StoreSource(store, embeddings.astype(np.float32))


def test_device_batches_match_the_numpy_reference(store, source):
    indices = torch.arange(len(store))
    got = source.batch(indices)
    expected = store.epochs(indices.numpy())
    assert got.shape == expected.shape == (len(store), 3, 120)
    assert got.dtype == torch.float32
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-5)
    # The fixture must actually contain the special cases this guards.
    assert (store.plan[:, 2] != 0).any(), "no padded tie-case epoch in the fixture"
    assert (np.abs(expected) == 20.0).any(), "no clipped sample in the fixture"


def test_batches_follow_the_requested_order(source):
    indices = torch.tensor([7, 3, 3, 250])
    batch = source.batch(indices)
    torch.testing.assert_close(batch[1], batch[2])
    torch.testing.assert_close(batch[0], source.batch(torch.tensor([7]))[0])


def test_embedding_count_must_match_the_images(store):
    with pytest.raises(ValueError, match="embeddings for"):
        StoreSource(store, np.zeros((len(store.images) - 1, 4), dtype=np.float32))


def test_splits_are_disjoint_in_subject_and_image(source):
    config = {"holdout_subjects": [5, 6], "holdout_image_fraction": 0.2, "images_per_group": 5,
              "seed": 33}  # fmt: skip
    splits = make_splits(source, config)
    assert set(splits) == {"train", "val", "val_test"}
    subject = {name: set(source.subject[idx].tolist()) for name, idx in splits.items()}
    image = {name: set(source.image[idx].tolist()) for name, idx in splits.items()}
    assert subject["train"] == {1, 2, 3, 4}
    assert subject["val"] == subject["val_test"] == {5, 6}
    assert not image["train"] & image["val"]
    assert not image["train"] & image["val_test"]
    assert not image["val"] & image["val_test"]
    assert not source.is_test[splits["train"]].any()
    assert source.is_test[splits["val_test"]].all()
    # Held-out images come in whole groups of 5 consecutive stimulus ids.
    key = source.image_key
    held_groups = {int(key[i]) // 5 for i in image["val"]}
    assert not {int(key[i]) // 5 for i in image["train"]} & held_groups


def test_grouping_uses_the_stimulus_id_not_the_list_position(store):
    # Image "007.jpg" is absent from this fixture or not, the key is still the name.
    source = StoreSource(store, np.zeros((len(store.images), 4), dtype=np.float32))
    assert source.image_key.tolist() == [int(name[:3]) for name in store.images]


def test_final_training_can_include_the_test_images(source):
    config = {"holdout_subjects": [6], "holdout_image_fraction": 0.2, "seed": 1,
              "train_on_test_images": True}  # fmt: skip
    splits = make_splits(source, config)
    assert source.is_test[splits["train"]].any()
    assert not set(source.subject[splits["train"]].tolist()) & {6}


def test_split_is_seeded(source):
    config = {"holdout_subjects": [6], "holdout_image_fraction": 0.3, "seed": 5}
    first, again = make_splits(source, config), make_splits(source, config)
    other = make_splits(source, {**config, "seed": 6})
    assert torch.equal(first["val"], again["val"])
    assert not torch.equal(first["val"], other["val"])


@pytest.mark.parametrize("subjects", [[], [1, 2, 3, 4, 5, 6], [99]])
def test_degenerate_subject_holdout_is_rejected(source, subjects):
    with pytest.raises(ValueError, match="holdout_subjects"):
        make_splits(source, {"holdout_subjects": subjects, "holdout_image_fraction": 0.2,
                             "seed": 0})  # fmt: skip


def test_array_source_round_trips_an_epochset(small_epochs):
    source = ArraySource(small_epochs)
    indices = torch.tensor([4, 0, 9])
    np.testing.assert_array_equal(source.batch(indices).numpy(), small_epochs.eeg[[4, 0, 9]])
    subset = source.epochset(indices).validate()
    np.testing.assert_array_equal(subset.image_index, small_epochs.image_index[[4, 0, 9]])
