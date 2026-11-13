"""The on-disk store: round trip, resumability and the Alljoined event rules."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mne")

from eeg2img.data import alljoined
from eeg2img.data.preprocess import PreprocessConfig, extract_epochs
from eeg2img.data.store import EpochStore, ProcessedRecording, build_store

CONFIG = PreprocessConfig()
CH = ("Cz", "Oz", "Pz")


def _recording(name: str, subject: int, seed: int) -> ProcessedRecording:
    rng = np.random.default_rng(seed)
    continuous = (3.0 * rng.standard_normal((3, 4000))).astype(np.float32)
    onsets = np.sort(rng.uniform(1.0, 30.0, size=25))
    onsets[3] = 3.0625  # a half-sample tie, to exercise the padded path
    stims = [f"{int(i)}.jpg" for i in rng.integers(0, 12, size=25)]
    return ProcessedRecording(name, subject, 1, 1, continuous, np.sort(onsets), stims,
                              [s in ("0.jpg", "1.jpg") for s in stims], CH)  # fmt: skip


@pytest.fixture
def built(tmp_path):
    recordings = {"a": _recording("a", 1, 0), "b": _recording("b", 2, 1)}
    calls = []

    def load(name):
        calls.append(name)
        return recordings[name]

    path = build_store(tmp_path / "store", ["a", "b"], load, CONFIG, provenance={"src": "test"})
    return path, recordings, calls, load


def test_store_round_trip_matches_direct_extraction(built):
    path, recordings, _, _ = built
    store = EpochStore(path)
    assert len(store) == 50 and store.ch_names == CH and store.n_times == 120
    direct = np.concatenate([extract_epochs(r.continuous, r.onsets, CONFIG)[0]
                             for r in recordings.values()])  # fmt: skip
    got = store.epochs(np.arange(len(store)))
    assert got.dtype == np.float32 and got.shape == direct.shape
    assert (store.plan[:, 2] != 0).any(), "no padded tie-case epoch in the fixture"
    # float16 storage: error stays far below the unit noise scale.
    assert np.abs(got - direct).max() < 0.02
    assert np.corrcoef(got.ravel(), direct.ravel())[0, 1] > 0.999999


def test_store_index_is_consistent(built):
    path, recordings, _, _ = built
    store = EpochStore(path)
    assert set(np.unique(store.subject)) == {1, 2}
    assert store.subject.tolist() == [1] * 25 + [2] * 25
    stored = [store.images[i] for i in store.image[:25]]
    assert stored == list(recordings["a"].stim_files)
    assert store.is_test.sum() == sum(sum(r.is_test) for r in recordings.values())
    assert store.images == sorted(store.images)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["preprocess"]["sfreq"] == 120.0 and meta["provenance"] == {"src": "test"}
    assert meta["n_events"] == 50 and meta["n_subjects"] == 2


def test_build_resumes_from_existing_shards(built):
    path, _, calls, load = built
    assert calls == ["a", "b"]
    before = (path / "eeg.npy").read_bytes()
    build_store(path, ["a", "b"], load, CONFIG)
    assert calls == ["a", "b"], "recordings with a shard must not be reprocessed"
    assert (path / "eeg.npy").read_bytes() == before


def test_to_epochset_validates_and_aligns_targets(built):
    path, *_ = built
    store = EpochStore(path)
    embeddings = np.random.default_rng(0).standard_normal((len(store.images), 8)).astype(np.float32)
    subset = store.to_epochset(np.flatnonzero(store.is_test), embeddings)
    assert len(subset) == store.is_test.sum()
    np.testing.assert_array_equal(subset.targets[0], embeddings[subset.image_index[0]])
    with pytest.raises(ValueError, match="embeddings for"):
        store.to_epochset([0], embeddings[:-1])


def test_store_survives_pickling_for_dataloader_workers(built):
    import pickle

    path, *_ = built
    store = EpochStore(path)
    first = store.epoch(0)
    clone = pickle.loads(pickle.dumps(store))
    np.testing.assert_array_equal(clone.epoch(0), first)


def test_events_follow_the_organisers_selection_rules(tmp_path):
    eeg_dir = alljoined.dataset_dir(tmp_path) / "sub-03" / "ses-02" / "eeg"
    eeg_dir.mkdir(parents=True)
    stem = "sub-03_ses-02_task-images_run-07"
    (eeg_dir / f"{stem}_eeg.edf").write_bytes(b"")
    (eeg_dir / f"{stem}_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\tmarker_index\tstim_file\n"
        "0.5\t0.0\tstim_test\t1\t128\t1\t16641.jpg\n"
        "0.75\t0.0\tstim_train\t2\t192\t2\t12.jpg\n"
        "1.0\t0.0\tstim_train\t3\t256\t3\t13.jpg\n"  # collides with the next marker
        "1.0\t0.0\tbehav\t4\t256\t4\tn/a\n"
        "1.25\t0.0\tbehav\t5\t320\t5\tn/a\n"  # no stimulus
        "1.5\t0.0\toddball\t6\t384\t6\t99.jpg\n"  # not a stim_train/stim_test trial
        "1.75\t0.0\tstim_train\t7\t448\t7\t14.jpg\n"
    )
    (recording,) = alljoined.list_recordings(tmp_path)
    assert (recording.subject, recording.session, recording.run) == (3, 2, 7)
    assert recording.name == "sub-03_ses-02_run-07"
    onsets, stims, labels = alljoined.load_events(recording)
    assert onsets.tolist() == [0.5, 0.75, 1.75]
    assert stims == ["16641.jpg", "12.jpg", "14.jpg"]
    assert labels == ["stim_test", "stim_train", "stim_train"]


def test_download_requires_accepting_the_licence(tmp_path, monkeypatch):
    monkeypatch.delenv(alljoined.LICENCE_ENV, raising=False)
    with pytest.raises(RuntimeError, match=r"CC-BY-NC-ND-4\.0"):
        alljoined.download(tmp_path)
