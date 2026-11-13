"""Parity of our preprocessing with the organisers' extractor, on real recordings.

For every Alljoined-1.6M recording found under ``data/``, epochs are produced
twice: by this repository, and by the organisers' reader and extractor
(``neuralfetch`` + ``neuralset`` 0.4.0) configured with the values of
NeuralBench's image task. They must agree to float32 rounding.

Skipped unless both the organisers' stack and some local data are present
(``make parity`` sets that up). Data is never committed, so this does not run
in CI; its result is recorded in ``data/README.md``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.official

DATA_ROOT = Path(os.environ.get("EEG2IMG_DATA_ROOT", "data"))
MAX_RECORDINGS = int(os.environ.get("EEG2IMG_PARITY_RECORDINGS", "2"))


def _recordings():
    try:
        from eeg2img.data import alljoined
    except ImportError:
        return []
    return alljoined.list_recordings(DATA_ROOT)[:MAX_RECORDINGS]


@pytest.fixture(scope="module")
def official():
    pytest.importorskip("neuralset")
    pytest.importorskip("neuralfetch")
    from neuralfetch.studies.xu2025alljoined import Xu2025Alljoined
    from neuralset.events import etypes
    from neuralset.extractors.neuro import EegExtractor

    # defaults/eeg/config.yaml merged with tasks/eeg/image/config.yaml (NeuralBench 0.4.0)
    extractor = EegExtractor(
        picks=("eeg",), frequency=120.0, filter=(0.1, 75.0), notch_filter=[50.0, 60.0],
        baseline=(0.0, 0.2), scaler="RobustScaler", clamp=20.0,
    )  # fmt: skip
    study = Xu2025Alljoined(path=DATA_ROOT)

    def reference(recording):
        timeline = {"subject": str(recording.subject), "session": recording.session,
                    "run": recording.run}  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            events = study._load_timeline_events(timeline)
            events["timeline"], events["subject"] = recording.name, timeline["subject"]
            eeg_event = etypes.Event.from_dict(events.iloc[0].to_dict())
            images = events[(events.type == "Image")
                            & events.label.isin(["stim_train", "stim_test"])]  # fmt: skip
            continuous = extractor._preprocess_raw(eeg_event.read().copy(), eeg_event)
            extractor._update_channels(continuous.ch_names)
            epochs = np.stack([
                extractor._get_timed_array(eeg_event, float(start) - 0.2, 1.0).data
                for start in images.start.to_numpy()
            ])  # fmt: skip
        return {
            "continuous": np.asarray(continuous.data, dtype=np.float32),
            "ch_names": list(continuous.ch_names),
            "epochs": epochs.astype(np.float32),
            "onsets": images.start.to_numpy(float),
            "stims": images.stim_file.tolist(),
            "labels": images.label.tolist(),
        }

    return reference


@pytest.mark.skipif(not _recordings(), reason=f"no Alljoined-1.6M recordings under {DATA_ROOT}")
@pytest.mark.parametrize("recording", _recordings(), ids=lambda rec: rec.name)
def test_epochs_match_the_organisers_extractor(official, recording):
    from eeg2img.data import alljoined
    from eeg2img.data.preprocess import extract_epochs, plan_windows, preprocess_raw

    expected = official(recording)

    raw = alljoined.load_raw(recording)
    assert list(alljoined.check_montage(raw)) == expected["ch_names"]
    onsets, stims, labels = alljoined.load_events(recording)
    np.testing.assert_array_equal(onsets, expected["onsets"])
    assert stims == expected["stims"] and labels == expected["labels"]

    continuous = preprocess_raw(raw)
    np.testing.assert_allclose(continuous, expected["continuous"], atol=1e-5, rtol=0)

    epochs, valid = extract_epochs(continuous, onsets)
    assert valid.all()
    assert epochs.shape == expected["epochs"].shape
    np.testing.assert_allclose(epochs, expected["epochs"], atol=1e-4, rtol=0)

    # The half-sample quirk must be present in the data for this test to cover it.
    plan, _ = plan_windows(onsets, continuous.shape[1])
    assert (plan.dst_offset != 0).any(), "expected some padded tie-case epochs"
