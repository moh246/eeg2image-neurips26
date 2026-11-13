"""Preprocessing on synthetic recordings: shapes, dtypes, finiteness and the slicing rules."""

from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from eeg2img.data.preprocess import (  # noqa: E402
    PreprocessConfig,
    cut_epoch,
    extract_epochs,
    plan_windows,
    preprocess_raw,
    robust_scale,
)

CONFIG = PreprocessConfig()


def _raw(seconds: float = 45.0, sfreq: float = 256.0, n_chans: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = np.arange(int(seconds * sfreq)) / sfreq
    data = 20e-6 * rng.standard_normal((n_chans, len(times)))
    data += 40e-6 * np.sin(2 * np.pi * 50.0 * times)  # mains interference
    data += 100e-6  # DC offset the high-pass must remove
    names = ["Cz", "Oz", "Pz", "POz"][:n_chans]
    info = mne.create_info(names, sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_robust_scale_matches_median_and_iqr():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4001)) * np.array([[1.0], [5.0], [0.1]]) + 7.0
    scaled = robust_scale(data)
    np.testing.assert_allclose(np.median(scaled, axis=1), 0, atol=1e-12)
    q25, q75 = np.percentile(scaled, [25, 75], axis=1)
    np.testing.assert_allclose(q75 - q25, 1.0, rtol=1e-12)


def test_robust_scale_leaves_a_flat_channel_finite():
    scaled = robust_scale(np.vstack([np.ones(100), np.arange(100.0)]))
    assert np.isfinite(scaled).all()
    assert (scaled[0] == 0).all()


def test_preprocess_raw_output_contract():
    raw = _raw()
    before = raw.get_data().copy()
    out = preprocess_raw(raw, CONFIG)
    assert out.shape == (4, round(45.0 * 120))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(np.median(out, axis=1), 0, atol=1e-5)
    np.testing.assert_array_equal(raw.get_data(), before)  # input untouched


def test_preprocess_raw_removes_mains_and_dc():
    out = preprocess_raw(_raw(), CONFIG)
    spectrum = np.abs(np.fft.rfft(out[0]))
    freqs = np.fft.rfftfreq(out.shape[1], 1 / 120.0)
    mains = spectrum[np.abs(freqs - 50.0) < 0.2].max()
    broadband = np.median(spectrum[(freqs > 5) & (freqs < 40)])
    assert mains < 3 * broadband  # the unfiltered 50 Hz line is ~100x the noise floor
    assert abs(out.mean()) < 0.1


def test_recording_shorter_than_the_filter_is_rejected():
    with pytest.raises(ValueError, match="shorter than"):
        preprocess_raw(_raw(seconds=5.0), CONFIG)


def test_regular_event_takes_120_samples_with_a_24_sample_baseline():
    onset = 10.0  # (10 - 0.2) * 120 = 1176 exactly
    plan, valid = plan_windows(np.array([onset]), 6000, CONFIG)
    assert valid.tolist() == [True]
    assert (plan.src_start[0], plan.src_len[0], plan.dst_offset[0]) == (1176, 120, 0)
    assert (plan.base_start[0], plan.base_len[0]) == (1176, 24)


def test_half_sample_tie_reproduces_the_organisers_slicing():
    # (42.0625 - 0.2) * 120 = 5023.5. Their extractor yields 119 samples behind a
    # leading zero, with a 23-sample baseline; verified against it on real data.
    plan, valid = plan_windows(np.array([42.0625]), 35640, CONFIG)
    assert valid.tolist() == [True]
    assert (plan.src_start[0], plan.src_len[0], plan.dst_offset[0]) == (5024, 119, 1)
    assert (plan.base_start[0], plan.base_len[0]) == (5024, 23)


def test_window_past_the_end_is_shifted_back_to_fit():
    plan, valid = plan_windows(np.array([49.9]), 6000, CONFIG)  # recording is 50 s long
    assert valid.tolist() == [True]
    assert plan.src_start[0] + plan.src_len[0] <= 6000
    assert plan.dst_offset[0] + plan.src_len[0] <= 120


def test_event_outside_the_recording_is_dropped():
    plan, valid = plan_windows(np.array([-5.0, 10.0, 500.0]), 6000, CONFIG)
    assert valid.tolist() == [False, True, False]
    assert len(plan) == 1


def test_extract_epochs_baselines_and_clips():
    rng = np.random.default_rng(1)
    continuous = (5.0 * rng.standard_normal((3, 6000)) + 3.0).astype(np.float32)
    continuous[0, 1250] = 500.0  # artefact inside the first window, after the baseline
    epochs, valid = extract_epochs(continuous, np.array([10.0, 20.0]), CONFIG)
    assert epochs.shape == (2, 3, 120) and epochs.dtype == np.float32
    assert valid.all()
    np.testing.assert_allclose(epochs[:, :, :24].mean(axis=2), 0, atol=1e-4)
    assert np.abs(epochs).max() == pytest.approx(20.0)
    expected = continuous[:, 1176:1296] - continuous[:, 1176:1200].mean(axis=1, keepdims=True)
    np.testing.assert_allclose(epochs[0], np.clip(expected, -20, 20), atol=1e-5)


def test_cut_epoch_zero_pads_a_short_window():
    continuous = np.arange(40, dtype=np.float32).reshape(20, 2)  # (T, C)
    epoch = cut_epoch(continuous, src_start=5, src_len=3, dst_offset=1, base_start=5, base_len=0,
                      n_times=6, clamp=100.0)  # fmt: skip
    assert epoch.shape == (2, 6)
    np.testing.assert_array_equal(epoch[0], [0, 10, 12, 14, 0, 0])
