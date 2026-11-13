"""Raw EEG -> model input, reproducing the organisers' NeuralBench pipeline.

The reference is ``neuralset.extractors.neuro.MneRaw`` (neuralset 0.4.0) with
the values of NeuralBench's ``defaults/eeg/config.yaml`` and
``tasks/eeg/image/config.yaml``. In order:

Per recording, on the continuous signal (:func:`preprocess_raw`):

1. keep EEG channels;
2. notch at 50 and 60 Hz and their harmonics below Nyquist (zero-phase FIR);
3. band-pass 0.1-75 Hz at the native rate (the low-pass edge is skipped only
   when it is at or above Nyquist);
4. resample to 120 Hz;
5. robust-scale each channel over the whole recording (median and IQR).

Per epoch (:func:`extract_epochs`):

6. cut 120 samples starting at ``round((onset - 0.2 s) * 120)``;
7. subtract each channel's mean over the first 0.2 s of the window;
8. clip to +/-20.

Steps 6-7 follow the organisers' slicing arithmetic exactly, including its edge
behaviour: when a window start falls on a half sample (about 1.4% of Alljoined
events) their code yields 119 samples behind one leading zero, with a 23-sample
baseline. :func:`plan_windows` ports that arithmetic, so those epochs look the
same in training as they will at evaluation.

``tests/test_preprocess_parity.py`` compares the output with epochs produced by
the organisers' own extractor on real Alljoined-1.6M recordings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import mne
import numpy as np

from eeg2img import contract


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing parameters; the defaults are the competition's."""

    sfreq: float = contract.SFREQ
    l_freq: float | None = 0.1
    h_freq: float | None = 75.0
    notch: tuple[float, ...] = (50.0, 60.0)
    epoch_start: float = contract.EPOCH_START
    epoch_duration: float = contract.EPOCH_DURATION
    baseline: tuple[float, float] = contract.BASELINE_WINDOW
    clamp: float = contract.CLAMP

    @property
    def n_times(self) -> int:
        return round(self.epoch_duration * self.sfreq)

    def to_dict(self) -> dict:
        return asdict(self)


def robust_scale(data: np.ndarray) -> np.ndarray:
    """Centre each channel on its median and scale by its interquartile range.

    ``data`` is ``(C, T)``. Matches ``sklearn.preprocessing.RobustScaler`` with
    default settings, including leaving constant channels unscaled.
    """
    center = np.nanmedian(data, axis=1, keepdims=True)
    q25, q75 = np.nanpercentile(data, [25.0, 75.0], axis=1, keepdims=True)
    scale = q75 - q25
    scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0
    return (data - center) / scale


def _notch_frequencies(notch: Sequence[float], sfreq: float) -> list[float]:
    freqs: list[float] = []
    for base in notch:
        freqs.extend(np.arange(base, min(sfreq / 2, 301), base).tolist())
    return freqs


def preprocess_raw(raw: mne.io.BaseRaw, config: PreprocessConfig | None = None) -> np.ndarray:
    """Filter, resample and scale one recording; return ``(C, T)`` float32.

    The input is not modified. Channels typed as EEG are kept, in their
    original order.
    """
    config = config or PreprocessConfig()
    raw = raw.copy().pick("eeg", verbose=False).load_data(verbose=False)
    sfreq = float(raw.info["sfreq"])

    freqs = _notch_frequencies(config.notch, sfreq)
    if freqs:
        raw.notch_filter(freqs, picks="all", phase="zero", filter_length="auto", verbose=False)

    l_freq, h_freq = config.l_freq, config.h_freq
    if h_freq is not None and h_freq >= sfreq / 2:
        h_freq = None
    if l_freq is not None or h_freq is not None:
        for low, high in ((l_freq, None), (None, h_freq)):
            if low is None and high is None:
                continue
            n_fir = len(mne.filter.create_filter(None, sfreq, low, high, verbose=False))
            if n_fir > raw.n_times:
                raise ValueError(
                    f"recording of {raw.n_times / sfreq:.1f} s is shorter than the "
                    f"{n_fir / sfreq:.1f} s filter needed for ({low}, {high}) Hz"
                )
        raw.filter(l_freq, h_freq, picks="all", skip_by_annotation=("edge", "bad_acq_skip"),
                   verbose=False)  # fmt: skip

    if sfreq != config.sfreq:
        raw.resample(config.sfreq, verbose=False)

    return robust_scale(raw.get_data()).astype(np.float32)


@dataclass(frozen=True)
class WindowPlan:
    """Where each epoch's samples come from, as integer indices into a recording.

    For event ``i`` the epoch is ``n_times`` zeros with
    ``recording[src_start[i] : src_start[i] + src_len[i]]`` written at
    ``dst_offset[i]``, after subtracting the mean of
    ``recording[base_start[i] : base_start[i] + base_len[i]]``.
    """

    src_start: np.ndarray
    src_len: np.ndarray
    dst_offset: np.ndarray
    base_start: np.ndarray
    base_len: np.ndarray

    def __len__(self) -> int:
        return len(self.src_start)


def _overlap(
    array_start: float, n: int, sfreq: float, start: float, duration: float
) -> tuple[float, int, int] | None:
    """Port of ``neuralset.base.TimedArray._overlap_slice``.

    For an array of ``n`` samples beginning at ``array_start`` seconds, return
    ``(start_seconds, first_index, length)`` of its part overlapping
    ``[start, start + duration)``, or None if there is none. The floating-point
    operation order is kept identical to the original, because results at exact
    half-sample ties depend on it.
    """
    array_duration = n / sfreq
    overlap_start = max(start, array_start)
    overlap_stop = min(start + duration, array_start + array_duration)
    if overlap_stop < overlap_start:
        return None
    if overlap_stop == overlap_start and array_duration and duration:
        return None
    first = round((overlap_start - array_start) * sfreq)
    length = round((overlap_stop - overlap_start) * sfreq)
    if length <= 0:
        length = 1
    if first > n - length:
        first = n - length
    if first < 0:
        return None
    return first / sfreq + array_start, first, length


def plan_windows(
    onsets: np.ndarray, n_samples: int, config: PreprocessConfig | None = None
) -> tuple[WindowPlan, np.ndarray]:
    """Plan the epoch window of each event in a recording of ``n_samples``.

    A port of ``neuralset.extractors.neuro.MneRaw._get_timed_array`` for a
    recording that starts at time zero. Returns the plan for the events that
    overlap the recording, and the boolean mask of those events.
    """
    config = config or PreprocessConfig()
    sfreq, n_times = float(config.sfreq), config.n_times
    base_lo, base_hi = config.baseline
    rows: list[tuple[int, int, int, int, int]] = []
    valid = np.zeros(len(onsets), dtype=bool)
    for i, onset in enumerate(np.asarray(onsets, dtype=np.float64).tolist()):
        start = onset + config.epoch_start
        duration = config.epoch_duration
        window_start = min(start, start + base_lo)
        window_stop = max(start + duration, start + base_hi)
        window = _overlap(0.0, n_samples, sfreq, window_start, window_stop - window_start)
        if window is None:
            continue
        t_start, t_first, t_len = window

        base = _overlap(t_start, t_len, sfreq, start + base_lo, base_hi - base_lo)
        base_start, base_len = (t_first + base[1], base[2]) if base else (t_first, 0)

        cut = _overlap(t_start, t_len, sfreq, start, duration)
        if cut is None:
            continue
        u_start, u_first, u_len = cut
        if u_start == start and u_len / sfreq == duration:
            dst_offset, src_first, length = 0, u_first, u_len
        else:
            # The cut is added into a zero window of n_times samples at `start`.
            into = _overlap(start, n_times, sfreq, u_start, u_len / sfreq)
            out_of = _overlap(u_start, u_len, sfreq, start, n_times / sfreq)
            if into is None or out_of is None:
                continue
            dst_offset, src_first, length = into[1], u_first + out_of[1], min(into[2], out_of[2])
        if length <= 0 or dst_offset + length > n_times:
            continue
        rows.append((t_first + src_first, length, dst_offset, base_start, base_len))
        valid[i] = True
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 5)
    return WindowPlan(*(table[:, column].copy() for column in range(5))), valid


def cut_epoch(
    continuous: np.ndarray,
    src_start: int,
    src_len: int,
    dst_offset: int,
    base_start: int,
    base_len: int,
    n_times: int,
    clamp: float,
) -> np.ndarray:
    """Build one ``(C, n_times)`` float32 epoch from a time-major ``(T, C)`` recording."""
    window = np.asarray(continuous[src_start : src_start + src_len], dtype=np.float32)
    if base_len > 0:
        baseline = np.asarray(continuous[base_start : base_start + base_len], dtype=np.float32)
        window = window - baseline.mean(axis=0, keepdims=True)
    epoch = np.zeros((window.shape[1], n_times), dtype=np.float32)
    epoch[:, dst_offset : dst_offset + src_len] = np.clip(window, -clamp, clamp).T
    return epoch


def extract_epochs(
    continuous: np.ndarray, onsets: np.ndarray, config: PreprocessConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cut model-input epochs out of a preprocessed ``(C, T)`` recording.

    Returns ``(epochs, valid)``: float32 ``(N_valid, C, n_times)`` and the
    boolean mask over ``onsets`` of the events that produced an epoch.
    """
    config = config or PreprocessConfig()
    plan, valid = plan_windows(onsets, continuous.shape[1], config)
    time_major = np.ascontiguousarray(continuous.T)
    epochs = np.zeros((len(plan), continuous.shape[0], config.n_times), dtype=np.float32)
    for i in range(len(plan)):
        epochs[i] = cut_epoch(
            time_major, int(plan.src_start[i]), int(plan.src_len[i]), int(plan.dst_offset[i]),
            int(plan.base_start[i]), int(plan.base_len[i]), config.n_times, config.clamp,
        )  # fmt: skip
    return epochs, valid
