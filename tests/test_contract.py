"""The contract constants are internally consistent."""

from __future__ import annotations

from eeg2img import contract


def test_montage_has_32_unique_channels():
    assert contract.N_CHANS == 32
    assert len(set(contract.EMOTIV_32)) == 32


def test_epoch_is_one_second_at_120_hz():
    assert contract.N_TIMES == 120
    assert contract.EPOCH_START + contract.EPOCH_DURATION == 0.8


def test_baseline_window_is_the_prestimulus_interval():
    start, stop = contract.BASELINE_WINDOW
    assert contract.EPOCH_START + start == -0.2
    assert abs(contract.EPOCH_START + stop) < 1e-9
