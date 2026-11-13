"""Channel matching by name."""

from __future__ import annotations

import pytest
import torch

from eeg2img.channels import ChannelAligner, canonical_channel_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Fp1", "fp1"), ("EEG Fp1-REF", "fp1"), (" AFz ", "afz"), ("Afz", "afz"), ("PO7-A1", "po7")],
)
def test_canonical_name(raw, expected):
    assert canonical_channel_name(raw) == expected


def test_same_order_is_a_no_op():
    aligner = ChannelAligner(["Cz", "Oz", "Pz"], ["Cz", "Oz", "Pz"])
    x = torch.randn(2, 3, 5)
    assert aligner.identity
    assert aligner(x) is x


def test_reorders_by_name_ignoring_case():
    aligner = ChannelAligner(["oz", "CZ", "Pz"], ["Cz", "Pz", "Oz"])
    x = torch.arange(3.0).view(1, 3, 1).expand(1, 3, 4)
    assert aligner(x)[0, :, 0].tolist() == [1.0, 2.0, 0.0]
    assert aligner.missing == ()


def test_missing_channel_is_zero_filled_and_reported():
    aligner = ChannelAligner(["Cz", "Oz"], ["Cz", "Fz", "Oz"])
    out = aligner(torch.ones(2, 2, 4))
    assert out.shape == (2, 3, 4)
    assert out[:, 1].abs().sum() == 0
    assert out[:, [0, 2]].eq(1).all()
    assert aligner.missing == ("Fz",)


def test_extra_input_channels_are_dropped():
    aligner = ChannelAligner(["Cz", "T7", "Oz"], ["Oz", "Cz"])
    x = torch.arange(3.0).view(1, 3, 1)
    assert aligner(x)[0, :, 0].tolist() == [2.0, 0.0]


def test_wrong_channel_count_is_rejected():
    aligner = ChannelAligner(["Cz", "Oz"], ["Cz", "Oz"])
    with pytest.raises(ValueError, match="expected 2 input channels"):
        aligner(torch.zeros(1, 3, 4))


def test_duplicate_input_names_are_rejected():
    with pytest.raises(ValueError, match="not unique"):
        ChannelAligner(["Cz", "cz"], ["Cz"])
