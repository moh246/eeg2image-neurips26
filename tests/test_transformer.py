"""Transformer encoders: the ablation is controlled, and the montage-agnostic model is."""

from __future__ import annotations

import pytest
import torch

from eeg2img import contract
from eeg2img.models import build_model
from eeg2img.models.transformer import electrode_positions
from eeg2img.montage import ELECTRODE_POSITIONS
from eeg2img.submission import load_packaged_model, write_package

SMALL = {"d_model": 32, "n_layers": 2, "n_heads": 2}
TOKENIZER = {"n_filters": 16, "temporal_kernel": 7, "pool_kernel": 6, "pool_stride": 3}
NAMES = contract.EMOTIV_32


def test_position_table_covers_the_montages_and_is_symmetric():
    assert all(name.lower() in ELECTRODE_POSITIONS for name in NAMES)
    positions = electrode_positions(["C3", "C4", "Cz", "Oz", "Fp1"])
    torch.testing.assert_close(positions[0, 0], -positions[1, 0], atol=0.02, rtol=0)  # left/right
    assert positions[2, 2] > 0.9  # vertex is at the top
    assert positions[3, 1] < 0 < positions[4, 1]  # occipital is behind, frontal-polar in front
    with pytest.raises(KeyError, match="NotAnElectrode"):
        electrode_positions(["Cz", "NotAnElectrode"])


def test_transformer_head_ablation_changes_only_the_head():
    conv = build_model("conv", 8, 60, 16, d_model=32, **TOKENIZER)
    transformer = build_model("conv_transformer", 8, 60, 16, **SMALL, **TOKENIZER)
    shapes = lambda m: {k: v.shape for k, v in m.tokenizer.state_dict().items()}  # noqa: E731
    assert shapes(conv) == shapes(transformer)
    assert conv.tokenizer.n_tokens == transformer.tokenizer.n_tokens


@pytest.mark.parametrize(("attention", "early_tokens_change"),
                         [("causal", False), ("bidirectional", True)])  # fmt: skip
def test_causal_attention_does_not_look_ahead(attention, early_tokens_change):
    torch.manual_seed(0)
    model = build_model("conv_transformer", 8, 120, 16, attention=attention, **SMALL,
                        **TOKENIZER).eval()  # fmt: skip
    x = torch.randn(2, 8, 120)
    later = x.clone()
    later[:, :, -8:] += 5.0  # change only the end of the epoch
    with torch.inference_mode():
        a, b = model.encode_tokens(x), model.encode_tokens(later)
    changed = not torch.allclose(a[:, :10], b[:, :10], atol=1e-6)
    assert changed is early_tokens_change
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6)  # the latest token always sees it


def test_unknown_attention_mode_is_rejected():
    with pytest.raises(ValueError, match="attention must be"):
        build_model("conv_transformer", 8, 60, 16, attention="sideways")


@pytest.fixture
def channel_model():
    torch.manual_seed(0)
    return build_model("channel_transformer", 32, 120, 24, ch_names=NAMES, **SMALL).eval()


def test_channel_transformer_is_invariant_to_channel_order(channel_model):
    x = torch.randn(3, 32, 120)
    with torch.inference_mode():
        reference = channel_model(x)
        perm = torch.randperm(32, generator=torch.Generator().manual_seed(1))
        channel_model.set_channels([NAMES[i] for i in perm.tolist()])
        permuted = channel_model(x[:, perm])
    torch.testing.assert_close(permuted, reference, atol=1e-5, rtol=0)


def test_channel_transformer_weights_do_not_depend_on_the_montage(channel_model):
    # The same weights load into a model built for a 63-channel research montage.
    things = [n for n in ("Fp1 Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT9 FT7 FC5 FC3 "
                          "FC1 FCz FC2 FC4 FC6 FT8 FT10 T7 C5 C3 C1 Cz C2 C4 C6 T8 TP9 TP7 CP5 CP3 "
                          "CP1 CPz CP2 CP4 CP6 TP8 TP10 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 "
                          "PO8 O1 Oz").split()]  # fmt: skip
    assert len(things) == 63
    wide = build_model("channel_transformer", 63, 120, 24, ch_names=things, **SMALL).eval()
    wide.load_state_dict(channel_model.state_dict())
    with torch.inference_mode():
        out = wide(torch.randn(2, 63, 120))
    assert out.shape == (2, 24) and torch.isfinite(out).all()


def test_channel_transformer_runs_on_a_subset_of_its_training_montage(channel_model):
    channel_model.set_channels(NAMES[:20])
    with torch.inference_mode():
        out = channel_model(torch.randn(2, 20, 120))
    assert out.shape == (2, 24) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="expected 20 channels"):
        channel_model(torch.randn(2, 32, 120))


def test_token_dropout_only_acts_in_training(channel_model):
    channel_model.token_dropout = 0.5
    x = torch.randn(2, 32, 120)
    with torch.inference_mode():
        torch.testing.assert_close(channel_model(x), channel_model(x))
    channel_model.train()
    assert channel_model(x).shape == (2, 24)


def _package(model, name, tmp_path):
    write_package(model, tmp_path, model_config={"name": name, "params": SMALL},
                  ch_names=NAMES, n_times=120, n_outputs=24, sfreq=120.0)  # fmt: skip


def test_packaged_position_aware_model_adapts_to_a_new_montage(channel_model, tmp_path):
    _package(channel_model, "channel_transformer", tmp_path)
    # A montage the model never saw: a 20-channel subset, two new electrodes, one unknown.
    montage = [*NAMES[5:25], "C3", "T8", "Mystery1"]
    meta = {"submission_dir": tmp_path, "device": "cpu", "ch_names": montage,
            "n_chans": len(montage), "n_times": 120, "n_outputs": 24}  # fmt: skip
    loaded = load_packaged_model(meta)
    assert loaded.network.positions.shape[0] == 22  # the unknown channel is dropped
    x = torch.randn(4, len(montage), 120)
    out = loaded.predict(x)
    assert out.shape == (4, 24) and torch.isfinite(out).all()
    # It really uses the new electrodes: changing C3 changes the prediction.
    x2 = x.clone()
    x2[:, montage.index("C3")] += 3.0
    assert not torch.allclose(loaded.predict(x2), out, atol=1e-4)
    # ...and ignores the channel it cannot place.
    x3 = x.clone()
    x3[:, montage.index("Mystery1")] += 3.0
    torch.testing.assert_close(loaded.predict(x3), out, atol=1e-6, rtol=0)


def test_packaged_position_aware_model_matches_on_the_training_montage(channel_model, tmp_path):
    _package(channel_model, "channel_transformer", tmp_path)
    meta = {"submission_dir": tmp_path, "device": "cpu", "ch_names": list(NAMES), "n_chans": 32,
            "n_times": 120, "n_outputs": 24}  # fmt: skip
    x = torch.randn(4, 32, 120)
    with torch.inference_mode():
        expected = torch.nn.functional.normalize(channel_model(x), dim=-1)
    torch.testing.assert_close(load_packaged_model(meta).predict(x), expected, atol=1e-6, rtol=0)
