"""Scoring a packaged model on a subset of its training electrodes."""

from __future__ import annotations

import pytest
import torch

from eeg2img.contract import EMOTIV_32
from eeg2img.data.source import ArraySource
from eeg2img.data.synthetic import make_synthetic
from eeg2img.engine.montage_eval import (
    channel_subset,
    evaluate_channel_subsets,
    score_with_channels,
)
from eeg2img.engine.train import evaluate_source
from eeg2img.models import build_model
from eeg2img.submission.builder import write_package

CH = tuple(EMOTIV_32[:8])
MODELS = {
    "conv": {"d_model": 16, "n_filters": 8, "temporal_kernel": 5, "pool_kernel": 6,
             "pool_stride": 3},
    "channel_transformer": {"d_model": 16, "patch_size": 10, "n_layers": 1, "n_heads": 2},
}  # fmt: skip


@pytest.fixture(scope="module")
def source():
    epochs = make_synthetic(n_subjects=3, n_images=30, n_repeats=2, n_chans=8, n_times=30,
                            embed_dim=12, seed=0)  # fmt: skip
    return ArraySource(type(epochs)(**{**epochs.__dict__, "ch_names": CH}))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_all_channels_match_the_in_memory_model(tmp_path, source, name):
    torch.manual_seed(0)
    model = build_model(name, n_chans=8, n_times=30, n_outputs=12, ch_names=CH, **MODELS[name])
    model.eval()
    write_package(model, tmp_path, model_config={"name": name, "params": MODELS[name]},
                  ch_names=CH, n_times=30, n_outputs=12, sfreq=30.0)  # fmt: skip
    indices = torch.arange(len(source))

    full = score_with_channels(tmp_path, source, indices, range(8))
    assert full == evaluate_source(model, source, indices)

    rows = evaluate_channel_subsets(tmp_path, source, indices, [8, 4], draws=2)
    assert [row["n_channels"] for row in rows] == [8, 4]
    assert rows[0]["values"] == [full["top5_acc_subject_agg"]] and rows[0]["std"] is None
    assert len(rows[1]["values"]) == 2 and all(len(s) == 4 for s in rows[1]["subsets"])
    assert set(rows[1]["subsets"][0]) <= set(CH)


def test_channel_subset_is_seeded_sorted_and_checked():
    assert channel_subset(32, 8, seed=1) == channel_subset(32, 8, seed=1)
    assert channel_subset(32, 8, seed=1) != channel_subset(32, 8, seed=2)
    subset = channel_subset(32, 8, seed=1)
    assert subset == sorted(set(subset)) and len(subset) == 8
    with pytest.raises(ValueError, match="keep"):
        channel_subset(32, 33, seed=0)


def test_ensemble_scores_members_and_their_mean(tmp_path, source):
    from eeg2img.engine.packages import evaluate_ensemble, predict_package, score

    packages = []
    for seed in range(2):
        torch.manual_seed(seed)
        model = build_model("conv", n_chans=8, n_times=30, n_outputs=12, ch_names=CH,
                            **MODELS["conv"])  # fmt: skip
        write_package(model.eval(), tmp_path / str(seed),
                      model_config={"name": "conv", "params": MODELS["conv"]},
                      ch_names=CH, n_times=30, n_outputs=12, sfreq=30.0)  # fmt: skip
        packages.append(tmp_path / str(seed))
    indices = torch.arange(len(source))

    report = evaluate_ensemble(packages, source, indices)
    singles = [predict_package(package, source, indices) for package in packages]
    assert report["members"] == [
        score(single, source, indices)["top5_acc_subject_agg"] for single in singles
    ]
    mean = score((singles[0] + singles[1]) / 2, source, indices)
    assert report["ensemble"] == mean["top5_acc_subject_agg"]
    assert report["n_candidates"] == 30

    alone = evaluate_ensemble(packages[:1], source, indices)
    assert alone["ensemble"] == alone["members"][0], "an ensemble of one is that model"
    with pytest.raises(ValueError, match="at least one"):
        evaluate_ensemble([], source, indices)
