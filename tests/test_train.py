"""The contrastive loss and the training loop."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from eeg2img.data.source import ArraySource, make_splits
from eeg2img.engine import train as train_module
from eeg2img.engine.losses import RetrievalLoss
from eeg2img.engine.train import _lr_factor, evaluate_source, train_model
from eeg2img.models import build_model
from eeg2img.tracking import RunTracker

SPLIT = {"holdout_subjects": [4, 5], "holdout_image_fraction": 0.25, "seed": 33}
CONFIG = {
    "seed": 0,
    "train": {"epochs": 4, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.05, "warmup_steps": 5,
              "log_every": 5, "monitor": "val/top5_acc_subject_agg", "patience": None,
              "amp": False},
    "loss": {"temperature": 0.07, "learn_temperature": True, "extra_negatives": 0,
             "mse_weight": 0.0},
}  # fmt: skip
PARAMS = {"d_model": 32, "n_filters": 16, "temporal_kernel": 7, "pool_kernel": 6, "pool_stride": 3}


def test_loss_is_low_for_correct_predictions_and_high_for_wrong_ones():
    torch.manual_seed(0)
    table = torch.randn(10, 16)
    image = torch.arange(10)
    loss = RetrievalLoss(temperature=0.05)
    good, stats = loss(table[image] * 3.0, image, table)
    bad, _ = loss(table[image.roll(1)], image, table)
    assert good < 0.1 < bad
    assert stats["batch_top1"] == 1.0 and stats["n_candidates"] == 10


def test_epochs_of_the_same_image_are_not_negatives_of_each_other():
    torch.manual_seed(0)
    table = torch.randn(6, 8)
    image = torch.tensor([2, 2, 2, 5])
    loss, stats = RetrievalLoss(temperature=0.05)(table[image], image, table)
    # Two candidates only (images 2 and 5); a naive batch loss would see 4 and
    # could never drive the loss on the three duplicates to zero.
    assert stats["n_candidates"] == 2
    assert loss < 0.05


def test_extra_negatives_enlarge_the_candidate_set_without_duplicating_positives():
    torch.manual_seed(0)
    table = torch.randn(50, 8)
    image = torch.tensor([0, 1, 2])
    generator = torch.Generator().manual_seed(0)
    criterion = RetrievalLoss(extra_negatives=30)
    _, stats = criterion(table[image], image, table, torch.arange(50), generator)
    assert 3 < stats["n_candidates"] <= 33
    _, stats = criterion(table[image], image, table, torch.tensor([0, 1, 2]), generator)
    assert stats["n_candidates"] == 3


def test_mse_term_and_temperature_clamp():
    table = torch.randn(4, 8)
    image = torch.arange(4)
    with_mse, stats = RetrievalLoss(mse_weight=1.0)(torch.randn(4, 8), image, table)
    assert "loss_mse" in stats and with_mse > stats["loss_contrastive"]
    criterion = RetrievalLoss()
    with torch.no_grad():
        criterion.log_scale.fill_(50.0)
    _, stats = criterion(table, image, table)
    assert stats["logit_scale"] == pytest.approx(100.0)
    assert not list(RetrievalLoss(learn_temperature=False).parameters())


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    assert _lr_factor(0, 10, 100, 0.01) == pytest.approx(0.1)
    assert _lr_factor(9, 10, 100, 0.01) == pytest.approx(1.0)
    assert _lr_factor(10, 10, 100, 0.01) == pytest.approx(1.0)
    assert _lr_factor(100, 10, 100, 0.01) == pytest.approx(0.01)
    factors = [_lr_factor(s, 10, 100, 0.01) for s in range(10, 101)]
    assert all(a >= b for a, b in itertools.pairwise(factors))


def _setup(small_epochs, tmp_path, name):
    torch.manual_seed(0)
    source = ArraySource(small_epochs)
    splits = make_splits(source, SPLIT)
    model = build_model("conv", source.n_chans, source.n_times, source.embed_dim, **PARAMS)
    tracker = RunTracker(name, CONFIG, root=tmp_path / "experiments", repo_dir=tmp_path)
    return source, splits, model, tracker


def test_training_beats_chance_and_restores_the_best_checkpoint(small_epochs, tmp_path):
    source, splits, model, tracker = _setup(small_epochs, tmp_path, "learn")
    before = evaluate_source(model, source, splits["val"])
    stats = train_model(model, source, splits, CONFIG, tracker, tmp_path / "ckpt")
    after = evaluate_source(model, source, splits["val"])

    assert stats["steps"] == stats["total_steps"] == 4 * math.ceil(len(splits["train"]) / 64)
    assert after["top5_acc_subject_agg"] > before["top5_acc_subject_agg"]
    assert after["top5_acc_subject_agg"] > 1.5 * after["chance_top5"]
    assert stats["best_monitor"] == pytest.approx(after["top5_acc_subject_agg"])
    assert (tmp_path / "ckpt" / "best.pt").exists() and (tmp_path / "ckpt" / "last.pt").exists()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_interrupted_run_resumes_to_the_same_result(small_epochs, tmp_path, monkeypatch):
    source, splits, model, tracker = _setup(small_epochs, tmp_path, "straight")
    train_model(model, source, splits, CONFIG, tracker, tmp_path / "straight")
    expected = {k: v.clone() for k, v in model.state_dict().items()}

    # Same run, killed during its third evaluation (after two checkpoints).
    source, splits, model, tracker = _setup(small_epochs, tmp_path, "killed")
    real, calls = train_module.evaluate_source, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate_source", flaky)
    with pytest.raises(KeyboardInterrupt):
        train_model(model, source, splits, CONFIG, tracker, tmp_path / "resumed")
    monkeypatch.setattr(train_module, "evaluate_source", real)

    source, splits, model, tracker = _setup(small_epochs, tmp_path, "resumed")
    stats = train_model(model, source, splits, CONFIG, tracker, tmp_path / "resumed")
    assert stats["steps"] == stats["total_steps"]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=1e-6, rtol=1e-5)


def test_patience_stops_training_early(small_epochs, tmp_path):
    source, splits, model, tracker = _setup(small_epochs, tmp_path, "patience")
    config = {**CONFIG, "train": {**CONFIG["train"], "epochs": 30, "lr": 0.0, "patience": 2}}
    stats = train_model(model, source, splits, config, tracker, tmp_path / "ckpt")
    assert stats["early_stopped"] is True
    assert stats["steps"] < stats["total_steps"]


def test_channel_dropout_runs_and_is_off_at_evaluation(small_epochs, tmp_path):
    source, splits, model, tracker = _setup(small_epochs, tmp_path, "dropout")
    config = {**CONFIG, "train": {**CONFIG["train"], "epochs": 1, "channel_dropout": 0.3}}
    train_model(model, source, splits, config, tracker, tmp_path / "ckpt")
    first = evaluate_source(model, source, splits["val"])
    assert first == evaluate_source(model, source, splits["val"])
