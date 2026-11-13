"""Forward-pass smoke tests, closed-form fitting and an output regression test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg2img import contract
from eeg2img.engine.evaluate import evaluate, predict
from eeg2img.engine.fit import fit_model
from eeg2img.models import MODEL_REGISTRY, build_model
from eeg2img.pipeline import make_split

SPLIT = {"holdout_subjects": [4, 5], "holdout_image_fraction": 0.25, "seed": 33}


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_forward_pass_on_contract_shaped_input(name):
    model = build_model(
        name, contract.N_CHANS, contract.N_TIMES, contract.EMBED_DIM, ch_names=contract.EMOTIV_32
    ).eval()
    x = torch.randn(3, contract.N_CHANS, contract.N_TIMES)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (3, contract.EMBED_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_unknown_model_lists_the_registry():
    with pytest.raises(KeyError, match="registered models"):
        build_model("nope", 2, 2, 2)


def test_mean_floor_predicts_the_training_mean(small_epochs):
    train, test = make_split(small_epochs, SPLIT)
    model = build_model("mean", train.n_chans, train.n_times, train.embed_dim)
    fit_model(model, train, {})
    out = predict(model, test.eeg)
    np.testing.assert_allclose(out[0], train.targets.mean(axis=0), rtol=1e-5)
    assert np.ptp(out, axis=0).max() == 0
    assert evaluate(model, test)["top5_acc"] == pytest.approx(evaluate(model, test)["chance_top5"])


def test_ridge_recovers_a_noiseless_linear_map():
    rng = np.random.default_rng(0)
    from eeg2img.data.epochs import EpochSet

    eeg = rng.standard_normal((500, 3, 4)).astype(np.float32)
    weight = rng.standard_normal((12, 5))
    targets = (eeg.reshape(500, 12) @ weight + 0.5).astype(np.float32)
    data = EpochSet(eeg, np.arange(500), np.zeros(500, dtype=np.int64), targets,
                    ("Cz", "Oz", "Pz"), 120.0).validate()  # fmt: skip
    model = build_model("ridge", 3, 4, 5)
    fit_model(model, data, {"alpha": 1e-6})
    np.testing.assert_allclose(predict(model, eeg), targets, atol=1e-3)
    np.testing.assert_allclose(model.linear.bias.detach().numpy(), 0.5, atol=1e-3)


def test_ridge_generalises_to_new_subjects_and_images(small_epochs):
    train, test = make_split(small_epochs, SPLIT)
    model = build_model("ridge", train.n_chans, train.n_times, train.embed_dim)
    fit_model(model, train, {"alpha": 100.0})
    scores = evaluate(model, test)
    assert scores["top5_acc_subject_agg"] > 2 * scores["chance_top5"]
    # Averaging repeats must help, not hurt, on noisy single trials.
    assert scores["top1_acc_subject_agg"] >= scores["top1_acc"]


def test_ridge_output_regression(small_epochs):
    """A fixed input must keep producing the same output (value, shape and range)."""
    train, _ = make_split(small_epochs, SPLIT)
    model = build_model("ridge", train.n_chans, train.n_times, train.embed_dim)
    fit_model(model, train, {"alpha": 100.0})
    out = predict(model, small_epochs.eeg[:4])
    assert out.shape == (4, 16)
    assert np.abs(out).max() < 20
    np.testing.assert_allclose(out[0, :3], RIDGE_REFERENCE, rtol=1e-3, atol=1e-4)


#: First three outputs for epoch 0 of the fixture. Re-pinned when the split moved to
#: ``data.source.make_splits``, which selects held-out images with a different generator.
RIDGE_REFERENCE = [1.04266, 0.51868, 0.06155]
