"""Packaging: a single self-contained file that reproduces the trained model."""

from __future__ import annotations

import ast
import json
import zipfile

import numpy as np
import pytest
import torch

from eeg2img.engine.evaluate import predict
from eeg2img.engine.fit import fit_model
from eeg2img.models import build_model
from eeg2img.pipeline import make_split
from eeg2img.submission import (
    check_submission,
    load_packaged_model,
    render_submission_source,
    write_package,
)

SPLIT = {"holdout_subjects": [4, 5], "holdout_image_fraction": 0.25, "seed": 33}


@pytest.fixture(scope="module")
def packaged(small_epochs, tmp_path_factory):
    train, test = make_split(small_epochs, SPLIT)
    model = build_model("ridge", train.n_chans, train.n_times, train.embed_dim)
    fit_model(model, train, {"alpha": 100.0})
    out_dir = tmp_path_factory.mktemp("package")
    zip_path = write_package(
        model, out_dir, model_config={"name": "ridge", "params": {}}, ch_names=train.ch_names,
        n_times=train.n_times, n_outputs=train.embed_dim, sfreq=train.sfreq,
        normalize_output=True, commit="abc1234",
    )  # fmt: skip
    return model, test, out_dir, zip_path


def _meta(test, out_dir, **overrides):
    meta = {"submission_dir": out_dir, "device": "cpu", "ch_names": list(test.ch_names),
            "n_chans": test.n_chans, "n_times": test.n_times, "n_outputs": test.embed_dim,
            "sfreq": test.sfreq}  # fmt: skip
    return {**meta, **overrides}


def test_generated_source_is_self_contained():
    source = render_submission_source(commit="abc1234")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "eeg2img" not in imported
    assert imported <= {"__future__", "benchmark_utils", "collections", "json", "pathlib", "torch"}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {"Solver", "PackagedModel", "ChannelAligner"} <= classes
    assert "abc1234" in source


def test_zip_is_flat_with_one_python_file(packaged):
    *_, zip_path = packaged
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["config.json", "submission.py", "weights.pt"]
        config = json.loads(archive.read("config.json"))
    assert config["model"]["name"] == "ridge"
    assert config["source_commit"] == "abc1234"


def test_packaged_model_reproduces_the_trained_model(packaged):
    model, test, out_dir, _ = packaged
    loaded = load_packaged_model(_meta(test, out_dir))
    expected = torch.nn.functional.normalize(torch.as_tensor(predict(model, test.eeg)), dim=-1)
    got = loaded.predict(torch.as_tensor(test.eeg))
    assert got.shape == (len(test), test.embed_dim)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(got.norm(dim=-1), torch.ones(len(test)), atol=1e-5, rtol=0)


def test_prediction_is_invariant_to_input_channel_order(packaged):
    _, test, out_dir, _ = packaged
    x = torch.as_tensor(test.eeg[:16])
    reference = load_packaged_model(_meta(test, out_dir)).predict(x)
    perm = torch.randperm(test.n_chans, generator=torch.Generator().manual_seed(0))
    shuffled = load_packaged_model(
        _meta(test, out_dir, ch_names=[test.ch_names[i] for i in perm.tolist()])
    )
    torch.testing.assert_close(shuffled.predict(x[:, perm, :]), reference, atol=1e-6, rtol=0)


def test_time_axis_is_resampled_to_the_trained_length(packaged):
    _, test, out_dir, _ = packaged
    loaded = load_packaged_model(_meta(test, out_dir, n_times=test.n_times * 2))
    out = loaded.predict(torch.randn(3, test.n_chans, test.n_times * 2))
    assert out.shape == (3, test.embed_dim)


def test_embedding_dimension_mismatch_fails_loudly(packaged):
    _, test, out_dir, _ = packaged
    with pytest.raises(ValueError, match="embeddings but the task expects"):
        load_packaged_model(_meta(test, out_dir, n_outputs=test.embed_dim + 1))


def test_predict_keeps_no_state_between_calls(packaged):
    _, test, out_dir, _ = packaged
    loaded = load_packaged_model(_meta(test, out_dir))
    x = torch.as_tensor(test.eeg[:8])
    first = loaded.predict(x)
    loaded.predict(torch.randn(5, test.n_chans, test.n_times) * 50)
    torch.testing.assert_close(loaded.predict(x), first, atol=0, rtol=0)
    # A sample's prediction must not depend on what else is in the batch.
    torch.testing.assert_close(loaded.predict(x[:1]), first[:1], atol=1e-6, rtol=0)


def test_contract_check_passes_in_an_isolated_interpreter(packaged):
    _, test, _, zip_path = packaged
    report = check_submission(
        zip_path, n_outputs=test.embed_dim, ch_names=list(test.ch_names), n_times=test.n_times,
        sfreq=test.sfreq,
    )  # fmt: skip
    assert report["ok"], report
    assert report["shape"] == [7, test.embed_dim]


def test_contract_check_flags_a_nested_zip(packaged, tmp_path):
    _, test, out_dir, _ = packaged
    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        for name in ("submission.py", "weights.pt", "config.json"):
            archive.write(out_dir / name, arcname=f"my_submission/{name}")
    with zipfile.ZipFile(nested) as archive:
        assert all("/" in name for name in archive.namelist())
    with pytest.raises((RuntimeError, FileNotFoundError)):
        check_submission(nested, n_outputs=test.embed_dim, ch_names=list(test.ch_names),
                         n_times=test.n_times)  # fmt: skip


def test_weights_load_with_weights_only(packaged):
    *_, out_dir, _ = packaged
    state = torch.load(out_dir / "weights.pt", weights_only=True)
    assert set(state) == {"linear.weight", "linear.bias"}
    assert all(np.isfinite(v.numpy()).all() for v in state.values())
