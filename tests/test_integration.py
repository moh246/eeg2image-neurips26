"""End-to-end: config -> train -> evaluate -> package -> contract check."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytestmark = pytest.mark.integration


def _run(config_name, tmp_path, *overrides):
    import train

    return train.main([
        "--config", f"configs/experiment/{config_name}.yaml",
        f"tracking.root={tmp_path / 'experiments'}",
        f"output_root={tmp_path / 'outputs'}",
        # Keep CI fast: a smaller problem than the shipped smoke config.
        "data.params.n_chans=8", "data.params.n_times=30", "data.params.embed_dim=16",
        "data.params.noise=6.0", "fit.alpha=100.0",
        *overrides,
    ])  # fmt: skip


def test_ridge_trains_scores_and_packages(tmp_path):
    record = _run("smoke_ridge", tmp_path)
    summary = record["summary"]
    test = summary["test"]

    # It learned something that transfers to unseen subjects and images.
    assert test["chance_top5"] == pytest.approx(0.25)
    assert test["top5_acc_subject_agg"] > 0.6
    assert test["top5_acc"] > 0.4
    # The uploaded artefact reproduces the measured model and passes the contract.
    assert summary["package_parity_max_abs_diff"] < 1e-5
    assert summary["contract_check_ok"] is True

    with zipfile.ZipFile(record["zip_path"]) as archive:
        assert "submission.py" in archive.namelist()
    run_dir = Path(record["run_dir"])
    for name in ("config.yaml", "meta.json", "metrics.jsonl", "summary.json"):
        assert (run_dir / name).is_file()
    assert json.loads((run_dir / "meta.json").read_text())["git_commit"]


def test_mean_floor_scores_chance(tmp_path):
    record = _run("smoke_mean", tmp_path)
    test = record["summary"]["test"]
    assert test["top5_acc_subject_agg"] == pytest.approx(test["chance_top5"])
    assert record["summary"]["contract_check_ok"] is True


def test_same_seed_gives_identical_metrics(tmp_path):
    first = _run("smoke_ridge", tmp_path / "a")["summary"]["test"]
    again = _run("smoke_ridge", tmp_path / "b")["summary"]["test"]
    assert first == again
