"""Run tracking: every run records its config, commit and metrics."""

from __future__ import annotations

import json
import subprocess

import yaml

from eeg2img.config import config_hash
from eeg2img.tracking import RunTracker, git_state


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_run_records_config_commit_and_metrics(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "file.txt").write_text("x")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")

    config = {"name": "demo", "seed": 3, "fit": {"alpha": 1.0}}
    tracker = RunTracker("demo", config, root=tmp_path / "experiments", repo_dir=repo)
    tracker.log(0, loss=1.5)
    tracker.log(1, loss=0.5)
    record = tracker.finish(top5=0.25)

    commit = git_state(repo)["commit"]
    assert len(commit) == 40
    assert record["git_commit"] == commit and record["git_dirty"] is False
    assert record["config_hash"] == config_hash(config)
    assert record["seed"] == 3
    assert tracker.run_id.endswith(f"_demo_{commit[:7]}")

    assert yaml.safe_load((tracker.run_dir / "config.yaml").read_text()) == config
    lines = (tracker.run_dir / "metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows == [{"step": 0, "loss": 1.5}, {"step": 1, "loss": 0.5}]
    assert json.loads((tracker.run_dir / "summary.json").read_text())["summary"] == {"top5": 0.25}
    registry = (tmp_path / "experiments" / "registry.jsonl").read_text().splitlines()
    assert json.loads(registry[-1])["run_id"] == tracker.run_id


def test_dirty_tree_is_flagged(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "untracked.txt").write_text("x")
    assert git_state(tmp_path)["dirty"] is True


def test_outside_a_repository_the_commit_is_unknown(tmp_path):
    state = git_state(tmp_path)
    assert state["commit"] == "unknown"
