"""Structured local experiment tracking.

Each run gets a directory under ``experiments/runs/`` holding small text files
that are committed to git, so a result can always be traced to the exact
config and code that produced it:

- ``config.yaml``   the fully resolved config;
- ``meta.json``     git commit, dirty flag, config hash, seed, library versions;
- ``metrics.jsonl`` one JSON object per logged step;
- ``summary.json``  the final metrics.

``experiments/registry.jsonl`` gets one line per finished run for side-by-side
comparison. Checkpoints and submission ZIPs are heavy and live under
``outputs/<run_id>/`` instead, which git ignores.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eeg2img.config import config_hash


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    """Commit hash and dirty flag of the repository containing ``cwd``."""
    cwd = Path(cwd)
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else None,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic | torch.Tensor):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


class RunTracker:
    """Record one run's config, provenance and metrics to disk."""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any],
        root: str | Path = "experiments",
        repo_dir: str | Path = ".",
    ) -> None:
        self.root = Path(root)
        git = git_state(repo_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{stamp}_{name}_{git['commit'][:7]}"
        self.run_dir = self.root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.config = dict(config)
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "started_utc": stamp,
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_hash": config_hash(config),
            "seed": config.get("seed"),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        }
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=True)
        self._write_json("meta.json", self.meta)

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
        (self.run_dir / filename).write_text(text + "\n", encoding="utf-8")

    def log(self, step: int, **metrics: Any) -> None:
        """Append one row of metrics for ``step``."""
        row = {"step": step, **metrics}
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=_jsonable) + "\n")

    def finish(self, **summary: Any) -> dict[str, Any]:
        """Write the final metrics and register the run."""
        record = {**self.meta, "summary": summary}
        self._write_json("summary.json", record)
        with (self.root / "registry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_jsonable) + "\n")
        return record
