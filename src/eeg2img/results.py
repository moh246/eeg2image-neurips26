"""Collect run records and turn the registry into a results table.

Runs happen on a remote GPU machine; their records (see ``tracking``) are
brought back as an archive of the ``experiments`` directory.
:func:`import_runs` merges such an archive into the repository's own
``experiments`` directory, and :func:`summarize` aggregates the registry over
seeds so that every number in the README and report is generated, not typed.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

#: The competition metric: Top-5 after averaging a participant's repeats of an image.
PRIMARY_METRIC = "top5_acc_subject_agg"


def read_registry(path: str | Path) -> list[dict[str, Any]]:
    """Registry lines as dicts, in file order; an absent file is an empty registry."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def import_runs(source: str | Path, experiments_dir: str | Path) -> list[str]:
    """Merge finished runs from ``source`` into ``experiments_dir``; return new run ids.

    ``source`` is an ``experiments`` directory or a ``.tar.gz`` of one. Only
    runs listed in the source registry are taken (a run still in progress has
    no registry line), runs already present are left untouched, and a run
    made from uncommitted code is refused because it cannot be reproduced.
    """
    source, experiments_dir = Path(source), Path(experiments_dir)
    with tempfile.TemporaryDirectory() as scratch:
        if source.is_file():
            with tarfile.open(source) as archive:
                archive.extractall(scratch, filter="data")
            candidates = [p.parent for p in Path(scratch).rglob("registry.jsonl")]
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one registry.jsonl in {source}, found {len(candidates)}"
                )
            source = candidates[0]

        known = {record["run_id"] for record in read_registry(experiments_dir / "registry.jsonl")}
        added = []
        for record in read_registry(source / "registry.jsonl"):
            run_id = record["run_id"]
            if run_id in known:
                continue
            if record.get("git_dirty"):
                raise ValueError(f"run {run_id} was made from uncommitted code")
            target = experiments_dir / "runs" / run_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / "runs" / run_id, target, dirs_exist_ok=True)
            with (experiments_dir / "registry.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            known.add(run_id)
            added.append(run_id)
    return added


def summarize(
    records: Sequence[dict[str, Any]],
    splits: Sequence[str] = ("val", "val_test"),
    metric: str = PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    """One row per experiment name: seeds, and mean/std/chance of ``metric`` per split.

    Runs are grouped by experiment name, so each experiment config must keep
    its hyperparameters fixed across seeds; a variant gets its own name.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["name"]].append(record)

    rows = []
    for name, runs in groups.items():
        runs = sorted(runs, key=lambda record: record["seed"])
        fit = runs[0]["summary"].get("fit", {})
        row: dict[str, Any] = {
            "name": name,
            "seeds": [record["seed"] for record in runs],
            "run_ids": [record["run_id"] for record in runs],
            "n_parameters": fit.get("n_parameters"),
        }
        for split in splits:
            values = [record["summary"][split][metric] for record in runs]
            row[split] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else None,
                "values": values,
                "chance": runs[0]["summary"][split]["chance_top5"],
                "n_candidates": runs[0]["summary"][split]["n_candidates"],
            }
        rows.append(row)
    return rows


def _percent(cell: dict[str, Any]) -> str:
    text = f"{100 * cell['mean']:.2f}"
    return text + (f" ± {100 * cell['std']:.2f}" if cell["std"] is not None else "")


def to_markdown(rows: Sequence[dict[str, Any]], splits: Sequence[str] = ("val", "val_test")) -> str:
    """Render :func:`summarize` rows as a Markdown table (accuracies in percent)."""
    if not rows:
        return "_no finished runs_\n"
    header = ["experiment", "seeds", "parameters"]
    for split in splits:
        cell = rows[0][split]
        header.append(
            f"{split} Top-5 % ({cell['n_candidates']}-way, chance {100 * cell['chance']:.2f})"
        )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        params = "—" if row["n_parameters"] is None else f"{row['n_parameters']:,}"
        cells = [row["name"], str(len(row["seeds"])), params]
        cells += [_percent(row[split]) for split in splits]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
