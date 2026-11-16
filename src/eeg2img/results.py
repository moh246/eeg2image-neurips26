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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from eeg2img.tracking import git_state

#: The competition metric: Top-5 after averaging a participant's repeats of an image.
PRIMARY_METRIC = "top5_acc_subject_agg"
#: Sub-directory of ``experiments`` holding evaluations of finished runs.
EVALUATIONS_DIR = "evaluations"


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

        # Evaluations of finished runs (ensembles, channel subsets) are single JSON files.
        for path in sorted((source / EVALUATIONS_DIR).glob("*.json")):
            target = experiments_dir / EVALUATIONS_DIR / path.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                added.append(f"{EVALUATIONS_DIR}/{path.name}")
    return added


def resolve_packages(
    records: Sequence[Mapping[str, Any]],
    outputs_root: str | Path,
    name: str,
    seeds: Sequence[int] | None = None,
) -> list[Path]:
    """Package directories of the finished runs of experiment ``name``, by seed."""
    runs = sorted((r for r in records if r["name"] == name), key=lambda r: r["seed"])
    if seeds is not None:
        runs = [record for record in runs if record["seed"] in set(seeds)]
    if not runs:
        raise LookupError(f"no finished run of {name!r} (seeds {seeds}) in the registry")
    packages = [Path(outputs_root) / record["run_id"] for record in runs]
    missing = [str(path) for path in packages if not (path / "weights.pt").exists()]
    if missing:
        raise FileNotFoundError(f"packages not found: {', '.join(missing)}")
    return packages


def write_evaluation(
    experiments_dir: str | Path,
    label: str,
    payload: Mapping[str, Any],
    repo_dir: str | Path = ".",
) -> Path:
    """Write an evaluation of finished runs to ``evaluations/<label>.json``.

    The record carries the commit of the evaluating code and the time, like a
    run record does.
    """
    state = git_state(repo_dir)
    record = {
        "label": label,
        "git_commit": state["commit"],
        "git_dirty": state["dirty"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        **payload,
    }
    path = Path(experiments_dir) / EVALUATIONS_DIR / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summarize(
    records: Sequence[dict[str, Any]],
    splits: Sequence[str] = ("val", "val_test"),
    metric: str = PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    """One row per experiment name: seeds, and mean/std/chance of ``metric`` per split.

    Runs are grouped by experiment name, so each experiment config must keep
    its hyperparameters fixed across seeds; a variant gets its own name. Only
    the earliest run of each seed enters the statistics.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["name"]].append(record)

    rows = []
    for name, runs in groups.items():
        # A seed that was run again (for example to retrain lost weights) counts once: the
        # earliest run is reported and later ones are listed as reruns.
        first: dict[int, dict[str, Any]] = {}
        reruns = []
        for record in sorted(runs, key=lambda record: record["run_id"]):
            if record["seed"] in first:
                reruns.append(record["run_id"])
            else:
                first[record["seed"]] = record
        runs = [first[seed] for seed in sorted(first)]
        fit = runs[0]["summary"].get("fit", {})
        row: dict[str, Any] = {
            "name": name,
            "seeds": [record["seed"] for record in runs],
            "run_ids": [record["run_id"] for record in runs],
            "reruns": reruns,
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


RESULTS_START = "<!-- results:start (generated by scripts/summarize_runs.py; do not edit) -->"
RESULTS_END = "<!-- results:end -->"


def render_results_block(table: str) -> str:
    """The generated block that documents embed, markers included."""
    return f"{RESULTS_START}\n{table}{RESULTS_END}"


def replace_results_block(document: str, table: str) -> str:
    """Return ``document`` with the text between the result markers replaced by ``table``."""
    start, end = document.find(RESULTS_START), document.find(RESULTS_END)
    if start < 0 or end < start:
        raise ValueError("document has no results block markers")
    return document[:start] + render_results_block(table) + document[end + len(RESULTS_END) :]
