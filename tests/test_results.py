"""Run-record import and the seed-aggregated results table."""

from __future__ import annotations

import json
import tarfile

import pytest

from eeg2img.results import import_runs, read_registry, summarize, to_markdown


def _record(name: str, seed: int, val: float, dirty: bool = False) -> dict:
    split = {"top5_acc_subject_agg": val, "chance_top5": 0.01, "n_candidates": 500}
    return {
        "run_id": f"2026_{name}_{seed}",
        "name": name,
        "seed": seed,
        "git_dirty": dirty,
        "summary": {"fit": {"n_parameters": 1234}, "val": split, "val_test": split},
    }


def _write_experiments(root, records):
    for record in records:
        run_dir = root / "runs" / record["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps(record))
    with (root / "registry.jsonl").open("w") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in records)
    # A run that is still in progress has a directory but no registry line.
    (root / "runs" / "unfinished").mkdir()


def test_import_merges_only_new_finished_runs(tmp_path):
    source, target = tmp_path / "remote" / "experiments", tmp_path / "experiments"
    _write_experiments(source, [_record("a", 0, 0.1), _record("a", 1, 0.3)])
    archive = tmp_path / "runs.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="experiments")

    assert import_runs(archive, target) == ["2026_a_0", "2026_a_1"]
    assert (target / "runs" / "2026_a_1" / "summary.json").exists()
    assert not (target / "runs" / "unfinished").exists()
    assert import_runs(source, target) == [], "importing twice must add nothing"
    assert len(read_registry(target / "registry.jsonl")) == 2


def test_import_refuses_runs_from_uncommitted_code(tmp_path):
    source = tmp_path / "experiments"
    _write_experiments(source, [_record("a", 0, 0.1, dirty=True)])
    with pytest.raises(ValueError, match="uncommitted"):
        import_runs(source, tmp_path / "target")


def test_summarize_aggregates_over_seeds():
    records = [_record("a", 1, 0.3), _record("a", 0, 0.1), _record("b", 0, 0.5)]
    rows = {row["name"]: row for row in summarize(records)}
    assert rows["a"]["seeds"] == [0, 1]
    assert rows["a"]["val"]["mean"] == pytest.approx(0.2)
    assert rows["a"]["val"]["std"] == pytest.approx(0.1414, abs=1e-4)
    assert rows["b"]["val"]["std"] is None, "one seed has no spread"

    table = to_markdown(list(rows.values()))
    assert "| a | 2 | 1,234 | 20.00 ± 14.14 |" in table
    assert "| b | 1 | 1,234 | 50.00 |" in table
    assert "500-way, chance 1.00" in table
    assert to_markdown([]) == "_no finished runs_\n"


def test_resolve_packages_and_evaluation_records(tmp_path):
    from eeg2img.results import resolve_packages, write_evaluation

    records = [_record("a", 1, 0.3), _record("a", 0, 0.1), _record("b", 0, 0.5)]
    for record in records[:2]:
        package = tmp_path / "outputs" / record["run_id"]
        package.mkdir(parents=True)
        (package / "weights.pt").write_bytes(b"")

    found = resolve_packages(records, tmp_path / "outputs", "a")
    assert [path.name for path in found] == ["2026_a_0", "2026_a_1"], "ordered by seed"
    assert len(resolve_packages(records, tmp_path / "outputs", "a", seeds=[1])) == 1
    with pytest.raises(LookupError):
        resolve_packages(records, tmp_path / "outputs", "missing")
    with pytest.raises(FileNotFoundError):
        resolve_packages(records, tmp_path / "outputs", "b")

    source, target = tmp_path / "remote", tmp_path / "local"
    _write_experiments(source, records[:1])
    path = write_evaluation(source, "ensemble_a", {"kind": "ensemble", "value": 1}, tmp_path)
    saved = json.loads(path.read_text())
    assert saved["label"] == "ensemble_a" and saved["value"] == 1 and "git_commit" in saved
    assert import_runs(source, target) == ["2026_a_1", "evaluations/ensemble_a.json"]
    assert (target / "evaluations" / "ensemble_a.json").exists()
    assert import_runs(source, target) == []


def test_results_block_is_replaced_between_markers():
    from eeg2img.results import RESULTS_END, RESULTS_START, replace_results_block

    document = f"intro\n{RESULTS_START}\nold\n{RESULTS_END}\noutro\n"
    updated = replace_results_block(document, "| new |\n")
    assert updated == f"intro\n{RESULTS_START}\n| new |\n{RESULTS_END}\noutro\n"
    assert replace_results_block(updated, "| new |\n") == updated
    with pytest.raises(ValueError, match="markers"):
        replace_results_block("no markers here", "x")


@pytest.mark.parametrize("document", ["README.md", "docs/report.md"])
def test_committed_results_tables_match_the_registry(document):
    """Numbers in the README and report are generated from run records, not typed."""
    from pathlib import Path

    from eeg2img.results import RESULTS_END, RESULTS_START

    root = Path(__file__).resolve().parents[1]
    text = (root / document).read_text(encoding="utf-8")
    rows = summarize(read_registry(root / "experiments" / "registry.jsonl"))
    table = to_markdown(rows)
    block = text[text.index(RESULTS_START) + len(RESULTS_START) : text.index(RESULTS_END)]
    assert block.strip() == table.strip(), f"run scripts/summarize_runs.py --write {document}"
