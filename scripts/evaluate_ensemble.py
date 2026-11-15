#!/usr/bin/env python
"""Score the seed ensemble of one or more experiments on the validation splits.

    python scripts/evaluate_ensemble.py --config configs/experiment/alljoined_conv.yaml \
        --names alljoined_conv --outputs outputs --label ensemble_conv

Averages the predictions of every finished run of the named experiments (all
seeds) and reports the competition metric next to each member's own score.
"""

from __future__ import annotations

import argparse
import json

from eeg2img.config import load_config, require
from eeg2img.data.source import build_source, make_splits
from eeg2img.engine.packages import evaluate_ensemble
from eeg2img.pipeline import resolve_device
from eeg2img.results import read_registry, resolve_packages, write_evaluation


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML (data and split)")
    parser.add_argument("--names", nargs="+", required=True, help="experiments to ensemble")
    parser.add_argument("--outputs", default="outputs", help="directory holding <run_id>/")
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--label", required=True, help="name of the evaluation record")
    parser.add_argument("--splits", nargs="+", default=["val", "val_test"])
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    records = read_registry(f"{args.experiments}/registry.jsonl")
    packages = [p for name in args.names for p in resolve_packages(records, args.outputs, name)]
    source = build_source(require(config, "data"), resolve_device(config.get("device", "auto")))
    splits = make_splits(source, require(config, "split"))
    batch_size = int((config.get("eval") or {}).get("batch_size", 1024))

    payload = {
        "kind": "ensemble",
        "experiments": args.names,
        "run_ids": [package.name for package in packages],
        "splits": {
            split: evaluate_ensemble(packages, source, splits[split], batch_size)
            for split in args.splits
        },
    }
    path = write_evaluation(args.experiments, args.label, payload)
    print(json.dumps(payload["splits"], indent=2))
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
