#!/usr/bin/env python
"""Train, evaluate and package one experiment.

python scripts/train.py --config configs/experiment/smoke_ridge.yaml
python scripts/train.py --config configs/experiment/smoke_ridge.yaml fit.alpha=10 seed=1
"""

from __future__ import annotations

import argparse
import json

from eeg2img.config import load_config
from eeg2img.pipeline import run_training


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    record = run_training(load_config(args.config, args.overrides))
    summary = record["summary"]
    scores = {name: summary[name] for name in summary["n_events"] if name != "train"}
    report = {"run_id": record["run_id"], "zip": record["zip_path"], "fit": summary["fit"]}
    print(json.dumps({**report, **scores}, indent=2))
    return record


if __name__ == "__main__":
    main()
