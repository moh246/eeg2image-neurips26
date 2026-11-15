#!/usr/bin/env python
"""Score a trained package on inputs with only some of the training electrodes.

    python scripts/evaluate_channels.py --config configs/experiment/alljoined_conv.yaml \
        --package outputs/<run_id> --keep 32 24 16 8 --out experiments/channels/<run_id>.json

Rebuilds the packaged model for each channel subset through the platform code
path and reports the competition metric on the chosen validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg2img.config import load_config, require
from eeg2img.data.source import build_source, make_splits
from eeg2img.engine.montage_eval import evaluate_channel_subsets
from eeg2img.pipeline import resolve_device


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML the package came from")
    parser.add_argument("--package", required=True, help="directory with weights.pt, config.json")
    parser.add_argument("--keep", type=int, nargs="+", required=True, help="channel counts")
    parser.add_argument("--draws", type=int, default=3, help="random subsets per channel count")
    parser.add_argument("--split", default="val_test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="write the report to this JSON file")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    source = build_source(require(config, "data"), resolve_device(config.get("device", "auto")))
    indices = make_splits(source, require(config, "split"))[args.split]
    package = json.loads((Path(args.package) / "config.json").read_text(encoding="utf-8"))
    report = {
        "run_id": package.get("run_id"),
        "model": package["model"]["name"],
        "split": args.split,
        "metric": "top5_acc_subject_agg",
        "seed": args.seed,
        "rows": evaluate_channel_subsets(
            args.package, source, indices, args.keep, draws=args.draws, seed=args.seed,
            batch_size=int((config.get("eval") or {}).get("batch_size", 1024)),
        ),
    }  # fmt: skip
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["rows"]:
        spread = "" if row["std"] is None else f" ± {100 * row['std']:.2f}"
        score = f"{100 * row['mean']:.2f}{spread}"
        print(f"{report['model']:>20} {row['n_channels']:>3} channels: {score}")
    return report


if __name__ == "__main__":
    main()
