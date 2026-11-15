#!/usr/bin/env python
"""Score a trained package on inputs with only some of the training electrodes.

    python scripts/evaluate_channels.py --config configs/experiment/alljoined_conv.yaml \
        --name alljoined_conv --seed 0 --outputs outputs --keep 32 24 16 8

Rebuilds the packaged model for each channel subset through the platform code
path and reports the competition metric on the chosen validation split.
"""

from __future__ import annotations

import argparse

from eeg2img.config import load_config, require
from eeg2img.data.source import build_source, make_splits
from eeg2img.engine.montage_eval import evaluate_channel_subsets
from eeg2img.pipeline import resolve_device
from eeg2img.results import read_registry, resolve_packages, write_evaluation


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML the run came from")
    parser.add_argument("--name", required=True, help="experiment whose package is scored")
    parser.add_argument("--seed", type=int, default=0, help="seed of the run to score")
    parser.add_argument("--outputs", default="outputs", help="directory holding <run_id>/")
    parser.add_argument("--experiments", default="experiments")
    parser.add_argument("--keep", type=int, nargs="+", required=True, help="channel counts")
    parser.add_argument("--draws", type=int, default=3, help="random subsets per channel count")
    parser.add_argument("--split", default="val_test")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    records = read_registry(f"{args.experiments}/registry.jsonl")
    (package,) = resolve_packages(records, args.outputs, args.name, seeds=[args.seed])
    source = build_source(require(config, "data"), resolve_device(config.get("device", "auto")))
    indices = make_splits(source, require(config, "split"))[args.split]
    report = {
        "kind": "channel_subsets",
        "experiment": args.name,
        "run_id": package.name,
        "model": require(config, "model")["name"],
        "split": args.split,
        "metric": "top5_acc_subject_agg",
        "subset_seed": args.subset_seed,
        "rows": evaluate_channel_subsets(
            package, source, indices, args.keep, draws=args.draws, seed=args.subset_seed,
            batch_size=int((config.get("eval") or {}).get("batch_size", 1024)),
        ),
    }  # fmt: skip
    path = write_evaluation(args.experiments, f"channels_{args.name}_seed{args.seed}", report)
    print(f"wrote {path}")
    for row in report["rows"]:
        spread = "" if row["std"] is None else f" ± {100 * row['std']:.2f}"
        score = f"{100 * row['mean']:.2f}{spread}"
        print(f"{report['model']:>20} {row['n_channels']:>3} channels: {score}")
    return report


if __name__ == "__main__":
    main()
