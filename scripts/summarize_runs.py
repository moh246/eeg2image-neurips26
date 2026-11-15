#!/usr/bin/env python
"""Print the results table: Top-5 per experiment, mean and std over seeds.

python scripts/summarize_runs.py                      # every experiment in the registry
python scripts/summarize_runs.py alljoined_conv alljoined_conv_transformer
"""

from __future__ import annotations

import argparse

from eeg2img.results import read_registry, summarize, to_markdown


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="experiments to include, in this order")
    parser.add_argument("--registry", default="experiments/registry.jsonl")
    parser.add_argument("--splits", nargs="+", default=["val", "val_test"])
    args = parser.parse_args(argv)

    rows = summarize(read_registry(args.registry), splits=args.splits)
    if args.names:
        order = {name: index for index, name in enumerate(args.names)}
        rows = sorted((row for row in rows if row["name"] in order), key=lambda r: order[r["name"]])
    table = to_markdown(rows, splits=args.splits)
    print(table, end="")
    return table


if __name__ == "__main__":
    main()
