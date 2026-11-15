#!/usr/bin/env python
"""Merge run records made on another machine into experiments/.

python scripts/import_runs.py ~/Downloads/eeg2img_runs.tar.gz
"""

from __future__ import annotations

import argparse

from eeg2img.results import import_runs


def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="an experiments directory or a .tar.gz of one")
    parser.add_argument("--experiments", default="experiments", help="directory to merge into")
    args = parser.parse_args(argv)
    added = import_runs(args.source, args.experiments)
    print(f"imported {len(added)} run(s)")
    for run_id in added:
        print(f"  {run_id}")
    return added


if __name__ == "__main__":
    main()
