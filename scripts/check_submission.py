#!/usr/bin/env python
"""Check a submission ZIP against the Track 01 contract before uploading it.

    python scripts/check_submission.py outputs/<run_id>/submission.zip

Defaults are the sealed-phase contract: 32 Emotiv channels, 120 samples,
1536-d output. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys

from eeg2img import contract
from eeg2img.submission.check import check_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("zip_path")
    parser.add_argument("--n-outputs", type=int, default=contract.EMBED_DIM)
    parser.add_argument("--n-times", type=int, default=contract.N_TIMES)
    args = parser.parse_args(argv)

    report = check_submission(args.zip_path, n_outputs=args.n_outputs, n_times=args.n_times)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
