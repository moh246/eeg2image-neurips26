#!/usr/bin/env python
"""Run a freshly packaged submission through the organisers' benchmark code.

This is the strongest local check available: the same Benchopt objective the
platform executes loads our ``submission.py`` and scores it. It uses the kit's
no-download ``Simulated`` dataset, so the score itself is meaningless; what it
proves is that the ZIP loads, predicts and is scored without error.

    make contract-replay

Requires the organisers' stack (``make setup-official``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from eeg2img.config import load_config
from eeg2img.pipeline import run_training

# The kit's Simulated dataset: 8 channels x 120 samples, 32-d targets.
SIMULATED_SHAPE = ["data.params.n_chans=8", "data.params.n_times=120", "data.params.embed_dim=32"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kit", required=True, help="checkout of the starter kit")
    parser.add_argument("--config", default="configs/experiment/smoke_ridge.yaml")
    args = parser.parse_args(argv)

    benchopt = Path(sys.executable).with_name("benchopt")
    if not benchopt.exists():
        print("benchopt is not installed; run `make setup-official` first", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        overrides = [*SIMULATED_SHAPE, f"tracking.root={tmp}/experiments",
                     f"output_root={tmp}/outputs", "name=replay"]  # fmt: skip
        record = run_training(load_config(args.config, overrides))
        submission_dir = Path(record["zip_path"]).parent
        env = {**os.environ, "COMPET_SUBMISSION_DIR": str(submission_dir), "COMPET_DEVICE": "cpu"}
        command = [str(benchopt), "run", "tracks/image_decoding", "-d", "Simulated",
                   "-s", str(submission_dir / "submission.py"), "--no-plot"]  # fmt: skip
        result = subprocess.run(command, cwd=args.kit, env=env, capture_output=True, text=True)
    print(result.stdout[-2000:])
    ok = result.returncode == 0 and "done" in result.stdout and "error" not in result.stdout.lower()
    if not ok:
        print(result.stderr[-4000:], file=sys.stderr)
    print("contract replay:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
