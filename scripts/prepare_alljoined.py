#!/usr/bin/env python
"""Download Alljoined-1.6M and preprocess it into a compact epoch store.

    ALLJOINED_ACCEPT_LICENCE=1 python scripts/prepare_alljoined.py \
        --config configs/data/alljoined16m.yaml

The data is CC-BY-NC-ND-4.0; setting ALLJOINED_ACCEPT_LICENCE=1 accepts that
licence. Both the download and the preprocessing resume if interrupted.
Needs the data extra: pip install -e ".[data]".
"""

from __future__ import annotations

import argparse
import json

from eeg2img.config import load_config, require
from eeg2img.data import alljoined
from eeg2img.data.prepare import prepare_alljoined
from eeg2img.data.store import EpochStore


def _progress(items):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="recordings")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/alljoined16m.yaml")
    parser.add_argument("--skip-download", action="store_true", help="use files already on disk")
    parser.add_argument("--no-stimuli", action="store_true", help="download EEG only")
    parser.add_argument("--download-only", action="store_true", help="skip preprocessing")
    parser.add_argument(
        "--part", default=None, metavar="I/N",
        help="preprocess only every N-th recording from the I-th and do not assemble; "
        "run N such processes in parallel, then once more without --part",
    )  # fmt: skip
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)
    part = tuple(int(v) for v in args.part.split("/")) if args.part else None

    data_config = require(load_config(args.config, args.overrides), "data")
    if not args.skip_download:
        alljoined.download(
            data_config["root"],
            subjects=data_config.get("subjects"),
            sessions=data_config.get("sessions"),
            runs=data_config.get("runs"),
            include_stimuli=not args.no_stimuli,
        )
    if args.download_only:
        return {"downloaded": str(alljoined.dataset_dir(data_config["root"]))}
    out_dir = prepare_alljoined(data_config, progress=_progress, part=part)
    if part is not None:
        return {"part": args.part, "store": str(out_dir)}
    store = EpochStore(out_dir)
    summary = {key: store.meta[key] for key in
               ("n_recordings", "n_subjects", "n_events", "n_images", "total_samples")}  # fmt: skip
    summary["n_test_events"] = int(store.is_test.sum())
    summary["store"] = str(store.path)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
