#!/usr/bin/env python
"""Compute the frozen DINOv2-giant target embedding of every stimulus in the store.

    python scripts/embed_images.py --config configs/data/alljoined16m.yaml

Uses the organisers' own extractor, so targets match the competition gallery.
A GPU is strongly recommended. Resumes if interrupted.
Needs the targets extra: pip install -e ".[data,targets]".
"""

from __future__ import annotations

import argparse

import numpy as np

from eeg2img.config import load_config, require
from eeg2img.data.prepare import embed_store_images


def _progress(items):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="image chunks")


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data/alljoined16m.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    data_config = require(load_config(args.config, args.overrides), "data")
    out_path = embed_store_images(data_config, device=args.device, progress=_progress)
    table = np.load(out_path, mmap_mode="r")
    print(f"wrote {out_path}: shape={table.shape} dtype={table.dtype}")
    return str(out_path)


if __name__ == "__main__":
    main()
