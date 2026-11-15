#!/usr/bin/env python
"""Check that consecutive stimulus ids form concept groups, as the split assumes.

    python scripts/check_image_groups.py --config configs/experiment/alljoined_conv.yaml

Compares the target embeddings of neighbouring training images inside and
across the groups of ``split.images_per_group`` ids (see ``group_coherence``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg2img.config import load_config, require
from eeg2img.data.checks import group_coherence
from eeg2img.data.store import EpochStore


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument("overrides", nargs="*", help="dotted.key=value config overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    data_config = require(config, "data")
    group_size = int(require(config, "split").get("images_per_group", 1))
    store = EpochStore(data_config["store"])
    embeddings = np.load(data_config["embeddings"])
    keys = np.array([int(Path(name).stem) for name in store.images])
    pool = np.zeros(len(store.images), dtype=bool)
    pool[np.unique(store.image[~store.is_test])] = True

    report = {"images_per_group": group_size, "n_pool_images": int(pool.sum())}
    report.update(group_coherence(embeddings[pool], keys[pool], group_size))
    for other in (group_size - 1, group_size + 1):
        if other > 1:
            alt = group_coherence(embeddings[pool], keys[pool], other)
            report[f"clean_boundary_fraction_if_{other}"] = alt["clean_boundary_fraction"]
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
