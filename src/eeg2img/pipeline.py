"""The train -> evaluate -> package pipeline behind the command-line scripts.

Scripts in ``scripts/`` only parse arguments and call into this module, so the
same code path runs from a terminal, from Colab and from the test suite.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg2img.config import require
from eeg2img.data.epochs import EpochSet
from eeg2img.data.synthetic import make_synthetic
from eeg2img.engine.evaluate import evaluate, predict
from eeg2img.engine.fit import fit_model
from eeg2img.models import build_model
from eeg2img.seed import seed_everything
from eeg2img.submission.builder import write_package
from eeg2img.submission.check import check_submission
from eeg2img.submission.runtime import load_packaged_model
from eeg2img.tracking import RunTracker


def resolve_device(name: str = "auto") -> str:
    """Map ``"auto"`` to the best available device; pass other names through."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_data(data_config: Mapping[str, Any]) -> EpochSet:
    """Load the dataset named by ``data.name``."""
    name = data_config["name"]
    params = dict(data_config.get("params") or {})
    if name == "synthetic":
        return make_synthetic(**params)
    raise KeyError(f"unknown dataset {name!r}")


def make_split(epochs: EpochSet, split_config: Mapping[str, Any]) -> tuple[EpochSet, EpochSet]:
    """Hold out whole subjects and whole images, as the sealed evaluation does.

    ``holdout_subjects`` is an explicit list. Held-out images are a seeded
    random ``holdout_image_fraction`` of the images present in the data.
    """
    holdout_subjects = frozenset(int(s) for s in split_config["holdout_subjects"])
    images = np.unique(epochs.image_index)
    rng = np.random.default_rng(int(split_config["seed"]))
    n_holdout = max(1, round(float(split_config["holdout_image_fraction"]) * len(images)))
    holdout_images = frozenset(rng.choice(images, size=n_holdout, replace=False).tolist())
    train, test = epochs.split(holdout_subjects, holdout_images)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"split produced an empty set (train={len(train)}, test={len(test)})")
    return train, test


def run_training(config: Mapping[str, Any], repo_dir: str | Path = ".") -> dict[str, Any]:
    """Run one experiment end to end and return its record.

    Steps: seed, load data, split, fit, score with the competition metric,
    package into a submission ZIP, verify that the packaged model reproduces
    the in-memory predictions, and log everything to the run tracker.
    """
    name = require(config, "name")
    seed = int(require(config, "seed"))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))

    epochs = build_data(require(config, "data"))
    train, test = make_split(epochs, require(config, "split"))

    model_config = require(config, "model")
    model = build_model(
        model_config["name"],
        n_chans=train.n_chans,
        n_times=train.n_times,
        n_outputs=train.embed_dim,
        **dict(model_config.get("params") or {}),
    )

    tracker = RunTracker(name, config, root=config.get("tracking", {}).get("root", "experiments"),
                         repo_dir=repo_dir)  # fmt: skip
    fit_stats = fit_model(model, train, config.get("fit") or {})

    eval_config = config.get("eval") or {}
    batch_size = int(eval_config.get("batch_size", 256))
    top_k = int(eval_config.get("top_k", 5))
    metrics = evaluate(model, test, batch_size=batch_size, device=device, k=top_k)
    train_metrics = evaluate(model, train, batch_size=batch_size, device=device, k=top_k)
    tracker.log(0, split="test", **metrics)
    tracker.log(0, split="train", **train_metrics)

    out_dir = Path(config.get("output_root", "outputs")) / tracker.run_id
    normalize_output = bool(model_config.get("normalize_output", True))
    zip_path = write_package(
        model.cpu(),
        out_dir,
        model_config=model_config,
        ch_names=train.ch_names,
        n_times=train.n_times,
        n_outputs=train.embed_dim,
        sfreq=train.sfreq,
        normalize_output=normalize_output,
        solver_name=f"eeg2img-{name}",
        commit=tracker.meta["git_commit"],
        extra={"run_id": tracker.run_id},
    )

    # The artefact we upload must score what we measured: reload it through the
    # platform code path and compare against the in-memory model.
    packaged = load_packaged_model(
        {"submission_dir": out_dir, "device": "cpu", "ch_names": list(test.ch_names),
         "n_chans": test.n_chans, "n_times": test.n_times, "n_outputs": test.embed_dim}
    )  # fmt: skip
    probe = torch.as_tensor(test.eeg[:64])
    reference = torch.as_tensor(predict(model, test.eeg[:64], device="cpu"))
    if normalize_output:
        reference = torch.nn.functional.normalize(reference, dim=-1)
    parity = float((packaged.predict(probe) - reference).abs().max())

    report = check_submission(
        zip_path, n_outputs=test.embed_dim, ch_names=list(test.ch_names),
        n_times=test.n_times, sfreq=test.sfreq,
    )  # fmt: skip

    summary = {
        "test": metrics,
        "train": train_metrics,
        "fit": fit_stats,
        "n_train": len(train),
        "n_test": len(test),
        "device": device,
        "package_parity_max_abs_diff": parity,
        "contract_check_ok": bool(report["ok"]),
    }
    record = tracker.finish(**summary)
    return {**record, "zip_path": str(zip_path), "run_dir": str(tracker.run_dir)}
