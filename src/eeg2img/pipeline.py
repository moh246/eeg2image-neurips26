"""The train -> evaluate -> package pipeline behind the command-line scripts.

Scripts in ``scripts/`` only parse arguments and call into this module, so the
same code path runs from a terminal, from Colab and from the test suite.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from eeg2img.config import require
from eeg2img.data.epochs import EpochSet
from eeg2img.data.source import ArraySource, build_source, make_splits
from eeg2img.engine.fit import CLOSED_FORM_MODELS, fit_model
from eeg2img.engine.train import evaluate_source, predict_source, train_model
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


def make_split(epochs: EpochSet, split_config: Mapping[str, Any]) -> tuple[EpochSet, EpochSet]:
    """Split an in-memory set into train and a held-out-subject, held-out-image set."""
    source = ArraySource(epochs)
    splits = make_splits(source, split_config)
    return source.epochset(splits["train"]), source.epochset(splits["val"])


def _subsample(indices: torch.Tensor, limit: int | None, seed: int) -> torch.Tensor:
    if limit is None or len(indices) <= limit:
        return indices
    generator = torch.Generator().manual_seed(seed)
    return indices[torch.randperm(len(indices), generator=generator)[:limit]]


def run_training(config: Mapping[str, Any], repo_dir: str | Path = ".") -> dict[str, Any]:
    """Run one experiment end to end and return its record.

    Steps: seed, load data, split, fit, score every validation split with the
    competition metric, package into a submission ZIP, verify that the packaged
    model reproduces the in-memory predictions, and log everything.
    """
    name = require(config, "name")
    seed = int(require(config, "seed"))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))

    source = build_source(require(config, "data"), device)
    splits = make_splits(source, require(config, "split"))

    model_config = require(config, "model")
    normalize_output = bool(model_config.get("normalize_output", True))
    model = build_model(
        model_config["name"],
        n_chans=source.n_chans,
        n_times=source.n_times,
        n_outputs=source.embed_dim,
        ch_names=source.ch_names,
        **dict(model_config.get("params") or {}),
    )

    tracker = RunTracker(
        name, config, root=config.get("tracking", {}).get("root", "experiments"), repo_dir=repo_dir
    )
    out_dir = Path(config.get("output_root", "outputs")) / tracker.run_id

    if model_config["name"] in CLOSED_FORM_MODELS:
        fit_config = config.get("fit") or {}
        subset = _subsample(splits["train"], fit_config.get("max_train_events"), seed)
        fit_stats = fit_model(model, source.epochset(subset), fit_config)
        fit_stats["n_fit_events"] = len(subset)
    else:
        checkpoint_dir = require(config, "train").get("checkpoint_dir") or out_dir / "checkpoints"
        fit_stats = train_model(
            model, source, splits, config, tracker, checkpoint_dir, normalize_output
        )
    model.to(device)

    eval_config = config.get("eval") or {}
    batch_size = int(eval_config.get("batch_size", 1024))
    top_k = int(eval_config.get("top_k", 5))
    scores: dict[str, Any] = {}
    for split_name, indices in splits.items():
        if split_name == "train":
            indices = _subsample(indices, int(eval_config.get("max_train_events", 20000)), seed)
        scores[split_name] = evaluate_source(
            model, source, indices, batch_size, normalize_output, k=top_k
        )
        tracker.log(fit_stats.get("steps", 0), split=f"final/{split_name}", **scores[split_name])

    probe = splits["val"][:64]
    reference = predict_source(model, source, probe, normalize=normalize_output)
    probe_input = source.batch(probe).cpu()

    zip_path = write_package(
        model.cpu(),
        out_dir,
        model_config=model_config,
        ch_names=source.ch_names,
        n_times=source.n_times,
        n_outputs=source.embed_dim,
        sfreq=source.sfreq,
        normalize_output=normalize_output,
        solver_name=f"eeg2img-{name}",
        commit=tracker.meta["git_commit"],
        extra={"run_id": tracker.run_id},
    )

    # The artefact we upload must score what we measured: reload it through the
    # platform code path and compare against the in-memory model.
    packaged = load_packaged_model(
        {
            "submission_dir": out_dir,
            "device": "cpu",
            "ch_names": list(source.ch_names),
            "n_chans": source.n_chans,
            "n_times": source.n_times,
            "n_outputs": source.embed_dim,
        }
    )
    parity = float((packaged.predict(probe_input) - reference).abs().max())

    report = check_submission(
        zip_path,
        n_outputs=source.embed_dim,
        ch_names=list(source.ch_names),
        n_times=source.n_times,
        sfreq=source.sfreq,
    )

    summary = {
        **scores,
        "fit": fit_stats,
        "n_events": {split_name: len(indices) for split_name, indices in splits.items()},
        "device": device,
        "package_parity_max_abs_diff": parity,
        "contract_check_ok": bool(report["ok"]),
    }
    record = tracker.finish(**summary)
    return {**record, "zip_path": str(zip_path), "run_dir": str(tracker.run_dir)}
