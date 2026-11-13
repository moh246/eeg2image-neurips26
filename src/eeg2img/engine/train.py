"""Gradient training of an EEG encoder against frozen image embeddings.

Design points that matter for this task:

- **Model selection is on the sealed-like split.** Checkpoints are chosen by
  subject-aggregated Top-5 on held-out subjects viewing held-out images,
  computed with the competition scorer, not by a batch-level proxy.
- **Resumable.** Colab sessions end without warning, so the full training state
  is written at every evaluation and a run restarted with the same
  ``checkpoint_dir`` continues from it. Data order is a function of
  ``(seed, epoch)``, so a resumed run visits the same batches.
- **No DataLoader.** Batches are cut on the device by the data source.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from eeg2img.data.source import ArraySource
from eeg2img.engine.losses import RetrievalLoss
from eeg2img.metrics import score_predictions_indexed
from eeg2img.tracking import RunTracker


@torch.inference_mode()
def predict_source(
    model: nn.Module,
    source: ArraySource,
    indices: torch.Tensor,
    batch_size: int = 1024,
    normalize: bool = True,
) -> torch.Tensor:
    """Predictions ``(N, D)`` on the CPU for event ``indices``.

    ``normalize`` L2-normalises each prediction, as the packaged model does
    before the platform averages repeats.
    """
    model.eval()
    outputs = []
    for start in range(0, len(indices), batch_size):
        out = model(source.batch(indices[start : start + batch_size])).float()
        outputs.append((F.normalize(out, dim=-1) if normalize else out).cpu())
    return torch.cat(outputs) if outputs else torch.zeros((0, source.embed_dim))


def evaluate_source(
    model: nn.Module,
    source: ArraySource,
    indices: torch.Tensor,
    batch_size: int = 1024,
    normalize: bool = True,
    k: int = 5,
    max_events: int | None = None,
    seed: int = 0,
) -> dict[str, float | int]:
    """Competition scores of ``model`` on event ``indices``.

    The gallery is every image occurring in ``indices``. ``max_events`` scores
    a seeded random subset, for cheap mid-training evaluation; note that this
    also thins the repeats that subject aggregation averages over.
    """
    if max_events is not None and len(indices) > max_events:
        generator = torch.Generator().manual_seed(seed)
        indices = indices[torch.randperm(len(indices), generator=generator)[:max_events]]
    pred = predict_source(model, source, indices, batch_size, normalize)
    return score_predictions_indexed(
        pred.numpy(),
        source.image[indices].numpy(),
        source.embeddings.cpu().numpy(),
        source.subject[indices].numpy(),
        k=k,
    )


def _lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _epoch_order(train: torch.Tensor, seed: int, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return train[torch.randperm(len(train), generator=generator)]


def train_model(
    model: nn.Module,
    source: ArraySource,
    splits: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    tracker: RunTracker,
    checkpoint_dir: str | Path,
    normalize_output: bool = True,
) -> dict[str, Any]:
    """Train ``model`` in place and leave it holding the best checkpoint.

    Reads the ``train`` and ``loss`` config sections. Returns fit statistics,
    including the best monitored value and the step it was reached at.
    """
    cfg = config["train"]
    seed = int(config["seed"])
    device = source.device
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train = splits["train"]
    monitor_split, monitor_key = cfg["monitor"].split("/")
    batch_size = int(cfg["batch_size"])
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = int(cfg["epochs"]) * steps_per_epoch
    eval_every = int(cfg.get("eval_every") or steps_per_epoch)
    log_every = int(cfg.get("log_every", 50))
    channel_dropout = float(cfg.get("channel_dropout", 0.0))

    model.to(device)
    criterion = RetrievalLoss(**dict(config.get("loss") or {})).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "weight_decay": float(cfg["weight_decay"])},
            {"params": list(criterion.parameters()), "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, int(cfg.get("warmup_steps", 0)), total_steps,
                                float(cfg.get("lr_floor", 0.01))),
    )  # fmt: skip
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    negative_pool = torch.unique(source.image[train])
    noise = torch.Generator().manual_seed(seed + 17)

    state = {"step": 0, "best": -math.inf, "best_step": -1, "bad_evals": 0}
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        criterion.load_state_dict(saved["criterion"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        noise.set_state(saved["noise"])
        # Dropout draws from the global generators; restore them so a resumed
        # run continues exactly where the interrupted one stopped.
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        state = saved["state"]
        tracker.log(state["step"], event="resumed")

    def run_evaluation(step: int) -> bool:
        """Score, checkpoint and report whether patience is exhausted."""
        scores = {}
        for name, indices in splits.items():
            if name == "train":
                continue
            scores[name] = evaluate_source(
                model, source, indices, int(cfg.get("eval_batch_size", 1024)), normalize_output,
                max_events=cfg.get("eval_max_events"), seed=seed,
            )  # fmt: skip
            tracker.log(step, split=name, **scores[name])
        value = float(scores[monitor_split][monitor_key])
        if value > state["best"]:
            state.update(best=value, best_step=step, bad_evals=0)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_path)
        else:
            state["bad_evals"] += 1
        torch.save(
            {"model": model.state_dict(), "criterion": criterion.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "noise": noise.get_state(), "state": state,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
            last_path,
        )  # fmt: skip
        patience = cfg.get("patience")
        return patience is not None and state["bad_evals"] >= int(patience)

    stop = False
    while state["step"] < total_steps and not stop:
        epoch, offset = divmod(state["step"], steps_per_epoch)
        order = _epoch_order(train, seed, epoch)
        for batch_index in range(offset, steps_per_epoch):
            indices = order[batch_index * batch_size : (batch_index + 1) * batch_size]
            model.train()
            x = source.batch(indices)
            if channel_dropout > 0:
                keep = torch.rand(x.shape[0], x.shape[1], 1, generator=noise) >= channel_dropout
                x = x * keep.to(x.device, x.dtype)
            image = source.image[indices].to(device)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                pred = model(x)
            loss, stats = criterion(pred, image, source.embeddings, negative_pool, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {state['step']}")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            state["step"] += 1

            if state["step"] % log_every == 0:
                tracker.log(state["step"], split="train", loss=float(loss.detach()),
                            lr=scheduler.get_last_lr()[0], epoch=epoch, **stats)  # fmt: skip
            end_of_run = state["step"] == total_steps
            if state["step"] % eval_every == 0 or end_of_run:
                stop = run_evaluation(state["step"])
                if stop or end_of_run:
                    break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return {
        "steps": state["step"],
        "total_steps": total_steps,
        "best_step": state["best_step"],
        "best_monitor": state["best"],
        "monitor": cfg["monitor"],
        "early_stopped": bool(stop and state["step"] < total_steps),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
