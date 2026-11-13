"""One-off data preparation: build the EEG store and the image-embedding table.

Both steps are resumable, because they are meant to run on Colab where
sessions can end without warning.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eeg2img import contract
from eeg2img.data.preprocess import PreprocessConfig
from eeg2img.data.store import EpochStore, ProcessedRecording, build_store


def preprocess_config(data_config: Mapping[str, Any]) -> PreprocessConfig:
    """Build a :class:`PreprocessConfig` from the ``data.preprocess`` config section."""
    params = dict(data_config.get("preprocess") or {})
    for key in ("notch", "baseline"):
        if key in params:
            params[key] = tuple(params[key])
    return PreprocessConfig(**params)


def prepare_alljoined(
    data_config: Mapping[str, Any],
    progress: Callable[[Iterable], Iterable] = lambda items: items,
) -> Path:
    """Preprocess every downloaded Alljoined-1.6M recording into a store.

    Reads ``data.root`` (the directory holding ``Xu2025Alljoined/``) and writes
    ``data.store``. ``data.subjects`` optionally restricts the participants.
    """
    import mne

    from eeg2img.data import alljoined
    from eeg2img.data.preprocess import preprocess_raw

    config = preprocess_config(data_config)
    recordings = alljoined.list_recordings(data_config["root"])
    subjects = data_config.get("subjects")
    if subjects:
        recordings = [rec for rec in recordings if rec.subject in set(subjects)]
    if not recordings:
        raise FileNotFoundError(
            f"no Alljoined-1.6M recordings under {alljoined.dataset_dir(data_config['root'])}"
        )
    by_name = {rec.name: rec for rec in recordings}

    def load(name: str) -> ProcessedRecording:
        rec = by_name[name]
        raw = alljoined.load_raw(rec)
        ch_names = alljoined.check_montage(raw)
        onsets, stims, labels = alljoined.load_events(rec)
        return ProcessedRecording(
            name=name, subject=rec.subject, session=rec.session, run=rec.run,
            continuous=preprocess_raw(raw, config), onsets=onsets, stim_files=stims,
            is_test=[label == "stim_test" for label in labels], ch_names=ch_names,
        )  # fmt: skip

    provenance = {
        "dataset": "Alljoined-1.6M",
        "nemar_id": alljoined.NEMAR_ID,
        "dataset_version": contract.ALLJOINED_16M_VERSION,
        "licence": alljoined.LICENCE,
        "reference_pipeline": f"neuralbench {contract.NEURALBENCH_VERSION} eeg/image",
        "mne": mne.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    return build_store(
        data_config["store"], list(by_name), load, config, provenance=provenance, progress=progress
    )


def embed_images(
    image_paths: Sequence[str | Path],
    target_config: Mapping[str, Any],
    out_path: str | Path,
    device: str = "auto",
    chunk_size: int = 256,
    progress: Callable[[Iterable], Iterable] = lambda items: items,
) -> np.ndarray:
    """Embed images with the organisers' own extractor; return ``(M, D)`` float32.

    Uses ``neuralset.extractors.image.HuggingFaceImage`` (neuralset 0.4.0) with
    the parameters of NeuralBench's image task, so the targets are produced by
    the same code as the competition gallery. Rows follow ``image_paths``.
    Progress is saved after every chunk; re-running resumes.
    """
    from neuralset.events import etypes
    from neuralset.extractors.image import HuggingFaceImage

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".progress.json")
    partial_path = out_path.with_suffix(".partial.npy")
    names = [str(path) for path in image_paths]

    extractor = HuggingFaceImage(
        model_name=target_config["model_name"],
        layers=float(target_config["layers"]),
        token_aggregation=target_config["token_aggregation"],
        imsize=target_config.get("imsize"),
        batch_size=int(target_config.get("batch_size", 32)),
        device=device,
    )

    done, table = 0, None
    if state_path.exists() and partial_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["names"] == names and state["target"] == dict(target_config):
            done = int(state["done"])
            table = np.load(partial_path, mmap_mode="r+")

    for start in progress(range(done, len(names), chunk_size)):
        chunk = names[start : start + chunk_size]
        events = [
            etypes.Image(start=0.0, duration=0.1, timeline="stimuli", filepath=name)
            for name in chunk
        ]
        latents = np.stack(list(extractor._iter_image_latents(events, aggregate_layers=True)))
        if table is None:
            table = np.lib.format.open_memmap(
                partial_path, mode="w+", dtype=np.float32, shape=(len(names), latents.shape[-1])
            )
        table[start : start + len(chunk)] = latents.astype(np.float32)
        table.flush()
        state = {"names": names, "target": dict(target_config), "done": start + len(chunk)}
        state_path.write_text(json.dumps(state), encoding="utf-8")

    if table is None:
        raise ValueError("no images to embed")
    result = np.asarray(table, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("embedding table contains NaN or infinite values")
    np.save(out_path, result)
    partial_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    return result


def embed_store_images(
    data_config: Mapping[str, Any],
    device: str = "auto",
    progress: Callable[[Iterable], Iterable] = lambda items: items,
) -> Path:
    """Embed every stimulus referenced by the store at ``data.store``."""
    from eeg2img.data import alljoined

    store = EpochStore(data_config["store"])
    stimuli_dir = alljoined.dataset_dir(data_config["root"]) / "stimuli"
    paths = [stimuli_dir / name for name in store.images]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} stimulus files are missing, e.g. {missing[0]}")
    out_path = Path(data_config["embeddings"])
    embed_images(paths, data_config["target"], out_path, device=device, progress=progress)
    return out_path
