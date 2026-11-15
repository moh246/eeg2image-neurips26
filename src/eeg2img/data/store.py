"""Compact on-disk store of preprocessed recordings, with lazy epoch access.

Storing every epoch of Alljoined-1.6M would take about 25 GB. Epochs overlap
heavily (images are 0.2-0.3 s apart, epochs last 1 s), so the store instead
keeps each preprocessed *recording* once, as float16, plus an integer plan of
where every epoch's samples sit (see ``preprocess.plan_windows``). That is
about 3.6 GB for the whole dataset, small enough for Colab memory and Google
Drive, and epochs are cut on the fly exactly as the organisers' loader does.

Layout of a store directory::

    eeg.bin       float16, shape (total_samples, n_chans), recordings concatenated
    index.npz     per-recording and per-event arrays
    images.json   stimulus file names; an event's ``image`` indexes this list
    meta.json     preprocessing parameters, channel names, versions, counts
    shards/       per-recording intermediates, kept so a build can resume

float16 keeps about three significant digits. Inputs are robust-scaled to
order one and clipped to +/-20 afterwards, so the rounding error (measured in
the tests) is far below the noise floor. Values are clipped to the float16
range before storage, which only affects extreme artefacts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eeg2img.data.epochs import EpochSet
from eeg2img.data.preprocess import PreprocessConfig, cut_epoch, plan_windows

STORE_VERSION = 1
_FLOAT16_LIMIT = 6.5e4


@dataclass(frozen=True)
class ProcessedRecording:
    """One preprocessed recording and its image events, ready to be stored."""

    name: str
    subject: int
    session: int
    run: int
    continuous: np.ndarray  # (C, T) float32
    onsets: np.ndarray  # (E,) seconds
    stim_files: Sequence[str]
    is_test: Sequence[bool]
    ch_names: Sequence[str]


def _shard_path(out_dir: Path, name: str) -> Path:
    return out_dir / "shards" / f"{name}.npz"


def write_shard(out_dir: Path, recording: ProcessedRecording, config: PreprocessConfig) -> Path:
    """Store one preprocessed recording and its epoch plan as an intermediate shard."""
    plan, valid = plan_windows(recording.onsets, recording.continuous.shape[1], config)
    data = np.clip(recording.continuous.T, -_FLOAT16_LIMIT, _FLOAT16_LIMIT).astype(np.float16)
    path = _shard_path(out_dir, recording.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        eeg=data,
        meta=np.array([recording.subject, recording.session, recording.run], dtype=np.int64),
        plan=np.stack([plan.src_start, plan.src_len, plan.dst_offset, plan.base_start,
                       plan.base_len], axis=1),
        onsets=np.asarray(recording.onsets, dtype=np.float64)[valid],
        stim_files=np.asarray(list(recording.stim_files), dtype=str)[valid],
        is_test=np.asarray(list(recording.is_test), dtype=bool)[valid],
        ch_names=np.asarray(list(recording.ch_names), dtype=str),
    )  # fmt: skip
    tmp.replace(path)
    return path


def build_store(
    out_dir: str | Path,
    names: Sequence[str],
    load: Callable[[str], ProcessedRecording],
    config: PreprocessConfig | None = None,
    provenance: dict | None = None,
    progress: Callable[[Iterable], Iterable] = lambda items: items,
    part: tuple[int, int] | None = None,
) -> Path:
    """Build a store from recordings; resumable and deterministic.

    ``load(name)`` preprocesses one recording. Recordings whose shard already
    exists are skipped, so an interrupted build continues where it stopped.
    ``names`` fixes the order of recordings in the store.

    ``part=(i, n)`` preprocesses only every ``n``-th recording starting at the
    ``i``-th and does not assemble, so ``n`` processes can share the work; a
    final call without ``part`` then finds every shard present and assembles.
    """
    config = config or PreprocessConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(names)
    if part is not None:
        index, count = part
        if not 0 <= index < count:
            raise ValueError(f"part must be (i, n) with 0 <= i < n, got {part}")
        todo = todo[index::count]
    for name in progress(todo):
        if not _shard_path(out_dir, name).exists():
            write_shard(out_dir, load(name), config)
    if part is not None:
        return out_dir
    return assemble_store(out_dir, names, config, provenance)


def assemble_store(
    out_dir: str | Path,
    names: Sequence[str],
    config: PreprocessConfig,
    provenance: dict | None = None,
) -> Path:
    """Concatenate shards into ``eeg.bin`` and write the index and metadata."""
    out_dir = Path(out_dir)
    shards = [np.load(_shard_path(out_dir, name)) for name in names]
    if not shards:
        raise ValueError("no recordings to assemble")
    ch_names = [str(c) for c in shards[0]["ch_names"]]
    lengths = [int(shard["eeg"].shape[0]) for shard in shards]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    images = sorted({str(f) for shard in shards for f in shard["stim_files"]})
    image_id = {name: i for i, name in enumerate(images)}

    eeg = np.lib.format.open_memmap(
        out_dir / "eeg.tmp.npy", mode="w+", dtype=np.float16, shape=(sum(lengths), len(ch_names))
    )
    rec_meta, ev_rec, ev_plan, ev_onset, ev_image, ev_test = [], [], [], [], [], []
    for i, (shard, offset) in enumerate(zip(shards, offsets, strict=True)):
        if [str(c) for c in shard["ch_names"]] != ch_names:
            raise ValueError(f"{names[i]} has a different channel order than {names[0]}")
        eeg[offset : offset + lengths[i]] = shard["eeg"]
        rec_meta.append(shard["meta"])
        plan = shard["plan"].astype(np.int64)
        plan[:, 0] += offset  # src_start
        plan[:, 3] += offset  # base_start
        ev_plan.append(plan)
        ev_rec.append(np.full(len(plan), i, dtype=np.int32))
        ev_onset.append(shard["onsets"])
        ev_image.append(np.array([image_id[str(f)] for f in shard["stim_files"]], dtype=np.int32))
        ev_test.append(shard["is_test"])
    eeg.flush()
    del eeg
    (out_dir / "eeg.tmp.npy").replace(out_dir / "eeg.npy")

    rec_meta_arr = np.stack(rec_meta)
    ev_rec_arr = np.concatenate(ev_rec)
    np.savez(
        out_dir / "index.npz",
        rec_subject=rec_meta_arr[:, 0], rec_session=rec_meta_arr[:, 1], rec_run=rec_meta_arr[:, 2],
        rec_offset=offsets, rec_len=np.asarray(lengths, dtype=np.int64),
        ev_recording=ev_rec_arr, ev_plan=np.concatenate(ev_plan),
        ev_onset=np.concatenate(ev_onset), ev_image=np.concatenate(ev_image),
        ev_is_test=np.concatenate(ev_test),
        ev_subject=rec_meta_arr[ev_rec_arr, 0],
    )  # fmt: skip
    (out_dir / "images.json").write_text(json.dumps(images), encoding="utf-8")
    meta = {
        "store_version": STORE_VERSION,
        "preprocess": config.to_dict(),
        "ch_names": ch_names,
        "n_recordings": len(names),
        "n_events": len(ev_rec_arr),
        "n_images": len(images),
        "n_subjects": len(np.unique(rec_meta_arr[:, 0])),
        "total_samples": int(sum(lengths)),
        "dtype": "float16",
        "recordings": list(names),
        "provenance": provenance or {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir


class EpochStore:
    """Read-only view of a store; cuts model-input epochs on demand."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        if self.meta["store_version"] != STORE_VERSION:
            raise ValueError(f"unsupported store version {self.meta['store_version']}")
        self.images: list[str] = json.loads((self.path / "images.json").read_text("utf-8"))
        index = np.load(self.path / "index.npz")
        self.plan = index["ev_plan"]
        self.subject = index["ev_subject"].astype(np.int64)
        self.image = index["ev_image"].astype(np.int64)
        self.is_test = index["ev_is_test"].astype(bool)
        self.recording = index["ev_recording"].astype(np.int64)
        self.onset = index["ev_onset"]
        self.ch_names: tuple[str, ...] = tuple(self.meta["ch_names"])
        self.sfreq = float(self.meta["preprocess"]["sfreq"])
        self.n_times = round(self.meta["preprocess"]["epoch_duration"] * self.sfreq)
        self.clamp = float(self.meta["preprocess"]["clamp"])
        self._eeg: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.plan)

    @property
    def eeg(self) -> np.ndarray:
        """The memory-mapped ``(total_samples, n_chans)`` float16 signal."""
        if self._eeg is None:
            self._eeg = np.load(self.path / "eeg.npy", mmap_mode="r")
        return self._eeg

    def __getstate__(self) -> dict:
        # Memory maps do not survive pickling into DataLoader workers; reopen lazily.
        state = self.__dict__.copy()
        state["_eeg"] = None
        return state

    def epoch(self, i: int) -> np.ndarray:
        """Event ``i`` as a ``(C, n_times)`` float32 model input."""
        src_start, src_len, dst_offset, base_start, base_len = (int(v) for v in self.plan[i])
        return cut_epoch(self.eeg, src_start, src_len, dst_offset, base_start, base_len,
                         self.n_times, self.clamp)  # fmt: skip

    def epochs(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Events ``indices`` as a ``(N, C, n_times)`` float32 array."""
        indices = np.asarray(indices, dtype=np.int64)
        out = np.zeros((len(indices), len(self.ch_names), self.n_times), dtype=np.float32)
        for row, i in enumerate(indices):
            out[row] = self.epoch(int(i))
        return out

    def to_epochset(self, indices: Sequence[int] | np.ndarray, embeddings: np.ndarray) -> EpochSet:
        """Materialise ``indices`` in memory with their targets, for evaluation.

        ``embeddings`` is the ``(n_images, D)`` table aligned with ``self.images``.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if len(embeddings) != len(self.images):
            raise ValueError(f"{len(embeddings)} embeddings for {len(self.images)} images")
        return EpochSet(
            eeg=self.epochs(indices),
            image_index=self.image[indices],
            subject=self.subject[indices],
            embeddings=np.asarray(embeddings, dtype=np.float32),
            ch_names=self.ch_names,
            sfreq=self.sfreq,
        ).validate()
