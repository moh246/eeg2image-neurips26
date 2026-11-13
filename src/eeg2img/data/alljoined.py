"""Reader for the Alljoined-1.6M BIDS release (NEMAR ``nm000134``).

Follows the organisers' reader (``neuralfetch.studies.xu2025alljoined``) so the
same recordings and events are selected:

- some files spell ``AFz`` as ``Afz``; it is renamed;
- channels present in the standard 10-05 montage are EEG, any other is ignored;
- markers that share an onset timestamp are dropped as ambiguous (the marker
  stream occasionally stalls and flushes several at once);
- markers with no stimulus file (responses, debug) are dropped;
- only ``stim_train`` and ``stim_test`` trials are kept, per the NeuralBench
  dataset config ``xu2025alljoined.yaml``.

Expected layout, identical to what the organisers' downloader produces::

    <data_root>/Xu2025Alljoined/download/nm000134/sub-XX/ses-XX/eeg/*_eeg.edf
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from eeg2img import contract

STUDY_DIR = "Xu2025Alljoined"
NEMAR_ID = "nm000134"
TRIAL_TYPES = ("stim_train", "stim_test")
LICENCE = "CC-BY-NC-ND-4.0"
LICENCE_ENV = "ALLJOINED_ACCEPT_LICENCE"

_RUN_PATTERN = re.compile(r"sub-(\d+)_ses-(\d+)_task-images_run-(\d+)_eeg\.edf$")


@dataclass(frozen=True)
class Recording:
    """One run of one session of one participant."""

    subject: int
    session: int
    run: int
    edf_path: Path

    @property
    def events_path(self) -> Path:
        return Path(str(self.edf_path).replace("_eeg.edf", "_events.tsv"))

    @property
    def name(self) -> str:
        return f"sub-{self.subject:02d}_ses-{self.session:02d}_run-{self.run:02d}"


def dataset_dir(data_root: str | Path) -> Path:
    """Directory of the BIDS tree under ``data_root``."""
    return Path(data_root) / STUDY_DIR / "download" / NEMAR_ID


def list_recordings(data_root: str | Path) -> list[Recording]:
    """All recordings on disk, sorted by subject, session, run."""
    found = []
    for path in dataset_dir(data_root).glob("sub-*/ses-*/eeg/*_eeg.edf"):
        match = _RUN_PATTERN.search(path.name)
        if match:
            subject, session, run = (int(group) for group in match.groups())
            found.append(Recording(subject, session, run, path))
    return sorted(found, key=lambda rec: (rec.subject, rec.session, rec.run))


def load_raw(recording: Recording) -> mne.io.BaseRaw:
    """Open a recording with channel names and types set as the organisers do."""
    raw = mne.io.read_raw_edf(recording.edf_path, preload=False, verbose=False)
    if "Afz" in raw.ch_names:
        raw.rename_channels({"Afz": "AFz"})
    known = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    types = {name: ("eeg" if name in known else "misc") for name in raw.ch_names}
    raw.set_channel_types(types, on_unit_change="ignore", verbose=False)
    return raw


def load_events(recording: Recording) -> tuple[np.ndarray, list[str], list[str]]:
    """Image events of a recording: ``(onsets_seconds, stim_files, trial_types)``."""
    with recording.events_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    onset_counts = Counter(row["onset"] for row in rows)
    onsets, stims, labels = [], [], []
    for row in rows:
        stim = (row.get("stim_file") or "").strip()
        if onset_counts[row["onset"]] > 1 or stim in ("", "n/a"):
            continue
        if row["trial_type"] not in TRIAL_TYPES:
            continue
        onsets.append(float(row["onset"]))
        stims.append(stim)
        labels.append(row["trial_type"])
    return np.asarray(onsets, dtype=np.float64), stims, labels


def check_montage(raw: mne.io.BaseRaw) -> tuple[str, ...]:
    """EEG channel names of ``raw``; raises if they are not the 32 Emotiv channels."""
    names = tuple(raw.copy().pick("eeg", verbose=False).ch_names)
    if set(names) != set(contract.EMOTIV_32):
        missing = sorted(set(contract.EMOTIV_32) - set(names))
        extra = sorted(set(names) - set(contract.EMOTIV_32))
        raise ValueError(f"unexpected montage: missing {missing}, extra {extra}")
    return names


def download(
    data_root: str | Path,
    subjects: list[int] | None = None,
    sessions: list[int] | None = None,
    runs: list[int] | None = None,
    include_stimuli: bool = True,
) -> Path:
    """Download the pinned release from NEMAR into the organisers' layout.

    Requires ``ALLJOINED_ACCEPT_LICENCE=1``: the data is CC-BY-NC-ND-4.0 and
    downloading it means accepting those terms. ``subjects``, ``sessions`` and
    ``runs`` restrict the EEG that is fetched (everything by default). Safe to
    re-run; files already present are kept.
    """
    if os.environ.get(LICENCE_ENV, "").lower() not in ("1", "true", "yes"):
        raise RuntimeError(
            f"Alljoined-1.6M is released under {LICENCE} (non-commercial use). "
            f"Set {LICENCE_ENV}=1 to accept the licence before downloading."
        )
    import nemar

    target = dataset_dir(data_root)
    target.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": NEMAR_ID,
        "tag": contract.ALLJOINED_16M_VERSION,
        "target_dir": target,
        "trust_existing": True,
    }
    selection = {"subject": subjects, "session": sessions, "run": runs}
    entities = {key: [f"{v:02d}" for v in values] for key, values in selection.items() if values}
    nemar.download(**common, scope=["raw"], datatype="eeg", **entities)
    if include_stimuli:
        nemar.download(**common, scope=["stimuli"])
    return target
