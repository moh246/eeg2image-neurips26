"""The Track 01 sealed-phase input/output contract, as constants.

Every value here was read from the organisers' sources rather than assumed:

- starter kit ``neural-interfaces26/2026-competition`` at commit ``bddaff3``
  (``codabench/pages/competition_image.md``, ``scientific_faq_image.md``,
  ``tracks/image_decoding/objective.py``);
- NeuralBench 0.4.0 (``tasks/eeg/image/config.yaml``, ``defaults/eeg/config.yaml``);
- the Alljoined-1.6M release ``nm000134`` v1.0.3 (``*_channels.tsv``).

Nothing at inference time may rely on these constants being true: the model is
built from the ``meta`` dict the platform passes in. They are the values we
train for and the defaults the tests exercise.
"""

from __future__ import annotations

#: Pinned sources this contract was verified against.
STARTER_KIT_COMMIT = "bddaff3dc6bd4b7300d531797685279a22c7bbf9"
NEURALBENCH_VERSION = "0.4.0"
ALLJOINED_16M_VERSION = "1.0.3"

#: The 32-channel Emotiv montage of Alljoined-1.6M, in recording order.
EMOTIV_32: tuple[str, ...] = (
    "Cz", "FCz", "AFz", "Fp1", "F5", "F1", "CP5", "CP3",
    "CP1", "P1", "P3", "P5", "P7", "PO7", "PO3", "O1",
    "Pz", "POz", "Oz", "O2", "PO4", "PO8", "P8", "P6",
    "P4", "P2", "CP2", "CP4", "CP6", "F2", "F6", "Fp2",
)  # fmt: skip

N_CHANS = len(EMOTIV_32)

#: Model input sampling rate, after the organisers' resampling.
SFREQ = 120.0
#: Epoch window in seconds relative to image onset.
EPOCH_START = -0.2
EPOCH_DURATION = 1.0
N_TIMES = round(EPOCH_DURATION * SFREQ)

#: Baseline-correction window in seconds, relative to the epoch start.
BASELINE_WINDOW = (0.0, 0.2)
#: Inputs are robust-scaled, then clipped to +/- this value.
CLAMP = 20.0

#: Target space: facebook/dinov2-giant, hidden state at 2/3 depth, mean over tokens.
TARGET_MODEL = "facebook/dinov2-giant"
TARGET_LAYER_FRACTION = 0.6667
#: The task's nominal resize. The HuggingFace processor that runs next resizes to 256 and
#: centre-crops to 224, so the model sees 224 x 224 inputs (see data/README.md, "Targets").
TARGET_IMAGE_SIZE = 518
EMBED_DIM = 1536

#: The sealed ranking metric is Top-K after averaging repeats per (subject, image).
TOP_K = 5
