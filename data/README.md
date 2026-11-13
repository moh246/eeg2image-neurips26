# Data provenance

No data is committed to this repository. This file records where each dataset comes from, the
version used, its licence, and how raw EEG becomes model input.

## Datasets

| Dataset | Role | Version | Montage | Licence |
|---|---|---|---|---|
| Alljoined-1.6M (`nm000134`) | Primary training and validation data; same hardware as the sealed cohort | NEMAR release 1.0.3 | 32-channel Emotiv, 256 Hz | CC-BY-NC-ND-4.0 |
| THINGS-EEG2 (`Gifford2022Large`) | Optional extra training data for the 63→32 transfer ablation | OSF `3jk45` | 63-channel, 1000 Hz | CC-BY-4.0 |

Alljoined-1.6M is licensed for non-commercial use and its downloader requires explicit
acceptance (`ALLJOINED_ACCEPT_LICENCE=1`). It has 20 participants, and its stimuli are the
THINGS-EEG2 image set (16,540 training and 200 test images), presented for 100 ms each.

The sealed evaluation cohort (11 participants, new images) is private and is never available
locally. Under the challenge terms, every corpus used for training must be declared in the
method description; this table is that declaration and must be kept current.

## From raw EEG to model input

The model input is defined by the organisers' NeuralBench 0.4.0 image task
(`defaults/eeg/config.yaml`, `tasks/eeg/image/config.yaml`) as executed by
`neuralset.extractors.neuro.MneRaw`. This repository reimplements it in
[`src/eeg2img/data/preprocess.py`](../src/eeg2img/data/preprocess.py).

Per recording, on the continuous signal:

1. Keep the EEG channels (all 32 Emotiv channels).
2. Notch-filter 50 and 60 Hz and their harmonics below Nyquist (50, 60, 100, 120 Hz at the
   native 256 Hz), zero-phase FIR.
3. Band-pass 0.1–75 Hz at the native rate, MNE default zero-phase FIR. The low-pass edge is
   skipped only for recordings whose Nyquist frequency is at or below 75 Hz.
4. Resample to 120 Hz.
5. Robust-scale each channel over the whole recording (subtract the median, divide by the
   interquartile range).

Per image event:

6. Cut 120 samples starting at `round((onset − 0.2 s) × 120)`.
7. Subtract each channel's mean over the first 0.2 s (24 samples) of the window.
8. Clip to ±20.

Event selection follows the organisers' reader: markers that share an onset timestamp are
dropped as ambiguous, markers without a stimulus file are dropped, and only `stim_train` and
`stim_test` trials are kept.

### Verified against the organisers' code

`tests/test_preprocess_parity.py` runs the organisers' reader and extractor and this
repository's pipeline on the same real recordings and compares the results. On
`sub-01/ses-01` runs 1 and 2 (2,028 epochs, 13 Nov 2026, mne 1.13.2):

| Stage | Result |
|---|---|
| Event selection (onsets, stimulus files, labels) | identical |
| Continuous signal after steps 1–5 | identical (max abs difference 0.0) |
| Epochs after steps 6–8 | max abs difference 3.8e-6 |

One behaviour of the organisers' slicing is reproduced deliberately. When a window start
falls exactly on a half sample (33 of the 2,028 epochs, 1.6%), their code returns 119 samples
behind one leading zero, with a 23-sample baseline. The sealed inputs are produced by the
same code, so training data keeps this quirk rather than "fixing" it.

Reproduce with `make setup-official && ALLJOINED_ACCEPT_LICENCE=1 make parity` (downloads
10 MB). The test needs real data, which cannot be redistributed, so it does not run in CI.

### Storage

The full dataset as individual epochs would be about 25 GB. Because consecutive epochs
overlap, the store keeps each preprocessed recording once as float16 plus an integer index of
every epoch's samples (about 3.6 GB), and cuts epochs on the fly. float16 rounding changes
epoch values by less than 0.02 on a signal of unit scale (`tests/test_store.py`).

### Preparing the data

On Colab, run [`notebooks/colab_prepare_data.ipynb`](../notebooks/colab_prepare_data.ipynb).
Locally (about 12 GB of disk):

```bash
ALLJOINED_ACCEPT_LICENCE=1 .venv/bin/python scripts/prepare_alljoined.py   # EEG store
.venv/bin/python scripts/embed_images.py --device cuda                      # image targets
```

## Targets

Image embeddings are 1536-dimensional `facebook/dinov2-giant` features: the hidden state at
2/3 depth, averaged over tokens. `scripts/embed_images.py` computes them by calling the
organisers' own extractor (`neuralset.extractors.image.HuggingFaceImage`, neuralset 0.4.0) with
the parameters of the NeuralBench image task, rather than a reimplementation, so the targets
are produced by the same code as the competition gallery. The table is computed once per
unique stimulus and stored next to the EEG store as `embeddings.npy`.

## Channel order

Models never assume an input channel order. At inference the platform supplies channel names,
and `eeg2img.channels.ChannelAligner` reorders the input to the training order by name.
The Emotiv order used for training is `eeg2img.contract.EMOTIV_32`.
