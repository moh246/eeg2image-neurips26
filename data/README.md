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

One script does the whole preparation on a GPU machine and resumes if interrupted:

```bash
pip install -e ".[data]" transformers torchvision pillow
bash scripts/colab_prepare.sh /content/work /content/persist   # scratch dir, output dir
```

It downloads the release, preprocesses the recordings with one process per CPU core, computes
the image targets, copies the finished store to the output directory and prints its SHA-256
sums. The individual steps are `scripts/prepare_alljoined.py` and `scripts/embed_images.py`.
It needs about 20 GB of scratch disk; the store itself is 3.7 GB.

### The prepared store (v1)

Built on 15 Nov 2026 on a Colab T4 runtime (2 CPU cores) from commit `427ece8`, with
Python 3.13.16, mne 1.13.2, torch 2.14.1+cu130 and transformers 5.22.1.

| | |
|---|---|
| Recordings | 1,520 = 20 participants × 4 sessions × 19 runs |
| Image events | 1,659,910, of which 324,294 show one of the 200 test images |
| Distinct images | 16,740 (16,540 training, 200 test) |
| Continuous samples | 55,765,800 at 120 Hz (129 hours), 32 channels, float16 |
| Wall time | download 10 min, preprocessing 12 min, image targets 38 min |

The release also contains `sub-08/ses-02old` (five runs of an abandoned session). The
organisers' reader enumerates sessions 1–4 and runs 1–19 only and never reads it; this
pipeline skips it too.

| File | Bytes | SHA-256 |
|---|---|---|
| `eeg.npy` | 3,569,011,328 | `47fa07c2328acaacea2f0f6bd31846439b5330af11f9c32e7f2cfd79c987c961` |
| `index.npz` | 107,957,726 | `fd6f6026f4f4fd66835f509d8e31c3a4a1c9fb22d91207cbcebbd29e85e9a5ab` |
| `images.json` | 217,620 | `12193429fed8011d7e34813c6d84d6a8346dfc3b9ea172bf35628546ae262ee0` |
| `meta.json` | 43,618 | `4bb2620a35083cb89bd115180a2429ed85d663fee8df6d85ff06da0732f127e8` |
| `embeddings.npy` | 102,850,688 | `2783af0104e3bc0263a960ef4f067ab331df790797c5d518e7d909c0fb231a68` |

The store is not redistributed (the licence forbids derivatives being shared); rebuild it with
the script above and compare the sums. `eeg.npy` and `index.npz` depend only on the release
and the preprocessing; `embeddings.npy` can differ in the last bits across GPU and library
versions.

### Image groups used by the validation split

Validation images are held out in groups of ten consecutive stimulus ids, on the assumption
that each group is the ten exemplars of one THINGS concept. `scripts/check_image_groups.py`
tests this on the target embeddings of the 16,540 training images:

| | |
|---|---|
| Similarity of neighbouring ids inside a group | 0.416 |
| Similarity of neighbouring ids across a group boundary | 0.025 |
| Boundaries less similar than both pairs next to them, groups of 10 | 92.6% |
| Same, if groups were 9 or 11 ids wide | 30.5%, 32.7% |

So groups of ten are concept groups, and no concept has images on both sides of the split.

## Targets

Image embeddings are 1536-dimensional `facebook/dinov2-giant` features: the hidden state after
26 of the 40 blocks (index `int(0.6667 × 40)`), averaged over all tokens.

The task config asks for `imsize: 518`, but that resize is undone by the HuggingFace
processor the organisers' extractor calls next, which resizes to 256 and centre-crops to 224.
The model therefore sees 224 × 224 inputs (257 tokens). This was checked on the processor's
output tensor, and this repository reproduces the same sequence of calls rather than the
nominal 518.

`scripts/embed_images.py` has two backends. `--backend reference` calls the organisers' own
extractor (`neuralset.extractors.image.HuggingFaceImage`, neuralset 0.4.0). `--backend
truncated` loads the same model, drops the 14 blocks after the layer that is read, and
otherwise makes the same calls; it is a third faster and its output was identical (max abs
difference 0.0) to the reference backend on sample images with `facebook/dinov2-small`. The
v1 store was built with the truncated backend. The table is computed once per unique stimulus
and stored next to the EEG store as `embeddings.npy`.

## Channel order

Models never assume an input channel order. At inference the platform supplies channel names,
and `eeg2img.channels.ChannelAligner` reorders the input to the training order by name.
The Emotiv order used for training is `eeg2img.contract.EMOTIV_32`.
