# eeg2img — EEG-to-image retrieval (EEG/EMG Foundation Challenge 2026, Track 01)

[![CI](https://github.com/moh246/eeg2image-neurips26/actions/workflows/ci.yml/badge.svg)](https://github.com/moh246/eeg2image-neurips26/actions/workflows/ci.yml)

Entry for [Track 01](https://neural-interfaces26.github.io/tracks.html) of the NeurIPS 2026
Brain & Body Workshop challenge. Given one second of 32-channel EEG recorded while a person
looked at an image, predict that image's frozen DINOv2-giant embedding; the platform ranks a
held-out gallery by cosine similarity and scores Top-5 accuracy.

**Status: all code through M4 is written and tested; no real results yet.** The scorer,
packaging, CI, real-data pipeline, training loop and every model are in place, and the
preprocessing is verified against the organisers' code on real recordings. Nothing has been
trained on the full dataset yet (that runs on Colab), so there is no leaderboard number and
no ablation result to report.
Results will be added to [Results](#results) as they are produced.

## The task, as verified

Every value below was read from the organisers' code and data, not assumed. Sources are pinned
in [`src/eeg2img/contract.py`](src/eeg2img/contract.py).

| | |
|---|---|
| Input | `(B, 32, 120)`: one second (−0.2 s to +0.8 s around image onset) at 120 Hz |
| Montage | 32-channel Emotiv, the channel names of Alljoined-1.6M |
| Output | `predict(X) -> (B, 1536)`, a DINOv2-giant embedding (hidden state at 2/3 depth, mean over tokens) |
| Metric | Top-5 after averaging predictions per (participant, image), then L2-normalising |
| Shift | New participants **and** new images |
| Submission | One ZIP with a single `submission.py` plus weights; inference only, 60 min on one A100 |
| Quota | 5 sealed submissions in total, at most 1 per day; logs hidden |

Two facts shape the approach:

- **The sealed task is much harder than the public one.** Reference Top-5 on THINGS-EEG2 is
  28% (EEGNet) and 85% (REVE frozen probe), within subject. On the sealed cohort the reference
  EEGNet scores 0.027% and the leaderboard leader 0.126% (12 Nov 2026). Public within-subject
  numbers do not predict sealed performance; validation here holds out both subjects and images.
- **The closest training data already uses the sealed montage.** Alljoined-1.6M is recorded on
  the same 32-channel Emotiv hardware, so the 63→32 channel mismatch only arises when adding
  THINGS-EEG2 as extra training data. It is handled by matching channels by name and position
  and is evaluated as an ablation, not assumed to help.

## Quick start

Requires Python 3.12 and `make`.

```bash
git clone https://github.com/moh246/eeg2image-neurips26.git && cd eeg2image-neurips26
make setup     # virtualenv + package + dev tools
make check     # lint + tests, exactly what CI runs
make train     # smoke experiment on synthetic data: train, score, package, contract-check
```

`make train` prints the run id, the Top-5 scores and the path of a ready-to-upload
`submission.zip`. Real experiments are `configs/experiment/alljoined_*.yaml`; they need the
prepared data (see [Data](#data)) and are launched on Colab with
[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb). Any config value can be overridden on the command line:

```bash
.venv/bin/python scripts/train.py --config configs/experiment/smoke_ridge.yaml fit.alpha=10 seed=1
```

## Layout

```
configs/        YAML only; every hyperparameter lives here (data/, model/, experiment/)
src/eeg2img/
  contract.py   the verified I/O contract as constants
  metrics.py    replica of the competition scorer
  data/         Alljoined-1.6M reader, preprocessing, on-disk store, targets, synthetic data
  models/       model registry: floors, EEGNet, conv baseline, transformer encoders
  montage.py    standard 3-D electrode positions for the position-aware embedding
  engine/       contrastive loss, resumable training loop, closed-form floors, evaluation
  submission/   single-file packager, inference runtime, local contract check
  tracking.py   per-run records: config, git commit, metrics
  pipeline.py   train -> evaluate -> package, shared by every entry point
scripts/        thin CLI wrappers: prepare_alljoined, embed_images, train, check_submission
notebooks/      Colab launcher for the scripts; no logic of its own
tests/          unit, regression and integration tests
experiments/    committed run records and the run registry
data/           provenance notes only; data is never committed
```

## Models

All models map `(B, 32, 120)` EEG to a 1536-d embedding and are trained with a contrastive
loss against the frozen image embeddings. Each config under `configs/model/` is one row of the
planned comparison:

| Config | What it is | Question it answers |
|---|---|---|
| `mean`, `ridge` | Constant and linear floors | Is the pipeline sound; does anything beat a linear map? |
| `eegnet` | The organisers' reference CNN, reimplemented | Do we reproduce the reference level? |
| `conv` | Temporal-spatial conv tokens, flattened and projected | CNN + contrastive baseline |
| `conv_transformer` | Same tokenizer, transformer over the tokens | Does a sequence-transformer head beat the CNN head? |
| `conv_transformer_causal` | Same, causal attention, predicts from the latest token | Does a language-model-style decoder head help? |
| `channel_transformer` | Tokens per (electrode, time patch) with 3-D position embeddings | Can a montage-agnostic encoder use other layouts? |

`conv`, `conv_transformer` and `conv_transformer_causal` share one tokenizer and differ only
in the head, so that comparison isolates the head. `channel_transformer` has no weight that
depends on the number or order of channels: the same checkpoint runs on 63 or 32 electrodes,
and the packaged model uses whichever input electrodes have a known position. Whether either
transformer actually helps is an open question until the runs are in.

## How results stay trustworthy

- **Preprocessing parity.** Training inputs are produced by a reimplementation of the
  organisers' pipeline that matches their extractor on real recordings to 4e-6
  (see [`data/README.md`](data/README.md)).
- **Scorer parity.** `metrics.py` reproduces the organisers' objective, and CI checks it against
  their actual implementation (starter kit pinned by commit) on random and degenerate inputs.
- **The uploaded artefact is the measured model.** Every run packages its model, reloads the
  package through the platform code path, and records the largest difference from the
  in-memory predictions.
- **Contract check before upload.** `scripts/check_submission.py` loads the ZIP in a fresh
  interpreter that cannot import this package and checks shape, dtype, finiteness, variable
  batch size, statelessness across `predict` calls, and invariance to input channel order.
  `make contract-replay` additionally runs the ZIP through the organisers' Benchopt benchmark.
- **Every run is traceable.** `experiments/runs/<run_id>/` stores the resolved config, git
  commit and dirty flag, seed, library versions and metrics; `experiments/registry.jsonl`
  lists all runs.
- **Seeds.** Data generation, splits, batch order and model initialisation take explicit seeds
  from the config; tests assert that equal seeds give identical metrics, and that a run
  interrupted mid-training resumes to the same weights as an uninterrupted one.
- **Validation mirrors the sealed shift.** Models are selected on held-out subjects viewing
  held-out images, scored with the competition metric, never on a batch-level proxy.

## Data

See [`data/README.md`](data/README.md) for datasets, versions, licences and the exact
raw-EEG-to-model-input chain.

## Results

None yet. This section will report, on a public split that holds out subjects and images and
on the sealed leaderboard where access allows: the linear floor, the organisers' EEGNet
baseline, a CNN + contrastive baseline, and the transformer-head ablation, each with seeds.

## Roadmap

| | Milestone | Status |
|---|---|---|
| M0 | Repo, CI, scorer replica, submission packager, contract checks | done |
| M1 | Alljoined-1.6M pipeline with a parity test against NeuralBench; DINOv2-giant targets | done (full-dataset run on Colab pending) |
| M2 | CNN + contrastive baseline and EEGNet on the same split | code done; training runs on Colab pending |
| M3 | Electrode-position-aware channel embedding; channel-dropout and 63→32 transfer tests | model done; transfer experiments pending |
| M4 | Transformer encoder; pooled-CNN head vs transformer over EEG latent tokens, 3 seeds | models done; runs pending |
| M5 | Final training, sealed submissions, one-page report | |
