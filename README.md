# eeg2img — EEG-to-image retrieval (EEG/EMG Foundation Challenge 2026, Track 01)

[![CI](https://github.com/moh246/eeg2image-neurips26/actions/workflows/ci.yml/badge.svg)](https://github.com/moh246/eeg2image-neurips26/actions/workflows/ci.yml)

Entry for [Track 01](https://neural-interfaces26.github.io/tracks.html) of the NeurIPS 2026
Brain & Body Workshop challenge. Given one second of 32-channel EEG recorded while a person
looked at an image, predict that image's frozen DINOv2-giant embedding; the platform ranks a
held-out gallery by cosine similarity and scores Top-5 accuracy.

**Status (16 Nov 2026).** The pipeline, models, tests and CI are complete, and the main
experiment has been run on the full public dataset: on participants and images never seen in
training, the CNN + contrastive baseline reaches 16.2 ± 0.3% Top-5 on a 200-image gallery
(chance 2.5%), and replacing its head with a transformer over the EEG latent tokens does not
improve on that (16.0 ± 1.2%). See [Results](#results) and the one-page
[report](docs/report.md) ([PDF](docs/report.pdf)). No leaderboard submission was possible: challenge registration had
closed before this work started.

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
prepared data and a GPU (see [Reproducing the results](#reproducing-the-results)). Any config
value can be overridden on the command line:

```bash
.venv/bin/python scripts/train.py --config configs/experiment/smoke_ridge.yaml fit.alpha=10 seed=1
```

## Layout

```
configs/        YAML only; every hyperparameter lives here (data/, model/, train/, experiment/)
src/eeg2img/
  contract.py   the verified I/O contract as constants
  metrics.py    replica of the competition scorer
  data/         Alljoined-1.6M reader, preprocessing, on-disk store, targets, checks, synthetic data
  models/       model registry: floors, EEGNet, conv baseline, transformer encoders
  montage.py    standard 3-D electrode positions for the position-aware embedding
  engine/       contrastive loss, resumable training loop, closed-form floors, evaluation of
                packaged models (seed ensembles, channel subsets)
  submission/   single-file packager, inference runtime, local contract check
  tracking.py   per-run records: config, git commit, metrics
  results.py    import of remote run records; seed-aggregated results table
  pipeline.py   train -> evaluate -> package, shared by every entry point
scripts/        thin CLI wrappers (prepare, embed, train, queue, evaluate, summarise)
notebooks/      Colab launchers for the scripts; no logic of their own
tests/          unit, regression and integration tests
experiments/    committed run records and the run registry
docs/           the one-page report
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
and the packaged model uses whichever input electrodes have a known position.

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

Top-5 retrieval accuracy in percent, scored with the competition metric (predictions averaged
over a participant's repeats of an image, then ranked). Both validation sets contain only
participants (4 of 20) and images that the model never saw in training, which is the shift of
the sealed test set. Mean ± standard deviation over seeds; every row is generated from the run
records in [`experiments/`](experiments/) by `make results`, and a test fails if it drifts.

<!-- results:start (generated by scripts/summarize_runs.py; do not edit) -->
| experiment | seeds | parameters | val Top-5 % (1650-way, chance 0.30) | val_test Top-5 % (200-way, chance 2.50) |
|---|---|---|---|---|
| alljoined_ridge | 1 | — | 0.32 | 2.62 |
| alljoined_eegnet | 1 | 76,848 | 1.17 | 11.62 |
| alljoined_conv | 3 | 6,748,384 | 1.72 ± 0.05 | 16.17 ± 0.29 |
| alljoined_conv_transformer | 3 | 3,415,776 | 1.63 ± 0.10 | 16.04 ± 1.16 |
| alljoined_conv_transformer_causal | 1 | 3,415,776 | 1.65 | 16.12 |
<!-- results:end -->

- `val`: 1,650 held-out training-pool images (165 object concepts), about 4 repeats per
  participant and image. Used for checkpoint selection, so it is slightly optimistic.
- `val_test`: the dataset's own 200 test images, about 81 repeats per participant and image.
  Not used for any selection. One run's sampling error is about ±1.3 points here and ±0.16
  on `val`.

What the numbers say:

- **A transformer head over EEG latent tokens does not beat the CNN head.** With the same
  tokenizer, loss, data and schedule, the bidirectional transformer scores 16.0 ± 1.2 against
  16.2 ± 0.3 (`val_test`) and 1.63 ± 0.10 against 1.72 ± 0.05 (`val`). The differences are
  inside the seed and sampling noise. The transformer reaches that level with half the
  parameters and peaks later in training (epochs 6 to 9) than the CNN head (epoch 3 or 4).
- **The causal, language-model-style head behaves the same** (16.1 and 1.65), from one seed.
- **Both are well above the organisers' reference architecture** (EEGNet, 11.6 and 1.17) and
  a linear map, which stays at chance.
- **Absolute accuracy is low because the task is hard**, not because of the split: the
  training loss stays close to its chance level for every model, i.e. single trials carry
  little image information that transfers across people. The sealed leaderboard shows the same
  (leader 0.126% Top-5, reference EEGNet 0.027%, on 12 Nov 2026).

Not done: the GPU session ended (Colab free-tier usage limit) before the rest of the queue
ran. Missing are seeds 1 and 2 of the causal head and of EEGNet, the montage-agnostic
`channel_transformer`, the seed ensembles, the channel-subset evaluation on real data, and the
longer-schedule transformer run. Their configs and scripts are in the repository
(`scripts/run_queue.sh`, `scripts/colab_followups.sh`). A foundation-model baseline (REVE) was
not attempted.

No sealed submission: challenge registration had closed before this work started, so no model
could be uploaded to the leaderboard. Every run above still produced a platform-format
`submission.zip` that reloads to the in-memory predictions (largest difference 2e-4) and
passes the contract check (`contract_check_ok` in each run record), and `make contract-replay`
runs a package through the organisers' Benchopt benchmark.

## Reproducing the results

On a machine with one GPU (the runs above used a Colab T4), 20 GB of scratch disk and the
repository cloned:

```bash
pip install -e ".[data]" transformers torchvision pillow

# 1. Data: download Alljoined-1.6M, preprocess, compute the image targets (about 1 hour).
#    Running this accepts the dataset's CC-BY-NC-ND-4.0 licence.
bash scripts/colab_prepare.sh /content/work /content/persist

# 2. Check the store against data/README.md, and the image-group assumption of the split.
cat /content/persist/stores/alljoined16m_v1/SHA256SUMS
STORE=/content/work/stores/alljoined16m_v1
python scripts/check_image_groups.py --config configs/experiment/alljoined_conv.yaml \
    data.store=$STORE data.embeddings=$STORE/embeddings.npy

# 3. Train. Each job resumes if interrupted and is skipped once finished.
#    T4 wall time: ridge 3 min, eegnet 18 min, conv 20 min, conv_transformer 35 min.
bash scripts/run_queue.sh /content/runs \
    alljoined_ridge:0 alljoined_eegnet:0 \
    alljoined_conv:0 alljoined_conv:1 alljoined_conv:2 \
    alljoined_conv_transformer:0 alljoined_conv_transformer:1 alljoined_conv_transformer:2 \
    alljoined_conv_transformer_causal:0 \
    -- data.store=$STORE data.embeddings=$STORE/embeddings.npy

# 4. Table.
python scripts/summarize_runs.py --registry /content/runs/experiments/registry.jsonl
```

Splits, batch order and initialisation are seeded, so the same commit, data and library
versions reproduce a run; across GPU models and library versions expect differences in the
last digits. The versions of each run are in its `meta.json`. To bring records from the GPU
machine into the repository, archive `/content/runs/experiments` and run
`python scripts/import_runs.py <archive>`, then `make results`.

[`notebooks/`](notebooks/) holds Colab launchers for steps 1 and 3.

## Roadmap

| | Milestone | Status |
|---|---|---|
| M0 | Repo, CI, scorer replica, submission packager, contract checks | done |
| M1 | Alljoined-1.6M pipeline with a parity test against NeuralBench; DINOv2-giant targets | done; full store built and checksummed |
| M2 | CNN + contrastive baseline and EEGNet on the same split | done (EEGNet has one seed) |
| M3 | Electrode-position-aware channel embedding; channel-subset and 63→32 transfer tests | model and evaluation code done; no real-data run yet |
| M4 | Pooled-CNN head vs transformer over EEG latent tokens, 3 seeds | done; causal variant has one seed |
| M5 | One-page report; submission | report done ([docs/report.md](docs/report.md)); submission not possible (registration closed), packaging validated locally |
