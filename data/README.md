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
(`defaults/eeg/config.yaml` and `tasks/eeg/image/config.yaml`). Reading those configs gives:

1. Pick EEG channels; the Emotiv recordings mark all 32 as EEG.
2. Band-pass 0.1–75 Hz. At the 120 Hz target rate the 75 Hz edge is above Nyquist, and the
   organisers' loader drops it, leaving a 0.1 Hz high-pass.
3. Notch at 50 and 60 Hz.
4. Resample to 120 Hz.
5. Epoch from −0.2 s to +0.8 s around each image onset: 120 samples.
6. Baseline-correct using the first 0.2 s of the epoch.
7. Robust scaling.
8. Clip to ±20.

Events that share an onset timestamp are dropped as ambiguous, following the organisers' reader.

**Status.** Steps 1–8 are transcribed from the configs. The exact semantics of steps 6 and 7
(per channel, per recording or per epoch) are not yet confirmed from the extractor source. The
M1 pipeline will settle them with a parity test that compares this repository's output against
NeuralBench's on the same recording, and this section will then be updated to state them.

## Targets

Image embeddings come from `facebook/dinov2-giant` at 518 px: the hidden state at 2/3 depth,
averaged over tokens, giving 1536 dimensions. They are computed once per unique image and cached.

## Channel order

Models never assume an input channel order. At inference the platform supplies channel names,
and `eeg2img.channels.ChannelAligner` reorders the input to the training order by name.
The Emotiv order used for training is `eeg2img.contract.EMOTIV_32`.
