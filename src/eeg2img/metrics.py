"""Retrieval metrics that reproduce the competition scorer.

The reference is ``tracks/image_decoding/objective.py`` and
``benchmark_utils/metrics.py`` in the organisers' starter kit. The sealed
ranking key is ``top5_acc_subject_agg``:

1. predictions for repeated presentations of one image are averaged *within a
   subject*, without normalising the individual predictions first;
2. each average and every gallery embedding is L2-normalised;
3. queries are ranked against the whole gallery by cosine similarity;
4. a query is a hit when its true image is among the top five, with tied
   scores sharing their average rank.

``tests/test_metrics.py`` checks these functions against hand-computed cases,
and ``tests/test_official_parity.py`` checks them against the organisers' own
implementation when the starter kit is available.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise the last axis, leaving all-zero rows at zero."""
    v = np.asarray(v)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def topk_accuracy(scores: np.ndarray, target_idx: np.ndarray, k: int = 5) -> float:
    """Top-``k`` retrieval accuracy from a ``(N, M)`` score matrix.

    Tied scores share their average rank, so a constant prediction is not
    rewarded by candidate order. A query whose true score is NaN is a miss.
    """
    scores = np.asarray(scores)
    target_idx = np.asarray(target_idx).reshape(-1)
    true = np.take_along_axis(scores, target_idx[:, None], axis=1)
    rank = ((scores > true).sum(axis=1) + (scores >= true).sum(axis=1) - 1) / 2
    rank[np.isnan(true[:, 0])] = np.inf
    return float((rank < k).mean())


def group_means(values: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average the rows of ``values`` that share a key.

    Returns the sorted distinct keys ``(G, K)`` and the float64 means ``(G, ...)``.
    """
    values = np.asarray(values)
    keys = np.asarray(keys).reshape(len(values), -1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(unique_keys), *values.shape[1:]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse, minlength=len(unique_keys))
    return unique_keys, sums / counts.reshape((-1,) + (1,) * (values.ndim - 1))


def _score(
    y_pred: np.ndarray,
    gallery: np.ndarray,
    target_idx: np.ndarray,
    subject: np.ndarray,
    k: int,
) -> dict[str, float | int]:
    gallery_n = l2_normalize(gallery)
    scores = l2_normalize(y_pred) @ gallery_n.T

    keys, pred_agg = group_means(y_pred, np.stack([subject, target_idx], axis=1))
    scores_agg = l2_normalize(pred_agg) @ gallery_n.T
    agg_idx = keys[:, 1]

    return {
        f"top{k}_acc": topk_accuracy(scores, target_idx, k=k),
        "top1_acc": topk_accuracy(scores, target_idx, k=1),
        f"top{k}_acc_subject_agg": topk_accuracy(scores_agg, agg_idx, k=k),
        "top1_acc_subject_agg": topk_accuracy(scores_agg, agg_idx, k=1),
        "n_candidates": len(gallery),
        "n_queries": len(y_pred),
        "n_queries_subject_agg": len(pred_agg),
        f"chance_top{k}": float(min(k, len(gallery)) / len(gallery)),
    }


def score_predictions(
    y_pred: np.ndarray, y_true: np.ndarray, subject: np.ndarray, k: int = 5
) -> dict[str, float | int]:
    """Score predictions exactly as the competition objective does.

    Parameters
    ----------
    y_pred : ``(N, D)`` predicted embeddings, one per EEG epoch.
    y_true : ``(N, D)`` embedding of the image shown on each epoch. The gallery
        is the set of distinct rows, as in the organisers' objective.
    subject : ``(N,)`` integer subject id of each epoch.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    subject = np.asarray(subject).reshape(-1)
    gallery, target_idx = np.unique(y_true, axis=0, return_inverse=True)
    return _score(y_pred, gallery, target_idx.reshape(-1), subject, k)


def score_predictions_indexed(
    y_pred: np.ndarray,
    image_index: np.ndarray,
    embeddings: np.ndarray,
    subject: np.ndarray,
    k: int = 5,
) -> dict[str, float | int]:
    """Same scores as :func:`score_predictions`, from image ids.

    Avoids materialising and de-duplicating an ``(N, D)`` target matrix. The
    gallery is the embeddings of the images that occur in ``image_index``. It
    matches the literal scorer whenever distinct images have distinct
    embeddings, which the tests assert.
    """
    y_pred = np.asarray(y_pred)
    image_index = np.asarray(image_index).reshape(-1)
    subject = np.asarray(subject).reshape(-1)
    present, target_idx = np.unique(image_index, return_inverse=True)
    return _score(y_pred, np.asarray(embeddings)[present], target_idx, subject, k)
