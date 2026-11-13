"""Parity with the organisers' own scorer.

Runs only when the starter kit is available: set ``STARTER_KIT_DIR`` to a
checkout of ``neural-interfaces26/2026-competition`` (``make contract-replay``
does this). The kit's metric helpers are plain NumPy, so nothing else is needed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from eeg2img.metrics import group_means, score_predictions, topk_accuracy

KIT = os.environ.get("STARTER_KIT_DIR")
pytestmark = [
    pytest.mark.official,
    pytest.mark.skipif(not KIT, reason="STARTER_KIT_DIR is not set"),
]


@pytest.fixture(scope="module")
def kit_metrics():
    path = Path(KIT) / "benchmark_utils" / "metrics.py"
    spec = importlib.util.spec_from_file_location("kit_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_scores(kit, y_pred, y_true, subject):
    """The body of ``Objective.evaluate_result`` in the kit's objective.py."""

    def normalize(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    candidates, target_idx = np.unique(y_true, axis=0, return_inverse=True)
    scores = normalize(y_pred) @ normalize(candidates).T
    keys, pred_sa = kit.group_means(y_pred, np.stack([subject, target_idx.reshape(-1)], axis=1))
    scores_sa = normalize(pred_sa) @ normalize(candidates).T
    return {
        "top5_acc": kit.topk_accuracy(scores, target_idx, k=5),
        "top1_acc": kit.topk_accuracy(scores, target_idx, k=1),
        "top5_acc_subject_agg": kit.topk_accuracy(scores_sa, keys[:, 1], k=5),
        "top1_acc_subject_agg": kit.topk_accuracy(scores_sa, keys[:, 1], k=1),
        "n_candidates": len(candidates),
    }


@pytest.mark.parametrize("seed", range(5))
def test_scores_match_the_organisers_objective(kit_metrics, seed):
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((60, 24)).astype(np.float32)
    image_index = rng.integers(0, 60, size=900)
    subject = rng.integers(0, 5, size=900)
    y_pred = (table[image_index] + 4.0 * rng.standard_normal((900, 24))).astype(np.float32)
    ours = score_predictions(y_pred, table[image_index], subject)
    theirs = _official_scores(kit_metrics, y_pred, table[image_index], subject)
    for key, value in theirs.items():
        assert ours[key] == pytest.approx(value, abs=0), key


def test_helpers_match_on_ties_and_nans(kit_metrics):
    scores = np.array([[1.0, 1.0, 0.5, np.nan], [np.nan, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]])
    target = np.array([0, 0, 2])
    for k in (1, 2, 3):
        assert topk_accuracy(scores, target, k) == kit_metrics.topk_accuracy(scores, target, k)
    values = np.arange(12.0).reshape(6, 2)
    keys = np.array([[0, 1], [0, 1], [1, 0], [0, 0], [1, 0], [0, 1]])
    ours, theirs = group_means(values, keys), kit_metrics.group_means(values, keys)
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
