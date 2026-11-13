"""The scorer replica, checked against hand-computed cases."""

from __future__ import annotations

import numpy as np
import pytest

from eeg2img.metrics import (
    group_means,
    l2_normalize,
    score_predictions,
    score_predictions_indexed,
    topk_accuracy,
)


def test_topk_counts_a_hit_when_true_rank_is_below_k():
    scores = np.array([[0.9, 0.1, 0.0], [0.2, 0.1, 0.9], [0.1, 0.5, 0.9]])
    target = np.array([0, 0, 0])  # ranks 0, 1, 2
    assert topk_accuracy(scores, target, k=1) == pytest.approx(1 / 3)
    assert topk_accuracy(scores, target, k=2) == pytest.approx(2 / 3)
    assert topk_accuracy(scores, target, k=3) == pytest.approx(1.0)


def test_tied_scores_share_their_average_rank():
    # All four candidates tie: average rank is 1.5, a hit for k=2 but not k=1.
    scores = np.ones((1, 4))
    assert topk_accuracy(scores, np.array([0]), k=1) == 0.0
    assert topk_accuracy(scores, np.array([0]), k=2) == 1.0
    # The result must not depend on which tied candidate is the target.
    assert topk_accuracy(scores, np.array([3]), k=2) == 1.0


def test_nan_true_score_is_a_miss():
    scores = np.array([[np.nan, 0.1, 0.2]])
    assert topk_accuracy(scores, np.array([0]), k=3) == 0.0


def test_group_means_averages_rows_sharing_a_key():
    values = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    keys, means = group_means(values, np.array([[0, 7], [0, 7], [1, 7]]))
    np.testing.assert_array_equal(keys, [[0, 7], [1, 7]])
    np.testing.assert_allclose(means, [[2.0, 0.0], [0.0, 5.0]])


def test_l2_normalize_leaves_zero_rows_finite():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_perfect_predictions_score_one():
    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((12, 6))
    idx = np.repeat(np.arange(12), 3)
    scores = score_predictions(gallery[idx] * 2.5, gallery[idx], np.zeros(36, dtype=int))
    assert scores["top5_acc"] == 1.0
    assert scores["top1_acc_subject_agg"] == 1.0
    assert scores["n_candidates"] == 12
    assert scores["n_queries_subject_agg"] == 12


def test_constant_prediction_scores_chance():
    rng = np.random.default_rng(1)
    gallery = rng.standard_normal((20, 8))
    idx = np.arange(20)
    constant = np.tile(rng.standard_normal(8), (20, 1))
    scores = score_predictions(constant, gallery[idx], np.zeros(20, dtype=int))
    # The same five candidates top every query, so exactly five queries hit.
    assert scores["top5_acc"] == pytest.approx(5 / 20)
    assert scores["chance_top5"] == pytest.approx(5 / 20)


def test_subject_aggregation_averages_before_normalising():
    # Image 0 has three repeats: one large-norm correct vote and two unit-norm
    # wrong ones. Averaging raw predictions gives [10/3, 2/3], a hit. Normalising
    # each repeat first would give [1/3, 2/3], a miss. The scorer does the former.
    table = np.array([[1.0, 0.0], [0.0, 1.0]])
    image_index = np.array([0, 0, 0, 1])
    y_pred = np.array([[10.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    scores = score_predictions_indexed(y_pred, image_index, table, np.zeros(4, dtype=int), k=1)
    assert scores["n_candidates"] == 2
    assert scores["top1_acc"] == pytest.approx(0.5)
    assert scores["top1_acc_subject_agg"] == 1.0

    normalised_first = l2_normalize(y_pred)
    scores = score_predictions_indexed(
        normalised_first, image_index, table, np.zeros(4, dtype=int), k=1
    )
    assert scores["top1_acc_subject_agg"] == pytest.approx(0.5)


def test_aggregation_is_within_subject_not_across():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_true = gallery[[0, 0, 1, 1]]
    # Subject 0 is right about image 0; subject 1 is wrong about it.
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    scores = score_predictions(y_pred, y_true, np.array([0, 1, 0, 1]), k=1)
    assert scores["n_queries_subject_agg"] == 4
    assert scores["top1_acc_subject_agg"] == pytest.approx(3 / 4)


def test_indexed_scorer_matches_literal_scorer():
    rng = np.random.default_rng(2)
    table = rng.standard_normal((30, 10)).astype(np.float32)
    image_index = rng.integers(0, 30, size=400)
    subject = rng.integers(0, 4, size=400)
    y_pred = table[image_index] + 2.0 * rng.standard_normal((400, 10)).astype(np.float32)
    literal = score_predictions(y_pred, table[image_index], subject)
    indexed = score_predictions_indexed(y_pred, image_index, table, subject)
    assert literal == indexed
    assert 0.0 < literal["top5_acc"] < 1.0, "case should be non-degenerate"


def test_indexed_gallery_only_contains_images_present():
    table = np.eye(6, dtype=np.float32)
    scores = score_predictions_indexed(
        table[[0, 2]], np.array([0, 2]), table, np.array([0, 0]), k=1
    )
    assert scores["n_candidates"] == 2
    assert scores["top1_acc"] == 1.0


def test_known_scores_regression():
    """Fixed input -> fixed scores; guards the scorer against silent change."""
    rng = np.random.default_rng(1234)
    table = rng.standard_normal((50, 12))
    image_index = np.repeat(np.arange(50), 4)
    subject = np.tile(np.arange(2), 100)
    y_pred = table[image_index] + 3.0 * rng.standard_normal((200, 12))
    scores = score_predictions_indexed(y_pred, image_index, table, subject)
    assert scores["top5_acc"] == pytest.approx(0.445)
    assert scores["top5_acc_subject_agg"] == pytest.approx(0.66)
    assert scores["top1_acc_subject_agg"] == pytest.approx(0.29)
