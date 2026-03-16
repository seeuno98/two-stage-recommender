"""Tests for ranking metrics."""

from src.eval.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_returns_expected_value() -> None:
    actual = [1, 2]
    predicted = [2, 3, 1]
    assert recall_at_k(actual=actual, predicted=predicted, k=2) == 0.5


def test_ndcg_at_k_returns_expected_value() -> None:
    actual = [1, 2]
    predicted = [2, 3, 1]
    score = ndcg_at_k(actual=actual, predicted=predicted, k=3)
    assert round(score, 4) == 0.9197
