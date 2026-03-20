"""Ranking metrics for offline evaluation."""

from __future__ import annotations

from math import log2


def recall_at_k(
    actual_items: list[object] | set[object],
    predicted_items: list[object],
    k: int,
) -> float:
    """Compute Recall@K for a single user."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    actual_set = set(actual_items)
    if not actual_set:
        return 0.0

    predicted_at_k = predicted_items[:k]
    hits = sum(1 for item in predicted_at_k if item in actual_set)
    return hits / len(actual_set)


def ndcg_at_k(
    actual_items: list[object] | set[object],
    predicted_items: list[object],
    k: int,
) -> float:
    """Compute NDCG@K for a single user with binary relevance."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    actual_set = set(actual_items)
    if not actual_set:
        return 0.0

    predicted_at_k = predicted_items[:k]
    dcg = 0.0
    for rank, item in enumerate(predicted_at_k, start=1):
        if item in actual_set:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_user_level(
    ground_truth: dict[int | str, list[object] | set[object]],
    predictions: dict[int | str, list[object]],
    k_values: list[int],
) -> dict[str, float]:
    """Evaluate Recall@K and NDCG@K by averaging across eligible users."""

    metrics: dict[str, float] = {}
    eligible_users = [
        user_id for user_id, actual_items in ground_truth.items() if len(set(actual_items)) > 0
    ]
    if not eligible_users:
        return {
            metric_name: 0.0
            for k in k_values
            for metric_name in (f"recall@{k}", f"ndcg@{k}")
        }

    for k in k_values:
        recall_scores: list[float] = []
        ndcg_scores: list[float] = []

        for user_id in eligible_users:
            actual_items = ground_truth[user_id]
            predicted_items = predictions.get(user_id, [])
            recall_scores.append(recall_at_k(actual_items=actual_items, predicted_items=predicted_items, k=k))
            ndcg_scores.append(ndcg_at_k(actual_items=actual_items, predicted_items=predicted_items, k=k))

        metrics[f"recall@{k}"] = sum(recall_scores) / len(recall_scores)
        metrics[f"ndcg@{k}"] = sum(ndcg_scores) / len(ndcg_scores)

    return metrics
