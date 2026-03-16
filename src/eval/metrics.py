"""Ranking metrics for offline evaluation."""

from math import log2


def recall_at_k(
    actual: list[object],
    predicted: list[object],
    k: int,
) -> float:
    """Compute Recall@K for a single example."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if not actual:
        return 0.0

    actual_set = set(actual)
    predicted_at_k = predicted[:k]
    hits = sum(1 for item in predicted_at_k if item in actual_set)
    return hits / len(actual_set)


def ndcg_at_k(
    actual: list[object],
    predicted: list[object],
    k: int,
) -> float:
    """Compute NDCG@K for a single example with binary relevance."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if not actual:
        return 0.0

    actual_set = set(actual)
    predicted_at_k = predicted[:k]

    dcg = 0.0
    for rank, item in enumerate(predicted_at_k, start=1):
        if item in actual_set:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0

    return dcg / idcg
