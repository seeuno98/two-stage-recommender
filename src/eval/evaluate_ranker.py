"""Ranking evaluation helpers."""

from src.eval.metrics import ndcg_at_k


def evaluate_ranked_list(
    actual: list[object],
    predicted: list[object],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate ranked outputs using NDCG@K."""
    return {"ndcg_at_k": ndcg_at_k(actual=actual, predicted=predicted, k=k)}
