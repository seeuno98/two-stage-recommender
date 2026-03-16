"""Candidate generation evaluation helpers."""

from src.eval.metrics import recall_at_k


def evaluate_candidate_lists(
    actual: list[object],
    predicted: list[object],
    k: int = 100,
) -> dict[str, float]:
    """Evaluate candidate generation outputs using Recall@K."""
    return {"recall_at_k": recall_at_k(actual=actual, predicted=predicted, k=k)}
