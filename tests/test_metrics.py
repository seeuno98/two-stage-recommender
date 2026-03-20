"""Tests for ranking metrics and popularity recommendation."""

from __future__ import annotations

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.eval.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_returns_expected_value() -> None:
    actual = {1, 2}
    predicted = [2, 3, 1]
    assert recall_at_k(actual_items=actual, predicted_items=predicted, k=2) == 0.5


def test_ndcg_at_k_rewards_earlier_hits() -> None:
    actual = {1, 2}
    earlier_hits = [1, 2, 3]
    later_hits = [3, 1, 2]

    assert ndcg_at_k(actual_items=actual, predicted_items=earlier_hits, k=3) > ndcg_at_k(
        actual_items=actual,
        predicted_items=later_hits,
        k=3,
    )


def test_popularity_recommender_excludes_seen_items() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [101, 102, 103],
            "event_weight": [1.0, 1.0, 1.0],
        }
    )

    recommender = PopularityRecommender().fit(train_df)
    recommendations = recommender.recommend_for_user(user_id=1, user_history={101, 102}, k=2)

    assert 101 not in recommendations
    assert 102 not in recommendations


def test_popularity_recommender_uses_weighted_popularity() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [101, 102, 101],
            "event_weight": [1.0, 4.0, 1.0],
        }
    )

    recommender = PopularityRecommender().fit(train_df)

    assert recommender.ranked_items_[0] == 102
