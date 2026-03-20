"""Tests for item-item co-occurrence recommendation."""

from __future__ import annotations

import pandas as pd

from src.candidate_gen.item_knn import ItemKNNRecommender


def test_items_that_cooccur_become_neighbors() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": ["A", "B", "A", "C"],
            "event_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    recommender = ItemKNNRecommender().fit(train_df)

    assert recommender.item_neighbors_["A"]["B"] > 0
    assert recommender.item_neighbors_["A"]["C"] > 0


def test_seen_items_are_not_recommended() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": ["A", "B", "A", "C"],
            "event_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    recommender = ItemKNNRecommender().fit(train_df)
    recommendations = recommender.recommend_for_user(user_id=1, user_history={"A", "B"}, k=5)

    assert "A" not in recommendations
    assert "B" not in recommendations


def test_users_with_different_histories_receive_different_recommendations() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["A", "B", "A", "C", "D", "E"],
            "event_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    recommender = ItemKNNRecommender().fit(train_df)
    user_one_recs = recommender.recommend_for_user(user_id=1, user_history={"A", "B"}, k=3)
    user_three_recs = recommender.recommend_for_user(user_id=3, user_history={"D", "E"}, k=3)

    assert user_one_recs != user_three_recs


def test_stronger_cooccurrence_ranks_higher() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "item_id": ["A", "B", "A", "B", "A", "C", "A", "C"],
            "event_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    recommender = ItemKNNRecommender().fit(train_df)
    recommendations = recommender.recommend_for_user(user_id=99, user_history={"A"}, k=2)

    assert recommendations[0] == "B"
