"""Tests for ranking dataset construction and reranking helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.dataset import (
    build_group_array,
    build_labeled_ranking_dataframe,
    make_candidate_pool_for_users,
)
from src.ranking.dataset import build_ranking_dataset_from_splits
from src.ranking.predict import rerank_candidates, topk_predictions_from_ranked_df
from src.ranking.train_ranker import split_ranking_dataset_by_user


class StubRetriever:
    """Minimal retriever that returns fixed candidates per user."""

    def __init__(self) -> None:
        self.popularity_scores_ = {10: 5.0, 20: 4.0, 30: 3.0}

    def recommend_for_user(
        self,
        user_id: int | str,
        user_history: set[object],
        k: int = 10,
    ) -> list[object]:
        del user_id, k
        return [item_id for item_id in [10, 20, 30] if item_id not in user_history]


def test_group_array_creation() -> None:
    df = pd.DataFrame({"user_id": [1, 1, 2, 3, 3, 3], "item_id": [1, 2, 3, 4, 5, 6]})

    assert build_group_array(df) == [2, 1, 3]


def test_user_level_split_has_no_user_leakage() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "item_id": [10, 11, 12, 13, 14, 15, 16, 17],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train_df, valid_df = split_ranking_dataset_by_user(df, valid_frac=0.25, random_state=42)

    assert set(train_df["user_id"]).isdisjoint(set(valid_df["user_id"]))


def test_reranking_sorts_by_descending_score_per_user() -> None:
    candidate_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 30, 40],
            "popularity_rank": [2, 1, 2, 1],
        }
    )
    scores = np.array([0.2, 0.8, 0.1, 0.9])

    ranked = rerank_candidates(candidate_df, scores)

    assert ranked[ranked["user_id"] == 1]["item_id"].tolist() == [20, 10]
    assert ranked[ranked["user_id"] == 2]["item_id"].tolist() == [40, 30]


def test_topk_extraction_works_correctly() -> None:
    ranked_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [10, 20, 30, 40, 50],
            "ranking_score": [0.9, 0.8, 0.1, 0.7, 0.6],
            "rank": [1, 2, 3, 1, 2],
        }
    )

    predictions = topk_predictions_from_ranked_df(ranked_df, k=2)

    assert predictions == {1: [10, 20], 2: [40, 50]}


def test_ranking_dataset_labels_are_correct_for_toy_example() -> None:
    candidate_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 30],
            "popularity_rank": [1, 2, 1],
            "popularity_score": [5.0, 4.0, 3.0],
        }
    )

    labeled = build_labeled_ranking_dataframe(candidate_df, ground_truth={1: {20}, 2: {30}})

    assert labeled["label"].tolist() == [0, 1, 1]


def test_candidate_pool_generation_excludes_seen_items() -> None:
    retriever = StubRetriever()
    candidates = make_candidate_pool_for_users(
        retriever=retriever,  # type: ignore[arg-type]
        user_histories={1: {10}, 2: {30}},
        top_n=3,
    )

    assert candidates[candidates["user_id"] == 1]["item_id"].tolist() == [20, 30]
    assert candidates[candidates["user_id"] == 2]["item_id"].tolist() == [10, 20]


def test_build_ranking_dataset_from_splits_train_and_test_behaviour() -> None:
    # toy interactions
    train = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "event_weight": [1.0, 1.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    val = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [30],
            "event_weight": [1.0],
            "timestamp": pd.to_datetime(["2020-01-03"]),
        }
    )
    test = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [40],
            "event_weight": [1.0],
            "timestamp": pd.to_datetime(["2020-01-04"]),
        }
    )

    # build train->val ranking dataset (labels come from val)
    ranking_train, feature_cols_train, summary_train = build_ranking_dataset_from_splits(
        history_df=train, target_df=val, item_features_df=None, candidate_top_n=10
    )
    # user 1 should have label=1 for item 30 in training ranking dataset
    labels_for_user1 = ranking_train[ranking_train["user_id"] == 1].set_index("item_id")["label"].to_dict()
    assert labels_for_user1.get(30) == 1

    # build (train+val)->test ranking dataset
    full_history = pd.concat([train, val], ignore_index=True)
    ranking_test, feature_cols_test, summary_test = build_ranking_dataset_from_splits(
        history_df=full_history, target_df=test, item_features_df=None, candidate_top_n=10
    )
    # user 1 should have label=1 for item 40 in test candidate dataset
    labels_for_user1_test = ranking_test[ranking_test["user_id"] == 1].set_index("item_id")["label"].to_dict()
    assert labels_for_user1_test.get(40) == 1
