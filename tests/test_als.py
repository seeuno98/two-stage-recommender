"""Tests for ALS candidate generation helpers."""

from __future__ import annotations

import pandas as pd

from src.candidate_gen.als import ALSCandidateGenerator
from src.data.preprocess import filter_interactions_by_event_types, remap_event_weights


class FakeALSModel:
    """Minimal stand-in for implicit ALS recommend()."""

    def recommend(self, userid: int, user_items, N: int, filter_already_liked_items: bool):  # type: ignore[no-untyped-def]
        del userid, user_items, N, filter_already_liked_items
        return [0, 3, 1, 2], [0.9, 0.8, 0.7, 0.6]


def make_train_df() -> pd.DataFrame:
    """Create a small deterministic training dataframe."""

    return pd.DataFrame(
        {
            "user_id": [10, 10, 20, 20, 20],
            "item_id": [100, 100, 100, 200, 300],
            "event_type": ["view", "addtocart", "view", "addtocart", "transaction"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                    "2020-01-01T02:00:00Z",
                    "2020-01-01T03:00:00Z",
                    "2020-01-01T04:00:00Z",
                ],
                utc=True,
            ),
            "event_weight": [1.0, 2.0, 1.0, 1.0, 1.0],
        }
    )


def test_filter_interactions_by_event_types_removes_views() -> None:
    train_df = make_train_df()

    filtered = filter_interactions_by_event_types(train_df, ["addtocart", "transaction"])

    assert filtered["event_type"].tolist() == ["addtocart", "addtocart", "transaction"]
    assert "view" not in set(filtered["event_type"])


def test_remap_event_weights_reassigns_values() -> None:
    train_df = make_train_df()
    filtered = filter_interactions_by_event_types(train_df, ["addtocart", "transaction"])

    remapped = remap_event_weights(
        filtered,
        {"addtocart": 10.0, "transaction": 30.0},
    )

    assert remapped["event_weight"].tolist() == [10.0, 10.0, 30.0]


def test_id_encoding_round_trip() -> None:
    train_df = make_train_df()

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(train_df)
    encoded = generator._build_id_maps(aggregated)

    assert encoded.idx_to_user_id[encoded.user_id_to_idx[10]] == 10
    assert encoded.idx_to_item_id[encoded.item_id_to_idx[200]] == 200


def test_user_item_matrix_shape_is_correct() -> None:
    train_df = make_train_df()

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(train_df)
    encoded = generator._build_id_maps(aggregated)
    generator.user_id_to_idx = encoded.user_id_to_idx
    generator.item_id_to_idx = encoded.item_id_to_idx

    matrix = generator._build_interaction_matrix(aggregated)

    assert matrix.shape == (2, 3)


def test_matrix_building_aggregates_duplicate_interactions() -> None:
    train_df = make_train_df()

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(train_df)
    encoded = generator._build_id_maps(aggregated)
    generator.user_id_to_idx = encoded.user_id_to_idx
    generator.item_id_to_idx = encoded.item_id_to_idx

    matrix = generator._build_interaction_matrix(aggregated)

    assert matrix[0, 0] == 3.0


def test_matrix_aggregation_sums_reweighted_duplicate_rows() -> None:
    train_df = make_train_df()
    remapped = remap_event_weights(
        train_df,
        {"view": 1.0, "addtocart": 10.0, "transaction": 30.0},
    )

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(remapped)

    user_10_item_100 = aggregated[
        (aggregated["user_id"] == 10) & (aggregated["item_id"] == 100)
    ]["interaction_strength"].iloc[0]

    assert user_10_item_100 == 11.0


def test_recommend_output_uses_original_item_ids() -> None:
    train_df = make_train_df()

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(train_df)
    encoded = generator._build_id_maps(aggregated)
    generator.user_id_to_idx = encoded.user_id_to_idx
    generator.item_id_to_idx = encoded.item_id_to_idx
    generator.idx_to_item_id = {0: 100, 1: 200, 2: 300}
    generator.user_item_matrix = generator._build_interaction_matrix(aggregated)
    generator.item_user_matrix = generator.user_item_matrix.T.tocsr()
    generator.model = FakeALSModel()

    recommendations = generator.recommend_for_user(user_id=10, user_history={100}, k=2)

    assert all(item_id in {100, 200, 300} for item_id in recommendations)


def test_seen_items_are_filtered_from_recommendations() -> None:
    train_df = make_train_df()

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(train_df)
    encoded = generator._build_id_maps(aggregated)
    generator.user_id_to_idx = encoded.user_id_to_idx
    generator.item_id_to_idx = encoded.item_id_to_idx
    generator.idx_to_item_id = {0: 100, 1: 200, 2: 300}
    generator.user_item_matrix = generator._build_interaction_matrix(aggregated)
    generator.item_user_matrix = generator.user_item_matrix.T.tocsr()
    generator.model = FakeALSModel()

    recommendations = generator.recommend_for_user(user_id=10, user_history={100}, k=3)

    assert 100 not in recommendations


def test_original_train_history_still_filters_seen_items_after_strong_event_training() -> None:
    train_df = pd.DataFrame(
        {
            "user_id": [10, 10, 10, 20],
            "item_id": [100, 200, 300, 400],
            "event_type": ["view", "addtocart", "transaction", "transaction"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                    "2020-01-01T02:00:00Z",
                    "2020-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "event_weight": [1.0, 3.0, 5.0, 5.0],
        }
    )
    filtered_train_df = filter_interactions_by_event_types(train_df, ["addtocart", "transaction"])

    generator = ALSCandidateGenerator()
    aggregated = generator._aggregate_interactions(filtered_train_df)
    encoded = generator._build_id_maps(aggregated)
    generator.user_id_to_idx = encoded.user_id_to_idx
    generator.item_id_to_idx = encoded.item_id_to_idx
    generator.idx_to_item_id = {0: 200, 1: 300, 2: 400}
    generator.user_item_matrix = generator._build_interaction_matrix(aggregated)
    generator.item_user_matrix = generator.user_item_matrix.T.tocsr()
    generator.model = FakeALSModel()

    original_train_history = {100, 200, 300}
    recommendations = generator.recommend_for_user(user_id=10, user_history=original_train_history, k=3)

    assert recommendations == [400]


def test_unseen_users_return_empty_recommendations() -> None:
    generator = ALSCandidateGenerator()
    generator.user_id_to_idx = {10: 0}
    generator.idx_to_item_id = {0: 100, 1: 200}
    generator.user_item_matrix = None
    generator.model = FakeALSModel()

    recommendations = generator.recommend_for_user(user_id=999, user_history=set(), k=5)

    assert recommendations == []
