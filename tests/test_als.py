"""Tests for ALS candidate generation helpers."""

from __future__ import annotations

import pandas as pd

from src.candidate_gen.als import ALSCandidateGenerator


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
            "event_weight": [1.0, 2.0, 1.0, 1.0, 1.0],
        }
    )


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


def test_unseen_users_return_empty_recommendations() -> None:
    generator = ALSCandidateGenerator()
    generator.user_id_to_idx = {10: 0}
    generator.idx_to_item_id = {0: 100, 1: 200}
    generator.user_item_matrix = None
    generator.model = FakeALSModel()

    recommendations = generator.recommend_for_user(user_id=999, user_history=set(), k=5)

    assert recommendations == []
