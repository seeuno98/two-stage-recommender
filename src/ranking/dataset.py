"""Utilities for assembling ranking datasets."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender


def build_user_histories(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build per-user interacted item sets from standardized interactions."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def build_ground_truth(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build per-user target item sets from standardized interactions."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def make_candidate_pool_for_users(
    retriever: PopularityRecommender,
    user_histories: Mapping[int | str, set[object]],
    top_n: int = 100,
) -> pd.DataFrame:
    """Generate a deterministic popularity-based candidate pool for users."""

    rows: list[dict[str, object]] = []
    for user_id in sorted(user_histories):
        user_history = user_histories[user_id]
        candidate_items = retriever.recommend_for_user(
            user_id=user_id,
            user_history=user_history,
            k=top_n,
        )
        for rank, item_id in enumerate(candidate_items, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "popularity_rank": rank,
                    "popularity_score": float(retriever.popularity_scores_.get(item_id, 0.0)),
                }
            )

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return pd.DataFrame(
            columns=["user_id", "item_id", "popularity_rank", "popularity_score"]
        )

    candidate_df = candidate_df.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
    candidate_df["popularity_rank"] = candidate_df["popularity_rank"].astype(int)
    candidate_df["popularity_score"] = candidate_df["popularity_score"].astype(float)
    return candidate_df


def build_labeled_ranking_dataframe(
    candidate_df: pd.DataFrame,
    ground_truth: Mapping[int | str, set[object]],
) -> pd.DataFrame:
    """Assign binary labels to candidate rows using per-user target item sets."""

    labeled = candidate_df.copy()
    if labeled.empty:
        labeled["label"] = pd.Series(dtype="int64")
        return labeled

    labeled["label"] = labeled.apply(
        lambda row: int(row["item_id"] in ground_truth.get(row["user_id"], set())),
        axis=1,
    )
    labeled["label"] = labeled["label"].astype(int)
    return labeled


def build_group_array(
    df: pd.DataFrame,
    group_col: str = "user_id",
) -> list[int]:
    """Return LightGBM group sizes ordered by dataframe row order."""

    if df.empty:
        return []

    ordered_groups = df[group_col].drop_duplicates().tolist()
    group_sizes = [
        int((df[group_col] == group_id).sum())
        for group_id in ordered_groups
    ]
    return group_sizes
