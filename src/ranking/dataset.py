"""Utilities for assembling ranking datasets."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.ranking.features import add_ranking_features


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


def build_ranking_dataset_from_splits(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    item_features_df: pd.DataFrame | None = None,
    candidate_top_n: int = 100,
    retriever_name: str = "popularity",
    retriever: object | None = None,
    include_labels: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Build a labeled candidate dataframe and ranking features from arbitrary splits.

    Parameters
    - history_df: interactions used to build user/item features and to fit retrievers
    - target_df: interactions that define positive labels for evaluation (one row per positive)
    - item_features_df: optional item metadata for features
    - candidate_top_n: top-N candidates per user to generate
    - retriever_name: name hint for the retriever (currently only "popularity")
    - retriever: optional pre-fit retriever to use
    - include_labels: when True attach `label` column based on target_df

    Returns (candidates_with_features, feature_columns, metadata_summary)
    """

    # build per-user histories from provided history_df
    user_histories = build_user_histories(history_df)
    # build ground truth from provided target_df
    ground_truth = build_ground_truth(target_df)

    # fit or use provided retriever
    if retriever is None:
        if retriever_name == "popularity":
            retriever = PopularityRecommender().fit(history_df)
        else:
            raise ValueError(f"Unsupported retriever_name: {retriever_name}")

    # restrict histories to only users in the target set to be deterministic
    eval_histories = {user_id: user_histories.get(user_id, set()) for user_id in sorted(ground_truth)}

    candidate_df = make_candidate_pool_for_users(
        retriever=retriever,
        user_histories=eval_histories,
        top_n=candidate_top_n,
    )

    if include_labels:
        labeled = build_labeled_ranking_dataframe(candidate_df, ground_truth=ground_truth)
    else:
        labeled = candidate_df.copy()

    # add ranking features computed on `history_df`
    features_df, feature_cols = add_ranking_features(
        candidates=labeled,
        train_df=history_df,
        item_features_df=item_features_df,
    )

    positives = int(features_df["label"].sum()) if include_labels and "label" in features_df.columns else 0
    negatives = int(len(features_df) - positives)
    summary: dict[str, object] = {
        "users": int(features_df["user_id"].nunique()) if not features_df.empty else 0,
        "total_candidate_rows": int(len(features_df)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / len(features_df)) if len(features_df) else 0.0,
        "average_candidates_per_user": float(len(features_df) / features_df["user_id"].nunique()) if len(features_df) else 0.0,
        "feature_count": int(len(feature_cols)),
        "top_n": int(candidate_top_n),
    }

    return features_df, feature_cols, summary
