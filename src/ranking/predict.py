"""Prediction utilities for the ranking model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_candidates(
    model: object,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Generate ranking scores for candidate rows."""

    return np.asarray(model.predict(X[feature_cols]))


def rerank_candidates(
    candidate_df: pd.DataFrame,
    scores: np.ndarray,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Sort candidate rows by descending model score within each user."""

    ranked = candidate_df.copy()
    ranked["ranking_score"] = scores

    sort_columns = [user_col, "ranking_score"]
    ascending = [True, False]
    if "popularity_rank" in ranked.columns:
        sort_columns.append("popularity_rank")
        ascending.append(True)
    sort_columns.append(item_col)
    ascending.append(True)

    ranked = ranked.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
    ranked["rank"] = ranked.groupby(user_col).cumcount() + 1
    return ranked


def topk_predictions_from_ranked_df(
    ranked_df: pd.DataFrame,
    k: int = 10,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[int | str, list[object]]:
    """Extract top-k item predictions per user from a ranked dataframe."""

    topk_df = ranked_df.groupby(user_col, sort=False).head(k)
    return topk_df.groupby(user_col)[item_col].apply(list).to_dict()
