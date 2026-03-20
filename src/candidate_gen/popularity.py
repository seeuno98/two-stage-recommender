"""Popularity-based candidate generation baseline."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class PopularityRecommender:
    """Recommend items by global weighted interaction popularity."""

    def __init__(
        self,
        item_col: str = "item_id",
        score_col: str = "popularity_score",
        weight_col: str = "event_weight",
    ) -> None:
        self.item_col = item_col
        self.score_col = score_col
        self.weight_col = weight_col
        self.ranked_items_: list[object] = []
        self.popularity_scores_: dict[object, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Fit popularity scores from training interactions."""

        if self.item_col not in train_df.columns:
            raise ValueError(f"Missing item column: {self.item_col}")

        if self.weight_col in train_df.columns:
            popularity = train_df.groupby(self.item_col)[self.weight_col].sum()
        else:
            popularity = train_df.groupby(self.item_col).size().astype(float)

        popularity = popularity.sort_values(ascending=False)
        self.popularity_scores_ = {item: float(score) for item, score in popularity.items()}
        self.ranked_items_ = list(popularity.index)
        return self

    def recommend_for_user(
        self,
        user_id: int | str,
        user_history: set[object],
        k: int = 10,
    ) -> list[object]:
        """Return the top-k unseen items for one user."""

        del user_id
        recommendations: list[object] = []
        seen_items = set(user_history)

        for item_id in self.ranked_items_:
            if item_id in seen_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_for_users(
        self,
        user_histories: dict[int | str, set[object]],
        k: int = 10,
    ) -> dict[int | str, list[object]]:
        """Generate top-k unseen recommendations for multiple users."""

        return {
            user_id: self.recommend_for_user(user_id=user_id, user_history=history, k=k)
            for user_id, history in user_histories.items()
        }

    def recommend(self, user_id: object, k: int = 10) -> list[dict[str, object]]:
        """Return top-k popular items without any user history filtering."""

        top_items = self.ranked_items_[:k]
        return [
            {self.item_col: item_id, self.score_col: self.popularity_scores_[item_id]}
            for item_id in top_items
        ]

    def batch_recommend(
        self,
        user_ids: Iterable[object],
        k: int = 10,
    ) -> dict[object, list[dict[str, object]]]:
        """Generate top-k popular items for multiple users without filtering."""

        return {user_id: self.recommend(user_id=user_id, k=k) for user_id in user_ids}
