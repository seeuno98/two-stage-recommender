"""Popularity-based candidate generation baseline."""

from collections.abc import Iterable

import pandas as pd


class PopularityRecommender:
    """Recommend items by global interaction frequency."""

    def __init__(
        self,
        item_col: str = "item_id",
        score_col: str = "score",
    ) -> None:
        self.item_col = item_col
        self.score_col = score_col
        self._popular_items: list[tuple[object, float]] = []

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Fit the recommender on implicit feedback interactions."""
        counts = interactions[self.item_col].value_counts()
        self._popular_items = [(item, float(score)) for item, score in counts.items()]
        return self

    def recommend(self, user_id: object, k: int = 10) -> list[dict[str, object]]:
        """Return top-k globally popular items for a given user."""
        del user_id
        top_items = self._popular_items[:k]
        return [
            {self.item_col: item_id, self.score_col: score}
            for item_id, score in top_items
        ]

    def batch_recommend(
        self,
        user_ids: Iterable[object],
        k: int = 10,
    ) -> dict[object, list[dict[str, object]]]:
        """Generate recommendations for multiple users."""
        return {user_id: self.recommend(user_id=user_id, k=k) for user_id in user_ids}
