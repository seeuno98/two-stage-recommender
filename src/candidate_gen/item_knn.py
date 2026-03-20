"""Item-item co-occurrence recommender baseline."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


class ItemKNNRecommender:
    """Recommend items using item-item co-occurrence across user histories."""

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = "event_weight",
    ) -> None:
        self.user_col = user_col
        self.item_col = item_col
        self.weight_col = weight_col
        self.user_histories_: dict[int | str, set[object]] = {}
        self.user_item_weights_: dict[int | str, dict[object, float]] = {}
        self.item_neighbors_: dict[object, Counter[object]] = {}
        self.items_: set[object] = set()

    def fit(self, train_df: pd.DataFrame) -> "ItemKNNRecommender":
        """Fit co-occurrence neighbors from training interactions."""

        self._validate_columns(train_df)
        self.user_histories_ = self._build_user_histories(train_df)
        self.user_item_weights_ = self._build_user_item_weights(train_df)
        self.item_neighbors_ = self._build_item_neighbors(self.user_histories_)
        self.items_ = set(train_df[self.item_col].unique())
        return self

    def _validate_columns(self, train_df: pd.DataFrame) -> None:
        """Validate required training columns."""

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(train_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training dataframe is missing required columns: {missing}")

    def _build_user_histories(
        self,
        train_df: pd.DataFrame,
    ) -> dict[int | str, set[object]]:
        """Build user -> unique interacted items."""

        return train_df.groupby(self.user_col)[self.item_col].apply(lambda items: set(items)).to_dict()

    def _build_user_item_weights(
        self,
        train_df: pd.DataFrame,
    ) -> dict[int | str, dict[object, float]]:
        """Build per-user item strengths for optional scoring weights."""

        if self.weight_col not in train_df.columns:
            grouped = train_df.groupby([self.user_col, self.item_col]).size().astype(float)
        else:
            grouped = train_df.groupby([self.user_col, self.item_col])[self.weight_col].sum()

        user_item_weights: dict[int | str, dict[object, float]] = defaultdict(dict)
        for (user_id, item_id), score in grouped.items():
            user_item_weights[user_id][item_id] = float(score)
        return dict(user_item_weights)

    def _build_item_neighbors(
        self,
        user_histories: dict[int | str, set[object]],
    ) -> dict[object, Counter[object]]:
        """Build sparse item-item co-occurrence counts."""

        item_neighbors: dict[object, Counter[object]] = defaultdict(Counter)
        for items in user_histories.values():
            unique_items = list(items)
            for item_id in unique_items:
                for neighbor_id in unique_items:
                    if item_id == neighbor_id:
                        continue
                    item_neighbors[item_id][neighbor_id] += 1.0
        return dict(item_neighbors)

    def _score_candidates(
        self,
        user_id: int | str,
        user_history: set[object],
    ) -> Counter[object]:
        """Accumulate candidate scores from the user's seen items."""

        candidate_scores: Counter[object] = Counter()
        user_item_weights = self.user_item_weights_.get(user_id, {})

        for item_id in user_history:
            seed_weight = user_item_weights.get(item_id, 1.0)
            for candidate_id, cooccurrence_score in self.item_neighbors_.get(item_id, Counter()).items():
                if candidate_id in user_history:
                    continue
                candidate_scores[candidate_id] += float(cooccurrence_score) * float(seed_weight)

        return candidate_scores

    def recommend_for_user(
        self,
        user_id: int | str,
        user_history: set[object],
        k: int = 10,
    ) -> list[object]:
        """Return top-k unseen co-occurrence candidates for one user."""

        candidate_scores = self._score_candidates(user_id=user_id, user_history=user_history)
        ranked_candidates = [
            item_id
            for item_id, _ in sorted(candidate_scores.items(), key=lambda pair: pair[1], reverse=True)
            if item_id not in user_history
        ]
        return ranked_candidates[:k]

    def recommend_for_users(
        self,
        user_histories: dict[int | str, set[object]],
        k: int = 10,
    ) -> dict[int | str, list[object]]:
        """Generate personalized recommendations for multiple users."""

        return {
            user_id: self.recommend_for_user(user_id=user_id, user_history=history, k=k)
            for user_id, history in user_histories.items()
        }
