"""ALS-based candidate generation for implicit feedback recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:  # pragma: no cover - exercised only in environments without implicit
    AlternatingLeastSquares = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EncodedIds:
    """Container for user/item ID mappings."""

    user_id_to_idx: dict[int | str, int]
    item_id_to_idx: dict[object, int]
    idx_to_user_id: dict[int, int | str]
    idx_to_item_id: dict[int, object]


class ALSCandidateGenerator:
    """Wrapper around implicit ALS for candidate generation."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 20.0,
        random_state: int = 42,
        user_col: str = "user_id",
        item_col: str = "item_id",
        weight_col: str = "event_weight",
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.user_col = user_col
        self.item_col = item_col
        self.weight_col = weight_col

        self.user_id_to_idx: dict[int | str, int] = {}
        self.item_id_to_idx: dict[object, int] = {}
        self.idx_to_user_id: dict[int, int | str] = {}
        self.idx_to_item_id: dict[int, object] = {}
        self.aggregated_interactions_: pd.DataFrame | None = None
        self.user_item_matrix: sparse.csr_matrix | None = None
        self.item_user_matrix: sparse.csr_matrix | None = None
        self.model: AlternatingLeastSquares | None = None
        self.invalid_recommendation_indices_: int = 0

    def fit(self, train_df: pd.DataFrame) -> "ALSCandidateGenerator":
        """Fit the ALS model from training interactions."""

        self._require_implicit()
        self._validate_columns(train_df)

        aggregated = self._aggregate_interactions(train_df)
        self.aggregated_interactions_ = aggregated.copy()
        encoded_ids = self._build_id_maps(aggregated)
        self.user_id_to_idx = encoded_ids.user_id_to_idx
        self.item_id_to_idx = encoded_ids.item_id_to_idx
        self.idx_to_user_id = encoded_ids.idx_to_user_id
        self.idx_to_item_id = encoded_ids.idx_to_item_id

        self.user_item_matrix = self._build_interaction_matrix(aggregated)
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        self.invalid_recommendation_indices_ = 0

        confidence_matrix = self.user_item_matrix.multiply(self.alpha).tocsr()
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.fit(confidence_matrix)
        return self

    def _require_implicit(self) -> None:
        """Raise a helpful error if implicit is unavailable."""

        if AlternatingLeastSquares is None:
            raise RuntimeError(
                "The `implicit` package is not installed. Install project dependencies and, "
                "if needed, use Python 3.11 for best compatibility."
            )

    def _validate_columns(self, train_df: pd.DataFrame) -> None:
        """Validate required training columns."""

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(train_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Training dataframe is missing required columns: {missing}")

    def _aggregate_interactions(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate duplicate user-item interactions into a single strength."""

        group_columns = [self.user_col, self.item_col]
        if self.weight_col in train_df.columns:
            aggregated = (
                train_df.groupby(group_columns, as_index=False)[self.weight_col]
                .sum()
                .rename(columns={self.weight_col: "interaction_strength"})
            )
        else:
            aggregated = (
                train_df.groupby(group_columns, as_index=False)
                .size()
                .rename(columns={"size": "interaction_strength"})
            )
        aggregated["interaction_strength"] = aggregated["interaction_strength"].astype("float32")
        return aggregated

    def _build_id_maps(self, aggregated_df: pd.DataFrame) -> EncodedIds:
        """Build user/item ID encodings."""

        user_ids = list(pd.Index(aggregated_df[self.user_col]).unique())
        item_ids = list(pd.Index(aggregated_df[self.item_col]).unique())

        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
        idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
        return EncodedIds(
            user_id_to_idx=user_id_to_idx,
            item_id_to_idx=item_id_to_idx,
            idx_to_user_id=idx_to_user_id,
            idx_to_item_id=idx_to_item_id,
        )

    def _build_interaction_matrix(self, aggregated_df: pd.DataFrame) -> sparse.csr_matrix:
        """Build a sparse user-item interaction matrix."""

        row_indices = aggregated_df[self.user_col].map(self.user_id_to_idx).to_numpy()
        col_indices = aggregated_df[self.item_col].map(self.item_id_to_idx).to_numpy()
        values = aggregated_df["interaction_strength"].astype("float32").to_numpy()

        matrix = sparse.coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_id_to_idx), len(self.item_id_to_idx)),
            dtype="float32",
        )
        return matrix.tocsr()

    def _decode_item_indices(
        self,
        item_indices: list[int],
        user_history: set[object],
        k: int,
    ) -> list[object]:
        """Decode internal item indices into original item IDs safely."""

        recommendations: list[object] = []
        for raw_item_idx in item_indices:
            item_idx = int(raw_item_idx)
            item_id = self.idx_to_item_id.get(item_idx)
            if item_id is None:
                self.invalid_recommendation_indices_ += 1
                continue
            if item_id in user_history:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def _recommend_from_internal_user_idx(
        self,
        user_idx: int,
        user_history: set[object],
        k: int,
    ) -> list[object]:
        """Recommend items for an encoded user index."""

        if self.model is None or self.user_item_matrix is None:
            raise RuntimeError("The ALS model has not been fit yet.")

        internal_n = max(k * 3, 100)
        item_indices, _ = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=internal_n,
            filter_already_liked_items=True,
        )
        if hasattr(item_indices, "tolist"):
            internal_item_indices = item_indices.tolist()
        else:
            internal_item_indices = list(item_indices)

        return self._decode_item_indices(
            item_indices=internal_item_indices,
            user_history=user_history,
            k=k,
        )

    def recommend_for_user(
        self,
        user_id: int | str,
        user_history: set[object],
        k: int = 10,
    ) -> list[object]:
        """Return top-k unseen ALS recommendations for one user."""

        if user_id not in self.user_id_to_idx:
            return []
        user_idx = self.user_id_to_idx[user_id]
        return self._recommend_from_internal_user_idx(
            user_idx=user_idx,
            user_history=user_history,
            k=k,
        )

    def recommend_for_users(
        self,
        user_histories: dict[int | str, set[object]],
        k: int = 10,
    ) -> dict[int | str, list[object]]:
        """Generate top-k unseen ALS recommendations for multiple users."""

        return {
            user_id: self.recommend_for_user(user_id=user_id, user_history=history, k=k)
            for user_id, history in user_histories.items()
        }

    def get_model_info(self) -> dict[str, int | float]:
        """Return basic model metadata."""

        return {
            "n_users": len(self.user_id_to_idx),
            "n_items": len(self.item_id_to_idx),
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
