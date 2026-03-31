"""Recommendation service used by the FastAPI inference layer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.ranking.features import add_ranking_features, get_feature_columns
from src.ranking.predict import rerank_candidates, score_candidates
from src.serving.schemas import RecommendationItem, RecommendationResponse

try:
    from lightgbm import Booster
except ImportError:  # pragma: no cover - exercised only in environments without lightgbm
    Booster = None  # type: ignore[assignment]


SUPPORTED_PIPELINES = {"popularity_only", "popularity_plus_ranker"}


@dataclass
class ServicePaths:
    """Filesystem paths used by the serving service."""

    project_root: Path

    @property
    def model_path(self) -> Path:
        return self.project_root / "artifacts" / "models" / "lightgbm_ranker.txt"

    @property
    def ranking_train_path(self) -> Path:
        return self.project_root / "artifacts" / "features" / "ranking_train.parquet"

    @property
    def ranking_test_path(self) -> Path:
        return self.project_root / "artifacts" / "features" / "ranking_test.parquet"

    @property
    def ranker_summary_path(self) -> Path:
        return self.project_root / "artifacts" / "reports" / "lightgbm_ranker_summary.json"

    @property
    def train_path(self) -> Path:
        return self.project_root / "data" / "processed" / "train.parquet"

    @property
    def val_path(self) -> Path:
        return self.project_root / "data" / "processed" / "val.parquet"

    @property
    def test_path(self) -> Path:
        return self.project_root / "data" / "processed" / "test.parquet"

    @property
    def item_features_path(self) -> Path:
        return self.project_root / "data" / "processed" / "item_features.parquet"


class RecommendationService:
    """Local production-style service for recommender inference."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        default_pipeline: str = "popularity_plus_ranker",
        fallback_pipeline: str = "popularity_only",
        candidate_pool_size: int = 150,
    ) -> None:
        if default_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported default pipeline: {default_pipeline}")
        if fallback_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported fallback pipeline: {fallback_pipeline}")

        self.paths = ServicePaths(Path(project_root or Path.cwd()))
        self.default_pipeline = default_pipeline
        self.fallback_pipeline = fallback_pipeline
        self.candidate_pool_size = candidate_pool_size

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.item_features_df: pd.DataFrame | None = None
        self.serving_history_df = pd.DataFrame()

        self.popularity_recommender: PopularityRecommender | None = None
        self.user_histories: dict[int | str, set[object]] = {}
        self.feature_columns: list[str] = []
        self.ranker_model: Booster | None = None

        self.model_loaded = False
        self.ranker_loaded = False
        self.user_id_dtype: str = "object"
        self.loaded_artifacts: dict[str, bool] = {}

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load processed data and optional model artifacts."""

        self.train_df = self._read_parquet(self.paths.train_path)
        self.val_df = self._read_parquet(self.paths.val_path)
        self.test_df = self._read_parquet(self.paths.test_path)
        self.item_features_df = self._read_optional_parquet(self.paths.item_features_path)

        self.loaded_artifacts = {
            "train": not self.train_df.empty,
            "val": not self.val_df.empty,
            "test": not self.test_df.empty,
            "item_features": self.item_features_df is not None and not self.item_features_df.empty,
            "ranking_train": self.paths.ranking_train_path.exists(),
            "ranking_test": self.paths.ranking_test_path.exists(),
            "ranker_model": self.paths.model_path.exists(),
        }

        history_frames = [df for df in [self.train_df, self.val_df] if not df.empty]
        self.serving_history_df = (
            pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
        )

        if not self.serving_history_df.empty:
            self.popularity_recommender = PopularityRecommender().fit(self.serving_history_df)
            self.user_histories = (
                self.serving_history_df.groupby("user_id")["item_id"].apply(set).to_dict()
            )
            self.user_id_dtype = str(self.serving_history_df["user_id"].dtype)
            self.model_loaded = True

        self.feature_columns = self._load_feature_columns()
        self.ranker_model = self._load_ranker_model()
        self.ranker_loaded = self.ranker_model is not None and bool(self.feature_columns)

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read a parquet file if present, otherwise return an empty dataframe."""

        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _read_optional_parquet(self, path: Path) -> pd.DataFrame | None:
        """Read an optional parquet file."""

        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _load_feature_columns(self) -> list[str]:
        """Infer the ranker feature schema from existing artifacts."""

        summary_columns = self._load_feature_columns_from_summary()
        if summary_columns:
            return summary_columns

        for path in [self.paths.ranking_train_path, self.paths.ranking_test_path]:
            if not path.exists():
                continue
            ranking_df = pd.read_parquet(path)
            columns = get_feature_columns(ranking_df)
            if columns:
                return columns
        return []

    def _load_feature_columns_from_summary(self) -> list[str]:
        """Load persisted ranker feature columns from the summary artifact."""

        if not self.paths.ranker_summary_path.exists():
            return []

        with self.paths.ranker_summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        feature_columns = payload.get("feature_columns")
        if isinstance(feature_columns, list):
            return [str(column) for column in feature_columns]
        return []

    def _load_ranker_model(self) -> Booster | None:
        """Load the LightGBM ranker if it is available and usable."""

        if Booster is None or not self.paths.model_path.exists() or not self.feature_columns:
            return None

        model = Booster(model_file=str(self.paths.model_path))
        if model.num_feature() != len(self.feature_columns):
            return None
        return model

    def get_health(self) -> dict[str, Any]:
        """Return service readiness and artifact state for health checks."""

        status = "ok" if self.model_loaded else "degraded"
        if self.model_loaded and not self.ranker_loaded:
            status = "degraded"

        return {
            "status": status,
            "model_loaded": self.model_loaded,
            "ranker_loaded": self.ranker_loaded,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "artifacts": self.loaded_artifacts,
            "default_pipeline": self.default_pipeline,
            "fallback_pipeline": self.fallback_pipeline,
            "candidate_pool_size": self.candidate_pool_size,
        }

    def recommend(
        self,
        user_id: int | str,
        top_k: int = 10,
        pipeline: str | None = None,
    ) -> RecommendationResponse:
        """Serve top-k recommendations with graceful fallback behavior."""

        if not self.model_loaded or self.popularity_recommender is None:
            raise RuntimeError(
                "Serving artifacts are not loaded. Run data preparation before starting the API."
            )

        requested_pipeline = pipeline or self.default_pipeline
        if requested_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported pipeline: {requested_pipeline}")

        normalized_user_id = self._normalize_user_id(user_id)
        start_time = time.perf_counter()

        if requested_pipeline == "popularity_only":
            items, metadata = self._recommend_popularity(normalized_user_id, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return RecommendationResponse(
                user_id=normalized_user_id,
                pipeline="popularity_only",
                top_k=top_k,
                recommendations=items,
                latency_ms=latency_ms,
                fallback_used=False,
                metadata=metadata,
            )

        fallback_reason: str | None = None
        if not self.ranker_loaded:
            fallback_reason = "ranker_unavailable"
        elif normalized_user_id not in self.user_histories:
            fallback_reason = "unseen_user"

        if fallback_reason is not None:
            items, metadata = self._recommend_popularity(normalized_user_id, top_k=top_k)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metadata["requested_pipeline"] = requested_pipeline
            metadata["fallback_reason"] = fallback_reason
            return RecommendationResponse(
                user_id=normalized_user_id,
                pipeline=self.fallback_pipeline,
                top_k=top_k,
                recommendations=items,
                latency_ms=latency_ms,
                fallback_used=True,
                metadata=metadata,
            )

        items, metadata = self._recommend_ranked(normalized_user_id, top_k=top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        metadata["requested_pipeline"] = requested_pipeline
        return RecommendationResponse(
            user_id=normalized_user_id,
            pipeline="popularity_plus_ranker",
            top_k=top_k,
            recommendations=items,
            latency_ms=latency_ms,
            fallback_used=False,
            metadata=metadata,
        )

    def _normalize_user_id(self, user_id: int | str) -> int | str:
        """Coerce incoming user identifiers to match the training artifact dtype."""

        if self.user_id_dtype.startswith(("int", "uint")):
            if isinstance(user_id, int):
                return user_id
            if isinstance(user_id, str) and user_id.isdigit():
                return int(user_id)
        return user_id

    def _recommend_popularity(
        self,
        user_id: int | str,
        top_k: int,
    ) -> tuple[list[RecommendationItem], dict[str, Any]]:
        """Return popularity-based recommendations, filtered for known users."""

        assert self.popularity_recommender is not None

        user_history = self.user_histories.get(user_id, set())
        item_ids = self.popularity_recommender.recommend_for_user(
            user_id=user_id,
            user_history=user_history,
            k=top_k,
        )

        recommendations = [
            RecommendationItem(
                item_id=item_id,
                score=float(self.popularity_recommender.popularity_scores_.get(item_id, 0.0)),
                rank=rank,
            )
            for rank, item_id in enumerate(item_ids, start=1)
        ]
        metadata = {
            "candidate_pool_size": len(item_ids),
            "ranker_used": False,
            "known_user": user_id in self.user_histories,
        }
        return recommendations, metadata

    def _recommend_ranked(
        self,
        user_id: int | str,
        top_k: int,
    ) -> tuple[list[RecommendationItem], dict[str, Any]]:
        """Generate popularity candidates, score them with the ranker, and rerank."""

        candidate_df = self._build_candidate_frame_for_user(
            user_id=user_id,
            candidate_k=max(self.candidate_pool_size, top_k),
        )
        if candidate_df.empty:
            return self._recommend_popularity(user_id=user_id, top_k=top_k)

        features_df = self._build_feature_frame(candidate_df)
        ranked_df = self._score_and_rerank_candidates(features_df).head(top_k).reset_index(drop=True)

        recommendations = [
            RecommendationItem(
                item_id=row["item_id"],
                score=float(row["ranking_score"]),
                rank=int(row["rank"]),
            )
            for _, row in ranked_df.iterrows()
        ]
        metadata = {
            "candidate_pool_size": int(len(candidate_df)),
            "ranker_used": True,
            "known_user": True,
            "feature_count": len(self.feature_columns),
        }
        return recommendations, metadata

    def _build_candidate_frame_for_user(
        self,
        user_id: int | str,
        candidate_k: int,
    ) -> pd.DataFrame:
        """Build a one-user popularity candidate pool."""

        assert self.popularity_recommender is not None

        user_history = self.user_histories.get(user_id, set())
        item_ids = self.popularity_recommender.recommend_for_user(
            user_id=user_id,
            user_history=user_history,
            k=candidate_k,
        )
        if not item_ids:
            return pd.DataFrame(
                columns=["user_id", "item_id", "popularity_rank", "popularity_score"]
            )

        return pd.DataFrame(
            {
                "user_id": [user_id] * len(item_ids),
                "item_id": item_ids,
                "popularity_rank": list(range(1, len(item_ids) + 1)),
                "popularity_score": [
                    float(self.popularity_recommender.popularity_scores_.get(item_id, 0.0))
                    for item_id in item_ids
                ],
            }
        )

    def _build_feature_frame(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """Attach the same style of numeric features used by the offline ranker."""

        features_df, _ = add_ranking_features(
            candidates=candidate_df,
            train_df=self.serving_history_df,
            item_features_df=self.item_features_df,
        )

        for column in self.feature_columns:
            if column not in features_df.columns:
                features_df[column] = 0.0

        return features_df

    def _score_and_rerank_candidates(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """Score candidates with LightGBM and return ranked rows."""

        if self.ranker_model is None:
            raise RuntimeError("Ranker model is not loaded.")

        scores = score_candidates(
            model=self.ranker_model,
            X=candidate_df,
            feature_cols=self.feature_columns,
        )
        ranked_df = rerank_candidates(candidate_df=candidate_df, scores=scores)
        return ranked_df
