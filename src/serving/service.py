"""Recommendation service used by the FastAPI inference layer."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidate_gen.popularity import PopularityRecommender
from src.ranking.features import (
    build_item_feature_table,
    build_user_feature_table,
    build_user_item_feature_table,
    encode_item_metadata,
    get_feature_columns,
)
from src.ranking.predict import rerank_candidates, score_candidates
from src.serving.config import ServingConfig
from src.serving.schemas import RecommendationItem, RecommendationResponse

try:
    from lightgbm import Booster
except ImportError:  # pragma: no cover - exercised only in environments without lightgbm
    Booster = None  # type: ignore[assignment]


SUPPORTED_PIPELINES = {"popularity_only", "popularity_plus_ranker"}


@dataclass(slots=True)
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
        config: ServingConfig | None = None,
    ) -> None:
        self.config = config or ServingConfig.from_env()
        if self.config.default_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported default pipeline: {self.config.default_pipeline}")
        if self.config.fallback_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported fallback pipeline: {self.config.fallback_pipeline}")

        self.paths = ServicePaths(Path(project_root or Path.cwd()))

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.item_features_df: pd.DataFrame | None = None
        self.serving_history_df = pd.DataFrame()

        self.popularity_recommender: PopularityRecommender | None = None
        self.popularity_ranked_items: list[object] = []
        self.popularity_scores: dict[object, float] = {}
        self.user_histories: dict[int | str, set[object]] = {}
        self.item_metadata_lookup: dict[object, dict[str, float]] = {}
        self.user_feature_lookup: dict[int | str, dict[str, float]] = {}
        self.item_feature_lookup: dict[object, dict[str, float]] = {}
        self.user_item_feature_lookup: dict[tuple[object, object], dict[str, float]] = {}
        self.feature_columns: list[str] = []
        self.ranker_model: Booster | None = None
        self.model_version: str | None = None

        self.model_loaded = False
        self.ranker_loaded = False
        self.user_id_dtype: str = "object"
        self.loaded_artifacts: dict[str, bool] = {}

        self._candidate_cache: OrderedDict[tuple[object, str, int], list[dict[str, object]]] = (
            OrderedDict()
        )

        self._load_static_artifacts()

    @property
    def default_pipeline(self) -> str:
        """Expose the default pipeline from config."""

        return self.config.default_pipeline

    @property
    def fallback_pipeline(self) -> str:
        """Expose the fallback pipeline from config."""

        return self.config.fallback_pipeline

    @property
    def candidate_pool_size(self) -> int:
        """Expose the default candidate pool size from config."""

        return self.config.default_candidate_pool_size

    def _load_static_artifacts(self) -> None:
        """Load all serving artifacts once at startup."""

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

        if self.serving_history_df.empty:
            return

        self.popularity_recommender = PopularityRecommender().fit(self.serving_history_df)
        self.popularity_ranked_items = list(self.popularity_recommender.ranked_items_)
        self.popularity_scores = dict(self.popularity_recommender.popularity_scores_)
        self.user_histories = (
            self.serving_history_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_id_dtype = str(self.serving_history_df["user_id"].dtype)

        self._precompute_feature_maps()
        self.feature_columns = self._load_feature_columns()
        self.ranker_model = self._load_ranker_model()
        self.ranker_loaded = self.ranker_model is not None and bool(self.feature_columns)
        self.model_loaded = True

    def _precompute_feature_maps(self) -> None:
        """Precompute user, item, and user-item feature lookups for request-time scoring."""

        user_features = build_user_feature_table(self.serving_history_df)
        item_features = build_item_feature_table(self.serving_history_df)
        user_item_features = build_user_item_feature_table(self.serving_history_df)
        metadata_features = encode_item_metadata(self.item_features_df)

        self.user_feature_lookup = self._frame_to_lookup(user_features, "user_id")
        self.item_feature_lookup = self._frame_to_lookup(item_features, "item_id")
        self.user_item_feature_lookup = self._frame_to_lookup(
            user_item_features,
            ["user_id", "item_id"],
        )
        self.item_metadata_lookup = self._frame_to_lookup(metadata_features, "item_id")

    def _frame_to_lookup(
        self,
        frame: pd.DataFrame,
        key_columns: str | list[str],
    ) -> dict[Any, dict[str, float]]:
        """Convert a dataframe to a nested lookup keyed by one or more columns."""

        if frame.empty:
            return {}

        columns = [key_columns] if isinstance(key_columns, str) else key_columns
        value_columns = [column for column in frame.columns if column not in columns]
        lookup: dict[Any, dict[str, float]] = {}
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            if len(columns) == 1:
                key: Any = row_dict[columns[0]]
            else:
                key = tuple(row_dict[column] for column in columns)
            lookup[key] = {
                column: float(row_dict[column])
                for column in value_columns
                if pd.notna(row_dict[column])
            }
        return lookup

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
        """Infer the ranker feature schema from persisted training artifacts."""

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
            self.model_version = str(payload.get("model_path", self.paths.model_path.name))
            return [str(column) for column in feature_columns]
        return []

    def _load_ranker_model(self) -> Booster | None:
        """Load the LightGBM ranker if it is available and usable."""

        if Booster is None or not self.paths.model_path.exists() or not self.feature_columns:
            return None

        model = Booster(model_file=str(self.paths.model_path))
        if model.num_feature() != len(self.feature_columns):
            return None
        if self.model_version is None:
            self.model_version = self.paths.model_path.name
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
        }

    def recommend(
        self,
        user_id: int | str,
        top_k: int | None = None,
        pipeline: str | None = None,
    ) -> RecommendationResponse:
        """Serve top-k recommendations with graceful fallback behavior."""

        if not self.model_loaded:
            raise RuntimeError(
                "Serving artifacts are not loaded. Run data preparation before starting the API."
            )

        requested_pipeline = pipeline or self.default_pipeline
        if requested_pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported pipeline: {requested_pipeline}")

        effective_top_k = top_k or self.config.default_top_k
        normalized_user_id = self._normalize_user_id(user_id)
        known_user = normalized_user_id in self.user_histories

        start_time = time.perf_counter()
        candidate_generation_ms = 0.0
        feature_build_ms = 0.0
        scoring_ms = 0.0
        fallback_reason: str | None = None
        effective_pipeline = requested_pipeline
        ranker_used = False

        if requested_pipeline == "popularity_plus_ranker":
            if not self.ranker_loaded:
                fallback_reason = "ranker_unavailable"
                effective_pipeline = self.fallback_pipeline
            elif not known_user:
                fallback_reason = "unseen_user"
                effective_pipeline = self.fallback_pipeline

        if effective_pipeline == "popularity_only":
            candidate_start = time.perf_counter()
            candidate_rows = self._get_candidate_items(
                user_id=normalized_user_id,
                pipeline="popularity_only",
                candidate_pool_size=effective_top_k,
            )
            candidate_generation_ms = self._elapsed_ms(candidate_start)
            recommendations = self._rows_to_recommendations(candidate_rows[:effective_top_k], "score")
        else:
            ranker_used = True
            candidate_pool_size = self._resolve_candidate_pool_size(effective_top_k)

            candidate_start = time.perf_counter()
            candidate_rows = self._get_candidate_items(
                user_id=normalized_user_id,
                pipeline="popularity_plus_ranker",
                candidate_pool_size=candidate_pool_size,
            )
            candidate_generation_ms = self._elapsed_ms(candidate_start)

            feature_start = time.perf_counter()
            feature_frame = self._build_candidate_feature_rows(
                user_id=normalized_user_id,
                candidate_rows=candidate_rows,
            )
            feature_build_ms = self._elapsed_ms(feature_start)

            scoring_start = time.perf_counter()
            ranked_frame = self._score_candidates(feature_frame).head(effective_top_k)
            scoring_ms = self._elapsed_ms(scoring_start)

            recommendations = self._rows_to_recommendations(
                ranked_frame.to_dict(orient="records"),
                "ranking_score",
            )

        total_latency_ms = self._elapsed_ms(start_time)
        metadata = {
            "candidate_pool_size": (
                len(candidate_rows)
                if effective_pipeline == "popularity_only"
                else self._resolve_candidate_pool_size(effective_top_k)
            ),
            "ranker_used": ranker_used,
            "known_user": known_user,
            "requested_pipeline": requested_pipeline,
            "fallback_reason": fallback_reason,
            "feature_count": len(self.feature_columns) if ranker_used else 0,
            "latency_breakdown": {
                "candidate_generation_ms": round(candidate_generation_ms, 3),
                "feature_build_ms": round(feature_build_ms, 3),
                "scoring_ms": round(scoring_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
            },
            "model_version": self.model_version,
            "artifact_source": {
                "model_path": str(self.paths.model_path),
                "ranking_train_path": str(self.paths.ranking_train_path),
                "item_features_path": str(self.paths.item_features_path),
            },
            "fast_mode": self.config.fast_mode,
            "candidate_cache_enabled": self.config.enable_candidate_cache,
        }

        return RecommendationResponse(
            user_id=normalized_user_id,
            pipeline=effective_pipeline,
            top_k=effective_top_k,
            recommendations=recommendations,
            latency_ms=round(total_latency_ms, 3),
            fallback_used=fallback_reason is not None,
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

    def _resolve_candidate_pool_size(self, top_k: int) -> int:
        """Resolve the candidate pool size for reranking."""

        base_size = self.candidate_pool_size
        if self.config.fast_mode:
            base_size = min(base_size, max(top_k, 50))
        return max(base_size, top_k)

    def _elapsed_ms(self, start_time: float) -> float:
        """Return elapsed wall time in milliseconds."""

        return (time.perf_counter() - start_time) * 1000.0

    def _cache_key(self, user_id: int | str, pipeline: str, candidate_pool_size: int) -> tuple[object, str, int]:
        """Build a stable in-memory cache key."""

        return (user_id, pipeline, candidate_pool_size)

    def _get_cached_candidates(
        self,
        user_id: int | str,
        pipeline: str,
        candidate_pool_size: int,
    ) -> list[dict[str, object]] | None:
        """Return cached candidate rows when available."""

        if not self.config.enable_candidate_cache:
            return None

        key = self._cache_key(user_id, pipeline, candidate_pool_size)
        cached = self._candidate_cache.get(key)
        if cached is None:
            return None

        self._candidate_cache.move_to_end(key)
        return [row.copy() for row in cached]

    def _set_cached_candidates(
        self,
        user_id: int | str,
        pipeline: str,
        candidate_pool_size: int,
        candidate_rows: list[dict[str, object]],
    ) -> None:
        """Store candidate rows in a bounded in-memory cache."""

        if not self.config.enable_candidate_cache:
            return

        key = self._cache_key(user_id, pipeline, candidate_pool_size)
        self._candidate_cache[key] = [row.copy() for row in candidate_rows]
        self._candidate_cache.move_to_end(key)

        while len(self._candidate_cache) > self.config.max_candidate_cache_size:
            self._candidate_cache.popitem(last=False)

    def _get_candidate_items(
        self,
        user_id: int | str,
        pipeline: str,
        candidate_pool_size: int,
    ) -> list[dict[str, object]]:
        """Build or fetch candidate items for a user."""

        cached = self._get_cached_candidates(user_id, pipeline, candidate_pool_size)
        if cached is not None:
            return cached

        user_history = self.user_histories.get(user_id, set())
        candidate_rows: list[dict[str, object]] = []
        rank = 1
        for item_id in self.popularity_ranked_items:
            if item_id in user_history:
                continue
            candidate_rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "popularity_rank": rank,
                    "popularity_score": float(self.popularity_scores.get(item_id, 0.0)),
                    "score": float(self.popularity_scores.get(item_id, 0.0)),
                }
            )
            rank += 1
            if len(candidate_rows) >= candidate_pool_size:
                break

        self._set_cached_candidates(user_id, pipeline, candidate_pool_size, candidate_rows)
        return candidate_rows

    def _build_candidate_feature_rows(
        self,
        user_id: int | str,
        candidate_rows: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Build model-ready candidate feature rows using in-memory lookup maps."""

        rows: list[dict[str, object]] = []
        user_features = self.user_feature_lookup.get(user_id, {})

        for row in candidate_rows:
            item_id = row["item_id"]
            feature_row: dict[str, object] = {
                "user_id": user_id,
                "item_id": item_id,
                "popularity_rank": float(row["popularity_rank"]),
                "popularity_score": float(row["popularity_score"]),
            }
            feature_row.update(user_features)
            feature_row.update(self.item_feature_lookup.get(item_id, {}))
            feature_row.update(self.user_item_feature_lookup.get((user_id, item_id), {}))
            feature_row.update(self.item_metadata_lookup.get(item_id, {}))
            rows.append(feature_row)

        feature_frame = pd.DataFrame(rows)
        if feature_frame.empty:
            return feature_frame

        for column in self.feature_columns:
            if column not in feature_frame.columns:
                feature_frame[column] = 0.0

        feature_frame[self.feature_columns] = (
            feature_frame[self.feature_columns].fillna(0.0).astype(float)
        )
        return feature_frame

    def _score_candidates(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Score candidates with the ranker and return ranked rows."""

        if self.ranker_model is None:
            raise RuntimeError("Ranker model is not loaded.")

        scores = score_candidates(
            model=self.ranker_model,
            X=feature_frame,
            feature_cols=self.feature_columns,
        )
        ranked_frame = rerank_candidates(candidate_df=feature_frame, scores=scores)
        return ranked_frame

    def _rows_to_recommendations(
        self,
        rows: list[dict[str, object]],
        score_column: str,
    ) -> list[RecommendationItem]:
        """Convert candidate rows into API response items."""

        recommendations: list[RecommendationItem] = []
        for rank, row in enumerate(rows, start=1):
            score = row.get(score_column)
            recommendations.append(
                RecommendationItem(
                    item_id=row["item_id"],
                    score=float(score) if score is not None else None,
                    rank=rank,
                )
            )
        return recommendations
