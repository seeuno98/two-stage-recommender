"""Tests for the FastAPI recommendation service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.config import ServingConfig
from src.serving.service import RecommendationService


class FakeRanker:
    """Small deterministic ranker used to exercise the reranked path."""

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return frame["popularity_score"] - frame["popularity_rank"] * 0.01


def _write_test_artifacts(project_root: Path) -> None:
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [101, 102, 101, 103, 104],
            "event_type": ["view", "transaction", "view", "addtocart", "view"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "event_weight": [1.0, 5.0, 1.0, 3.0, 1.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "user_id": [1, 4],
            "item_id": [105, 106],
            "event_type": ["view", "view"],
            "timestamp": pd.to_datetime(["2024-01-06", "2024-01-07"]),
            "event_weight": [1.0, 1.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "user_id": [2],
            "item_id": [107],
            "event_type": ["transaction"],
            "timestamp": pd.to_datetime(["2024-01-08"]),
            "event_weight": [5.0],
        }
    )
    item_features_df = pd.DataFrame(
        {
            "item_id": [101, 102, 103, 104, 105, 106, 107],
            "category_id": [1, 1, 2, 2, 3, 3, 4],
        }
    )

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    item_features_df.to_parquet(processed_dir / "item_features.parquet", index=False)


def _build_service(project_root: Path, enable_ranker: bool = False) -> RecommendationService:
    config = ServingConfig(
        enable_candidate_cache=True,
        max_candidate_cache_size=32,
        default_candidate_pool_size=5,
    )
    service = RecommendationService(project_root=project_root, config=config)
    if enable_ranker:
        service.feature_columns = [
            "popularity_rank",
            "popularity_score",
            "user_train_interaction_count",
            "item_popularity_count",
        ]
        service.ranker_model = FakeRanker()  # type: ignore[assignment]
        service.ranker_loaded = True
        service.model_version = "fake-ranker-v1"
    return service


def _build_test_client(project_root: Path, enable_ranker: bool = False) -> TestClient:
    service = _build_service(project_root, enable_ranker=enable_ranker)
    app = create_app(service=service, config=service.config)
    return TestClient(app)


def test_health_returns_expected_fields(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"status", "model_loaded", "ranker_loaded", "timestamp"}
    assert payload["model_loaded"] is True


def test_known_user_ranked_path_returns_latency_breakdown(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path, enable_ranker=True) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 1, "top_k": 3, "pipeline": "popularity_plus_ranker"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "popularity_plus_ranker"
    assert payload["fallback_used"] is False
    assert payload["metadata"]["ranker_used"] is True
    assert payload["metadata"]["feature_count"] == 4
    assert set(payload["metadata"]["latency_breakdown"]) == {
        "candidate_generation_ms",
        "feature_build_ms",
        "scoring_ms",
        "total_latency_ms",
    }


def test_unseen_user_fallback_still_works(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path, enable_ranker=True) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 9999, "top_k": 2, "pipeline": "popularity_plus_ranker"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "popularity_only"
    assert payload["fallback_used"] is True
    assert len(payload["recommendations"]) == 2


def test_fallback_response_includes_fallback_reason(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 1, "top_k": 3, "pipeline": "popularity_plus_ranker"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metadata"]["fallback_reason"] == "ranker_unavailable"


def test_candidate_cache_path_preserves_recommendation_correctness(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)
    service = _build_service(tmp_path, enable_ranker=True)

    first = service.recommend(user_id=1, top_k=3, pipeline="popularity_plus_ranker")
    second = service.recommend(user_id=1, top_k=3, pipeline="popularity_plus_ranker")

    assert [item.item_id for item in first.recommendations] == [
        item.item_id for item in second.recommendations
    ]
    assert len(service._candidate_cache) >= 1


def test_request_id_middleware_does_not_break_responses(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.get("/recommend/1?top_k=2&pipeline=popularity_only")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-MS" in response.headers
