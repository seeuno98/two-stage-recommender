"""Tests for the FastAPI recommendation service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.service import RecommendationService


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


def _build_test_client(project_root: Path) -> TestClient:
    service = RecommendationService(project_root=project_root, default_pipeline="popularity_plus_ranker")
    app = create_app(service=service)
    return TestClient(app)


def test_health_returns_expected_fields(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"status", "model_loaded", "ranker_loaded", "timestamp"}
    assert payload["model_loaded"] is True
    assert payload["ranker_loaded"] is False


def test_recommend_returns_200_for_known_user(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 1, "top_k": 3, "pipeline": "popularity_only"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 1
    assert payload["pipeline"] == "popularity_only"
    assert len(payload["recommendations"]) == 3
    assert payload["recommendations"][0]["rank"] == 1


def test_unseen_user_returns_non_empty_fallback_recommendations(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 9999, "top_k": 2, "pipeline": "popularity_only"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) == 2
    assert payload["fallback_used"] is False
    assert payload["metadata"]["known_user"] is False


def test_invalid_request_body_returns_validation_error(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.post("/recommend", json={"top_k": 0})

    assert response.status_code == 422


def test_pipeline_fallback_when_ranker_is_unavailable(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.post(
            "/recommend",
            json={"user_id": 1, "top_k": 3, "pipeline": "popularity_plus_ranker"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "popularity_only"
    assert payload["fallback_used"] is True
    assert payload["metadata"]["fallback_reason"] == "ranker_unavailable"


def test_response_includes_latency_and_fallback_fields(tmp_path: Path) -> None:
    _write_test_artifacts(tmp_path)

    with _build_test_client(tmp_path) as client:
        response = client.get("/recommend/1?top_k=2&pipeline=popularity_plus_ranker")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["latency_ms"], float)
    assert payload["latency_ms"] >= 0.0
    assert "fallback_used" in payload
