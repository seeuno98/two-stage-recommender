"""Structured logging helpers for the serving API."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.serving.schemas import RecommendationResponse


LOGGER_NAME = "two_stage_recommender.serving"


def get_serving_logger() -> logging.Logger:
    """Return a process-wide logger configured for local API serving."""

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def log_structured_event(event_type: str, **payload: Any) -> None:
    """Emit one JSON-style structured log line."""

    logger = get_serving_logger()
    logger.info(
        json.dumps(
            {"event": event_type, **payload},
            default=str,
            sort_keys=True,
        )
    )


def build_request_log_payload(
    response: RecommendationResponse,
    *,
    request_id: str | None,
    endpoint: str,
    status_code: int,
) -> dict[str, Any]:
    """Build a standard structured payload for recommendation logs."""

    metadata = response.metadata or {}
    latency_breakdown = metadata.get("latency_breakdown", {})

    return {
        "request_id": request_id,
        "endpoint": endpoint,
        "user_id": response.user_id,
        "pipeline": response.pipeline,
        "requested_pipeline": metadata.get("requested_pipeline"),
        "top_k": response.top_k,
        "candidate_pool_size": metadata.get("candidate_pool_size"),
        "known_user": metadata.get("known_user"),
        "fallback_used": response.fallback_used,
        "fallback_reason": metadata.get("fallback_reason"),
        "ranker_used": metadata.get("ranker_used"),
        "total_latency_ms": round(response.latency_ms, 3),
        "candidate_generation_ms": latency_breakdown.get("candidate_generation_ms"),
        "feature_build_ms": latency_breakdown.get("feature_build_ms"),
        "scoring_ms": latency_breakdown.get("scoring_ms"),
        "recommendation_count": len(response.recommendations),
        "status_code": status_code,
    }
