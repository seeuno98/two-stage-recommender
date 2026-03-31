"""Pydantic schemas for the FastAPI serving layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    """Request payload for a recommendation query."""

    user_id: int | str = Field(..., description="User identifier to score.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return.",
    )
    pipeline: str | None = Field(
        default=None,
        description="Optional serving pipeline override.",
    )


class RecommendationItem(BaseModel):
    """One recommended item in ranked order."""

    item_id: int | str = Field(..., description="Recommended item identifier.")
    score: float | None = Field(
        default=None,
        description="Optional retrieval or ranking score used for ordering.",
    )
    rank: int = Field(..., ge=1, description="One-based rank in the response list.")


class RecommendationResponse(BaseModel):
    """Structured response for a recommendation request."""

    user_id: int | str = Field(..., description="User identifier from the request.")
    pipeline: str = Field(..., description="Effective pipeline used to serve the response.")
    top_k: int = Field(..., ge=1, description="Requested number of recommendations.")
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="Ordered recommendation list.",
    )
    latency_ms: float = Field(..., ge=0.0, description="End-to-end service latency in milliseconds.")
    fallback_used: bool = Field(
        default=False,
        description="Whether the request was served by a fallback path.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional debugging and serving metadata.",
    )


class RequestContext(BaseModel):
    """Optional request-level metadata used in logs and debugging."""

    request_id: str | None = Field(default=None, description="Per-request identifier.")
    endpoint: str | None = Field(default=None, description="Requested endpoint path.")


class HealthResponse(BaseModel):
    """Health payload for the serving service."""

    status: str = Field(..., description="Overall service status.")
    model_loaded: bool = Field(..., description="Whether base serving artifacts are ready.")
    ranker_loaded: bool = Field(..., description="Whether the ranker is ready for inference.")
    timestamp: str = Field(..., description="Current UTC timestamp in ISO 8601 format.")
