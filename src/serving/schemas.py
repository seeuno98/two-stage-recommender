"""Pydantic schemas for the serving layer."""

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    """Response model for recommendation requests."""

    user_id: int
    recommendations: list[int]
