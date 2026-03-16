"""Minimal FastAPI application for recommendation serving."""

from fastapi import FastAPI

from src.serving.schemas import RecommendationResponse

app = FastAPI(title="Two-Stage Recommendation System")


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int) -> RecommendationResponse:
    """Return placeholder recommendations for a user."""
    recommendations = [101, 202, 303, 404, 505]
    return RecommendationResponse(user_id=user_id, recommendations=recommendations)
