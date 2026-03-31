"""FastAPI application for local recommendation serving."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from src.serving.logging import log_structured_event
from src.serving.schemas import HealthResponse, RecommendationRequest, RecommendationResponse
from src.serving.service import RecommendationService


def create_app(service: RecommendationService | None = None) -> FastAPI:
    """Create a FastAPI app with one shared recommendation service instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_service = service or RecommendationService(project_root=Path.cwd())
        app.state.service = active_service
        health = active_service.get_health()
        log_structured_event(
            "startup",
            status=health["status"],
            model_loaded=health["model_loaded"],
            ranker_loaded=health["ranker_loaded"],
            candidate_pool_size=active_service.candidate_pool_size,
        )
        yield

    app = FastAPI(
        title="Two-Stage Recommendation Service",
        description="Local production-style serving layer for the two-stage recommender.",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Return service readiness and artifact load status."""

        payload = request.app.state.service.get_health()
        return HealthResponse(
            status=payload["status"],
            model_loaded=payload["model_loaded"],
            ranker_loaded=payload["ranker_loaded"],
            timestamp=payload["timestamp"],
        )

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(
        recommendation_request: RecommendationRequest,
        request: Request,
    ) -> RecommendationResponse:
        """Serve recommendations from the requested pipeline."""

        service_instance: RecommendationService = request.app.state.service
        try:
            response = service_instance.recommend(
                user_id=recommendation_request.user_id,
                top_k=recommendation_request.top_k,
                pipeline=recommendation_request.pipeline,
            )
        except ValueError as exc:
            log_structured_event(
                "recommendation_request",
                user_id=recommendation_request.user_id,
                pipeline=recommendation_request.pipeline or service_instance.default_pipeline,
                top_k=recommendation_request.top_k,
                latency_ms=0.0,
                fallback_used=False,
                recommendation_count=0,
                status="bad_request",
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log_structured_event(
                "recommendation_request",
                user_id=recommendation_request.user_id,
                pipeline=recommendation_request.pipeline or service_instance.default_pipeline,
                top_k=recommendation_request.top_k,
                latency_ms=0.0,
                fallback_used=False,
                recommendation_count=0,
                status="error",
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log_structured_event(
            "recommendation_request",
            user_id=response.user_id,
            pipeline=response.pipeline,
            top_k=response.top_k,
            latency_ms=round(response.latency_ms, 3),
            fallback_used=response.fallback_used,
            recommendation_count=len(response.recommendations),
            status="ok",
        )
        return response

    @app.get("/recommend/{user_id}", response_model=RecommendationResponse)
    def recommend_for_user(
        user_id: str,
        request: Request,
        top_k: int = Query(default=10, ge=1, le=100),
        pipeline: str | None = Query(default=None),
    ) -> RecommendationResponse:
        """Convenience endpoint for browser-friendly recommendation calls."""

        return recommend(
            recommendation_request=RecommendationRequest(
                user_id=user_id,
                top_k=top_k,
                pipeline=pipeline,
            ),
            request=request,
        )

    return app


app = create_app()
