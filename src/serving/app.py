"""FastAPI application for local recommendation serving."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.serving.config import ServingConfig
from src.serving.logging import (
    build_request_log_payload,
    log_structured_event,
)
from src.serving.schemas import HealthResponse, RecommendationRequest, RecommendationResponse
from src.serving.service import RecommendationService


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and request timing metadata."""

    async def dispatch(self, request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.request_start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-MS"] = f"{(time.perf_counter() - request.state.request_start) * 1000.0:.3f}"
        return response


def create_app(
    service: RecommendationService | None = None,
    config: ServingConfig | None = None,
) -> FastAPI:
    """Create a FastAPI app with one shared recommendation service instance."""

    effective_config = config or ServingConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_service = service or RecommendationService(
            project_root=Path.cwd(),
            config=effective_config,
        )
        app.state.service = active_service
        app.state.config = effective_config

        health = active_service.get_health()
        log_structured_event(
            "startup",
            status=health["status"],
            model_loaded=health["model_loaded"],
            ranker_loaded=health["ranker_loaded"],
            default_pipeline=active_service.default_pipeline,
            fallback_pipeline=active_service.fallback_pipeline,
            candidate_pool_size=active_service.candidate_pool_size,
            fast_mode=effective_config.fast_mode,
            candidate_cache_enabled=effective_config.enable_candidate_cache,
        )
        yield

    app = FastAPI(
        title="Two-Stage Recommendation Service",
        description="Local production-style serving layer for the two-stage recommender.",
        version="0.3.5",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Return service readiness and artifact load status."""

        payload = request.app.state.service.get_health()
        return HealthResponse(**payload)

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
                request_id=request.state.request_id,
                endpoint=str(request.url.path),
                user_id=recommendation_request.user_id,
                pipeline=recommendation_request.pipeline or service_instance.default_pipeline,
                status_code=400,
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log_structured_event(
                "recommendation_request",
                request_id=request.state.request_id,
                endpoint=str(request.url.path),
                user_id=recommendation_request.user_id,
                pipeline=recommendation_request.pipeline or service_instance.default_pipeline,
                status_code=500,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        log_structured_event(
            "recommendation_request",
            **build_request_log_payload(
                response,
                request_id=request.state.request_id,
                endpoint=str(request.url.path),
                status_code=200,
            ),
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
