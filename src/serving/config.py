"""Configuration helpers for the FastAPI serving layer."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PIPELINE = "popularity_plus_ranker"
FALLBACK_PIPELINE = "popularity_only"
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_POOL_SIZE = 150
ENABLE_CANDIDATE_CACHE = True
MAX_CANDIDATE_CACHE_SIZE = 10000
LOG_LATENCY_BREAKDOWN = True
FAST_MODE = False

def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a safe default."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe default."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(slots=True)
class ServingConfig:
    """Runtime configuration for local recommendation serving."""

    default_pipeline: str = DEFAULT_PIPELINE
    fallback_pipeline: str = FALLBACK_PIPELINE
    default_top_k: int = DEFAULT_TOP_K
    default_candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    enable_candidate_cache: bool = ENABLE_CANDIDATE_CACHE
    max_candidate_cache_size: int = MAX_CANDIDATE_CACHE_SIZE
    log_latency_breakdown: bool = LOG_LATENCY_BREAKDOWN
    fast_mode: bool = FAST_MODE

    @classmethod
    def from_env(cls) -> "ServingConfig":
        return cls(
            default_pipeline=os.getenv(
                "SERVING_DEFAULT_PIPELINE",
                DEFAULT_PIPELINE,
            ),
            fallback_pipeline=os.getenv(
                "SERVING_FALLBACK_PIPELINE",
                FALLBACK_PIPELINE,
            ),
            default_top_k=_get_int_env(
                "SERVING_DEFAULT_TOP_K",
                DEFAULT_TOP_K,
            ),
            default_candidate_pool_size=_get_int_env(
                "SERVING_DEFAULT_CANDIDATE_POOL_SIZE",
                DEFAULT_CANDIDATE_POOL_SIZE,
            ),
            enable_candidate_cache=_get_bool_env(
                "SERVING_ENABLE_CANDIDATE_CACHE",
                ENABLE_CANDIDATE_CACHE,
            ),
            max_candidate_cache_size=_get_int_env(
                "SERVING_MAX_CANDIDATE_CACHE_SIZE",
                MAX_CANDIDATE_CACHE_SIZE,
            ),
            log_latency_breakdown=_get_bool_env(
                "SERVING_LOG_LATENCY_BREAKDOWN",
                LOG_LATENCY_BREAKDOWN,
            ),
            fast_mode=_get_bool_env(
                "SERVING_FAST_MODE",
                FAST_MODE,
            ),
        )