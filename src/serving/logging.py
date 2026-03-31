"""Structured logging helpers for the serving API."""

from __future__ import annotations

import json
import logging
from typing import Any


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
    """Emit one JSON-like structured log line."""

    logger = get_serving_logger()
    message = json.dumps(
        {"event": event_type, **payload},
        default=str,
        sort_keys=True,
    )
    logger.info(message)
