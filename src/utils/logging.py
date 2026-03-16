"""Logging utilities."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a configured logger."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
