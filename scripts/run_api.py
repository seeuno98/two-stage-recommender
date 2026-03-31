"""Development entrypoint for the FastAPI recommendation service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the local FastAPI service with reload enabled."""

    uvicorn.run(
        "src.serving.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() in {"1", "true", "yes", "on"},
    )


if __name__ == "__main__":
    main()
