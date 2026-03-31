"""Development entrypoint for the FastAPI recommendation service."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the local FastAPI service with reload enabled."""

    uvicorn.run(
        "src.serving.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
