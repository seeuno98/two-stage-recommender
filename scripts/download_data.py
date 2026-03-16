"""Placeholder script for downloading or preparing raw data."""

from pathlib import Path


def main() -> None:
    """Create the raw data directory if it does not exist."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Raw data directory is ready at: {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
