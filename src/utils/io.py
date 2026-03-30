"""Simple I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a dataframe to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_json(payload: object, path: str | Path) -> None:
    """Persist a JSON-serializable payload."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_text(text: str, path: str | Path) -> None:
    """Persist a UTF-8 text payload."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(text)
