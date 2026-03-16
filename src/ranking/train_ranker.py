"""Training utilities for the ranking model."""

import pandas as pd


def train_ranker(features: pd.DataFrame) -> dict[str, object]:
    """Return a placeholder trained model artifact."""
    return {"model_type": "placeholder_ranker", "num_rows": len(features)}
