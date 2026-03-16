"""Prediction utilities for the ranking model."""

import pandas as pd


def predict_scores(model: dict[str, object], features: pd.DataFrame) -> pd.Series:
    """Generate placeholder scores for ranking inference."""
    del model
    return pd.Series([0.0] * len(features), index=features.index, name="score")
