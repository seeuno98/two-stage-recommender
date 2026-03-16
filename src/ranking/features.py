"""Feature generation for ranking models."""

import pandas as pd


def add_ranker_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder ranking features."""
    features = candidates.copy()
    if "candidate_score" not in features.columns:
        features["candidate_score"] = 0.0
    return features
