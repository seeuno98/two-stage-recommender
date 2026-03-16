"""Utilities for assembling ranking datasets."""

import pandas as pd


def build_ranking_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Return the input features unchanged as a placeholder dataset builder."""
    return features.copy()
