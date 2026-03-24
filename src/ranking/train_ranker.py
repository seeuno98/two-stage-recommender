"""Training utilities for the ranking model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover - exercised only in environments without lightgbm
    LGBMRanker = None  # type: ignore[assignment]

from src.eval.metrics import evaluate_user_level
from src.ranking.predict import rerank_candidates, score_candidates, topk_predictions_from_ranked_df
from src.utils.io import save_dataframe, save_json


DEFAULT_K_VALUES = [10, 20, 50]


def require_lightgbm() -> None:
    """Raise a helpful error if LightGBM is unavailable."""

    if LGBMRanker is None:
        raise RuntimeError(
            "The `lightgbm` package is not installed. Install project dependencies first."
        )


def split_ranking_dataset_by_user(
    ranking_df: pd.DataFrame,
    valid_frac: float = 0.2,
    random_state: int = 42,
    user_col: str = "user_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ranking rows by user so groups do not leak across train and validation."""

    if not 0 < valid_frac < 1:
        raise ValueError("valid_frac must be between 0 and 1.")

    user_ids = ranking_df[user_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(user_ids)
    valid_count = max(1, int(len(shuffled) * valid_frac))
    valid_users = set(shuffled[:valid_count].tolist())

    train_df = ranking_df[~ranking_df[user_col].isin(valid_users)].copy()
    valid_df = ranking_df[ranking_df[user_col].isin(valid_users)].copy()
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def train_lgbm_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    group_valid: list[int] | None = None,
    random_state: int = 42,
) -> LGBMRanker:
    """Train an LGBMRanker with a small, stable default configuration."""

    require_lightgbm()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )

    fit_kwargs: dict[str, object] = {
        "X": X_train,
        "y": y_train,
        "group": group_train,
    }
    if X_valid is not None and y_valid is not None and group_valid is not None:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_group"] = [group_valid]
        fit_kwargs["eval_at"] = DEFAULT_K_VALUES

    model.fit(**fit_kwargs)
    return model


def evaluate_lgbm_ranker(
    model: LGBMRanker,
    candidate_df: pd.DataFrame,
    feature_cols: list[str],
    ground_truth: dict[int | str, set[object]],
    k_values: list[int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score, rerank, and evaluate candidate rows against per-user ground truth."""

    effective_k_values = k_values or DEFAULT_K_VALUES
    scores = score_candidates(model=model, X=candidate_df, feature_cols=feature_cols)
    ranked_df = rerank_candidates(candidate_df=candidate_df, scores=scores)
    predictions = topk_predictions_from_ranked_df(ranked_df, k=max(effective_k_values))
    metrics = evaluate_user_level(
        ground_truth=ground_truth,
        predictions=predictions,
        k_values=effective_k_values,
    )
    return metrics, ranked_df


def save_feature_importance(
    model: LGBMRanker,
    feature_cols: list[str],
    output_path: str | Path,
) -> pd.DataFrame:
    """Persist feature importance values as a CSV report."""

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False)
    save_dataframe(importance_df, output_path)
    return importance_df.reset_index(drop=True)


def save_lgbm_model(model: LGBMRanker, output_path: str | Path) -> None:
    """Persist the trained LightGBM model."""

    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))


def save_lgbm_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Persist ranker metrics as JSON."""

    save_json(metrics, output_path)


def maybe_load_metrics(path: str | Path) -> dict[str, object] | None:
    """Load a JSON metrics payload if it exists."""

    metrics_path = Path(path)
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)
