"""Run filtered ALS retrieval experiments on processed RetailRocket data."""

from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from src.candidate_gen.als import ALSCandidateGenerator
from src.data.preprocess import filter_interactions_by_event_types, remap_event_weights
from src.eval.metrics import evaluate_user_level
from src.utils.io import save_dataframe, save_json


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("artifacts/reports")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
EXPERIMENTS_JSON_PATH = REPORTS_DIR / "als_experiments.json"
EXPERIMENTS_CSV_PATH = REPORTS_DIR / "als_experiments.csv"
BEST_EXPERIMENT_PATH = REPORTS_DIR / "als_best_experiment.json"
POPULARITY_METRICS_PATH = REPORTS_DIR / "popularity_baseline_metrics.json"
ITEMKNN_METRICS_PATH = REPORTS_DIR / "itemknn_baseline_metrics.json"
ALS_BASELINE_METRICS_PATH = REPORTS_DIR / "als_baseline_metrics.json"
K_VALUES = [10, 20, 50]
PRIMARY_METRIC = "recall@20"

INTERACTION_VARIANTS = {
    "all_events": ["view", "addtocart", "transaction"],
    "strong_events": ["addtocart", "transaction"],
    "transaction_only": ["transaction"],
}
WEIGHT_VARIANTS = {
    "baseline_weights": {
        "view": 1.0,
        "addtocart": 3.0,
        "transaction": 5.0,
    },
    "strong_intent_weights": {
        "view": 1.0,
        "addtocart": 10.0,
        "transaction": 30.0,
    },
}
EXPERIMENT_VARIANTS = [
    ("all_events", "baseline_weights"),
    ("strong_events", "baseline_weights"),
    ("strong_events", "strong_intent_weights"),
    ("transaction_only", "baseline_weights"),
]
ALS_PARAM_GRID = {
    "factors": [64],
    "regularization": [0.01, 0.05],
    "iterations": [20],
    "alpha": [10.0, 20.0, 40.0],
}

# ALS_PARAM_GRID = {
#     "factors": [32, 64, 128],
#     "regularization": [0.01, 0.05, 0.1],
#     "iterations": [20],
#     "alpha": [10.0, 20.0, 40.0],
# }

def build_user_history(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build per-user interacted item sets."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def build_ground_truth(df: pd.DataFrame) -> dict[int | str, set[object]]:
    """Build validation ground-truth item sets."""

    return df.groupby("user_id")["item_id"].apply(set).to_dict()


def load_processed_split(path: Path) -> pd.DataFrame:
    """Load a processed parquet split."""

    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found at {path}. Run `python -m scripts.prepare_data` first."
        )
    return pd.read_parquet(path)


def maybe_load_metrics(path: Path) -> dict[str, float] | None:
    """Load a metrics JSON file if it exists."""

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return {str(key): float(value) for key, value in payload.items()}


def build_experiment_name(config: dict[str, str | int | float]) -> str:
    """Create a short readable experiment identifier."""

    return (
        f"{config['interaction_variant']}__{config['weight_variant']}"
        f"__f{config['factors']}"
        f"_reg{config['regularization']}"
        f"_iter{config['iterations']}"
        f"_a{config['alpha']}"
    )


def iter_experiment_configs() -> list[dict[str, str | int | float]]:
    """Enumerate the small ALS experiment grid."""

    configs: list[dict[str, str | int | float]] = []
    for interaction_variant, weight_variant in EXPERIMENT_VARIANTS:
        for factors, regularization, iterations, alpha in product(
            ALS_PARAM_GRID["factors"],
            ALS_PARAM_GRID["regularization"],
            ALS_PARAM_GRID["iterations"],
            ALS_PARAM_GRID["alpha"],
        ):
            config = {
                "interaction_variant": interaction_variant,
                "weight_variant": weight_variant,
                "factors": factors,
                "regularization": regularization,
                "iterations": iterations,
                "alpha": alpha,
            }
            config["experiment_name"] = build_experiment_name(config)
            configs.append(config)
    return configs


def prepare_experiment_train_df(
    train_df: pd.DataFrame,
    interaction_variant: str,
    weight_variant: str,
) -> pd.DataFrame:
    """Filter and reweight training interactions for one ALS run."""

    allowed_event_types = INTERACTION_VARIANTS[interaction_variant]
    weight_map = WEIGHT_VARIANTS[weight_variant]
    filtered_df = filter_interactions_by_event_types(train_df, allowed_event_types)
    remapped_df = remap_event_weights(filtered_df, weight_map)
    return remapped_df


def evaluate_experiment(
    train_df: pd.DataFrame,
    eval_histories: dict[int | str, set[object]],
    ground_truth: dict[int | str, set[object]],
    config: dict[str, str | int | float],
) -> dict[str, object]:
    """Fit ALS for one config and return metrics plus metadata."""

    experiment_train_df = prepare_experiment_train_df(
        train_df=train_df,
        interaction_variant=str(config["interaction_variant"]),
        weight_variant=str(config["weight_variant"]),
    )
    if experiment_train_df.empty:
        raise ValueError(
            "Experiment produced no training rows after filtering:"
            f" interaction_variant={config['interaction_variant']}"
            f" weight_variant={config['weight_variant']}"
        )

    generator = ALSCandidateGenerator(
        factors=int(config["factors"]),
        regularization=float(config["regularization"]),
        iterations=int(config["iterations"]),
        alpha=float(config["alpha"]),
    ).fit(experiment_train_df)

    predictions = generator.recommend_for_users(eval_histories, k=max(K_VALUES))
    metrics = evaluate_user_level(
        ground_truth=ground_truth,
        predictions=predictions,
        k_values=K_VALUES,
    )

    result: dict[str, object] = {
        "experiment_name": str(config["experiment_name"]),
        "interaction_variant": str(config["interaction_variant"]),
        "weight_variant": str(config["weight_variant"]),
        "train_rows_used": int(len(experiment_train_df)),
        "unique_train_users_used": int(experiment_train_df["user_id"].nunique()),
        "unique_train_items_used": int(experiment_train_df["item_id"].nunique()),
        "factors": int(config["factors"]),
        "regularization": float(config["regularization"]),
        "iterations": int(config["iterations"]),
        "alpha": float(config["alpha"]),
        "invalid_recommendation_indices": int(generator.invalid_recommendation_indices_),
        "metrics": metrics,
    }
    return result


def flatten_experiment_results(results: list[dict[str, object]]) -> pd.DataFrame:
    """Convert nested experiment results into a flat report table."""

    rows: list[dict[str, object]] = []
    for result in results:
        row = {key: value for key, value in result.items() if key != "metrics"}
        metrics = result["metrics"]
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def print_leaderboard(results_df: pd.DataFrame, top_n: int = 10) -> None:
    """Print a concise leaderboard sorted by the primary metric."""

    leaderboard = results_df.sort_values(
        by=[PRIMARY_METRIC, "ndcg@20", "recall@50"],
        ascending=[False, False, False],
    ).head(top_n)

    print("[als-experiments] leaderboard")
    for _, row in leaderboard.iterrows():
        print(
            "[als-experiments]"
            f" {row['experiment_name']}"
            f" Recall@20={row['recall@20']:.4f}"
            f" NDCG@20={row['ndcg@20']:.4f}"
            f" rows={int(row['train_rows_used'])}"
        )


def print_baseline_comparison(best_result: dict[str, object]) -> None:
    """Print the best ALS experiment against prior baseline reports if available."""

    baseline_paths = {
        "popularity": POPULARITY_METRICS_PATH,
        "itemknn": ITEMKNN_METRICS_PATH,
        "als_baseline": ALS_BASELINE_METRICS_PATH,
    }
    baseline_metrics = {
        name: metrics
        for name, path in baseline_paths.items()
        if (metrics := maybe_load_metrics(path)) is not None
    }
    if not baseline_metrics:
        return

    best_metrics = best_result["metrics"]
    if not isinstance(best_metrics, dict):
        return

    print("[als-experiments] comparison")
    print(
        "[als-experiments]"
        f" best_experiment={best_result['experiment_name']}"
        f" {PRIMARY_METRIC}={best_metrics[PRIMARY_METRIC]:.4f}"
    )
    for baseline_name, metrics in baseline_metrics.items():
        print(
            "[als-experiments]"
            f" baseline={baseline_name}"
            f" Recall@20={metrics.get('recall@20', 0.0):.4f}"
            f" NDCG@20={metrics.get('ndcg@20', 0.0):.4f}"
        )


def main() -> None:
    """Run the ALS filtering and hyperparameter experiments on validation data."""

    train_df = load_processed_split(TRAIN_PATH)
    val_df = load_processed_split(VAL_PATH)

    train_histories = build_user_history(train_df)
    val_ground_truth = build_ground_truth(val_df)
    eval_histories = {
        user_id: train_histories.get(user_id, set())
        for user_id in val_ground_truth.keys()
    }

    print(f"[als-experiments] train_rows={len(train_df)}")
    print(f"[als-experiments] val_rows={len(val_df)}")
    print(f"[als-experiments] validation_users={len(val_ground_truth)}")
    print(f"[als-experiments] num_experiments={len(iter_experiment_configs())}")

    results: list[dict[str, object]] = []
    experiment_configs = iter_experiment_configs()
    for index, config in enumerate(experiment_configs, start=1):
        print(
            "[als-experiments]"
            f" run={index}/{len(experiment_configs)}"
            f" name={config['experiment_name']}"
        )
        result = evaluate_experiment(
            train_df=train_df,
            eval_histories=eval_histories,
            ground_truth=val_ground_truth,
            config=config,
        )
        metrics = result["metrics"]
        if isinstance(metrics, dict):
            print(
                "[als-experiments]"
                f" Recall@20={metrics['recall@20']:.4f}"
                f" NDCG@20={metrics['ndcg@20']:.4f}"
            )
        results.append(result)

    results_df = flatten_experiment_results(results)
    results_df = results_df.sort_values(
        by=[PRIMARY_METRIC, "ndcg@20", "recall@50"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_result = results_df.iloc[0].to_dict()
    best_metrics = {
        metric_name: float(best_result[metric_name])
        for metric_name in results_df.columns
        if metric_name.startswith("recall@") or metric_name.startswith("ndcg@")
    }
    best_payload = {
        "experiment_name": best_result["experiment_name"],
        "interaction_variant": best_result["interaction_variant"],
        "weight_variant": best_result["weight_variant"],
        "train_rows_used": int(best_result["train_rows_used"]),
        "unique_train_users_used": int(best_result["unique_train_users_used"]),
        "unique_train_items_used": int(best_result["unique_train_items_used"]),
        "factors": int(best_result["factors"]),
        "regularization": float(best_result["regularization"]),
        "iterations": int(best_result["iterations"]),
        "alpha": float(best_result["alpha"]),
        "invalid_recommendation_indices": int(best_result["invalid_recommendation_indices"]),
        "metrics": best_metrics,
    }

    save_json(results, EXPERIMENTS_JSON_PATH)
    save_dataframe(results_df, EXPERIMENTS_CSV_PATH)
    save_json(best_payload, BEST_EXPERIMENT_PATH)

    print(f"[als-experiments] json_report={EXPERIMENTS_JSON_PATH}")
    print(f"[als-experiments] csv_report={EXPERIMENTS_CSV_PATH}")
    print(f"[als-experiments] best_report={BEST_EXPERIMENT_PATH}")
    print_leaderboard(results_df)
    print_baseline_comparison(best_payload)


if __name__ == "__main__":
    main()
