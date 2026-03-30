"""Offline experiment orchestration for recommendation variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from src.ranking.dataset import build_ranking_dataset_from_splits
from scripts.build_ranking_dataset import load_parquet
from src.candidate_gen.als import ALSCandidateGenerator
from src.candidate_gen.item_knn import ItemKNNRecommender
from src.candidate_gen.popularity import PopularityRecommender
from src.eval.assignment import assign_users, validate_split_config
from src.eval.metrics import evaluate_user_level
from src.ranking.dataset import build_ground_truth, build_user_histories, make_candidate_pool_for_users
from src.ranking.features import add_ranking_features
from src.ranking.predict import rerank_candidates, score_candidates, topk_predictions_from_ranked_df
from src.ranking.train_ranker import split_ranking_dataset_by_user, train_lgbm_ranker


PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
ITEM_FEATURES_PATH = PROCESSED_DIR / "item_features.parquet"
DEFAULT_K_VALUES = [10, 20, 50]
SUPPORTED_PIPELINES = {
    "popularity_only",
    "itemknn_only",
    "als_only",
    "popularity_plus_ranker",
}


@dataclass
class RankerArtifacts:
    """Cached artifacts for one trained ranking variant family."""

    model: object
    feature_cols: list[str]
    candidate_k: int
    excluded_features: tuple[str, ...]
    train_rows: int
    valid_rows: int


@dataclass
class ExperimentContext:
    """Shared dataframes, histories, and model caches for offline experiments."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    item_features_df: pd.DataFrame
    fit_df: pd.DataFrame
    eval_ground_truth: dict[int | str, set[object]]
    eval_histories: dict[int | str, set[object]]
    retriever_cache: dict[str, object] = field(default_factory=dict)
    ranker_cache: dict[tuple[int, tuple[str, ...]], RankerArtifacts] = field(default_factory=dict)


def load_experiment_config(path: str | Path) -> list[dict[str, object]]:
    """Load and validate experiment definitions from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Config must define a non-empty `experiments` list.")

    validated: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise TypeError("Each experiment entry must be a mapping.")

        name = str(experiment.get("name", "")).strip()
        if not name:
            raise ValueError("Each experiment must define a non-empty `name`.")
        if name in seen_names:
            raise ValueError(f"Duplicate experiment name: {name}")
        seen_names.add(name)

        assignment = experiment.get("assignment", {})
        if not isinstance(assignment, dict):
            raise TypeError(f"Experiment `{name}` assignment must be a mapping.")
        method = assignment.get("method", "hash_mod")
        if method != "hash_mod":
            raise ValueError(f"Experiment `{name}` only supports assignment.method=hash_mod.")
        split = validate_split_config(assignment.get("split", {}))

        variants = experiment.get("variants", {})
        if not isinstance(variants, dict) or not variants:
            raise ValueError(f"Experiment `{name}` must define variants.")
        if set(split) != set(variants):
            raise ValueError(
                f"Experiment `{name}` must use the same variant names in assignment.split and variants."
            )

        normalized_variants: dict[str, dict[str, object]] = {}
        for variant_name, variant_config in variants.items():
            if not isinstance(variant_config, dict):
                raise TypeError(
                    f"Experiment `{name}` variant `{variant_name}` must be a mapping."
                )
            pipeline = str(variant_config.get("pipeline", "")).strip()
            if pipeline not in SUPPORTED_PIPELINES:
                raise ValueError(
                    f"Experiment `{name}` variant `{variant_name}` uses unsupported pipeline `{pipeline}`."
                )
            candidate_k = int(variant_config.get("candidate_k", max(DEFAULT_K_VALUES)))
            if candidate_k < max(DEFAULT_K_VALUES):
                raise ValueError(
                    f"Experiment `{name}` variant `{variant_name}` must set candidate_k >= {max(DEFAULT_K_VALUES)}."
                )
            excluded_features = variant_config.get("exclude_features", [])
            if excluded_features is None:
                excluded_features = []
            if not isinstance(excluded_features, list):
                raise TypeError(
                    f"Experiment `{name}` variant `{variant_name}` exclude_features must be a list."
                )
            normalized_variants[str(variant_name)] = {
                "pipeline": pipeline,
                "candidate_k": candidate_k,
                "exclude_features": [str(feature) for feature in excluded_features],
            }

        validated.append(
            {
                "name": name,
                "description": str(experiment.get("description", "")),
                "assignment": {
                    "method": "hash_mod",
                    "id_col": str(assignment.get("id_col", "user_id")),
                    "split": split,
                },
                "variants": normalized_variants,
            }
        )

    return validated


def build_experiment_context(
    train_path: Path = TRAIN_PATH,
    val_path: Path = VAL_PATH,
    test_path: Path = TEST_PATH,
    item_features_path: Path = ITEM_FEATURES_PATH,
) -> ExperimentContext:
    """Load the default offline experiment data context."""

    train_df = load_parquet(train_path)
    val_df = load_parquet(val_path)
    test_df = load_parquet(test_path)
    item_features_df = load_parquet(item_features_path)

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    full_histories = build_user_histories(fit_df)
    test_ground_truth = build_ground_truth(test_df)
    eval_histories = {
        user_id: full_histories.get(user_id, set())
        for user_id in test_ground_truth
    }

    return ExperimentContext(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        item_features_df=item_features_df,
        fit_df=fit_df,
        eval_ground_truth=test_ground_truth,
        eval_histories=eval_histories,
    )


def get_experiment_by_name(
    experiments: list[dict[str, object]],
    name: str,
) -> dict[str, object]:
    """Return one configured experiment by name."""

    for experiment in experiments:
        if experiment.get("name") == name:
            return experiment
    raise KeyError(f"Experiment `{name}` not found in config.")


def _build_group_sizes(df: pd.DataFrame, group_col: str = "user_id") -> list[int]:
    """Compute user-level group sizes for ranking model training."""

    if df.empty:
        return []
    return df.groupby(group_col, sort=False).size().astype(int).tolist()


def _ensure_retriever(pipeline_name: str, context: ExperimentContext) -> object:
    """Fit and cache a retrieval model needed by a variant pipeline."""

    if pipeline_name in {"popularity_only", "popularity_plus_ranker"}:
        cache_key = "popularity"
        if cache_key not in context.retriever_cache:
            context.retriever_cache[cache_key] = PopularityRecommender().fit(context.fit_df)
        return context.retriever_cache[cache_key]

    if pipeline_name == "itemknn_only":
        cache_key = "itemknn"
        if cache_key not in context.retriever_cache:
            context.retriever_cache[cache_key] = ItemKNNRecommender().fit(context.fit_df)
        return context.retriever_cache[cache_key]

    if pipeline_name == "als_only":
        cache_key = "als"
        if cache_key not in context.retriever_cache:
            context.retriever_cache[cache_key] = ALSCandidateGenerator().fit(context.fit_df)
        return context.retriever_cache[cache_key]

    raise ValueError(f"Unsupported pipeline: {pipeline_name}")


def _ensure_ranker_artifacts(
    context: ExperimentContext,
    candidate_k: int,
    excluded_features: tuple[str, ...] = (),
) -> RankerArtifacts:
    """Train and cache a LightGBM ranker for one candidate/feature setting."""

    cache_key = (candidate_k, excluded_features)
    if cache_key in context.ranker_cache:
        return context.ranker_cache[cache_key]

    # build ranker training dataset from train -> val (labels from val), deterministic retriever fit on train
    ranking_df, feature_cols, _ = build_ranking_dataset_from_splits(
        history_df=context.train_df,
        target_df=context.val_df,
        item_features_df=context.item_features_df,
        candidate_top_n=candidate_k,
        retriever_name="popularity",
        retriever=None,
        include_labels=True,
    )
    filtered_feature_cols = [
        feature for feature in feature_cols if feature not in set(excluded_features)
    ]
    if not filtered_feature_cols:
        raise ValueError("Ranker feature exclusion removed all usable features.")

    train_df, valid_df = split_ranking_dataset_by_user(
        ranking_df,
        valid_frac=0.2,
        random_state=42,
    )
    model = train_lgbm_ranker(
        X_train=train_df[filtered_feature_cols],
        y_train=train_df["label"],
        group_train=_build_group_sizes(train_df),
        X_valid=valid_df[filtered_feature_cols],
        y_valid=valid_df["label"],
        group_valid=_build_group_sizes(valid_df),
    )
    artifacts = RankerArtifacts(
        model=model,
        feature_cols=filtered_feature_cols,
        candidate_k=candidate_k,
        excluded_features=excluded_features,
        train_rows=int(len(train_df)),
        valid_rows=int(len(valid_df)),
    )
    context.ranker_cache[cache_key] = artifacts
    return artifacts


def _align_feature_columns(candidate_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensure candidate scoring frames contain the expected feature columns."""

    aligned = candidate_features.copy()
    for column in feature_cols:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned


def build_variant_predictions(
    variant_config: Mapping[str, object],
    user_ids: list[int | str],
    context: ExperimentContext,
    k_values: list[int] | None = None,
) -> dict[int | str, list[object]]:
    """Generate top-k recommendations for one experiment variant."""

    effective_k_values = k_values or DEFAULT_K_VALUES
    max_k = max(effective_k_values)
    pipeline_name = str(variant_config["pipeline"])
    candidate_k = int(variant_config.get("candidate_k", max_k))
    user_histories = {
        user_id: context.eval_histories.get(user_id, set())
        for user_id in user_ids
    }

    if pipeline_name in {"popularity_only", "itemknn_only", "als_only"}:
        retriever = _ensure_retriever(pipeline_name, context)
        return retriever.recommend_for_users(user_histories, k=max_k)

    if pipeline_name == "popularity_plus_ranker":
        excluded_features = tuple(sorted(str(item) for item in variant_config.get("exclude_features", [])))
        popularity_retriever = _ensure_retriever("popularity_plus_ranker", context)
        ranker_artifacts = _ensure_ranker_artifacts(
            context=context,
            candidate_k=candidate_k,
            excluded_features=excluded_features,
        )
        # Build the held-out test candidate set from (train+val) -> test
        candidate_features, _, _ = build_ranking_dataset_from_splits(
            history_df=context.fit_df,
            target_df=context.test_df,
            item_features_df=context.item_features_df,
            candidate_top_n=candidate_k,
            retriever_name="popularity",
            retriever=popularity_retriever,
            include_labels=True,
        )
        candidate_features = _align_feature_columns(candidate_features, ranker_artifacts.feature_cols)
        scores = score_candidates(
            model=ranker_artifacts.model,
            X=candidate_features,
            feature_cols=ranker_artifacts.feature_cols,
        )
        ranked_df = rerank_candidates(candidate_df=candidate_features, scores=scores)
        return topk_predictions_from_ranked_df(ranked_df, k=max_k)

    raise ValueError(f"Unsupported pipeline: {pipeline_name}")


def evaluate_variant(
    variant_name: str,
    variant_config: Mapping[str, object],
    user_ids: list[int | str],
    context: ExperimentContext,
    k_values: list[int] | None = None,
) -> dict[str, object]:
    """Evaluate one variant on its assigned users."""

    effective_k_values = k_values or DEFAULT_K_VALUES
    predictions = build_variant_predictions(
        variant_config=variant_config,
        user_ids=user_ids,
        context=context,
        k_values=effective_k_values,
    )
    ground_truth = {
        user_id: context.eval_ground_truth[user_id]
        for user_id in user_ids
        if user_id in context.eval_ground_truth
    }
    metrics = evaluate_user_level(
        ground_truth=ground_truth,
        predictions=predictions,
        k_values=effective_k_values,
    )
    return {
        "variant": variant_name,
        "pipeline": str(variant_config["pipeline"]),
        "candidate_k": int(variant_config.get("candidate_k", max(effective_k_values))),
        "exclude_features": list(variant_config.get("exclude_features", [])),
        "user_count": len(user_ids),
        "metrics": metrics,
    }


def compare_variants(
    variant_results: Mapping[str, Mapping[str, object]],
    control_name: str = "control",
) -> dict[str, object]:
    """Compute absolute and relative metric lift for each treatment vs control."""

    if control_name not in variant_results:
        raise KeyError(f"Control variant `{control_name}` not found in experiment results.")

    control_metrics = variant_results[control_name].get("metrics", {})
    if not isinstance(control_metrics, dict):
        raise TypeError("Control variant metrics must be a mapping.")

    comparison: dict[str, object] = {
        "control_variant": control_name,
        "lift_vs_control": {},
    }
    for variant_name, payload in variant_results.items():
        if variant_name == control_name:
            continue

        variant_metrics = payload.get("metrics", {})
        if not isinstance(variant_metrics, dict):
            continue

        metric_lift: dict[str, dict[str, float | None]] = {}
        for metric_name, treatment_value in variant_metrics.items():
            control_value = float(control_metrics.get(metric_name, 0.0))
            treatment_value_float = float(treatment_value)
            absolute_diff = treatment_value_float - control_value
            relative_diff = None if control_value == 0.0 else absolute_diff / control_value
            metric_lift[metric_name] = {
                "control": control_value,
                "treatment": treatment_value_float,
                "absolute_diff": absolute_diff,
                "relative_diff": relative_diff,
            }

        comparison["lift_vs_control"][variant_name] = metric_lift

    return comparison


def run_single_experiment(
    experiment_config: Mapping[str, object],
    context: ExperimentContext | None = None,
    k_values: list[int] | None = None,
) -> dict[str, object]:
    """Run one offline experiment end to end."""

    experiment_context = context or build_experiment_context()
    effective_k_values = k_values or DEFAULT_K_VALUES

    name = str(experiment_config["name"])
    description = str(experiment_config.get("description", ""))
    assignment = experiment_config.get("assignment", {})
    if not isinstance(assignment, dict):
        raise TypeError("experiment_config.assignment must be a mapping.")
    split_config = validate_split_config(assignment.get("split", {}))

    variants = experiment_config.get("variants", {})
    if not isinstance(variants, dict):
        raise TypeError("experiment_config.variants must be a mapping.")

    eval_user_ids = sorted(experiment_context.eval_ground_truth)
    user_assignments = assign_users(eval_user_ids, split_config=split_config)
    users_by_variant = {
        variant_name: sorted(
            user_id
            for user_id, assigned_variant in user_assignments.items()
            if assigned_variant == variant_name
        )
        for variant_name in variants
    }

    variant_results: dict[str, dict[str, object]] = {}
    for variant_name, variant_config in variants.items():
        variant_results[variant_name] = evaluate_variant(
            variant_name=variant_name,
            variant_config=variant_config,
            user_ids=users_by_variant.get(variant_name, []),
            context=experiment_context,
            k_values=effective_k_values,
        )

    return {
        "name": name,
        "description": description,
        "assignment": {
            "method": assignment.get("method", "hash_mod"),
            "id_col": assignment.get("id_col", "user_id"),
            "split": split_config,
            "assigned_users": {
                variant_name: len(user_ids)
                for variant_name, user_ids in users_by_variant.items()
            },
        },
        "evaluation": {
            "fit_data": "train_plus_val",
            "ranker_training": "train_to_val",
            "evaluation_split": "test",
            "metrics": effective_k_values,
        },
        "variants": variant_results,
        "comparison": compare_variants(variant_results),
    }
