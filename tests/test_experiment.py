"""Tests for offline experiment helpers."""

from __future__ import annotations

import pytest
import pandas as pd

from src.eval.assignment import (
    assign_users,
    assign_variant,
    stable_hash_to_bucket,
    validate_split_config,
)
from src.eval.experiment import compare_variants
from src.eval.reporting import flatten_experiment_results
from src.eval.experiment import run_single_experiment, ExperimentContext
from src.ranking.dataset import build_ground_truth, build_user_histories


def test_stable_hash_to_bucket_is_deterministic() -> None:
    """The same value should always map to the same bucket."""

    assert stable_hash_to_bucket(12345) == stable_hash_to_bucket(12345)
    assert stable_hash_to_bucket("user-42", modulo=17) == stable_hash_to_bucket("user-42", modulo=17)


def test_validate_split_config_requires_sum_of_100() -> None:
    """Split percentages must sum to 100."""

    with pytest.raises(ValueError):
        validate_split_config({"control": 40, "treatment": 40})


def test_same_user_is_always_assigned_to_same_variant() -> None:
    """Assignment should be deterministic for one user."""

    split = {"control": 50, "treatment": 50}
    assigned = assign_variant(987654, split)
    for _ in range(5):
        assert assign_variant(987654, split) == assigned


def test_assign_users_returns_expected_variant_names() -> None:
    """Bulk assignment should preserve the declared variant names."""

    assignments = assign_users([1, 2, 3, 4], {"control": 50, "treatment": 50})
    assert set(assignments).issuperset({1, 2, 3, 4})
    assert set(assignments.values()).issubset({"control", "treatment"})


def test_compare_variants_computes_absolute_and_relative_lift() -> None:
    """Variant comparison should compute lift vs control."""

    comparison = compare_variants(
        {
            "control": {"metrics": {"recall@10": 0.10, "ndcg@10": 0.05}},
            "treatment": {"metrics": {"recall@10": 0.12, "ndcg@10": 0.06}},
        }
    )

    recall_lift = comparison["lift_vs_control"]["treatment"]["recall@10"]
    assert recall_lift["absolute_diff"] == pytest.approx(0.02)
    assert recall_lift["relative_diff"] == pytest.approx(0.2)


def test_flatten_experiment_results_returns_variant_rows() -> None:
    """Reporting should flatten nested experiment payloads into tabular rows."""

    result = {
        "name": "demo_experiment",
        "variants": {
            "control": {
                "pipeline": "popularity_only",
                "candidate_k": 100,
                "user_count": 10,
                "metrics": {"recall@10": 0.10, "ndcg@10": 0.05},
            },
            "treatment": {
                "pipeline": "popularity_plus_ranker",
                "candidate_k": 100,
                "user_count": 10,
                "metrics": {"recall@10": 0.15, "ndcg@10": 0.07},
            },
        },
        "comparison": {
            "lift_vs_control": {
                "treatment": {
                    "recall@10": {
                        "absolute_diff": 0.05,
                        "relative_diff": 0.5,
                    }
                }
            }
        },
    }

    flattened = flatten_experiment_results(result)
    assert list(flattened["variant"]) == ["control", "treatment"]
    treatment_row = flattened.loc[flattened["variant"] == "treatment"].iloc[0]
    assert treatment_row["recall@10"] == pytest.approx(0.15)
    assert treatment_row["recall@10_absolute_diff"] == pytest.approx(0.05)


def test_experiment_metadata_and_temporal_consistency() -> None:
    # toy interactions
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "event_weight": [1.0], "timestamp": pd.to_datetime(["2020-01-01"])})
    val = pd.DataFrame({"user_id": [1], "item_id": [20], "event_weight": [1.0], "timestamp": pd.to_datetime(["2020-01-02"])})
    test = pd.DataFrame({"user_id": [1], "item_id": [30], "event_weight": [1.0], "timestamp": pd.to_datetime(["2020-01-03"])})
    item_features = pd.DataFrame({"item_id": [10, 20, 30]})

    fit_df = pd.concat([train, val], ignore_index=True)
    eval_ground_truth = build_ground_truth(test)
    full_histories = build_user_histories(fit_df)
    eval_histories = {user_id: full_histories.get(user_id, set()) for user_id in eval_ground_truth}

    context = ExperimentContext(
        train_df=train,
        val_df=val,
        test_df=test,
        item_features_df=item_features,
        fit_df=fit_df,
        eval_ground_truth=eval_ground_truth,
        eval_histories=eval_histories,
    )

    experiment_config = {
        "name": "toy_experiment",
        "assignment": {"method": "hash_mod", "id_col": "user_id", "split": {"control": 50, "treatment": 50}},
        "variants": {
            "control": {"pipeline": "popularity_only", "candidate_k": 10},
            "treatment": {"pipeline": "popularity_plus_ranker", "candidate_k": 10},
        },
    }

    result = run_single_experiment(experiment_config, context=context)
    assert result["evaluation"]["ranker_training"] == "train_to_val"
    assert result["evaluation"]["evaluation_split"] == "test"
