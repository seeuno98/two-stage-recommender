"""Run offline control-vs-treatment recommendation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.experiment import (
    build_experiment_context,
    get_experiment_by_name,
    load_experiment_config,
    run_single_experiment,
)
from src.eval.reporting import (
    save_experiment_csv,
    save_experiment_json,
    save_experiment_summary,
)


DEFAULT_CONFIG_PATH = Path("configs/experiments.yaml")
EXPERIMENT_REPORTS_DIR = Path("artifacts/reports/experiments")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run offline recommendation experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment name to run from configs/experiments.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment config YAML file.",
    )
    return parser.parse_args()


def print_experiment_logs(result: dict[str, object], output_dir: Path) -> None:
    """Print concise console logs for one experiment."""

    print(f"[experiment] name={result['name']}")
    assignment = result.get("assignment", {})
    if isinstance(assignment, dict):
        assigned_users = assignment.get("assigned_users", {})
        if isinstance(assigned_users, dict):
            for variant_name, user_count in assigned_users.items():
                print(f"[experiment] users_{variant_name}={user_count}")

    variants = result.get("variants", {})
    if isinstance(variants, dict):
        for variant_name, payload in variants.items():
            if not isinstance(payload, dict):
                continue
            metrics = payload.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            print(
                f"[experiment] {variant_name}"
                f" pipeline={payload.get('pipeline')}"
                f" Recall@10={float(metrics.get('recall@10', 0.0)):.4f}"
                f" NDCG@10={float(metrics.get('ndcg@10', 0.0)):.4f}"
            )

    comparison = result.get("comparison", {})
    if isinstance(comparison, dict):
        lift_vs_control = comparison.get("lift_vs_control", {})
        if isinstance(lift_vs_control, dict):
            for variant_name, payload in lift_vs_control.items():
                if not isinstance(payload, dict):
                    continue
                recall_payload = payload.get("recall@10", {})
                if isinstance(recall_payload, dict):
                    print(
                        f"[experiment] lift {variant_name}"
                        f" Recall@10_abs={float(recall_payload.get('absolute_diff', 0.0)):.4f}"
                    )

    print(f"[experiment] saved_results={output_dir}")


def main() -> None:
    """Run one or more offline experiments and persist their reports."""

    args = parse_args()
    experiments = load_experiment_config(args.config)
    if args.experiment:
        experiments_to_run = [get_experiment_by_name(experiments, args.experiment)]
    else:
        experiments_to_run = experiments

    context = build_experiment_context()
    for experiment in experiments_to_run:
        result = run_single_experiment(experiment, context=context)
        output_dir = EXPERIMENT_REPORTS_DIR / str(result["name"])
        save_experiment_json(result, output_dir)
        save_experiment_csv(result, output_dir)
        save_experiment_summary(result, output_dir)
        print_experiment_logs(result, output_dir)


if __name__ == "__main__":
    main()
