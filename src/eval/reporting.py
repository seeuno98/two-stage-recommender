"""Reporting helpers for offline recommendation experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_dataframe, save_json, save_text


def flatten_experiment_results(result: dict[str, object]) -> pd.DataFrame:
    """Flatten nested experiment results into a one-row-per-variant dataframe."""

    comparison = result.get("comparison", {})
    lift_vs_control = comparison.get("lift_vs_control", {}) if isinstance(comparison, dict) else {}
    rows: list[dict[str, object]] = []

    variants = result.get("variants", {})
    if not isinstance(variants, dict):
        return pd.DataFrame()

    for variant_name, payload in variants.items():
        if not isinstance(payload, dict):
            continue

        row: dict[str, object] = {
            "experiment_name": result.get("name"),
            "variant": variant_name,
            "pipeline": payload.get("pipeline"),
            "candidate_k": payload.get("candidate_k"),
            "user_count": payload.get("user_count"),
        }

        metrics = payload.get("metrics", {})
        if isinstance(metrics, dict):
            row.update(metrics)

        variant_lift = lift_vs_control.get(variant_name, {})
        if isinstance(variant_lift, dict):
            for metric_name, metric_payload in variant_lift.items():
                if not isinstance(metric_payload, dict):
                    continue
                row[f"{metric_name}_absolute_diff"] = metric_payload.get("absolute_diff")
                row[f"{metric_name}_relative_diff"] = metric_payload.get("relative_diff")

        rows.append(row)

    return pd.DataFrame(rows)


def save_experiment_json(result: dict[str, object], output_dir: str | Path) -> Path:
    """Save the full experiment payload as JSON."""

    output_path = Path(output_dir) / "results.json"
    save_json(result, output_path)
    return output_path


def save_experiment_csv(result: dict[str, object], output_dir: str | Path) -> Path:
    """Save flattened variant metrics as CSV."""

    output_path = Path(output_dir) / "variant_metrics.csv"
    save_dataframe(flatten_experiment_results(result), output_path)
    return output_path


def summarize_experiment_to_console(result: dict[str, object]) -> str:
    """Render a concise human-readable summary of experiment outcomes."""

    lines = [
        f"experiment: {result.get('name', 'unknown')}",
        f"description: {result.get('description', '')}",
    ]

    variants = result.get("variants", {})
    if isinstance(variants, dict):
        for variant_name, payload in variants.items():
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"{variant_name}: pipeline={payload.get('pipeline')} "
                f"candidate_k={payload.get('candidate_k')} "
                f"users={payload.get('user_count')}"
            )
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                for metric_name in sorted(metrics):
                    lines.append(f"  {metric_name}={float(metrics[metric_name]):.4f}")

    comparison = result.get("comparison", {})
    if isinstance(comparison, dict):
        lift_vs_control = comparison.get("lift_vs_control", {})
        if isinstance(lift_vs_control, dict):
            lines.append("lift_vs_control:")
            for variant_name, metric_payloads in lift_vs_control.items():
                lines.append(f"  {variant_name}:")
                if not isinstance(metric_payloads, dict):
                    continue
                for metric_name in sorted(metric_payloads):
                    payload = metric_payloads[metric_name]
                    if not isinstance(payload, dict):
                        continue
                    absolute_diff = payload.get("absolute_diff")
                    relative_diff = payload.get("relative_diff")
                    relative_text = (
                        "n/a" if relative_diff is None else f"{float(relative_diff):.4f}"
                    )
                    lines.append(
                        f"    {metric_name}: absolute_diff={float(absolute_diff):.4f} "
                        f"relative_diff={relative_text}"
                    )

    return "\n".join(lines)


def save_experiment_summary(result: dict[str, object], output_dir: str | Path) -> Path:
    """Save a plain-text summary next to the structured experiment outputs."""

    output_path = Path(output_dir) / "summary.txt"
    save_text(summarize_experiment_to_console(result), output_path)
    return output_path
