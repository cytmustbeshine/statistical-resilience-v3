"""Aggregate thesis-level D-STSGCN experiment results.

This script turns run_experiments.py summaries and weather slice metrics into
CSV tables and a compact Markdown report suitable for thesis drafting.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from statistical_tests import (
    format_test_result,
    holm_bonferroni_correction,
    wilcoxon_paired_test,
)


MAIN_METRICS = ["mae", "rmse", "smape", "wape"]
HORIZON_METRICS = ["h1_mae", "h3_mae", "h6_mae", "h12_mae"]
WEATHER_DATASETS = {"rainstorm", "typhoon"}
NORMAL_DATASETS = {"pems04", "pems08"}
KEY_SLICES = ["weather_nonzero", "weather_top10", "high_shift_top10"]


def to_float(value: object) -> float:
    """Parse a CSV value as float, returning NaN for blanks."""
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values = [to_float(row.get(metric)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def summarize_dataset_model(
    rows: list[dict[str, str]],
    base_model: str,
    proposed_model: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model") in {base_model, proposed_model}:
            grouped[(row["dataset"], row["model"])].append(row)

    stats: dict[tuple[str, str], dict[str, float]] = {}
    table_rows: list[dict[str, object]] = []
    for (dataset, model), group_rows in sorted(grouped.items()):
        item: dict[str, float] = {"n_seeds": float(len(group_rows))}
        output: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "n_seeds": len(group_rows),
        }
        for metric in MAIN_METRICS:
            values = finite_values(group_rows, metric)
            item[f"{metric}_mean"] = mean(values) if values else math.nan
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            output[f"{metric}_mean"] = item[f"{metric}_mean"]
            output[f"{metric}_std"] = item[f"{metric}_std"]
        stats[(dataset, model)] = item
        table_rows.append(output)

    for dataset in sorted({key[0] for key in grouped}):
        base = stats.get((dataset, base_model))
        proposed = stats.get((dataset, proposed_model))
        if not base or not proposed:
            continue
        comparison: dict[str, object] = {
            "dataset": dataset,
            "model": f"{proposed_model}_vs_{base_model}",
            "n_seeds": int(proposed.get("n_seeds", 0)),
        }
        for metric in MAIN_METRICS:
            base_value = base.get(f"{metric}_mean", math.nan)
            proposed_value = proposed.get(f"{metric}_mean", math.nan)
            improvement = (
                (base_value - proposed_value) / base_value * 100.0
                if math.isfinite(base_value) and base_value > 0
                else math.nan
            )
            comparison[f"{metric}_mean"] = proposed_value
            comparison[f"{metric}_std"] = proposed.get(f"{metric}_std", math.nan)
            comparison[f"{metric}_rel_improve_pct"] = improvement
        table_rows.append(comparison)
    return table_rows, stats


def group_for_dataset(dataset: str) -> str:
    if dataset in WEATHER_DATASETS:
        return "weather"
    if dataset in NORMAL_DATASETS:
        return "normal"
    return "other"


def build_group_rows(
    stats: dict[tuple[str, str], dict[str, float]],
    base_model: str,
    proposed_model: str,
) -> list[dict[str, object]]:
    datasets = sorted({dataset for dataset, _ in stats})
    groups = {
        "weather": [dataset for dataset in datasets if group_for_dataset(dataset) == "weather"],
        "normal": [dataset for dataset in datasets if group_for_dataset(dataset) == "normal"],
        "all": datasets,
    }
    rows: list[dict[str, object]] = []
    for group_name, group_datasets in groups.items():
        if not group_datasets:
            continue
        row: dict[str, object] = {
            "group": group_name,
            "datasets": ",".join(group_datasets),
            "n_datasets": len(group_datasets),
        }
        for metric in MAIN_METRICS:
            base_values = [
                stats[(dataset, base_model)][f"{metric}_mean"]
                for dataset in group_datasets
                if (dataset, base_model) in stats
            ]
            proposed_values = [
                stats[(dataset, proposed_model)][f"{metric}_mean"]
                for dataset in group_datasets
                if (dataset, proposed_model) in stats
            ]
            base_values = [value for value in base_values if math.isfinite(value)]
            proposed_values = [value for value in proposed_values if math.isfinite(value)]
            base_mean = mean(base_values) if base_values else math.nan
            proposed_mean = mean(proposed_values) if proposed_values else math.nan
            row[f"base_{metric}"] = base_mean
            row[f"proposed_{metric}"] = proposed_mean
            row[f"{metric}_rel_improve_pct"] = (
                (base_mean - proposed_mean) / base_mean * 100.0
                if math.isfinite(base_mean) and base_mean > 0
                else math.nan
            )
        rows.append(row)
    return rows


def build_horizon_rows(
    rows: list[dict[str, str]],
    base_model: str,
    proposed_model: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("model") in {base_model, proposed_model}:
            grouped[(row["dataset"], row["model"])].append(row)
    out: list[dict[str, object]] = []
    for dataset in sorted({key[0] for key in grouped}):
        base_rows = grouped.get((dataset, base_model), [])
        proposed_rows = grouped.get((dataset, proposed_model), [])
        for horizon in HORIZON_METRICS:
            base_values = finite_values(base_rows, horizon)
            proposed_values = finite_values(proposed_rows, horizon)
            base_mean = mean(base_values) if base_values else math.nan
            proposed_mean = mean(proposed_values) if proposed_values else math.nan
            out.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "base_mean": base_mean,
                    "proposed_mean": proposed_mean,
                    "rel_improve_pct": (
                        (base_mean - proposed_mean) / base_mean * 100.0
                        if math.isfinite(base_mean) and base_mean > 0
                        else math.nan
                    ),
                }
            )
    return out


def read_slice_metrics(slice_report_dir: Path) -> list[dict[str, str]]:
    path = slice_report_dir / "weather_shift_slice_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def slice_summary(slice_rows: list[dict[str, str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for slice_name in KEY_SLICES:
        values = [
            to_float(row.get("rel_mae_improve_pct"))
            for row in slice_rows
            if row.get("slice") == slice_name and to_float(row.get("count")) > 0
        ]
        values = [value for value in values if math.isfinite(value)]
        result[slice_name] = mean(values) if values else math.nan
    for dataset in WEATHER_DATASETS:
        for slice_name in KEY_SLICES:
            values = [
                to_float(row.get("rel_mae_improve_pct"))
                for row in slice_rows
                if row.get("dataset") == dataset
                and row.get("slice") == slice_name
                and to_float(row.get("count")) > 0
            ]
            values = [value for value in values if math.isfinite(value)]
            result[f"{dataset}_{slice_name}"] = mean(values) if values else math.nan
    return result


def seed_stability(
    rows: list[dict[str, str]],
    base_model: str,
    proposed_model: str,
    datasets: set[str],
) -> tuple[int, int]:
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("dataset") in datasets and row.get("model") in {base_model, proposed_model}:
            grouped[(row["dataset"], int(row["seed"]))][row["model"]] = to_float(row["mae"])
    wins = 0
    total = 0
    for values in grouped.values():
        if base_model not in values or proposed_model not in values:
            continue
        total += 1
        if values[proposed_model] < values[base_model]:
            wins += 1
    return wins, total


def seed_level_tests(
    rows: list[dict[str, str]],
    base_model: str,
    proposed_model: str,
) -> tuple[list[dict[str, object]], str]:
    """Run paired Wilcoxon tests across seeds for each dataset and metric.

    Args:
        rows: Experiment summary rows with one row per dataset/model/seed.
        base_model: Baseline model name.
        proposed_model: Proposed model name.

    Returns:
        A tuple containing test rows and a note. The note explains when
        seed-level paired tests cannot be run because raw per-seed metrics are
        unavailable or too sparse.
    """
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    has_seed_column = all("seed" in row and str(row.get("seed", "")).strip() for row in rows)
    if not has_seed_column:
        return [], "Seed-level paired test requires raw per-seed metrics."

    for row in rows:
        if row.get("model") not in {base_model, proposed_model}:
            continue
        try:
            seed = int(row["seed"])
        except (KeyError, ValueError):
            continue
        grouped[(row["dataset"], seed)][row["model"]] = row

    test_rows: list[dict[str, object]] = []
    for dataset in sorted({dataset for dataset, _ in grouped}):
        paired = [
            values for (row_dataset, _), values in grouped.items()
            if row_dataset == dataset
            and base_model in values
            and proposed_model in values
        ]
        if len(paired) < 2:
            continue
        for metric in MAIN_METRICS:
            base_values = []
            proposed_values = []
            for values in paired:
                base_value = to_float(values[base_model].get(metric))
                proposed_value = to_float(values[proposed_model].get(metric))
                if math.isfinite(base_value) and math.isfinite(proposed_value):
                    base_values.append(base_value)
                    proposed_values.append(proposed_value)
            if len(base_values) < 2:
                continue
            result = wilcoxon_paired_test(
                np.asarray(proposed_values, dtype=float),
                np.asarray(base_values, dtype=float),
            )
            test_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": int(result["n_pairs"]),
                    "statistic": result["statistic"],
                    "p_value": result["p_value"],
                    "p_value_adj": math.nan,
                    "effect_size_r": result["effect_size_r"],
                    "median_delta": result["median_diff"],
                    "ci_lower_95": result["ci_lower_95"],
                    "ci_upper_95": result["ci_upper_95"],
                }
            )

    adjusted = holm_bonferroni_correction(
        [float(row["p_value"]) for row in test_rows]
    )
    for row, adjusted_p in zip(test_rows, adjusted):
        row["p_value_adj"] = adjusted_p

    note = (
        ""
        if test_rows
        else "Seed-level paired test requires raw per-seed metrics."
    )
    return test_rows, note


def make_report(
    output_dir: Path,
    main_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    horizon_rows: list[dict[str, object]],
    slice_stats: dict[str, float],
    significance_rows: list[dict[str, object]],
    significance_note: str,
    weather_wins: tuple[int, int],
    base_model: str,
    proposed_model: str,
) -> None:
    group_by_name = {row["group"]: row for row in group_rows}
    weather = group_by_name.get("weather", {})
    normal = group_by_name.get("normal", {})
    weather_mae = to_float(weather.get("mae_rel_improve_pct"))
    normal_mae = to_float(normal.get("mae_rel_improve_pct"))
    weather_rmse = to_float(weather.get("rmse_rel_improve_pct"))
    normal_rmse = to_float(normal.get("rmse_rel_improve_pct"))
    weather_wape = to_float(weather.get("wape_rel_improve_pct"))
    normal_wape = to_float(normal.get("wape_rel_improve_pct"))
    positive_key_slices = sum(
        1
        for key in KEY_SLICES
        if math.isfinite(slice_stats.get(key, math.nan)) and slice_stats[key] > 0
    )
    typhoon_rows = [
        row
        for row in main_rows
        if row.get("dataset") == "typhoon"
        and str(row.get("model", "")).startswith(f"{proposed_model}_vs_")
    ]
    typhoon_mae = to_float(typhoon_rows[0].get("mae_rel_improve_pct")) if typhoon_rows else math.nan
    rainstorm_slice = max(
        [
            slice_stats.get("rainstorm_weather_nonzero", math.nan),
            slice_stats.get("rainstorm_weather_top10", math.nan),
            slice_stats.get("rainstorm_high_shift_top10", math.nan),
        ],
        key=lambda value: value if math.isfinite(value) else -1e9,
    )
    wins, total = weather_wins
    accepted = (
        math.isfinite(weather_mae)
        and weather_mae > 0
        and (
            (math.isfinite(typhoon_mae) and typhoon_mae > 0)
            or (math.isfinite(rainstorm_slice) and rainstorm_slice > 0)
        )
        and positive_key_slices >= 2
        and (not math.isfinite(normal_mae) or normal_mae >= -0.5)
        and (not math.isfinite(weather_rmse) or weather_rmse > -0.5)
        and (not math.isfinite(weather_wape) or weather_wape > -0.5)
        and (not math.isfinite(normal_rmse) or normal_rmse > -0.5)
        and (not math.isfinite(normal_wape) or normal_wape > -0.5)
        and total > 0
        and wins >= max(1, math.ceil(total / 2))
    )
    lines = [
        "# Thesis Result Report",
        "",
        f"Base model: `{base_model}`",
        f"Proposed model: `{proposed_model}`",
        "",
        "## Group Summary",
        "",
        "| Group | Datasets | MAE Improve | RMSE Improve | WAPE Improve |",
        "|---|---|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {row['datasets']} | "
            f"{to_float(row.get('mae_rel_improve_pct')):+.2f}% | "
            f"{to_float(row.get('rmse_rel_improve_pct')):+.2f}% | "
            f"{to_float(row.get('wape_rel_improve_pct')):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Key Slice Summary",
            "",
            "| Slice | Mean MAE Improve |",
            "|---|---:|",
        ]
    )
    for key in KEY_SLICES:
        value = slice_stats.get(key, math.nan)
        lines.append(f"| {key} | {value:+.2f}% |")
    lines.extend(
        [
            "",
            "## Horizon Summary",
            "",
            "| Dataset | Horizon | Base MAE | Proposed MAE | Improve |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in horizon_rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | "
            f"{to_float(row.get('base_mean')):.4f} | "
            f"{to_float(row.get('proposed_mean')):.4f} | "
            f"{to_float(row.get('rel_improve_pct')):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Statistical Significance",
            "",
        ]
    )
    if significance_rows:
        lines.extend(
            [
                "Paired Wilcoxon signed-rank tests compare proposed minus baseline "
                "seed-level metrics. Negative median delta means the proposed model "
                "has lower error. Adjusted p-values use Holm-Bonferroni correction.",
                "",
                "| Dataset | Metric | N | Test Result |",
                "|---|---|---:|---|",
            ]
        )
        for row in significance_rows:
            test_result = {
                "statistic": row["statistic"],
                "p_value": row["p_value"],
                "p_value_adj": row["p_value_adj"],
                "effect_size_r": row["effect_size_r"],
                "median_diff": row["median_delta"],
                "ci_lower_95": row["ci_lower_95"],
                "ci_upper_95": row["ci_upper_95"],
            }
            lines.append(
                f"| {row['dataset']} | {row['metric']} | {row['n_pairs']} | "
                f"{format_test_result(test_result)} |"
            )
    else:
        lines.append(significance_note or "Seed-level paired test requires raw per-seed metrics.")
    lines.extend(
        [
            "",
            "## Acceptance Check",
            "",
            f"- Weather average MAE improvement: {weather_mae:+.2f}%",
            f"- Normal average MAE improvement: {normal_mae:+.2f}%",
            f"- Positive key slices: {positive_key_slices}/3",
            f"- Weather seed wins: {wins}/{total}",
            f"- Thesis method accepted: {'yes' if accepted else 'no'}",
            "",
        ]
    )
    if not accepted:
        lines.append(
            "Current evidence supports the model mainly as a weather-disturbance "
            "slice enhancement method, not yet as a universal all-scenario "
            "forecasting model."
        )
    else:
        lines.append(
            "The model satisfies the preliminary thesis acceptance criteria and "
            "can move to longer-epoch final experiments."
        )
    (output_dir / "thesis_result_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather-summary", required=True)
    parser.add_argument("--normal-summary", required=True)
    parser.add_argument("--slice-report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="quality_v2")
    parser.add_argument("--proposed-model", default="quality_stat_shift_auto_v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weather_rows = read_summary(Path(args.weather_summary))
    normal_rows = read_summary(Path(args.normal_summary))
    all_rows = weather_rows + normal_rows

    main_rows, stats = summarize_dataset_model(
        all_rows,
        args.base_model,
        args.proposed_model,
    )
    group_rows = build_group_rows(stats, args.base_model, args.proposed_model)
    horizon_rows = build_horizon_rows(all_rows, args.base_model, args.proposed_model)
    slice_rows = read_slice_metrics(Path(args.slice_report_dir))
    slices = slice_summary(slice_rows)
    significance_rows, significance_note = seed_level_tests(
        all_rows,
        args.base_model,
        args.proposed_model,
    )
    weather_wins = seed_stability(
        all_rows,
        args.base_model,
        args.proposed_model,
        WEATHER_DATASETS,
    )

    write_csv(output_dir / "thesis_main_metrics.csv", main_rows)
    write_csv(output_dir / "thesis_group_metrics.csv", group_rows)
    write_csv(output_dir / "thesis_horizon_metrics.csv", horizon_rows)
    write_csv(output_dir / "thesis_statistical_significance.csv", significance_rows)
    make_report(
        output_dir,
        main_rows,
        group_rows,
        horizon_rows,
        slices,
        significance_rows,
        significance_note,
        weather_wins,
        args.base_model,
        args.proposed_model,
    )
    print(f"Saved thesis report: {output_dir / 'thesis_result_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
