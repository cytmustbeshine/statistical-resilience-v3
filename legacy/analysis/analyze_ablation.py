"""Analyze D-STSGCN thesis ablation experiment summaries."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ABLATION_MODELS = ["static", "quality_v2", "quality_stat_shift_auto_v2"]
FULL_MODEL = "quality_stat_shift_auto_v2"
MODEL_LABELS = {
    "static": "No dynamic",
    "quality_v2": "No shift",
    "quality_stat_shift_auto_v2": "Full auto\\_v2",
}
METRICS = ["mae", "rmse", "smape"]


def to_float(value: object) -> float:
    """Parse a value as float, returning NaN for blanks or invalid strings.

    Args:
        value: Raw CSV value.

    Returns:
        Parsed float or ``math.nan``.
    """
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
    """Read one run_experiments summary CSV.

    Args:
        path: Path to ``experiment_summary.csv``.

    Returns:
        Rows with ``status == ok``.
    """
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "ok"]


def summarize(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    """Summarize ablation metrics by dataset and model.

    Args:
        rows: Summary CSV rows.

    Returns:
        Mapping from ``(dataset, model)`` to mean/std metric statistics.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "")
        if model in ABLATION_MODELS:
            grouped[(row.get("dataset", ""), model)].append(row)

    stats: dict[tuple[str, str], dict[str, float]] = {}
    for key, group_rows in grouped.items():
        item: dict[str, float] = {"n_seeds": float(len(group_rows))}
        for metric in METRICS:
            values = [
                to_float(row.get(metric))
                for row in group_rows
                if math.isfinite(to_float(row.get(metric)))
            ]
            item[f"{metric}_mean"] = mean(values) if values else math.nan
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        stats[key] = item
    return stats


def build_csv_rows(
    stats: dict[tuple[str, str], dict[str, float]]
) -> list[dict[str, object]]:
    """Build CSV rows including full-model gains over ablation variants.

    Args:
        stats: Dataset/model summary statistics.

    Returns:
        CSV-ready rows with mean/std and absolute/relative gains.
    """
    rows: list[dict[str, object]] = []
    datasets = sorted({dataset for dataset, _ in stats})
    for dataset in datasets:
        full = stats.get((dataset, FULL_MODEL), {})
        for model in ABLATION_MODELS:
            item = stats.get((dataset, model), {})
            row: dict[str, object] = {
                "dataset": dataset,
                "model": model,
                "model_label": MODEL_LABELS[model].replace("\\_", "_"),
                "n_seeds": int(item.get("n_seeds", 0)),
            }
            for metric in METRICS:
                model_mean = item.get(f"{metric}_mean", math.nan)
                full_mean = full.get(f"{metric}_mean", math.nan)
                row[f"{metric}_mean"] = model_mean
                row[f"{metric}_std"] = item.get(f"{metric}_std", math.nan)
                abs_gain = (
                    model_mean - full_mean
                    if math.isfinite(model_mean) and math.isfinite(full_mean)
                    else math.nan
                )
                rel_gain = (
                    abs_gain / model_mean * 100.0
                    if math.isfinite(abs_gain) and model_mean > 0
                    else math.nan
                )
                row[f"full_over_model_{metric}_abs_gain"] = abs_gain
                row[f"full_over_model_{metric}_rel_gain_pct"] = rel_gain
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV.

    Args:
        path: Output CSV path.
        rows: Rows to write.

    Returns:
        None.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_cell(item: dict[str, float]) -> str:
    """Format one LaTeX table cell as MAE±std (RMSE±std).

    Args:
        item: Summary statistics for one dataset/model pair.

    Returns:
        LaTeX table cell text.
    """
    mae = item.get("mae_mean", math.nan)
    mae_std = item.get("mae_std", math.nan)
    rmse = item.get("rmse_mean", math.nan)
    rmse_std = item.get("rmse_std", math.nan)
    if not math.isfinite(mae) or not math.isfinite(rmse):
        return "--"
    return f"{mae:.4f}$\\pm${mae_std:.4f} ({rmse:.4f}$\\pm${rmse_std:.4f})"


def write_latex_table(
    path: Path,
    stats: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Write a thesis-ready LaTeX ablation table.

    Args:
        path: Output ``.tex`` path.
        stats: Dataset/model summary statistics.

    Returns:
        None.
    """
    datasets = sorted({dataset for dataset, _ in stats})
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation study of D-STSGCN components. Cells report MAE$\\pm$std (RMSE$\\pm$std).}",
        "\\label{tab:dstsgcn_ablation}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Dataset & No dynamic & No shift & Full auto\\_v2 \\\\",
        "\\hline",
    ]
    for dataset in datasets:
        row_items = {model: stats.get((dataset, model), {}) for model in ABLATION_MODELS}
        finite_mae = {
            model: item.get("mae_mean", math.nan)
            for model, item in row_items.items()
            if math.isfinite(item.get("mae_mean", math.nan))
        }
        best_model = min(finite_mae, key=finite_mae.get) if finite_mae else None
        cells = []
        for model in ABLATION_MODELS:
            cell = metric_cell(row_items[model])
            if model == best_model and cell != "--":
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{dataset} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weather-dir",
        default=r"D:\TrafficGNN\outputs\experiments\thesis_ablation_weather",
    )
    parser.add_argument(
        "--normal-dir",
        default=r"D:\TrafficGNN\outputs\experiments\thesis_ablation_normal",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\TrafficGNN\outputs\ablation_analysis",
    )
    return parser.parse_args()


def main() -> int:
    """Run ablation analysis.

    Args:
        None.

    Returns:
        Process exit code.
    """
    args = parse_args()
    weather_dir = Path(args.weather_dir)
    normal_dir = Path(args.normal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(read_summary(weather_dir / "experiment_summary.csv"))
    rows.extend(read_summary(normal_dir / "experiment_summary.csv"))
    stats = summarize(rows)
    csv_rows = build_csv_rows(stats)

    csv_path = output_dir / "ablation_table.csv"
    tex_path = output_dir / "ablation_table.tex"
    write_csv(csv_path, csv_rows)
    write_latex_table(tex_path, stats)
    print(f"Saved ablation CSV: {csv_path}")
    print(f"Saved ablation LaTeX table: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
