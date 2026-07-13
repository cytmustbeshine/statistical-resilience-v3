"""Analyze STSGCN-style and DGCN-style public baseline experiments.

The reported baselines are implemented under the same data preprocessing,
splitting, scaling, and training pipeline used by the thesis models. They are
paper-inspired baselines, not full source-level reproductions of the original
STSGCN or DGCN repositories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENT_DATASETS = {"rainstorm", "typhoon", "bridge"}
NORMAL_DATASETS = {"pems04", "pems08"}
BASELINE_MODELS = ["stsgcn", "dgcn"]
METRIC_COLS = ["mae", "rmse", "smape", "wape", "h1_mae", "h3_mae", "h6_mae", "h12_mae"]


def dataset_group(dataset: str) -> str:
    dataset = str(dataset).lower()
    if dataset in EVENT_DATASETS:
        return "event"
    if dataset in NORMAL_DATASETS:
        return "normal"
    return "other"


def read_summaries(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("At least one summary CSV is required.")
    df = pd.concat(frames, ignore_index=True)
    if "canonical_model_name" in df.columns:
        df["model"] = df["canonical_model_name"].fillna(df["model"])
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"].isin(BASELINE_MODELS)].copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.startswith("ok")].copy()
    if df.empty:
        raise ValueError("No completed stsgcn/dgcn baseline rows were found.")
    df["group"] = df["dataset"].map(dataset_group)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in METRIC_COLS if col in df.columns and df[col].notna().any()]


def aggregate(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_metrics = (
        df.groupby(["group", "dataset", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_std": (metric, "std") for metric in metrics},
        )
        .sort_values(["group", "dataset", "model"])
    )
    dataset_means = dataset_metrics[[*["group", "dataset", "model"], *[f"{m}_mean" for m in metrics]]]
    group_metrics = (
        dataset_means.groupby(["group", "model"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            **{f"{metric}_mean": (f"{metric}_mean", "mean") for metric in metrics},
        )
        .sort_values(["group", "model"])
    )
    return dataset_metrics, group_metrics


def relative_improvement(dataset_metrics: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (group, dataset), sub in dataset_metrics.groupby(["group", "dataset"]):
        values = sub.set_index("model")
        if not {"stsgcn", "dgcn"}.issubset(values.index):
            continue
        row: dict[str, object] = {"group": group, "dataset": dataset}
        for metric in metrics:
            stsgcn = float(values.loc["stsgcn", f"{metric}_mean"])
            dgcn = float(values.loc["dgcn", f"{metric}_mean"])
            row[f"dgcn_vs_stsgcn_{metric}_improvement_pct"] = (
                (stsgcn - dgcn) / stsgcn * 100.0 if stsgcn != 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "dataset"]) if rows else pd.DataFrame()


def format_float(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(format_float(value, digits) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    dataset_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    improvements: pd.DataFrame,
    metrics: list[str],
) -> None:
    lines = [
        "# Public Baseline Analysis",
        "",
        "## Scope",
        "- STSGCN baseline: STSGCN-style static graph baseline under the unified D-STSGCN data pipeline.",
        "- DGCN baseline: DGCN-style dynamic graph baseline under the unified D-STSGCN data pipeline.",
        "- These are not claimed as full source-level reproductions of the original papers.",
        "",
        "## Group Mean Metrics",
    ]
    group_cols = ["group", "model", "datasets", *[f"{metric}_mean" for metric in metrics]]
    lines.append(markdown_table(group_metrics, group_cols))
    lines.extend(["", "## Dataset Mean Metrics"])
    dataset_cols = ["group", "dataset", "model", "seeds", *[f"{metric}_mean" for metric in metrics]]
    lines.append(markdown_table(dataset_metrics, dataset_cols))
    if not improvements.empty:
        lines.extend(["", "## DGCN-Style Improvement Over STSGCN-Style"])
        improvement_cols = ["group", "dataset", *[f"dgcn_vs_stsgcn_{metric}_improvement_pct" for metric in metrics]]
        lines.append(markdown_table(improvements, improvement_cols))
    lines.extend(
        [
            "",
            "## Thesis Interpretation",
            "- If DGCN-style is better than STSGCN-style, the dynamic graph learner contributes useful adaptive spatial structure.",
            "- The proposed model must be compared against both baselines to show that it is not only inheriting gains from one published component.",
        ]
    )
    (output_dir / "public_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", action="append", default=[])
    parser.add_argument("--event-summary", default="")
    parser.add_argument("--normal-summary", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary_paths = list(args.summary_csv)
    if args.event_summary:
        summary_paths.append(args.event_summary)
    if args.normal_summary:
        summary_paths.append(args.normal_summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_summaries(summary_paths)
    metrics = metric_columns(df)
    dataset_metrics, group_metrics = aggregate(df, metrics)
    improvements = relative_improvement(dataset_metrics, metrics)

    dataset_metrics.to_csv(output_dir / "public_baseline_dataset_metrics.csv", index=False, encoding="utf-8-sig")
    group_metrics.to_csv(output_dir / "public_baseline_group_metrics.csv", index=False, encoding="utf-8-sig")
    improvements.to_csv(output_dir / "public_baseline_relative_improvement.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, dataset_metrics, group_metrics, improvements, metrics)

    print(f"Public baseline report saved to: {output_dir / 'public_baseline_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
