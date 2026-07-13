"""Profile traffic datasets for statistical graph-fusion experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data import load_wide_traffic_csv, read_csv_with_fallback  # noqa: E402
from run_experiments import DATASETS, datasets_for_suite, parse_csv_list  # noqa: E402


def finite_summary(values: np.ndarray) -> dict[str, float]:
    """Return robust scalar summaries for finite values."""
    flat = values[np.isfinite(values)].astype(np.float64)
    if flat.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "skew": math.nan,
            "kurtosis": math.nan,
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
        }
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    centered = flat - mean
    if std < 1e-12:
        skew = 0.0
        kurtosis = 0.0
    else:
        z = centered / std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3.0)
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }


def split_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same time-ordered 60/20/20 split convention as train.py."""
    n = values.shape[0]
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def correlation_profile(train_traffic: np.ndarray, top_k: int = 5) -> dict[str, float]:
    """Summarize node-to-node correlation structure on the training split."""
    series = train_traffic[:, :, 0]
    if series.shape[1] < 2:
        return {
            "corr_mean_abs_offdiag": math.nan,
            "corr_mean_positive_offdiag": math.nan,
            "corr_topk_mean": math.nan,
            "corr_topk_p95": math.nan,
        }
    corr = np.corrcoef(series.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    offdiag = corr[mask]
    positive = np.maximum(offdiag, 0.0)
    top_values = []
    k = min(max(int(top_k), 1), max(n - 1, 1))
    for row in np.maximum(corr, 0.0):
        row = row.copy()
        row[np.argmax(row)] = 0.0
        top_values.extend(np.sort(row)[-k:])
    top_values = np.asarray(top_values, dtype=np.float64)
    return {
        "corr_mean_abs_offdiag": float(np.mean(np.abs(offdiag))),
        "corr_mean_positive_offdiag": float(np.mean(positive)),
        "corr_topk_mean": float(np.mean(top_values)) if top_values.size else math.nan,
        "corr_topk_p95": float(np.quantile(top_values, 0.95)) if top_values.size else math.nan,
    }


def shift_summary(train: np.ndarray, other: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize distribution shift between train and another split."""
    train_flat = train[..., 0].reshape(-1).astype(np.float64)
    other_flat = other[..., 0].reshape(-1).astype(np.float64)
    train_flat = train_flat[np.isfinite(train_flat)]
    other_flat = other_flat[np.isfinite(other_flat)]
    if train_flat.size == 0 or other_flat.size == 0:
        return {
            f"{prefix}_mean_shift_z": math.nan,
            f"{prefix}_std_ratio": math.nan,
            f"{prefix}_median_shift_z": math.nan,
        }
    train_mean = float(np.mean(train_flat))
    train_std = float(np.std(train_flat))
    if train_std < 1e-12:
        train_std = 1.0
    return {
        f"{prefix}_mean_shift_z": float((np.mean(other_flat) - train_mean) / train_std),
        f"{prefix}_std_ratio": float(np.std(other_flat) / train_std),
        f"{prefix}_median_shift_z": float((np.median(other_flat) - np.median(train_flat)) / train_std),
    }


def missing_rate(dataset_name: str, max_nodes: int | None) -> float:
    """Estimate missing rate in the primary traffic columns before imputation."""
    cfg = DATASETS[dataset_name]
    df = read_csv_with_fallback(cfg.csv_path)
    value_suffix = cfg.value_suffix
    cols = []
    for col in df.columns:
        if col == cfg.time_col:
            continue
        if value_suffix and not str(col).endswith(value_suffix):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if max_nodes is not None:
        cols = cols[:max_nodes]
    if not cols:
        return math.nan
    return float(df[cols].isna().to_numpy().mean())


def profile_dataset(dataset_name: str, max_nodes: int | None) -> dict[str, object]:
    """Load and profile one dataset."""
    cfg = DATASETS[dataset_name]
    data, node_names, feature_names = load_wide_traffic_csv(
        cfg.csv_path,
        value_suffix=cfg.value_suffix,
        time_col=cfg.time_col,
        max_nodes=max_nodes,
        extra_feature_cols=parse_csv_list(cfg.extra_feature_cols),
        node_feature_suffixes=parse_csv_list(cfg.node_feature_suffixes),
        add_time_features=cfg.add_time_features,
        exclude_cols=["ID", "id"],
        return_feature_names=True,
    )
    train, val, test = split_arrays(data)
    traffic = data[..., :1]
    train_traffic = train[..., :1]
    profile: dict[str, object] = {
        "dataset": dataset_name,
        "csv_path": cfg.csv_path,
        "time_length": int(data.shape[0]),
        "num_nodes": int(data.shape[1]),
        "num_features": int(data.shape[2]),
        "feature_names": feature_names,
        "event_col": cfg.event_col,
        "missing_rate": missing_rate(dataset_name, max_nodes),
    }
    profile.update({f"traffic_{k}": v for k, v in finite_summary(traffic).items()})
    profile.update({f"train_traffic_{k}": v for k, v in finite_summary(train_traffic).items()})
    profile.update(correlation_profile(train_traffic))
    profile.update(shift_summary(train, val, "val"))
    profile.update(shift_summary(train, test, "test"))

    if cfg.event_col and cfg.event_col in feature_names:
        event_idx = feature_names.index(cfg.event_col)
        event_values = data[..., event_idx]
        event_summary = finite_summary(event_values)
        profile.update({f"event_{k}": v for k, v in event_summary.items()})
        threshold = max(1e-8, float(np.quantile(event_values, 0.80)))
        profile["event_period_ratio"] = float(np.mean(event_values > threshold))
        profile["event_threshold_p80"] = threshold
    else:
        profile["event_period_ratio"] = math.nan
        profile["event_threshold_p80"] = math.nan
    return profile


def write_report(profiles: list[dict[str, object]], output_dir: Path) -> None:
    """Write a compact Markdown distribution report."""
    lines = [
        "# Dataset Shift Report",
        "",
        "This report profiles traffic datasets before statistical graph-prior experiments.",
        "",
        "| dataset | type | T | N | mean | std | skew | kurtosis | mean abs corr | val shift z | test shift z | event ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    event_names = {"rainstorm", "bridge", "typhoon"}
    for p in profiles:
        name = str(p["dataset"])
        dtype = "event" if name in event_names else "normal"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    dtype,
                    str(p["time_length"]),
                    str(p["num_nodes"]),
                    f"{float(p['traffic_mean']):.4f}",
                    f"{float(p['traffic_std']):.4f}",
                    f"{float(p['traffic_skew']):.3f}",
                    f"{float(p['traffic_kurtosis']):.3f}",
                    f"{float(p['corr_mean_abs_offdiag']):.3f}",
                    f"{float(p['val_mean_shift_z']):.3f}",
                    f"{float(p['test_mean_shift_z']):.3f}",
                    f"{float(p['event_period_ratio']):.3f}"
                    if np.isfinite(float(p["event_period_ratio"]))
                    else "NA",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation checklist:",
            "",
            "- Larger absolute train/test shift suggests stronger distribution mismatch.",
            "- Higher skew/kurtosis suggests heavier-tailed traffic states.",
            "- Event datasets should be interpreted separately from normal PEMS datasets.",
            "- Typhoon is included only when a configured CSV is available.",
        ]
    )
    (output_dir / "dataset_shift_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all_quick")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\dataset_profile")
    args = parser.parse_args()

    if args.datasets.strip():
        dataset_names = parse_csv_list(args.datasets)
    else:
        dataset_names = datasets_for_suite(args.suite)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = [profile_dataset(name, args.max_nodes) for name in dataset_names]
    df = pd.DataFrame(profiles)
    df.to_csv(output_dir / "dataset_profile.csv", index=False, encoding="utf-8-sig")
    (output_dir / "dataset_profile.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(profiles, output_dir)
    print(f"Dataset profiles saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
