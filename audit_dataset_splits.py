"""Audit dataset time order and event split structure for D-STSGCN experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import read_csv_with_fallback


@dataclass(frozen=True)
class AuditDataset:
    name: str
    csv_path: str
    time_col: str
    value_suffix: str | None
    event_col: str | None = None


DATASETS = [
    AuditDataset(
        "rainstorm",
        r"D:\TrafficGNN\data\rainstorm_traffic_state.csv",
        "Time",
        "_volume",
        "precip_accum_one_hour",
    ),
    AuditDataset(
        "typhoon",
        r"D:\TrafficGNN\data\typhoon_traffic_state_standard.csv",
        "Time",
        "_volume",
        "typhoon_intensity",
    ),
    AuditDataset(
        "bridge",
        r"D:\TrafficGNN\data\bridge_collapse_flow.csv",
        "Time",
        None,
        None,
    ),
    AuditDataset(
        "pems04",
        r"D:\TrafficGNN\data\public\PEMS04\pems04.csv",
        "Time",
        "_volume",
        None,
    ),
    AuditDataset(
        "pems08",
        r"D:\TrafficGNN\data\public\PEMS08\pems08.csv",
        "Time",
        "_volume",
        None,
    ),
]


def sort_by_time(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, bool, float]:
    if time_col not in df.columns:
        return df.reset_index(drop=True), False, 0.0
    parsed = pd.to_datetime(df[time_col], errors="coerce")
    parse_ratio = float(parsed.notna().mean())
    if parse_ratio >= 0.95:
        sorted_df = (
            df.assign(__parsed_time_for_sort=parsed)
            .sort_values("__parsed_time_for_sort")
            .drop(columns="__parsed_time_for_sort")
            .reset_index(drop=True)
        )
        return sorted_df, True, parse_ratio
    return df.sort_values(time_col).reset_index(drop=True), False, parse_ratio


def target_columns(df: pd.DataFrame, dataset: AuditDataset) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == dataset.time_col or str(col).lower() == "id":
            continue
        if dataset.value_suffix and not str(col).endswith(dataset.value_suffix):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def event_window_values(
    event_series: np.ndarray,
    history: int,
    num_windows: int,
) -> np.ndarray:
    if num_windows <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [event_series[min(idx + history - 1, len(event_series) - 1)] for idx in range(num_windows)],
        dtype=np.float64,
    )


def audit_one(
    dataset: AuditDataset,
    history: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    event_threshold: float,
) -> dict[str, object]:
    df_raw = read_csv_with_fallback(dataset.csv_path)
    df, used_datetime_sort, parse_ratio = sort_by_time(df_raw, dataset.time_col)
    time_values = pd.to_datetime(df[dataset.time_col], errors="coerce") if dataset.time_col in df else None
    monotonic = bool(time_values.is_monotonic_increasing) if time_values is not None else False
    time_start = str(df[dataset.time_col].iloc[0]) if dataset.time_col in df and len(df) else ""
    time_end = str(df[dataset.time_col].iloc[-1]) if dataset.time_col in df and len(df) else ""
    nodes = target_columns(df, dataset)
    num_windows = max(0, len(df) - history - horizon + 1)
    train_end = int(num_windows * train_ratio)
    val_end = int(num_windows * (train_ratio + val_ratio))
    train_windows = train_end
    val_windows = max(0, val_end - train_end)
    test_windows = max(0, num_windows - val_end)

    train_event_windows = ""
    val_event_windows = ""
    test_event_windows = ""
    total_event_windows = ""
    ood_structure = ""
    event_max = ""
    if dataset.event_col and dataset.event_col in df.columns:
        event_series = pd.to_numeric(df[dataset.event_col], errors="coerce").fillna(0.0).to_numpy()
        event_values = event_window_values(event_series, history, num_windows)
        train_events = int(np.sum(event_values[:train_end] > event_threshold))
        val_events = int(np.sum(event_values[train_end:val_end] > event_threshold))
        test_events = int(np.sum(event_values[val_end:] > event_threshold))
        train_event_windows = train_events
        val_event_windows = val_events
        test_event_windows = test_events
        total_event_windows = int(np.sum(event_values > event_threshold))
        event_max = float(np.nanmax(event_series)) if len(event_series) else 0.0
        ood_structure = bool(train_events == 0 and test_events > 0)

    return {
        "dataset": dataset.name,
        "csv_path": dataset.csv_path,
        "rows": int(len(df)),
        "nodes": int(len(nodes)),
        "time_start": time_start,
        "time_end": time_end,
        "time_parse_ratio": parse_ratio,
        "used_datetime_sort": used_datetime_sort,
        "time_monotonic_after_sort": monotonic,
        "history": history,
        "horizon": horizon,
        "num_windows": int(num_windows),
        "train_windows": int(train_windows),
        "val_windows": int(val_windows),
        "test_windows": int(test_windows),
        "event_col": dataset.event_col or "",
        "event_threshold": event_threshold if dataset.event_col else "",
        "event_max": event_max,
        "train_event_windows": train_event_windows,
        "val_event_windows": val_event_windows,
        "test_event_windows": test_event_windows,
        "total_event_windows": total_event_windows,
        "train_no_event_test_has_event": ood_structure,
    }


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    headers = [
        "dataset",
        "rows",
        "nodes",
        "time_start",
        "time_end",
        "time_monotonic_after_sort",
        "event_col",
        "train_event_windows",
        "val_event_windows",
        "test_event_windows",
        "train_no_event_test_has_event",
    ]
    lines = [
        "# Dataset Split Audit",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "- Time is sorted with parsed datetimes when at least 95% of values parse successfully.",
            "- Event windows use the last observed step of each input history window.",
            "- `train_no_event_test_has_event=True` indicates an unknown-disturbance/OOD evaluation structure.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\dataset_audit")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--event-threshold", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        audit_one(
            dataset,
            history=args.history,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            event_threshold=args.event_threshold,
        )
        for dataset in DATASETS
    ]
    csv_path = output_dir / "dataset_split_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md_path = output_dir / "dataset_split_audit.md"
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
