"""Diagnose auto_v2 shift calibration decisions from saved run_config files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PEMS_DATASETS = {"pems04", "pems08"}
SUMMARY_FILENAME = "diagnose_shift_profile_summary.csv"
SEED_RE = re.compile(r"seed(?P<seed>\d+)", re.IGNORECASE)


def normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("run_config.json must contain a JSON object")
    return data


def path_text(config: dict[str, Any], run_config_path: Path) -> str:
    parts = [
        str(run_config_path),
        str(run_config_path.parent),
        str(config.get("output_dir", "")),
        str(config.get("csv", "")),
    ]
    return " ".join(parts).lower()


def infer_dataset(config: dict[str, Any], run_config_path: Path) -> str:
    for key in ("dataset", "dataset_name", "data_name"):
        value = normalize_text(config.get(key))
        if value:
            return value
    text = path_text(config, run_config_path)
    for dataset in ("pems04", "pems08", "rainstorm", "typhoon", "bridge"):
        if dataset in text:
            return dataset
    csv_path = normalize_text(config.get("csv"))
    if csv_path:
        return Path(csv_path).stem.lower()
    return ""


def infer_seed(config: dict[str, Any], run_config_path: Path) -> str:
    if config.get("seed") is not None:
        return str(config["seed"])
    match = SEED_RE.search(path_text(config, run_config_path))
    return match.group("seed") if match else ""


def is_quality_stat_shift_auto_v2(config: dict[str, Any], run_config_path: Path) -> bool:
    model_keys = (
        "model",
        "model_name",
        "experiment_model",
        "candidate",
        "model_candidate",
    )
    for key in model_keys:
        if normalize_text(config.get(key)) == "quality_stat_shift_auto_v2":
            return True

    if "quality_stat_shift_auto_v2" in path_text(config, run_config_path):
        return True

    return (
        normalize_text(config.get("fusion_type")) == "quality_stat"
        and normalize_text(config.get("stat_feature_mode")) == "shift"
        and normalize_text(config.get("shift_calibration_mode")) == "auto_v2"
    )


def profile_value(profile: dict[str, Any], key: str) -> object:
    return profile.get(key, "")


def collect_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_config_path in sorted(output_root.rglob("run_config.json")):
        try:
            config = read_json(run_config_path)
        except Exception as exc:
            print(f"[WARNING] Failed to read {run_config_path}: {exc}")
            continue

        if not is_quality_stat_shift_auto_v2(config, run_config_path):
            continue

        shift_profile = config.get("train_shift_profile") or config.get("shift_profile") or {}
        if not isinstance(shift_profile, dict):
            shift_profile = {}

        rows.append(
            {
                "dataset": infer_dataset(config, run_config_path),
                "seed": infer_seed(config, run_config_path),
                "selected_shift_calibration": config.get(
                    "selected_shift_calibration", ""
                ),
                "cv": profile_value(shift_profile, "cv"),
                "p90_p50_ratio": profile_value(shift_profile, "p90_p50_ratio"),
                "mean": profile_value(shift_profile, "mean"),
                "p50": profile_value(shift_profile, "p50"),
                "p90": profile_value(shift_profile, "p90"),
                "run_config_path": str(run_config_path),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "selected_shift_calibration",
        "cv",
        "p90_p50_ratio",
        "mean",
        "p50",
        "p90",
        "run_config_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No quality_stat_shift_auto_v2 run_config.json files found.")
        return

    print(
        "dataset, seed, selected_shift_calibration, cv, "
        "p90_p50_ratio, mean, p50, p90"
    )
    for row in rows:
        print(
            f"{row['dataset']}, {row['seed']}, "
            f"{row['selected_shift_calibration']}, {row['cv']}, "
            f"{row['p90_p50_ratio']}, {row['mean']}, "
            f"{row['p50']}, {row['p90']}"
        )


def print_warnings(rows: list[dict[str, object]]) -> None:
    for row in rows:
        dataset = normalize_text(row.get("dataset"))
        selected = normalize_text(row.get("selected_shift_calibration"))
        if dataset in PEMS_DATASETS and selected != "disabled":
            print(
                "[WARNING] "
                f"{dataset.upper()} seed={row.get('seed', '')} selected "
                f"shift calibration is {selected or '<missing>'}, not disabled: "
                f"{row.get('run_config_path', '')}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose quality_stat_shift_auto_v2 shift profiles from experiment "
            "run_config.json files."
        )
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Experiment output directory, e.g. "
            r"D:\TrafficGNN\outputs\experiments\thesis_normal_seeds"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"--output-root does not exist: {output_root}")

    rows = collect_rows(output_root)
    print_rows(rows)
    print_warnings(rows)

    summary_path = output_root / SUMMARY_FILENAME
    write_summary(summary_path, rows)
    print(f"Wrote summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
