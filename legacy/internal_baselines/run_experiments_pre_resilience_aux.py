"""Run clean thesis experiment suites for D-STSGCN.

Supported models:
  - stsgcn
  - dgcn
  - quality_v2
  - quality_ood_scaled_v1
  - dcrnn_resilience

The public baseline suites intentionally include only STSGCN-style and
DGCN-style baselines. They are paper-inspired baselines under this unified
data/training framework, not full source-level reproductions of the papers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


TEST_RE = re.compile(
    r"Test MAE=(?P<mae>[0-9.]+), "
    r"RMSE=(?P<rmse>[0-9.]+), "
    r"SMAPE=(?P<smape>[0-9.]+)%, "
    r"WAPE=(?P<wape>[0-9.]+)%"
)
HORIZON_RE = re.compile(
    r"\|\s*h1=(?P<h1_mae>[0-9.]+)\s+"
    r"h3=(?P<h3_mae>[0-9.]+)\s+"
    r"h6=(?P<h6_mae>[0-9.]+)\s+"
    r"h12=(?P<h12_mae>[0-9.]+)"
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    csv_path: str
    time_col: str
    value_suffix: str | None
    extra_feature_cols: str = ""
    node_feature_suffixes: str = ""
    add_time_features: bool = True
    event_col: str = ""


DATASETS = {
    "rainstorm": DatasetConfig(
        name="rainstorm",
        csv_path=r"D:\TrafficGNN\data\rainstorm_traffic_state.csv",
        time_col="Time",
        value_suffix="_volume",
        extra_feature_cols=(
            "altimeter,air_temp,relative_humidity,wind_speed,"
            "precip_accum_one_hour,visibility"
        ),
        node_feature_suffixes="_speed",
        event_col="precip_accum_one_hour",
    ),
    "bridge": DatasetConfig(
        name="bridge",
        csv_path=r"D:\TrafficGNN\data\bridge_collapse_flow.csv",
        time_col="Time",
        value_suffix=None,
    ),
    "typhoon": DatasetConfig(
        name="typhoon",
        csv_path=r"D:\TrafficGNN\data\typhoon_traffic_state_standard.csv",
        time_col="Time",
        value_suffix="_volume",
        extra_feature_cols="typhoon_intensity",
        node_feature_suffixes="_speed",
        event_col="typhoon_intensity",
    ),
    "pems04": DatasetConfig(
        name="pems04",
        csv_path=r"D:\TrafficGNN\data\public\PEMS04\pems04.csv",
        time_col="Time",
        value_suffix="_volume",
        node_feature_suffixes="_occupancy,_speed",
    ),
    "pems08": DatasetConfig(
        name="pems08",
        csv_path=r"D:\TrafficGNN\data\public\PEMS08\pems08.csv",
        time_col="Time",
        value_suffix="_volume",
        node_feature_suffixes="_occupancy,_speed",
    ),
}

MODEL_ALIASES = {
    "stsgcn": "stsgcn",
    "stsgn": "stsgcn",
    "static": "stsgcn",
    "dgcn": "dgcn",
    "dcrnn_resilience": "dcrnn_resilience",
    "quality_v2": "quality_v2",
    "quality_ood_scaled_v1": "quality_ood_scaled_v1",
}
SUPPORTED_MODELS = set(MODEL_ALIASES)


def canonical_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(
            f"Unsupported model {model_name!r}. Use one of: "
            + ",".join(sorted(SUPPORTED_MODELS))
        )
    return MODEL_ALIASES[key]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def default_seeds_for_suite(suite: str, explicit: str) -> list[int]:
    if explicit.strip():
        return [int(seed) for seed in parse_csv_list(explicit)]
    if suite == "resilience_event_quick":
        return [42]
    if suite in {"resilience_event_seeds", "resilience_normal_seeds"}:
        return [42, 2024, 3407]
    if suite.endswith("_seeds"):
        return [42, 2024, 3407]
    return [42]


def datasets_for_suite(suite: str) -> list[str]:
    if suite in {"resilience_event_quick", "resilience_event_seeds"}:
        return ["bridge", "rainstorm", "typhoon"]
    if suite == "resilience_normal_seeds":
        return ["pems04", "pems08"]
    if suite in {
        "main_event_quick",
        "main_event_seeds",
        "public_baseline_event_quick",
        "public_baseline_event_seeds",
        "ood_event_quick",
        "ood_event_seeds",
        "event_quick",
        "event_seeds",
    }:
        return ["rainstorm", "typhoon", "bridge"]
    if suite in {
        "main_normal_quick",
        "main_normal_seeds",
        "public_baseline_normal_quick",
        "public_baseline_normal_seeds",
        "ood_normal_quick",
        "ood_normal_seeds",
        "normal_quick",
        "normal_seeds",
    }:
        return ["pems04", "pems08"]
    if suite in {
        "main_all_quick",
        "main_all_seeds",
        "public_baseline_all_quick",
        "public_baseline_all_seeds",
        "thesis_all_quick",
        "thesis_all_seeds",
        "all_quick",
        "all_seeds",
    }:
        return ["rainstorm", "bridge", "typhoon", "pems04", "pems08"]
    if suite in {"thesis_weather_quick", "thesis_weather_seeds", "weather_quick", "weather_seeds"}:
        return ["rainstorm", "typhoon"]
    if suite.startswith("rainstorm"):
        return ["rainstorm"]
    if suite.startswith("bridge"):
        return ["bridge"]
    if suite.startswith("typhoon"):
        return ["typhoon"]
    if suite.startswith("pems"):
        return ["pems04", "pems08"]
    raise ValueError(f"Unknown suite: {suite}")


def split_mode_for_dataset(args: argparse.Namespace, dataset_name: str) -> str:
    if args.split_mode is not None:
        return args.split_mode
    if dataset_name in {"rainstorm", "typhoon"}:
        return "event_aware"
    return "chronological"


def default_models_for_suite(suite: str) -> list[str]:
    if suite.startswith("resilience_"):
        return ["dcrnn_resilience"]
    if suite.startswith("public_baseline"):
        return ["stsgcn", "dgcn"]
    return ["stsgcn", "dgcn", "quality_v2", "quality_ood_scaled_v1"]


def model_paper_name(model_name: str) -> str:
    return {
        "stsgcn": "STSGCN-style static graph baseline",
        "dgcn": "DGCN-style dynamic graph baseline",
        "dcrnn_resilience": "DCRNN-Resilience public baseline",
        "quality_v2": "Static-dynamic quality fusion baseline",
        "quality_ood_scaled_v1": "OOD-scaled quality fusion extension",
    }[canonical_model_name(model_name)]


def model_implementation_scope(model_name: str) -> str:
    model_name = canonical_model_name(model_name)
    if model_name in {"stsgcn", "dgcn"}:
        return "paper-inspired unified-framework baseline"
    if model_name == "dcrnn_resilience":
        return "lightweight public resilience baseline re-implementation"
    return "internal D-STSGCN variant"


def build_command(
    args: argparse.Namespace,
    dataset: DatasetConfig,
    model_name: str,
    seed: int,
    output_dir: Path,
    split_mode: str,
) -> list[str]:
    requested_model_name = model_name
    model_name = canonical_model_name(model_name)

    if model_name == "dcrnn_resilience":
        cmd = [
            args.python,
            str(Path(args.code_dir) / "baselines" / "dcrnn_resilience" / "train.py"),
            "--dataset-name",
            dataset.name,
            "--csv",
            dataset.csv_path,
            "--time-col",
            dataset.time_col,
            "--value-suffix",
            dataset.value_suffix or "",
            "--max-nodes",
            str(args.max_nodes),
            "--history-steps",
            str(args.history),
            "--horizon",
            str(args.horizon),
            "--hidden-dim",
            str(args.hidden_dim),
            "--diffusion-steps",
            str(args.num_diffusion_steps),
            "--split-mode",
            split_mode,
            "--event-window-threshold",
            str(args.event_window_threshold),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if dataset.event_col:
            cmd += ["--event-col", args.event_col or dataset.event_col]
        return cmd

    cmd = [
        args.python,
        str(Path(args.code_dir) / "train.py"),
        "--dataset-name",
        dataset.name,
        "--model-name",
        requested_model_name,
        "--canonical-model-name",
        model_name,
        "--csv",
        dataset.csv_path,
        "--time-col",
        dataset.time_col,
        "--value-suffix",
        dataset.value_suffix or "",
        "--history",
        str(args.history),
        "--horizon",
        str(args.horizon),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-blocks",
        str(args.num_blocks),
        "--matrix-hidden-dim",
        str(args.matrix_hidden_dim),
        "--max-nodes",
        str(args.max_nodes),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--split-mode",
        split_mode,
        "--event-window-threshold",
        str(args.event_window_threshold),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if dataset.extra_feature_cols:
        cmd += ["--extra-feature-cols", dataset.extra_feature_cols]
    if dataset.node_feature_suffixes:
        cmd += ["--node-feature-suffixes", dataset.node_feature_suffixes]
    if dataset.add_time_features:
        cmd += ["--add-time-features"]
    if dataset.event_col:
        cmd += ["--event-col", args.event_col or dataset.event_col]

    cmd += [
        "--adj-source",
        args.adj_source,
        "--corr-threshold",
        str(args.corr_threshold),
        "--corr-method",
        args.corr_method,
        "--corr-shrinkage-lambda",
        str(args.corr_shrinkage_lambda),
        "--num-diffusion-steps",
        str(args.num_diffusion_steps),
        "--temporal-reg-weight",
        str(args.temporal_reg_weight),
        "--dynamic-top-k",
        str(args.dynamic_top_k),
        "--quality-gate-bias",
        str(args.quality_gate_bias),
    ]
    if args.auto_corr_shrinkage:
        cmd += ["--auto-corr-shrinkage"]
    if args.static_adj_train_only:
        cmd += ["--static-adj-train-only"]

    if model_name == "stsgcn":
        cmd += ["--fusion-mode", "static", "--fusion-type", "quality"]
    elif model_name == "dgcn":
        cmd += ["--fusion-mode", "dynamic", "--fusion-type", "quality"]
    elif model_name == "quality_v2":
        cmd += ["--fusion-mode", "fusion", "--fusion-type", "quality"]
    elif model_name == "quality_ood_scaled_v1":
        cmd += [
            "--fusion-mode",
            "fusion",
            "--fusion-type",
            "quality_ood_scaled",
            "--ood-gamma-max",
            str(args.ood_gamma_max),
            "--ood-clip",
            str(args.ood_clip),
            "--ood-use-volatility",
        ]
    return cmd


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    output_lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            output_lines.append(line)
    return process.wait(), "".join(output_lines)


def parse_test_metrics(output: str) -> dict[str, float | None]:
    match = TEST_RE.search(output)
    metrics: dict[str, float | None] = {
        "mae": None,
        "rmse": None,
        "smape": None,
        "wape": None,
        "h1_mae": None,
        "h3_mae": None,
        "h6_mae": None,
        "h12_mae": None,
    }
    if match:
        metrics.update({name: float(value) for name, value in match.groupdict().items()})
    horizon = HORIZON_RE.search(output)
    if horizon:
        metrics.update({name: float(value) for name, value in horizon.groupdict().items()})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "rainstorm_quick",
            "rainstorm_seeds",
            "bridge_quick",
            "bridge_seeds",
            "typhoon_quick",
            "typhoon_seeds",
            "public_baseline_event_quick",
            "public_baseline_event_seeds",
            "public_baseline_normal_quick",
            "public_baseline_normal_seeds",
            "public_baseline_all_quick",
            "public_baseline_all_seeds",
            "resilience_event_quick",
            "resilience_event_seeds",
            "resilience_normal_seeds",
            "main_event_quick",
            "main_event_seeds",
            "main_normal_quick",
            "main_normal_seeds",
            "main_all_quick",
            "main_all_seeds",
            "weather_quick",
            "weather_seeds",
            "event_quick",
            "event_seeds",
            "ood_event_quick",
            "ood_event_seeds",
            "normal_quick",
            "normal_seeds",
            "ood_normal_quick",
            "ood_normal_seeds",
            "pems_quick",
            "pems_seeds",
            "thesis_weather_quick",
            "thesis_weather_seeds",
            "thesis_all_quick",
            "thesis_all_seeds",
            "all_quick",
            "all_seeds",
        ],
        default="main_event_quick",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated model list. Defaults to dcrnn_resilience for "
            "resilience_* suites, stsgcn,dgcn for public_baseline_* suites, "
            "and all clean mainline models otherwise."
        ),
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--matrix-hidden-dim", type=int, default=256)
    parser.add_argument("--num-diffusion-steps", type=int, default=2)
    parser.add_argument("--temporal-reg-weight", type=float, default=1e-3)
    parser.add_argument("--event-col", default="")
    parser.add_argument("--dynamic-top-k", type=int, default=3)
    parser.add_argument("--quality-gate-bias", type=float, default=-1.0)
    parser.add_argument("--ood-gamma-max", type=float, default=0.35)
    parser.add_argument("--ood-clip", type=float, default=5.0)
    parser.add_argument("--adj-source", choices=["corr", "road", "granger", "mix"], default="corr")
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    parser.add_argument("--corr-method", choices=["pearson", "shrinkage", "partial"], default="pearson")
    parser.add_argument("--corr-shrinkage-lambda", type=float, default=0.1)
    parser.add_argument("--auto-corr-shrinkage", action="store_true")
    parser.add_argument("--static-adj-train-only", action="store_true", default=True)
    parser.add_argument("--no-static-adj-train-only", dest="static_adj_train_only", action="store_false")
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "event_aware"],
        default=None,
    )
    parser.add_argument("--event-window-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=r"D:\soft\Python310\python.exe")
    parser.add_argument("--code-dir", default=r"D:\TrafficGNN\dstsgcn_code")
    parser.add_argument("--output-root", default=r"D:\TrafficGNN\outputs\experiments")
    args = parser.parse_args()

    requested_model_names = (
        parse_csv_list(args.models)
        if args.models.strip()
        else default_models_for_suite(args.suite)
    )
    unsupported = [name for name in requested_model_names if name.strip().lower() not in SUPPORTED_MODELS]
    if unsupported:
        raise ValueError(
            f"Unsupported models: {unsupported}. Mainline supports only "
            "stsgcn, dgcn, dcrnn_resilience, quality_v2, quality_ood_scaled_v1."
        )
    model_names = list(dict.fromkeys(canonical_model_name(name) for name in requested_model_names))

    code_dir = Path(args.code_dir)
    output_root = Path(args.output_root) / args.suite
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_names = datasets_for_suite(args.suite)
    seeds = default_seeds_for_suite(args.suite, args.seeds)
    rows: list[dict[str, object]] = []

    for dataset_name in dataset_names:
        dataset = DATASETS[dataset_name]
        split_mode = split_mode_for_dataset(args, dataset_name)
        for model_name in model_names:
            for seed in seeds:
                run_name = (
                    f"{dataset.name}_{model_name}_seed{seed}_"
                    f"n{args.max_nodes}_e{args.epochs}_adj{args.adj_source}"
                )
                run_dir = output_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = build_command(args, dataset, model_name, seed, run_dir, split_mode)
                print("\n" + "=" * 80)
                print(f"Run: {run_name}")
                print("Command:")
                print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

                if args.dry_run:
                    metrics = parse_test_metrics("")
                    status = "dry_run"
                    run_config = {}
                else:
                    return_code, output = run_command(cmd, code_dir, run_dir / "train.log")
                    metrics = parse_test_metrics(output)
                    status = "ok" if return_code == 0 else f"failed:{return_code}"
                    run_config_path = run_dir / "run_config.json"
                    if run_config_path.exists():
                        with run_config_path.open("r", encoding="utf-8") as f:
                            run_config = json.load(f)
                    else:
                        run_config = {}

                ood_profile = run_config.get("ood_profile", {})
                rows.append(
                    {
                        "suite": args.suite,
                        "dataset": dataset.name,
                        "model": model_name,
                        "canonical_model_name": model_name,
                        "experiment_role": "public_baseline"
                        if model_name in {"stsgcn", "dgcn", "dcrnn_resilience"}
                        else "internal_variant",
                        "paper_model_name": model_paper_name(model_name),
                        "implementation_scope": model_implementation_scope(model_name),
                        "seed": seed,
                        "status": status,
                        "max_nodes": args.max_nodes,
                        "epochs": args.epochs,
                        "split_mode": split_mode,
                        "event_col": args.event_col or dataset.event_col,
                        "fusion_type": (
                            "dcrnn_resilience"
                            if model_name == "dcrnn_resilience"
                            else
                            "quality_ood_scaled"
                            if model_name == "quality_ood_scaled_v1"
                            else "quality"
                        ),
                        "fusion_mode": (
                            "static"
                            if model_name == "stsgcn"
                            else "dynamic"
                            if model_name == "dgcn"
                            else "dcrnn_resilience"
                            if model_name == "dcrnn_resilience"
                            else "fusion"
                        ),
                        "static_adj_train_only": args.static_adj_train_only,
                        "num_diffusion_steps": args.num_diffusion_steps,
                        "temporal_reg_weight": args.temporal_reg_weight,
                        "dynamic_top_k": args.dynamic_top_k,
                        "quality_gate_bias": args.quality_gate_bias,
                        "ood_enabled": run_config.get("ood_enabled", ""),
                        "ood_gamma_max": run_config.get("ood_gamma_max", "")
                        if model_name == "quality_ood_scaled_v1" else "",
                        "ood_clip": run_config.get("ood_clip", "")
                        if model_name == "quality_ood_scaled_v1" else "",
                        "ood_combined_q95": ood_profile.get("combined_ood_q95", "")
                        if model_name == "quality_ood_scaled_v1" else "",
                        "ood_combined_q99": ood_profile.get("combined_ood_q99", "")
                        if model_name == "quality_ood_scaled_v1" else "",
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "smape": metrics["smape"],
                        "wape": metrics["wape"],
                        "h1_mae": metrics["h1_mae"],
                        "h3_mae": metrics["h3_mae"],
                        "h6_mae": metrics["h6_mae"],
                        "h12_mae": metrics["h12_mae"],
                        "output_dir": str(run_dir),
                    }
                )

    summary_path = output_root / "experiment_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSummary saved to:")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
