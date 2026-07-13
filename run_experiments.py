"""Run active traffic-resilience experiment suites.

The active thesis direction no longer treats DGCN/quality_v2 variants as
main models. They remain available as legacy internal baselines only when
``--allow-legacy-internal-models`` is explicitly set.
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
    explicit_event_time: str = ""


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
        explicit_event_time="2007-07-29 00:00:00",
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
    "dcrnn_resilience": "dcrnn_resilience_official_adapted",
    "dcrnn_resilience_official_adapted": "dcrnn_resilience_official_adapted",
    "quality_v2": "quality_v2",
    "quality_ood_scaled_v1": "quality_ood_scaled_v1",
    "quality_resilience_aux_v1": "quality_resilience_aux_v1",
    "resilience_aux_dstsgcn_v1": "quality_resilience_aux_v1",
    "stat_resilience_dstsgcn_v2": "stat_resilience_dstsgcn_v2",
}
SUPPORTED_MODELS = set(MODEL_ALIASES)
LEGACY_INTERNAL_MODELS = {"dgcn", "quality_v2", "quality_ood_scaled_v1"}


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
    if suite in {"resilience_event_quick", "stat_resilience_quick"}:
        return [42]
    if suite in {"resilience_event_seeds", "resilience_normal_seeds", "stat_resilience_seeds", "stat_resilience_normal_seeds"}:
        return [42, 2024, 3407]
    if suite.endswith("_seeds"):
        return [42, 2024, 3407]
    return [42]


def datasets_for_suite(suite: str) -> list[str]:
    if suite in {"resilience_event_quick", "resilience_event_seeds", "stat_resilience_quick", "stat_resilience_seeds"}:
        return ["bridge", "rainstorm", "typhoon"]
    if suite in {"resilience_normal_seeds", "stat_resilience_normal_seeds"}:
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
    if dataset_name == "bridge":
        return "event_aligned"
    return "chronological"


def default_models_for_suite(suite: str) -> list[str]:
    if suite.startswith("stat_resilience_"):
        return ["dcrnn_resilience_official_adapted", "quality_resilience_aux_v1", "stat_resilience_dstsgcn_v2"]
    if suite.startswith("resilience_"):
        return ["dcrnn_resilience_official_adapted", "quality_resilience_aux_v1"]
    if suite.startswith("public_baseline"):
        return ["stsgcn"]
    return ["dcrnn_resilience_official_adapted"]


def model_paper_name(model_name: str) -> str:
    return {
        "stsgcn": "STSGCN-style static graph baseline",
        "dgcn": "Legacy DGCN-style dynamic graph baseline",
        "dcrnn_resilience_official_adapted": "Official-code-adapted DCRNN-Resilience",
        "quality_resilience_aux_v1": "Proposed resilience-auxiliary D-STSGCN",
        "stat_resilience_dstsgcn_v2": "Statistical resilience D-STSGCN v2",
        "quality_v2": "Legacy static-dynamic quality fusion baseline",
        "quality_ood_scaled_v1": "Legacy OOD-scaled quality fusion extension",
    }[canonical_model_name(model_name)]


def model_implementation_scope(model_name: str) -> str:
    model_name = canonical_model_name(model_name)
    if model_name in LEGACY_INTERNAL_MODELS:
        return "legacy internal model archived outside the current resilience mainline"
    if model_name == "stsgcn":
        return "static graph sanity baseline under the unified framework"
    if model_name == "dcrnn_resilience_official_adapted":
        return "official-code-adapted public DCRNN-Resilience baseline"
    if model_name == "quality_resilience_aux_v1":
        return "active statistical resilience auxiliary model"
    return "active traffic-resilience model"


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

    if model_name == "dcrnn_resilience_official_adapted":
        cmd = [
            args.python,
            str(Path(args.code_dir) / "baselines" / "dcrnn_resilience_official_adapted" / "train.py"),
            "--dataset-name", dataset.name,
            "--csv", dataset.csv_path,
            "--time-col", dataset.time_col,
            "--value-suffix", dataset.value_suffix or "none",
            "--max-nodes", str(args.max_nodes),
            "--history-steps", str(args.history),
            "--horizon", str(args.horizon),
            "--rnn-units", "256",
            "--num-rnn-layers", "2",
            "--max-diffusion-step", "1",
            "--cl-decay-steps", "2000",
            "--protocol", "fair",
            "--split-mode", split_mode,
            "--event-window-threshold", str(args.event_window_threshold),
            "--output-dir", str(output_dir),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--lr", "0.01",
            "--batch-size", str(args.batch_size),
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
        if dataset.explicit_event_time:
            cmd += ["--explicit-event-time", dataset.explicit_event_time]
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
        "--experiment-stage",
        (args.stat_ablation_stage.upper() if model_name == "stat_resilience_dstsgcn_v2" else ""),
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
    if dataset.explicit_event_time:
        cmd += ["--explicit-event-time", dataset.explicit_event_time]

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
    elif model_name == "stat_resilience_dstsgcn_v2":
        stage = args.stat_ablation_stage
        consistency_weight = args.resilience_consistency_weight if stage in {"a2", "a3", "a4", "full"} else 0.0
        cmd += [
            "--fusion-mode", "fusion",
            "--fusion-type", "quality",
            "--resilience-aux",
            "--resilience-target-mode", "log_ratio",
            "--resilience-baseline-mode", "robust_shrunk",
            "--resilience-shrinkage-candidates", args.resilience_shrinkage_candidates,
            "--resilience-log-ratio-mad-clip", str(args.resilience_log_ratio_mad_clip),
            "--resilience-consistency-weight", str(consistency_weight),
            "--resilience-consistency-warmup-epochs", str(args.resilience_consistency_warmup_epochs),
            "--traffic-cvar-alpha", str(args.traffic_cvar_alpha),
            "--traffic-cvar-weight", str(args.traffic_cvar_weight),
            "--resilience-cvar-alpha", str(args.resilience_cvar_alpha),
            "--resilience-cvar-weight", str(args.resilience_cvar_weight),
        ]
        if stage in {"a1", "a2"}:
            cmd += ["--disable-cvar"]
        if stage in {"a1", "a2", "a3"}:
            cmd += ["--disable-resilience-uncertainty-weighting"]
        else:
            cmd += ["--resilience-uncertainty-weighting"]
    elif model_name == "quality_resilience_aux_v1":
        cmd += [
            "--fusion-mode",
            "fusion",
            "--fusion-type",
            "quality",
            "--resilience-aux",
            "--resilience-loss-weight",
            str(args.resilience_loss_weight),
            "--resilience-consistency-weight",
            str(args.resilience_consistency_weight),
            "--resilience-time-of-day-bins",
            str(args.resilience_time_of_day_bins),
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
            "stat_resilience_quick",
            "stat_resilience_seeds",
            "stat_resilience_normal_seeds",
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
            "Comma-separated model list. Defaults to active resilience models. "
            "Legacy internal models such as dgcn and quality_v2 require "
            "--allow-legacy-internal-models."
        ),
    )
    parser.add_argument(
        "--allow-legacy-internal-models",
        action="store_true",
        help="Allow archived internal baselines: dgcn, quality_v2, quality_ood_scaled_v1.",
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
    parser.add_argument("--resilience-loss-weight", type=float, default=0.3)
    parser.add_argument("--resilience-consistency-weight", type=float, default=0.1)
    parser.add_argument("--resilience-time-of-day-bins", type=int, default=288)
    parser.add_argument("--resilience-shrinkage-candidates", default="0,1,3,7,14,28")
    parser.add_argument("--resilience-log-ratio-mad-clip", type=float, default=6.0)
    parser.add_argument("--resilience-consistency-warmup-epochs", type=int, default=5)
    parser.add_argument("--stat-ablation-stage", choices=["a1", "a2", "a3", "a4", "full"], default="full")
    parser.add_argument("--traffic-cvar-alpha", type=float, default=0.90)
    parser.add_argument("--traffic-cvar-weight", type=float, default=0.20)
    parser.add_argument("--resilience-cvar-alpha", type=float, default=0.90)
    parser.add_argument("--resilience-cvar-weight", type=float, default=0.20)
    parser.add_argument("--adj-source", choices=["corr", "road", "granger", "mix"], default="corr")
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    parser.add_argument("--corr-method", choices=["pearson", "shrinkage", "partial", "oas_partial"], default="pearson")
    parser.add_argument("--corr-shrinkage-lambda", type=float, default=0.1)
    parser.add_argument("--auto-corr-shrinkage", action="store_true")
    parser.add_argument("--static-adj-train-only", action="store_true", default=True)
    parser.add_argument("--no-static-adj-train-only", dest="static_adj_train_only", action="store_false")
    parser.add_argument(
        "--split-mode",
        choices=["chronological", "event_aware", "event_aligned"],
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
            f"Unsupported models: {unsupported}. Known models are: "
            + ",".join(sorted(SUPPORTED_MODELS))
        )
    model_names = list(dict.fromkeys(canonical_model_name(name) for name in requested_model_names))
    legacy_requested = [name for name in model_names if name in LEGACY_INTERNAL_MODELS]
    if legacy_requested and not args.allow_legacy_internal_models:
        raise ValueError(
            "Legacy internal models are no longer part of the active traffic-resilience mainline: "
            f"{legacy_requested}. Re-run with --allow-legacy-internal-models only for reproduction "
            "or appendix experiments."
        )

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
                stage_tag = (
                    f"_{args.stat_ablation_stage}"
                    if model_name == "stat_resilience_dstsgcn_v2" and args.stat_ablation_stage != "full"
                    else ""
                )
                run_name = (
                    f"{dataset.name}_{model_name}{stage_tag}_seed{seed}_"
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
                        "experiment_stage": (
                            args.stat_ablation_stage.upper()
                            if model_name == "stat_resilience_dstsgcn_v2" else ""
                        ),
                        "experiment_role": (
                            "legacy_internal"
                            if model_name in LEGACY_INTERNAL_MODELS
                            else "public_baseline"
                            if model_name in {"stsgcn", "dcrnn_resilience_official_adapted"}
                            else "active_resilience_model"
                        ),
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
                            if model_name == "dcrnn_resilience_official_adapted"
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
                            if model_name == "dcrnn_resilience_official_adapted"
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
                        "resilience_aux_enabled": run_config.get("resilience_aux_enabled", "")
                        if model_name == "quality_resilience_aux_v1" else "",
                        "resilience_loss_weight": run_config.get("resilience_loss_weight", "")
                        if model_name == "quality_resilience_aux_v1" else "",
                        "resilience_consistency_weight": run_config.get("resilience_consistency_weight", "")
                        if model_name == "quality_resilience_aux_v1" else "",
                        "resilience_time_of_day_bins": run_config.get("resilience_time_of_day_bins", "")
                        if model_name == "quality_resilience_aux_v1" else "",
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

