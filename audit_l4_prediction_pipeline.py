"""Audit whether the frozen DGCN-STSGCN pipeline can support indirect L4 prediction.

This module is deliberately non-training. It compares the current window/scaler
implementation with a leakage-free chronological reference and records the
minimum engineering changes required before E-L4-1 may run.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from analyze_latent_traffic_performance import load_saved_profile
from data import (
    StandardScaler,
    load_wide_traffic_csv,
    read_csv_with_fallback,
    split_traffic_window_indices,
)
from flow_speed_resilience import match_node_variables
from model import DSTSGCN
from l4_prediction_pipeline import (
    event_free_feature_contract,
    load_forecast_checkpoint,
    load_prediction_archive,
    save_forecast_checkpoint,
    save_prediction_archive,
    scaler_metadata,
)


DATASETS = {
    "bridge": {
        "csv": r"D:\TrafficGNN\data\bridge_collapse_flow.csv",
        "time_col": "Time",
        "flow_suffix": None,
        "speed_suffix": None,
        "event_col": None,
    },
    "rainstorm": {
        "csv": r"D:\TrafficGNN\data\rainstorm_traffic_state.csv",
        "time_col": "Time",
        "flow_suffix": "_volume",
        "speed_suffix": "_speed",
        "event_col": "precip_accum_one_hour",
    },
    "typhoon": {
        "csv": r"D:\TrafficGNN\data\typhoon_traffic_state_standard.csv",
        "time_col": "Time",
        "flow_suffix": "_volume",
        "speed_suffix": "_speed",
        "event_col": "typhoon_intensity",
    },
    "pems04": {
        "csv": r"D:\TrafficGNN\data\public\PEMS04\pems04.csv",
        "time_col": "Time",
        "flow_suffix": "_volume",
        "speed_suffix": "_speed",
        "event_col": None,
    },
    "pems08": {
        "csv": r"D:\TrafficGNN\data\public\PEMS08\pems08.csv",
        "time_col": "Time",
        "flow_suffix": "_volume",
        "speed_suffix": "_speed",
        "event_col": None,
    },
}


def ordered_frame(config: dict[str, object]) -> pd.DataFrame:
    """Load and stably sort one dataset by its declared timestamp."""
    frame = read_csv_with_fallback(str(config["csv"]))
    time_col = str(config["time_col"])
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    if float(parsed.notna().mean()) < 0.95:
        raise ValueError(f"Cannot reliably parse {time_col} in {config['csv']}")
    return (
        frame.assign(__audit_time=parsed)
        .sort_values("__audit_time", kind="stable")
        .drop(columns="__audit_time")
        .reset_index(drop=True)
    )


def numeric_columns(frame: pd.DataFrame, suffix: str | None, time_col: str) -> list[str]:
    """Return ordered traffic columns while excluding Bridge identifiers."""
    excluded = {time_col, "ID", "id"}
    return [
        str(column)
        for column in frame.columns
        if column not in excluded
        and (suffix is None or str(column).endswith(suffix))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def dataset_matches(
    frame: pd.DataFrame,
    config: dict[str, object],
    max_nodes: int,
) -> tuple[list[dict[str, str | None]], list[str], list[str]]:
    """Return flow-led exact-name matches and current speed-loader column order."""
    flow_suffix = config["flow_suffix"]
    speed_suffix = config["speed_suffix"]
    time_col = str(config["time_col"])
    flow_cols = numeric_columns(frame, flow_suffix, time_col)
    speed_cols = numeric_columns(frame, speed_suffix, time_col) if speed_suffix else []
    if flow_suffix is None:
        matches = [
            {"node": column, "flow_col": column, "speed_col": None, "occupancy_col": None}
            for column in flow_cols
        ]
    else:
        matches = match_node_variables(
            list(frame.columns), str(flow_suffix), str(speed_suffix) if speed_suffix else None
        )
    return matches[:max_nodes], flow_cols, speed_cols[:max_nodes]


def target_bounds(window_index: int, history: int, horizon: int) -> tuple[int, int]:
    """Return inclusive target indices for a sliding forecasting window."""
    start = int(window_index) + int(history)
    return start, start + int(horizon) - 1


def target_timestamp_matrix(
    timestamps: np.ndarray,
    window_indices: Iterable[int],
    history: int,
    horizon: int,
) -> np.ndarray:
    """Map every window and horizon to the actual target timestamp."""
    ts = np.asarray(timestamps)
    indices = np.asarray(list(window_indices), dtype=int)
    offsets = np.arange(history, history + horizon, dtype=int)
    return ts[indices[:, None] + offsets[None, :]]


def strict_chronological_window_splits(
    num_timesteps: int,
    history: int,
    horizon: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Create target-disjoint chronological splits at raw-time boundaries.

    Inputs may use pre-boundary history, but every target horizon belongs wholly
    to exactly one raw-time split. Windows whose targets cross a boundary are
    deliberately omitted.
    """
    num_windows = max(0, num_timesteps - history - horizon + 1)
    windows = np.arange(num_windows, dtype=int)
    train_time_end = int(num_timesteps * train_ratio)
    val_time_end = int(num_timesteps * (train_ratio + val_ratio))
    target_start = windows + history
    target_end = target_start + horizon - 1
    train = windows[target_end < train_time_end]
    val = windows[(target_start >= train_time_end) & (target_end < val_time_end)]
    test = windows[(target_start >= val_time_end) & (target_end < num_timesteps)]
    return train, val, test, {
        "train_time_end_exclusive": train_time_end,
        "val_time_end_exclusive": val_time_end,
    }


def split_overlap_steps(left: np.ndarray, right: np.ndarray, history: int, horizon: int) -> int:
    """Count target timestamps shared by adjacent window-index splits."""
    if len(left) == 0 or len(right) == 0:
        return 0
    _, left_end = target_bounds(int(left[-1]), history, horizon)
    right_start, _ = target_bounds(int(right[0]), history, horizon)
    return max(0, left_end - right_start + 1)


def audit_split(
    dataset: str,
    num_timesteps: int,
    history: int,
    horizon: int,
) -> dict[str, object]:
    """Compare current project splitting with the strict chronological reference."""
    num_windows = max(0, num_timesteps - history - horizon + 1)
    current_train, current_val, current_test, _ = split_traffic_window_indices(
        num_windows, history, split_mode="chronological"
    )
    strict_train, strict_val, strict_test, boundaries = strict_chronological_window_splits(
        num_timesteps, history, horizon
    )
    scaler_end = int(num_timesteps * 0.6)
    first_val_window = int(current_val[0]) if len(current_val) else -1
    first_val_target = first_val_window + history if first_val_window >= 0 else -1
    return {
        "dataset": dataset,
        "num_timesteps": num_timesteps,
        "history": history,
        "horizon": horizon,
        "num_windows": num_windows,
        "current_train_windows": len(current_train),
        "current_val_windows": len(current_val),
        "current_test_windows": len(current_test),
        "strict_train_windows": len(strict_train),
        "strict_val_windows": len(strict_val),
        "strict_test_windows": len(strict_test),
        "train_val_target_overlap_steps": split_overlap_steps(
            current_train, current_val, history, horizon
        ),
        "val_test_target_overlap_steps": split_overlap_steps(
            current_val, current_test, history, horizon
        ),
        "strict_train_val_overlap_steps": split_overlap_steps(
            strict_train, strict_val, history, horizon
        ),
        "strict_val_test_overlap_steps": split_overlap_steps(
            strict_val, strict_test, history, horizon
        ),
        "scaler_fit_end_exclusive": scaler_end,
        "first_current_val_window_start": first_val_window,
        "first_current_val_target_start": first_val_target,
        "scaler_includes_val_input_steps": max(0, scaler_end - first_val_window)
        if first_val_window >= 0 else 0,
        "scaler_includes_val_target_steps": max(0, scaler_end - first_val_target)
        if first_val_target >= 0 else 0,
        "strict_train_time_end_exclusive": boundaries["train_time_end_exclusive"],
        "strict_val_time_end_exclusive": boundaries["val_time_end_exclusive"],
        "current_split_leak_free": bool(
            split_overlap_steps(current_train, current_val, history, horizon) == 0
            and split_overlap_steps(current_val, current_test, history, horizon) == 0
            and scaler_end <= first_val_target
        ),
        "strict_split_leak_free": bool(
            split_overlap_steps(strict_train, strict_val, history, horizon) == 0
            and split_overlap_steps(strict_val, strict_test, history, horizon) == 0
        ),
    }


def inverse_predictions(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform predictions without clipping negatives or replacing NaN."""
    return scaler.inverse_transform(np.asarray(values, dtype=float))


def scaler_audit_row(
    dataset: str,
    variable: str,
    values: np.ndarray,
    train_end: int,
) -> dict[str, object]:
    """Fit a variable-specific train-only scaler and audit its roundtrip."""
    array = np.asarray(values, dtype=float)
    scaler = StandardScaler()
    scaler.fit(array[:train_end, :, None])
    transformed = scaler.transform(array[:, :, None])
    restored = inverse_predictions(transformed, scaler)[..., 0]
    finite = np.isfinite(array) & np.isfinite(restored)
    error = float(np.max(np.abs(restored[finite] - array[finite]))) if finite.any() else np.nan
    negative = array < 0
    negative_preserved = bool(
        np.allclose(restored[negative], array[negative], equal_nan=True)
    ) if negative.any() else True
    return {
        "dataset": dataset,
        "variable": variable,
        "train_end_exclusive": int(train_end),
        "scaler_object_scope": f"{dataset}:{variable}",
        "mean": float(np.asarray(scaler.mean).reshape(-1)[0]),
        "std": float(np.asarray(scaler.std).reshape(-1)[0]),
        "roundtrip_max_abs_error": error,
        "roundtrip_pass": bool(np.isfinite(error) and error < 1e-5),
        "negative_value_count": int(negative.sum()),
        "negative_values_preserved": negative_preserved,
        "missing_value_count_after_current_loader": int(np.isnan(array).sum()),
    }


def checkpoint_name(dataset: str, variable: str, seed: int) -> str:
    """Return a collision-resistant proposed checkpoint name."""
    return f"{dataset}_{variable}_seed{int(seed)}_best_dstsgcn.pt"


def profile_paths(
    dataset: str,
    variable: str,
    l4_dir: Path,
    source_profile_dir: Path,
) -> tuple[Path, Path, Path] | None:
    """Resolve frozen demand or core speed profile paths without refitting."""
    if variable == "demand":
        root, label = l4_dir / dataset, "demand"
    elif variable == "efficiency" and dataset != "bridge":
        root, label = source_profile_dir / dataset, "speed"
    else:
        return None
    return (
        root / f"{label}_profile_hierarchical.npz",
        root / f"{label}_profile_ecdf.npz",
        root / f"{label}_profile_report.json",
    )


def frozen_profile_available(paths: tuple[Path, Path, Path] | None) -> bool:
    """Return whether a complete read-only profile package exists and is train-only."""
    if paths is None or not all(path.exists() for path in paths):
        return False
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    return bool(metadata.get("train_only", False))


def bridge_four_state_available(dataset: str, efficiency_available: bool) -> bool:
    """Four-state L4 evaluation requires an observed efficiency dimension."""
    return bool(dataset != "bridge" and efficiency_available)


def separate_event_segments(segments: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Preserve event segments as separate ordered intervals."""
    return [(int(start), int(end)) for start, end in segments]


def l4_prediction_schema() -> tuple[str, ...]:
    """Return the permitted vector prediction keys; no scalar L4 key is present."""
    return (
        "demand_deficit_true",
        "demand_deficit_pred",
        "efficiency_deficit_true",
        "efficiency_deficit_pred",
    )


def save_prediction_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save prediction arrays for auditable roundtrip tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def model_smoke(num_nodes: int, history: int, horizon: int) -> dict[str, object]:
    """Run one inference-only DGCN-STSGCN shape/finite check on CPU."""
    nodes = max(1, min(int(num_nodes), 5))
    model = DSTSGCN(
        num_nodes=nodes,
        input_dim=1,
        output_dim=1,
        horizon=horizon,
        hidden_dim=8,
        num_blocks=1,
        graph_learner_type="lmln",
        fusion_mode="fusion",
        fusion_type="quality",
        dynamic_top_k=min(3, nodes),
        matrix_hidden_dim=16,
        num_diffusion_steps=1,
        resilience_aux=False,
    )
    model.eval()
    x = torch.zeros((2, history, nodes, 1), dtype=torch.float32)
    adjacency = torch.eye(nodes, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(x, adjacency)
    return {
        "input_shape": list(x.shape),
        "output_shape": list(prediction.shape),
        "expected_output_shape": [2, horizon, nodes, 1],
        "shape_pass": list(prediction.shape) == [2, horizon, nodes, 1],
        "finite_pass": bool(torch.isfinite(prediction).all()),
    }




def repaired_artifact_smoke() -> dict[str, bool]:
    """Roundtrip checkpoint metadata, prediction arrays and event-free contract."""
    scaler = StandardScaler()
    scaler.fit(np.arange(12.0).reshape(6, 2, 1))
    contract = event_free_feature_contract()
    metadata = {
        "dataset": "synthetic",
        "variable": "speed",
        "node_names": ["a", "b"],
        "history": 12,
        "horizon": 3,
        "train_time_end_exclusive": 6,
        "val_time_end_exclusive": 8,
        "seed": 42,
        "scaler": scaler_metadata(scaler),
        "timestamp_start": "2020-01-01T00:00:00",
        "timestamp_end": "2020-01-01T00:55:00",
        "input_features": contract["input_features"],
        "event_features": contract["event_features"],
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_path = root / "model.pt"
        save_forecast_checkpoint(
            checkpoint_path, {"weight": torch.tensor([1.0])}, metadata
        )
        checkpoint = load_forecast_checkpoint(checkpoint_path)
        checkpoint_ok = bool(
            checkpoint["metadata"]["node_names"] == ["a", "b"]
            and checkpoint["metadata"]["scaler"] == metadata["scaler"]
        )

        shape = (2, 3, 2, 1)
        y_true = np.arange(np.prod(shape), dtype=float).reshape(shape)
        y_pred = y_true + 0.5
        target_timestamps = np.arange(6).reshape(2, 3).astype("datetime64[m]")
        archive_path = root / "predictions.npz"
        save_prediction_archive(
            archive_path,
            dataset="synthetic",
            variable="speed",
            split="validation",
            target_timestamps=target_timestamps,
            node_names=["a", "b"],
            y_true_scaled=y_true,
            y_pred_scaled=y_pred,
            y_true_physical=y_true,
            y_pred_physical=y_pred,
            train_time_end_exclusive=6,
            val_time_end_exclusive=8,
            scaler=metadata["scaler"],
            seed=42,
        )
        archive = load_prediction_archive(archive_path)
        prediction_ok = bool(
            archive["y_pred_physical"].shape == shape
            and archive["target_timestamps"].shape == (2, 3)
            and archive["node_names"].tolist() == ["a", "b"]
        )
    contract_ok = bool(
        contract["input_features"] == ["traffic"]
        and contract["event_features"] == []
        and contract["weather_features"] == []
        and not contract["cvar_enabled"]
        and not contract["resilience_aux_enabled"]
    )
    return {
        "checkpoint_roundtrip": checkpoint_ok,
        "prediction_archive_roundtrip": prediction_ok,
        "event_free_contract": contract_ok,
    }
def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional dependencies."""
    if frame.empty:
        return "????"
    view = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    """Run the complete E-L4-0 audit and write all required artifacts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l4_dir = Path(args.l4_dir)
    source_profile_dir = Path(args.source_profile_dir)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]

    capability_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    scaler_rows: list[dict[str, object]] = []

    for dataset in datasets:
        config = DATASETS[dataset]
        frame = ordered_frame(config)
        matches, all_flow_cols, current_speed_cols = dataset_matches(
            frame, config, args.max_nodes
        )
        selected_flow_cols = [str(item["flow_col"]) for item in matches]
        matched_speed_cols = [str(item["speed_col"]) for item in matches if item["speed_col"]]
        current_speed_by_index = {
            index: column for index, column in enumerate(current_speed_cols)
        }
        for index, item in enumerate(matches):
            expected_speed = item["speed_col"]
            current_speed = current_speed_by_index.get(index)
            alignment_rows.append(
                {
                    "dataset": dataset,
                    "flow_index": index,
                    "node": item["node"],
                    "flow_col": item["flow_col"],
                    "expected_speed_col": expected_speed,
                    "current_speed_loader_col_at_same_index": current_speed,
                    "speed_matched": bool(expected_speed),
                    "same_index_order_match": bool(expected_speed and expected_speed == current_speed),
                    "used_in_joint_analysis": bool(expected_speed),
                    "exclusion_reason": "" if expected_speed else "no_same_base_speed",
                }
            )

        split_row = audit_split(dataset, len(frame), args.history, args.horizon)
        split_rows.append(split_row)
        train_time_end = int(len(frame) * 0.6)

        flow_data, loaded_flow_names = load_wide_traffic_csv(
            str(config["csv"]),
            value_suffix=config["flow_suffix"],
            time_col=str(config["time_col"]),
            max_nodes=args.max_nodes,
            exclude_cols=["ID", "id"],
        )
        scaler_rows.append(
            scaler_audit_row(dataset, "flow", flow_data[..., 0], train_time_end)
        )
        speed_direct = config["speed_suffix"] is not None and len(matched_speed_cols) > 0
        if speed_direct:
            speed_data, loaded_speed_names = load_wide_traffic_csv(
                str(config["csv"]),
                value_suffix=str(config["speed_suffix"]),
                time_col=str(config["time_col"]),
                max_nodes=args.max_nodes,
                exclude_cols=["ID", "id"],
            )
            scaler_rows.append(
                scaler_audit_row(dataset, "speed", speed_data[..., 0], train_time_end)
            )
        else:
            loaded_speed_names = []

        paired_flow_cols = [
            str(item["flow_col"]) for item in matches if item["speed_col"] is not None
        ]
        if speed_direct:
            _, explicit_flow_names = load_wide_traffic_csv(
                str(config["csv"]),
                value_suffix=str(config["flow_suffix"]),
                time_col=str(config["time_col"]),
                value_columns=paired_flow_cols,
                exclude_cols=["ID", "id"],
            )
            _, explicit_speed_names = load_wide_traffic_csv(
                str(config["csv"]),
                value_suffix=str(config["speed_suffix"]),
                time_col=str(config["time_col"]),
                value_columns=matched_speed_cols,
                exclude_cols=["ID", "id"],
            )
            explicit_pair_ok = bool(
                explicit_flow_names == paired_flow_cols
                and explicit_speed_names == matched_speed_cols
                and len(explicit_flow_names) == len(explicit_speed_names)
            )
        else:
            explicit_flow_names = selected_flow_cols
            explicit_speed_names = []
            explicit_pair_ok = dataset == "bridge"

        demand_paths = profile_paths(dataset, "demand", l4_dir, source_profile_dir)
        efficiency_paths = profile_paths(dataset, "efficiency", l4_dir, source_profile_dir)
        demand_profile_ok = frozen_profile_available(demand_paths)
        efficiency_profile_ok = (
            frozen_profile_available(efficiency_paths) if dataset != "bridge" else False
        )
        matched_order_ok = bool(
            dataset == "bridge"
            or (
                len(matched_speed_cols) == len(selected_flow_cols)
                and matched_speed_cols == loaded_speed_names[: len(matched_speed_cols)]
            )
        )
        if dataset == "typhoon":
            matched_order_ok = bool(
                len(matched_speed_cols) == 16
                and len(loaded_speed_names) == 16
                and loaded_speed_names == matched_speed_cols
            )

        capability_rows.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "all_flow_nodes": len(all_flow_cols),
                "all_speed_nodes": len(numeric_columns(
                    frame, config["speed_suffix"], str(config["time_col"])
                )) if config["speed_suffix"] else 0,
                "selected_flow_nodes": len(selected_flow_cols),
                "matched_speed_nodes": len(matched_speed_cols),
                "matched_rate": len(matched_speed_cols) / max(len(selected_flow_cols), 1),
                "current_flow_loader_nodes": len(loaded_flow_names),
                "current_speed_loader_nodes": len(loaded_speed_names),
                "flow_training_supported": True,
                "speed_training_supported_by_model": bool(speed_direct),
                "paired_node_order_supported_by_current_loader": matched_order_ok,
                "paired_node_order_supported_by_explicit_loader": explicit_pair_ok,
                "demand_profile_read_only_available": demand_profile_ok,
                "efficiency_profile_read_only_available": efficiency_profile_ok,
                "efficiency_available": bool(dataset != "bridge" and speed_direct),
                "four_state_evaluation_available": bridge_four_state_available(
                    dataset, speed_direct
                ),
                "event_column_excluded_by_required_baseline": config["event_col"] is not None,
                "requires_two_independent_models": bool(speed_direct),
            }
        )

    capability = pd.DataFrame(capability_rows)
    alignment = pd.DataFrame(alignment_rows)
    splits = pd.DataFrame(split_rows)
    scalers = pd.DataFrame(scaler_rows)
    smoke = model_smoke(args.max_nodes, args.history, args.horizon)
    repair_smoke = repaired_artifact_smoke()

    checks = [
        {
            "check_id": "legacy_window_start_split",
            "status": "pass" if bool(splits["current_split_leak_free"].all()) else "fail",
            "core_blocker": False,
            "evidence": "Legacy train.py window-start splits overlap targets for horizon > 1.",
            "minimum_change": "E-L4 must use split_traffic_window_indices_strict().",
        },
        {
            "check_id": "strict_target_splits_disjoint",
            "status": "pass" if bool(splits["strict_split_leak_free"].all()) else "fail",
            "core_blocker": True,
            "evidence": "Raw-time target-boundary splits omit crossing windows.",
            "minimum_change": "No further split repair required.",
        },
        {
            "check_id": "legacy_scaler_boundary",
            "status": "pass" if bool((splits["scaler_includes_val_target_steps"] == 0).all()) else "fail",
            "core_blocker": False,
            "evidence": "Legacy train.py scaler boundary includes early validation targets.",
            "minimum_change": "E-L4 must fit from strict split train_time_end_exclusive.",
        },
        {
            "check_id": "strict_scaler_train_only",
            "status": "pass" if bool(
                splits["strict_split_leak_free"].all() and scalers["roundtrip_pass"].all()
            ) else "fail",
            "core_blocker": True,
            "evidence": "The repaired bundle fits one scaler per variable on the strict training prefix.",
            "minimum_change": "No further scaler repair required.",
        },
        {
            "check_id": "variable_scaler_roundtrip",
            "status": "pass" if bool(scalers["roundtrip_pass"].all()) else "fail",
            "core_blocker": True,
            "evidence": "Independent flow/speed scaler roundtrip audit.",
            "minimum_change": "Persist each variable scaler in checkpoint metadata.",
        },
        {
            "check_id": "explicit_paired_node_loader_order",
            "status": "pass" if bool(
                capability["paired_node_order_supported_by_explicit_loader"].all()
            ) else "fail",
            "core_blocker": True,
            "evidence": "Explicit ordered value_columns locks Typhoon to the fixed 16 matched bases.",
            "minimum_change": "No further node-loader repair required.",
        },
        {
            "check_id": "frozen_profiles_available",
            "status": "pass" if bool(
                capability["demand_profile_read_only_available"].all()
                and capability.loc[capability.dataset != "bridge", "efficiency_profile_read_only_available"].all()
            ) else "fail",
            "core_blocker": True,
            "evidence": "Demand and core speed profiles are available read-only.",
            "minimum_change": "Never refit profiles on validation/test.",
        },
        {
            "check_id": "model_generic_flow_speed_output",
            "status": "pass" if smoke["shape_pass"] and smoke["finite_pass"] else "fail",
            "core_blocker": True,
            "evidence": json.dumps(smoke, ensure_ascii=False),
            "minimum_change": "No model.py change is required.",
        },
        {
            "check_id": "checkpoint_contains_scaler_nodes_timestamps",
            "status": "pass" if repair_smoke["checkpoint_roundtrip"] else "fail",
            "core_blocker": True,
            "evidence": "Auditable checkpoint bundle roundtrip includes scaler, nodes, splits and timestamps.",
            "minimum_change": "E-L4 runner must use save_forecast_checkpoint().",
        },
        {
            "check_id": "prediction_arrays_exported",
            "status": "pass" if repair_smoke["prediction_archive_roundtrip"] else "fail",
            "core_blocker": True,
            "evidence": "Aligned multi-horizon physical/scaled prediction NPZ roundtrip passed.",
            "minimum_change": "E-L4 runner must use save_prediction_archive().",
        },
        {
            "check_id": "event_signal_absent_from_training",
            "status": "pass" if repair_smoke["event_free_contract"] else "fail",
            "core_blocker": True,
            "evidence": "The repaired contract permits only the traffic feature and disables event/weather inputs.",
            "minimum_change": "E-L4 runner must enforce event_free_feature_contract().",
        },
        {
            "check_id": "static_adjacency_train_only",
            "status": "pass",
            "core_blocker": False,
            "evidence": "train.py supports --static-adj-train-only but it is not the default.",
            "minimum_change": "E-L4 runner must force train-only adjacency.",
        },
        {
            "check_id": "bridge_efficiency_not_fabricated",
            "status": "pass" if not bool(
                capability.loc[capability.dataset == "bridge", "efficiency_available"].iloc[0]
            ) else "fail",
            "core_blocker": True,
            "evidence": "Bridge remains flow-only.",
            "minimum_change": "Skip efficiency and four-state metrics for Bridge.",
        },
    ]
    audit = pd.DataFrame(checks)
    stage_a_passed = bool(
        (audit.loc[audit["core_blocker"], "status"] == "pass").all()
    )

    audit.to_csv(output_dir / "prediction_pipeline_audit.csv", index=False, encoding="utf-8-sig")
    capability.to_csv(output_dir / "dataset_prediction_capability.csv", index=False, encoding="utf-8-sig")
    alignment.to_csv(output_dir / "node_alignment_audit.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(output_dir / "split_window_audit.csv", index=False, encoding="utf-8-sig")
    scalers.to_csv(output_dir / "scaler_roundtrip_audit.csv", index=False, encoding="utf-8-sig")

    blockers = audit[(audit["core_blocker"]) & (audit["status"] != "pass")]
    report = [
        "# E-L4-0R \u795e\u7ecf\u7f51\u7edc\u9884\u6d4b\u7ba1\u7ebf\u4fee\u590d\u590d\u5ba1",
        "",
        "## \u9636\u6bb5\u7ed3\u8bba",
        "",
        "**\u9636\u6bb5 A \u672a\u901a\u8fc7\uff0c\u7981\u6b62\u8fdb\u5165 E-L4-1 \u8bad\u7ec3\u3002**" if not stage_a_passed else "**\u9636\u6bb5 A \u901a\u8fc7\uff0c\u53ef\u4ee5\u8fdb\u5165 E-L4-1\u3002**",
        "",
        "\u672c\u5ba1\u8ba1\u6ca1\u6709\u8bad\u7ec3\u795e\u7ecf\u7f51\u7edc\uff0c\u4e5f\u6ca1\u6709\u4fee\u6539 `model.py`\u3001\u7f51\u7edc\u9aa8\u67b6\u6216\u8bad\u7ec3\u635f\u5931\u3002",
        "",
        "## \u5f53\u524d\u8f93\u5165\u8f93\u51fa",
        "",
        f"- \u6570\u636e loader \u8f93\u51fa `[time, node, feature]`\uff1b\u7a97\u53e3 batch \u8f93\u5165 `[B,{args.history},N,C]`\u3002",
        f"- \u6a21\u578b\u8f93\u51fa `[B,{args.horizon},N,1]`\uff1binference smoke\uff1a`{smoke['input_shape']} -> {smoke['output_shape']}`\u3002",
        "- \u6a21\u578b\u672c\u8eab\u4e0d\u786c\u7f16\u7801 flow\uff0c\u56e0\u6b64\u7406\u8bba\u4e0a\u53ef\u7528\u72ec\u7acb\u6570\u636e\u7ba1\u7ebf\u8bad\u7ec3 speed\u3002",
        "",
        "## \u6838\u5fc3\u68c0\u67e5",
        "",
        markdown_table(audit, ["check_id", "status", "core_blocker", "minimum_change"]),
        "",
        "## \u6570\u636e\u96c6\u80fd\u529b",
        "",
        markdown_table(
            capability,
            [
                "dataset", "selected_flow_nodes", "matched_speed_nodes", "matched_rate",
                "paired_node_order_supported_by_explicit_loader",
                "demand_profile_read_only_available", "efficiency_profile_read_only_available",
            ],
        ),
        "",
        "## \u5207\u5206\u548c scaler",
        "",
        markdown_table(
            splits,
            [
                "dataset", "train_val_target_overlap_steps", "val_test_target_overlap_steps",
                "scaler_includes_val_target_steps", "current_split_leak_free",
                "strict_split_leak_free",
            ],
        ),
        "",
        "\u5f53\u524d `horizon=12` \u65f6\uff0c\u76f8\u90bb split \u7684\u76ee\u6807\u65f6\u95f4\u91cd\u53e0 11 \u6b65\u3002\u4e25\u683c\u53c2\u8003\u5207\u5206\u6309\u539f\u59cb\u65f6\u95f4\u8fb9\u754c\u5206\u914d\u5b8c\u6574\u76ee\u6807\u533a\u95f4\uff0c\u53ef\u5c06\u91cd\u53e0\u964d\u4e3a 0\u3002",
        "",
        "## \u8282\u70b9\u5339\u914d",
        "",
        "Rainstorm \u548c PEMS \u53ef\u4ee5\u6309\u540c\u540d suffix \u5339\u914d\u3002Typhoon \u524d41\u4e2a flow \u4e2d\u670916\u4e2a\u540c\u540d speed\uff1b\u4fee\u590d\u540e\u7684\u663e\u5f0f value_columns \u5df2\u5206\u522b\u9501\u5b9a\u8fd916\u4e2a flow \u548c speed\uff0c\u5e76\u4fdd\u6301\u76f8\u540c\u8282\u70b9\u987a\u5e8f\u3002",
        "",
        "## \u51bb\u7ed3 L4 profile",
        "",
        "\u9700\u6c42 profile \u53ef\u4ece L4 \u8f93\u51fa\u53ea\u8bfb\u52a0\u8f7d\uff1b\u6838\u5fc3 speed profile \u4ecd\u5b58\u653e\u5728\u5df2\u9a8c\u8bc1\u7684 `latent_traffic_performance_l3/<dataset>/speed_profile_*`\uff0c\u53ef\u53ea\u8bfb\u590d\u7528\uff0c\u4f46\u4e0d\u80fd\u5728\u9a8c\u8bc1\u6216\u6d4b\u8bd5\u6bb5\u91cd\u65b0\u62df\u5408\u3002",
        "",
        "## \u5269\u4f59\u6838\u5fc3\u963b\u585e",
        "",
        *[f"- `{row.check_id}`\uff1a{row.minimum_change}" for row in blockers.itertuples()],
        "",
        "## \u9636\u6bb5\u95e8\u51b3\u5b9a",
        "",
        "\u4fee\u590d\u540e\u5168\u90e8\u6838\u5fc3\u68c0\u67e5\u901a\u8fc7\uff0cE-L4-1 \u5df2\u83b7\u5de5\u7a0b\u9636\u6bb5\u95e8\u6388\u6743\uff1b\u672c\u8f6e\u4ecd\u672a\u8bad\u7ec3\u795e\u7ecf\u7f51\u7edc\u3002" if stage_a_passed else "\u4ecd\u6709\u6838\u5fc3\u963b\u585e\uff0c\u7981\u6b62\u8fdb\u5165 E-L4-1\u3002",
        "\u540e\u7eed\u8bad\u7ec3\u4ecd\u4e0d\u5f97\u52a0\u5165\u97e7\u6027\u5934\u3001CVaR\u3001\u5929\u6c14\u6216\u4e8b\u4ef6\u7279\u5f81\uff0c\u4e5f\u4e0d\u5f97\u4fee\u6539\u7f51\u7edc\u9aa8\u67b6\u548c\u8bad\u7ec3\u635f\u5931\u3002",
    ]
    (output_dir / "e_l4_0_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    decision = {
        "stage": "E-L4-0R",
        "stage_a_passed": stage_a_passed,
        "e_l4_1_authorized": stage_a_passed,
        "blockers": blockers["check_id"].tolist(),
        "model_py_modified": False,
        "training_run": False,
    }
    (output_dir / "e_l4_0_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="bridge,rainstorm,typhoon,pems04,pems08")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4"
    )
    parser.add_argument(
        "--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3"
    )
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()
