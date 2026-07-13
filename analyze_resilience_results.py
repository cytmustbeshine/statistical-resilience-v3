"""Analyze traffic resilience indicators from trained model predictions.

The script loads completed experiment directories, runs inference, converts
multi-horizon forecasts into a time-indexed prediction series, and evaluates
traffic resilience indicators on top of the predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from data import (  # noqa: E402
    SplitScaler,
    TrafficWindowDataset,
    build_static_adjacency,
    load_wide_traffic_csv,
)
from model import DSTSGCN  # noqa: E402
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN  # noqa: E402
from resilience_metrics import (  # noqa: E402
    compute_R_series,
    compute_baseline,
    compute_node_vulnerability,
    compute_resilience_indicators,
    compute_system_R,
    detect_event_window,
    detect_event_windows,
    select_dominant_event_window,
)
from statistical_tests import (
    diebold_mariano_test,
    holm_bonferroni_correction,
    moving_block_bootstrap_difference,
    wilcoxon_paired_test,
)  # noqa: E402


DATASET_CONFIGS = {
    "rainstorm": {
        "csv": r"D:\TrafficGNN\data\rainstorm_traffic_state.csv",
        "time_col": "Time",
        "value_suffix": "_volume",
        "extra_feature_cols": "altimeter,air_temp,relative_humidity,wind_speed,precip_accum_one_hour,visibility",
        "node_feature_suffixes": "_speed",
        "add_time_features": True,
        "event_col": "precip_accum_one_hour",
    },
    "bridge": {
        "csv": r"D:\TrafficGNN\data\bridge_collapse_flow.csv",
        "time_col": "Time",
        "value_suffix": "",
        "extra_feature_cols": "",
        "node_feature_suffixes": "",
        "add_time_features": True,
        "event_col": "",
    },
    "typhoon": {
        "csv": r"D:\TrafficGNN\data\typhoon_traffic_state_standard.csv",
        "time_col": "Time",
        "value_suffix": "_volume",
        "extra_feature_cols": "typhoon_intensity",
        "node_feature_suffixes": "_speed",
        "add_time_features": True,
        "event_col": "typhoon_intensity",
    },
    "pems04": {
        "csv": r"D:\TrafficGNN\data\public\PEMS04\pems04.csv",
        "time_col": "Time",
        "value_suffix": "_volume",
        "extra_feature_cols": "",
        "node_feature_suffixes": "_occupancy,_speed",
        "add_time_features": True,
        "event_col": "",
    },
    "pems08": {
        "csv": r"D:\TrafficGNN\data\public\PEMS08\pems08.csv",
        "time_col": "Time",
        "value_suffix": "_volume",
        "extra_feature_cols": "",
        "node_feature_suffixes": "_occupancy,_speed",
        "add_time_features": True,
        "event_col": "",
    },
}

METRIC_NAMES = ["mae", "rmse", "wape", "tail_mae_q90"]
RESILIENCE_ERROR_COLS = [
    "performance_loss_abs_error",
    "recovery_time_abs_error",
    "resilience_index_abs_error",
]


@dataclass
class RunRecord:
    """Metadata for a completed experiment run."""

    run_dir: Path
    config: dict[str, Any]
    dataset: str
    model: str
    seed: int
    weight_path: Path


def parse_csv_list(value: str) -> list[str]:
    """Parse comma-separated CLI values."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_suffix(value: Any) -> str | None:
    """Convert empty/null suffix values to ``None``."""
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like config values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "enabled"}


def finite_mean(values: np.ndarray) -> float:
    """Mean over finite values, or NaN if no finite values exist."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def finite_std(values: np.ndarray) -> float:
    """Standard deviation over finite values, or NaN if unavailable."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=0))


def find_runs(results_root: Path, datasets: set[str], models: set[str]) -> list[RunRecord]:
    """Find completed experiment directories under ``results_root``."""
    records: list[RunRecord] = []
    for config_path in sorted(results_root.rglob("run_config.json")):
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        dataset = str(config.get("dataset_name") or config.get("dataset") or "").lower()
        model = str(
            config.get("canonical_model_name")
            or config.get("model_name")
            or config.get("model")
            or ""
        ).lower()
        if dataset not in datasets or model not in models:
            continue
        run_dir = config_path.parent
        weight_path = locate_weight_file(run_dir, model)
        if weight_path is None:
            print(f"[WARNING] Missing model weights, skipping: {run_dir}")
            continue
        seed = int(config.get("seed", -1))
        records.append(
            RunRecord(
                run_dir=run_dir,
                config=config,
                dataset=dataset,
                model=model,
                seed=seed,
                weight_path=weight_path,
            )
        )
    return records


def array_cache_key(record: RunRecord) -> tuple[Any, ...]:
    """Build a cache key for model input arrays."""
    config = record.config
    return (
        record.dataset,
        record.model,
        str(config.get("csv", "")),
        str(config.get("value_suffix", "")),
        str(config.get("max_nodes", "")),
        str(config.get("extra_feature_cols", "")),
        str(config.get("node_feature_suffixes", "")),
        str(config.get("add_time_features", "")),
    )


def truth_cache_key(record: RunRecord, arrays: dict[str, Any]) -> tuple[Any, ...]:
    """Build a cache key for ground-truth resilience profiles."""
    return (
        record.dataset,
        int(arrays["raw_data"].shape[1]),
    )


def locate_weight_file(run_dir: Path, model: str) -> Path | None:
    """Locate the best checkpoint for a run."""
    candidates = []
    if model == "dcrnn_resilience_official_adapted":
        candidates.append(run_dir / "best_dcrnn_resilience_official_adapted.pt")
    else:
        candidates.append(run_dir / "best_dstsgcn.pt")
    candidates.extend(sorted(run_dir.glob("best_*.pt")))
    for path in candidates:
        if path.exists():
            return path
    return None


def config_value(config: dict[str, Any], dataset: str, key: str, default: Any = None) -> Any:
    """Read a config value, falling back to built-in dataset defaults."""
    if key in config and config[key] not in {None, ""}:
        return config[key]
    return DATASET_CONFIGS.get(dataset, {}).get(key, default)


def load_canonical_resilience_arrays(dataset: str, max_nodes: int | None) -> dict[str, Any]:
    """Load shared ground-truth arrays from the canonical dataset definition.

    Model input configurations can differ. The official-code-adapted DCRNN baseline is
    trained with traffic values only, while D-STSGCN runs may include event or
    weather features. Ground-truth resilience windows must be identical across
    models, so this loader always uses ``DATASET_CONFIGS`` and keeps event
    features when they exist.
    """
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset for resilience analysis: {dataset}")
    config = DATASET_CONFIGS[dataset]
    raw_data, node_names, feature_names = load_wide_traffic_csv(
        str(config["csv"]),
        value_suffix=parse_optional_suffix(config.get("value_suffix", "_volume")),
        time_col=str(config.get("time_col", "Time")),
        max_nodes=max_nodes,
        extra_feature_cols=parse_csv_list(str(config.get("extra_feature_cols", ""))),
        node_feature_suffixes=parse_csv_list(str(config.get("node_feature_suffixes", ""))),
        add_time_features=as_bool(config.get("add_time_features", False)),
        exclude_cols=parse_csv_list("ID,id"),
        return_feature_names=True,
    )
    event_col = str(config.get("event_col", "") or "")
    event_feature_idx = feature_names.index(event_col) if event_col in feature_names else None
    event_series = raw_data[:, 0, event_feature_idx].copy() if event_feature_idx is not None else None
    return {
        "raw_data": raw_data,
        "node_names": node_names,
        "feature_names": feature_names,
        "event_feature_idx": event_feature_idx,
        "event_series": event_series,
        "train_time_end": int(len(raw_data) * 0.6),
        "time_values": pd.to_datetime(
            pd.read_csv(str(config["csv"]))[str(config.get("time_col", "Time"))],
            errors="coerce",
        ).to_numpy(),
    }


def load_run_arrays(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Load raw and scaled arrays for a run using the original training config."""
    model_name = str(
        config.get("canonical_model_name")
        or config.get("model_name")
        or config.get("model")
        or ""
    ).lower()
    csv_path = str(config_value(config, dataset, "csv"))
    time_col = str(config_value(config, dataset, "time_col", "Time"))
    value_suffix = parse_optional_suffix(config_value(config, dataset, "value_suffix", "_volume"))
    max_nodes = config.get("max_nodes", None)
    if max_nodes in {"", "None", "none"}:
        max_nodes = None
    max_nodes = int(max_nodes) if max_nodes is not None else None

    extra_feature_cols = parse_csv_list(str(config_value(config, dataset, "extra_feature_cols", "")))
    node_feature_suffixes = parse_csv_list(str(config_value(config, dataset, "node_feature_suffixes", "")))
    add_time_features = as_bool(config_value(config, dataset, "add_time_features", False))

    raw_data, node_names, feature_names = load_wide_traffic_csv(
        csv_path,
        value_suffix=value_suffix,
        time_col=time_col,
        max_nodes=max_nodes,
        extra_feature_cols=extra_feature_cols,
        node_feature_suffixes=node_feature_suffixes,
        add_time_features=add_time_features,
        exclude_cols=parse_csv_list(str(config.get("exclude_cols", "ID,id"))),
        return_feature_names=True,
    )
    event_col = str(config_value(config, dataset, "event_col", "") or "")
    event_feature_idx = feature_names.index(event_col) if event_col in feature_names else None
    event_series = raw_data[:, 0, event_feature_idx].copy() if event_feature_idx is not None else None

    train_time_end = int(len(raw_data) * 0.6)
    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(raw_data[:train_time_end])
    scaled_data = scaler.transform(raw_data)
    return {
        "raw_data": raw_data,
        "scaled_data": scaled_data,
        "node_names": node_names,
        "feature_names": feature_names,
        "event_feature_idx": event_feature_idx,
        "event_series": event_series,
        "train_time_end": train_time_end,
        "scaler": scaler,
    }


def event_window_values(event_series: np.ndarray, history: int, num_windows: int) -> np.ndarray:
    """Map raw event series to sliding-window event values."""
    return np.asarray(
        [event_series[min(idx + history - 1, len(event_series) - 1)] for idx in range(num_windows)],
        dtype=np.float64,
    )


def split_indices(
    num_windows: int,
    history: int,
    split_mode: str,
    event_series: np.ndarray | None,
    event_window_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the train/val/test split used during training."""
    train_end = int(num_windows * 0.6)
    val_end = int(num_windows * 0.8)
    indices = np.arange(num_windows)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    if split_mode != "event_aware" or event_series is None:
        return train_indices, val_indices, test_indices

    event_values = event_window_values(event_series, history, num_windows)
    post_train = indices[train_end:]
    post_event = post_train[event_values[post_train] > event_window_threshold]
    if post_event.size == 0:
        return train_indices, val_indices, test_indices

    test_count = len(test_indices)
    selected_events = post_event[-min(post_event.size, test_count) :]
    selected_mask = np.zeros(num_windows, dtype=bool)
    selected_mask[selected_events] = True
    remaining_slots = test_count - selected_events.size
    if remaining_slots > 0:
        fill_candidates = post_train[~selected_mask[post_train]]
        selected_mask[fill_candidates[-remaining_slots:]] = True
    proposed_test = np.sort(post_train[selected_mask[post_train]])
    remaining_post = post_train[~selected_mask[post_train]]
    val_count = val_end - train_end
    if remaining_post.size < val_count:
        return train_indices, val_indices, test_indices
    return train_indices, remaining_post[:val_count], proposed_test


def build_static_adj(config: dict[str, Any], arrays: dict[str, Any], dataset: str, device: str) -> torch.Tensor:
    """Build the static adjacency matrix needed for inference."""
    scaled_data = arrays["scaled_data"]
    train_time_end = arrays["train_time_end"]
    node_names = arrays["node_names"]
    model_name = str(
        config.get("canonical_model_name")
        or config.get("model_name")
        or config.get("model")
        or ""
    ).lower()
    if model_name in {"dcrnn_resilience_official_adapted"}:
        adj_np = build_static_adjacency(scaled_data[:train_time_end], node_names, source="corr", corr_threshold=0.2)
        return torch.tensor(adj_np, dtype=torch.float32, device=device)

    adj_data = scaled_data[:train_time_end] if as_bool(config.get("static_adj_train_only"), False) else scaled_data
    adj_np = build_static_adjacency(
        adj_data,
        node_names,
        source=str(config.get("adj_source", "corr")),
        corr_threshold=float(config.get("corr_threshold", 0.2)),
        corr_method=str(config.get("corr_method", "pearson")),
        corr_shrinkage_lambda=float(config.get("corr_shrinkage_lambda", 0.1)),
        auto_corr_shrinkage=as_bool(config.get("auto_corr_shrinkage"), False),
        adj_path=config.get("adj_path") or None,
        road_weight=float(config.get("road_weight", 0.5)),
        granger_lag=int(config.get("granger_lag", 3)),
        granger_p_threshold=float(config.get("granger_p_threshold", 0.05)),
        granger_cache_path=config.get("granger_cache_path") or None,
        from_col=str(config.get("adj_from_col", "from_node")),
        to_col=str(config.get("adj_to_col", "to_node")),
        weight_col=config.get("adj_weight_col") or None,
        link_id_col=config.get("adj_link_id_col") or "link_id",
        directed=not as_bool(config.get("adj_undirected"), False),
        road_knn=int(config.get("road_knn", 3)),
        geo_wkt_col=str(config.get("adj_geo_wkt_col", "geo_wkt")),
    )
    return torch.tensor(adj_np, dtype=torch.float32, device=device)


def instantiate_model(record: RunRecord, arrays: dict[str, Any], static_adj: torch.Tensor, device: str) -> torch.nn.Module:
    """Instantiate and load a trained model checkpoint."""
    config = record.config
    model_name = record.model
    num_nodes = len(arrays["node_names"])
    input_dim = int(arrays["scaled_data"].shape[-1])
    horizon = int(config.get("horizon", config.get("history_steps", 12)))
    if model_name == "dcrnn_resilience_official_adapted":
        model = OfficialAdaptedDCRNN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=1,
            rnn_units=int(config.get("rnn_units", 256)),
            num_rnn_layers=int(config.get("num_rnn_layers", 2)),
            horizon=int(config.get("horizon", 12)),
            max_diffusion_step=int(config.get("max_diffusion_step", 1)),
            cl_decay_steps=int(config.get("cl_decay_steps", 2000)),
            use_curriculum_learning=not as_bool(config.get("disable_curriculum_learning"), False),
        ).to(device)
    else:
        graph_learner_type = str(
            config.get("requested_graph_learner_type")
            or config.get("graph_learner_type")
            or "lmln"
        )
        model = DSTSGCN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=1,
            horizon=horizon,
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_blocks=int(config.get("num_blocks", 2)),
            graph_learner_type=graph_learner_type,
            fusion_mode=str(config.get("fusion_mode", "fusion")),
            fusion_type=str(config.get("fusion_type", config.get("effective_fusion_type", "quality"))),
            dynamic_top_k=int(config.get("dynamic_top_k", 3)) if int(config.get("dynamic_top_k", 3)) > 0 else None,
            quality_gate_bias=float(config.get("quality_gate_bias", -1.0)),
            matrix_hidden_dim=int(config.get("matrix_hidden_dim", 256)),
            static_adj_is_normalized=False,
            num_diffusion_steps=int(config.get("num_diffusion_steps", 2)),
            event_dim=1 if arrays["event_feature_idx"] is not None else 0,
            ood_gamma_max=float(config.get("ood_gamma_max", 0.35)),
            resilience_aux=as_bool(config.get("resilience_aux_enabled"), False),
            resilience_target_mode=str(config.get("resilience_target_mode", "ratio")),
            resilience_uncertainty_weighting=as_bool(
                config.get("resilience_uncertainty_weighting"), False
            ),
        ).to(device)
    state = torch.load(record.weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def extract_event_signal(x: torch.Tensor, event_feature_idx: int | None) -> torch.Tensor | None:
    """Extract batch event signal for D-STSGCN graph fusion."""
    if event_feature_idx is None:
        return None
    return x[:, -1, :, event_feature_idx].mean(dim=1, keepdim=True)


def infer_predictions(record: RunRecord, arrays: dict[str, Any], device: str) -> dict[str, Any]:
    """Run inference on a run's test split and aggregate horizon forecasts by time."""
    config = record.config
    scaled_data = arrays["scaled_data"]
    raw_traffic = arrays["raw_data"][..., 0]
    history = int(config.get("history", config.get("history_steps", 12)))
    horizon = int(config.get("horizon", 12))
    dataset = TrafficWindowDataset(scaled_data, history=history, horizon=horizon, target_dim=0)
    split_info = config.get("split_info", {})
    if (
        str(config.get("split_mode", "chronological")) == "event_aligned"
        and split_info.get("test_start_window") is not None
    ):
        test_indices = np.arange(int(split_info["test_start_window"]), len(dataset), dtype=int)
    else:
        _, _, test_indices = split_indices(
            len(dataset),
            history=history,
            split_mode=str(config.get("split_mode", "chronological")),
            event_series=arrays["event_series"],
            event_window_threshold=float(config.get("event_window_threshold", 0.0)),
        )
    static_adj = build_static_adj(config, arrays, record.dataset, device)
    model = instantiate_model(record, arrays, static_adj, device)

    pred_sum = np.zeros_like(raw_traffic, dtype=float)
    pred_count = np.zeros_like(raw_traffic, dtype=float)
    batch_size = int(config.get("batch_size", 64) or 64)
    with torch.no_grad():
        for start in range(0, len(test_indices), batch_size):
            batch_indices = test_indices[start : start + batch_size]
            x_np = np.stack([scaled_data[idx : idx + history] for idx in batch_indices], axis=0)
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            if record.model in {"dcrnn_resilience_official_adapted"}:
                pred_scaled = model(x, static_adj)
            else:
                event_signal = extract_event_signal(x, arrays["event_feature_idx"])
                pred_scaled = model(x, static_adj, event_signal=event_signal)
            pred_np = pred_scaled.detach().cpu().numpy()
            pred_raw = arrays["scaler"].inverse_transform_traffic(pred_np)[..., 0]
            for batch_pos, window_idx in enumerate(batch_indices):
                for h in range(horizon):
                    target_t = int(window_idx + history + h)
                    if target_t >= pred_sum.shape[0]:
                        continue
                    pred_sum[target_t] += pred_raw[batch_pos, h]
                    pred_count[target_t] += 1.0

    pred_series = np.full_like(raw_traffic, np.nan, dtype=float)
    covered = pred_count > 0
    pred_series[covered] = pred_sum[covered] / pred_count[covered]
    finite = np.isfinite(pred_series) & np.isfinite(raw_traffic)
    mae = float(np.mean(np.abs(pred_series[finite] - raw_traffic[finite]))) if finite.any() else float("nan")
    rmse = (
        float(np.sqrt(np.mean((pred_series[finite] - raw_traffic[finite]) ** 2)))
        if finite.any()
        else float("nan")
    )
    if finite.any():
        abs_errors = np.abs(pred_series[finite] - raw_traffic[finite])
        denominator = float(np.sum(np.abs(raw_traffic[finite])))
        wape = float(np.sum(abs_errors) / max(denominator, 1e-6) * 100.0)
        q90 = float(np.quantile(abs_errors, 0.90))
        tail = abs_errors[abs_errors >= q90]
        tail_mae_q90 = float(np.mean(tail)) if tail.size else float("nan")
    else:
        wape = float("nan")
        tail_mae_q90 = float("nan")
    return {
        "pred_series": pred_series,
        "covered_mask": covered,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "tail_mae_q90": tail_mae_q90,
    }


def compute_R_preserve_nan(data: np.ndarray, baseline: dict[str, np.ndarray]) -> np.ndarray:
    """Compute performance ratio while preserving missing predictions as NaN."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    tod_mean = np.asarray(baseline["tod_mean"], dtype=float)
    bins = np.arange(data.shape[0]) % tod_mean.shape[0]
    denom = np.maximum(tod_mean[bins], 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = data / denom
    ratio = np.where(np.isfinite(data), ratio, np.nan)
    return np.clip(ratio, 0.0, 2.0)


def system_R_preserve_nan(R: np.ndarray) -> np.ndarray:
    """Aggregate node ratios while preserving all-missing time steps."""
    finite = np.isfinite(R)
    counts = finite.sum(axis=1)
    sums = np.where(finite, R, 0.0).sum(axis=1)
    result = np.full(R.shape[0], np.nan, dtype=float)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result


def ground_truth_for_dataset(
    dataset: str,
    arrays: dict[str, Any],
    max_gap_steps: int = 12,
    min_event_steps: int = 3,
    recovery_hold_steps: int = 6,
) -> dict[str, Any]:
    """Compute segmented ground-truth resilience events for a dataset."""
    raw_traffic = arrays["raw_data"][..., 0]
    baseline = compute_baseline(raw_traffic, arrays["train_time_end"])
    true_R = compute_R_series(raw_traffic, baseline, use_tod=True)
    true_R_sys = compute_system_R(true_R)
    explicit_start = None
    detection_source = "event_signal" if arrays["event_series"] is not None else "performance_drop"
    if dataset == "bridge":
        target_time = np.datetime64("2007-07-29T00:00:00")
        time_values = np.asarray(arrays.get("time_values", []), dtype="datetime64[ns]")
        finite = ~np.isnat(time_values)
        if finite.any():
            explicit_start = int(np.argmin(np.abs(time_values - target_time)))
            detection_source = "explicit_event_time"
    windows = detect_event_windows(
        arrays["event_series"],
        true_R_sys,
        max_gap_steps=max_gap_steps,
        min_event_steps=min_event_steps,
        explicit_event_start=explicit_start,
    )
    dominant = select_dominant_event_window(windows, arrays["event_series"], true_R_sys)
    if dominant is None:
        raise ValueError(f"No resilience event detected for {dataset}.")
    event_start, event_end = dominant
    indicators = compute_resilience_indicators(
        true_R_sys,
        event_start=event_start,
        event_end=event_end,
        baseline_end=arrays["train_time_end"],
        recovery_hold_steps=recovery_hold_steps,
    )
    vulnerability = compute_node_vulnerability(true_R, event_start, event_end)
    top5 = np.argsort(-vulnerability, kind="stable")[: min(5, vulnerability.size)].astype(int).tolist()
    row = {
        "dataset": dataset,
        "n_nodes": int(raw_traffic.shape[1]),
        "event_start": int(event_start),
        "event_end": int(event_end),
        "event_count": int(len(windows)),
        "event_detection_source": detection_source,
        "min_R": indicators["min_R"],
        "performance_loss_hours": indicators["performance_loss"],
        "recovery_time_hours": indicators["recovery_time_hours"],
        "recovery_slope": indicators["recovery_slope"],
        "resilience_index": indicators["resilience_index"],
        "nadir_idx": indicators["nadir_idx"],
        "recovery_idx": indicators["recovery_idx"],
        "observed_recovery": indicators["observed_recovery"],
        "event_duration_hours": indicators["event_duration_hours"],
        "top5_vulnerable_nodes": ",".join(str(idx) for idx in top5),
    }
    return {
        "baseline": baseline,
        "true_R": true_R,
        "true_R_sys": true_R_sys,
        "event_start": event_start,
        "event_end": event_end,
        "event_windows": windows,
        "event_detection_source": detection_source,
        "indicators": indicators,
        "vulnerability": vulnerability,
        "top5": top5,
        "row": row,
    }

def resilience_for_prediction(
    pred_series: np.ndarray,
    truth: dict[str, Any],
    train_time_end: int,
    min_event_coverage: float = 0.90,
    recovery_hold_steps: int = 6,
) -> dict[str, float | bool]:
    """Compute indicators on the exact observed/predicted common event support."""
    pred_R = compute_R_preserve_nan(pred_series, truth["baseline"])
    pred_R_sys = system_R_preserve_nan(pred_R)
    true_R_sys = np.asarray(truth["true_R_sys"], dtype=float)
    start = int(truth["event_start"])
    recovery_idx = truth["indicators"].get("recovery_idx", float("nan"))
    end = (
        int(recovery_idx) + max(int(recovery_hold_steps), 1) - 1
        if np.isfinite(recovery_idx)
        else int(truth["event_end"])
    )
    end = min(end, pred_R_sys.size - 1)
    interval = np.arange(start, end + 1)
    common = np.isfinite(pred_R_sys[interval]) & np.isfinite(true_R_sys[interval])
    event_interval = np.arange(start, int(truth["event_end"]) + 1)
    event_common = np.isfinite(pred_R_sys[event_interval]) & np.isfinite(true_R_sys[event_interval])
    event_coverage = float(event_common.mean()) if event_common.size else 0.0
    recovery_interval = np.arange(int(truth["event_end"]), end + 1)
    recovery_common = np.isfinite(pred_R_sys[recovery_interval]) & np.isfinite(true_R_sys[recovery_interval])
    recovery_coverage = float(recovery_common.mean()) if recovery_common.size else float("nan")
    if event_coverage < min_event_coverage:
        raise ValueError(
            f"event prediction coverage {event_coverage:.3f} is below {min_event_coverage:.3f}"
        )
    aligned_pred = np.full_like(pred_R_sys, np.nan, dtype=float)
    aligned_true = np.full_like(true_R_sys, np.nan, dtype=float)
    aligned_pred[interval[common]] = pred_R_sys[interval[common]]
    aligned_true[interval[common]] = true_R_sys[interval[common]]
    pred_indicators = compute_resilience_indicators(
        aligned_pred,
        event_start=start,
        event_end=int(truth["event_end"]),
        baseline_end=train_time_end,
        recovery_hold_steps=recovery_hold_steps,
    )
    true_indicators = compute_resilience_indicators(
        aligned_true,
        event_start=start,
        event_end=int(truth["event_end"]),
        baseline_end=train_time_end,
        recovery_hold_steps=recovery_hold_steps,
    )
    pred_indicators.update(
        {
            "event_coverage_ratio": event_coverage,
            "recovery_coverage_ratio": recovery_coverage,
            "evaluation_start_idx": float(start),
            "evaluation_end_idx": float(end),
            "aligned_true_performance_loss": true_indicators["performance_loss"],
            "aligned_true_recovery_time_hours": true_indicators["recovery_time_hours"],
            "aligned_true_resilience_index": true_indicators["resilience_index"],
        }
    )
    return pred_indicators

def node_error_profile(
    pred_series: np.ndarray,
    raw_traffic: np.ndarray,
    event_start: int,
    event_end: int,
    top5: list[int],
) -> dict[str, float]:
    """Compare prediction error on vulnerable nodes against other nodes."""
    event_start = max(0, int(event_start))
    event_end = min(int(event_end), raw_traffic.shape[0] - 1)
    err = np.abs(pred_series[event_start : event_end + 1] - raw_traffic[event_start : event_end + 1])
    top5 = [idx for idx in top5 if 0 <= idx < raw_traffic.shape[1]]
    other = [idx for idx in range(raw_traffic.shape[1]) if idx not in set(top5)]
    return {
        "top5_event_mae": finite_mean(err[:, top5]) if top5 else float("nan"),
        "other_event_mae": finite_mean(err[:, other]) if other else float("nan"),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    """Render a compact Markdown table."""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float) or isinstance(value, np.floating):
                cells.append("" if math.isnan(float(value)) else f"{float(value):.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_prediction_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prediction and resilience indicator errors across seeds."""
    if df.empty:
        return df
    agg_cols = [*METRIC_NAMES, *RESILIENCE_ERROR_COLS]
    return (
        df.groupby(["dataset", "n_nodes", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in agg_cols},
            **{f"{col}_std": (col, "std") for col in agg_cols},
        )
        .sort_values(["dataset", "n_nodes", "model"])
    )



def time_series_tests(
    error_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    block_length: int,
    repetitions: int,
    hac_lag: int,
) -> pd.DataFrame:
    """Compare paired per-time absolute-error series with dependence-aware tests."""
    rows: list[dict[str, Any]] = []
    if error_df.empty or "time_abs_errors" not in error_df.columns:
        return pd.DataFrame(rows)
    keys = ["dataset", "n_nodes", "seed"]
    for key, group in error_df.groupby(keys):
        mapping = {str(row["model"]): np.asarray(row["time_abs_errors"], dtype=float) for _, row in group.iterrows()}
        if model_a not in mapping or model_b not in mapping:
            continue
        a, b = mapping[model_a], mapping[model_b]
        length = min(a.size, b.size)
        valid = np.isfinite(a[:length]) & np.isfinite(b[:length])
        a, b = a[:length][valid], b[:length][valid]
        if a.size == 0:
            continue
        boot = moving_block_bootstrap_difference(
            a, b, block_length=block_length, repetitions=repetitions
        )
        dm = diebold_mariano_test(a, b, hac_lag=hac_lag)
        rows.append(
            {
                "dataset": key[0],
                "n_nodes": key[1],
                "seed": key[2],
                "model_a": model_a,
                "model_b": model_b,
                "mean_abs_error_diff": boot["estimate"],
                "block_ci_lower_95": boot["ci_lower_95"],
                "block_ci_upper_95": boot["ci_upper_95"],
                "dm_statistic": dm["statistic"],
                "dm_p_value": dm["p_value"],
                "dm_warning": dm.get("warning", ""),
                "n_time_points": int(a.size),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["dm_p_value_adj_holm"] = holm_bonferroni_correction(result["dm_p_value"].tolist())
        result["dm_significance"] = result["dm_p_value_adj_holm"].map(significance_marker)
    return result

def significance_marker(p_value: float) -> str:
    """Return a conventional significance marker."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def paired_tests(error_df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Run paired Wilcoxon tests for two configured models."""
    rows = []
    model_a = model_a.lower()
    model_b = model_b.lower()
    if error_df.empty or not {model_a, model_b}.issubset(set(error_df["model"])):
        return pd.DataFrame(rows)
    for metric in RESILIENCE_ERROR_COLS:
        pivot = error_df.pivot_table(index=["dataset", "n_nodes", "seed"], columns="model", values=metric, aggfunc="mean")
        if not {model_a, model_b}.issubset(pivot.columns):
            continue
        paired = pivot[[model_a, model_b]].dropna()
        if paired.empty:
            continue
        result = wilcoxon_paired_test(paired[model_a].to_numpy(), paired[model_b].to_numpy())
        rows.append(
            {
                "indicator": metric,
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": int(result["n_pairs"]),
                "model_a_mean_abs_error": float(paired[model_a].mean()),
                "model_b_mean_abs_error": float(paired[model_b].mean()),
                "median_diff_a_minus_b": result["median_diff"],
                "ci_lower_95": result["ci_lower_95"],
                "ci_upper_95": result["ci_upper_95"],
                "p_value": result["p_value"],
                "effect_size_r": result["effect_size_r"],
            }
        )
    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df["p_value_adj_holm"] = holm_bonferroni_correction(result_df["p_value"].tolist())
        result_df["significance"] = result_df["p_value_adj_holm"].map(significance_marker)
    return result_df


def write_latex_table(agg_df: pd.DataFrame, output_path: Path, models: list[str]) -> None:
    """Write a LaTeX-ready comparison table."""
    metrics = [
        ("mae_mean", "MAE"),
        ("performance_loss_abs_error_mean", "Loss Err."),
        ("recovery_time_abs_error_mean", "Rec. Err."),
        ("resilience_index_abs_error_mean", "RIdx Err."),
    ]
    lines = [
        "\\begin{tabular}{l" + "rrrr" * len(models) + "}",
        "\\toprule",
        "Dataset & " + " & ".join([f"\\multicolumn{{4}}{{c}}{{{model}}}" for model in models]) + " \\\\",
        " & " + " & ".join([label for _ in models for _, label in metrics]) + " \\\\",
        "\\midrule",
    ]
    for (dataset, n_nodes), sub in agg_df.groupby(["dataset", "n_nodes"]):
        row = [f"{dataset} ({int(n_nodes)})"]
        best_by_metric: dict[str, float] = {}
        for metric, _ in metrics:
            values = [float(sub[sub["model"] == model][metric].iloc[0]) for model in models if not sub[sub["model"] == model].empty]
            finite = [value for value in values if np.isfinite(value)]
            best_by_metric[metric] = min(finite) if finite else float("nan")
        for model in models:
            model_row = sub[sub["model"] == model]
            for metric, _ in metrics:
                if model_row.empty:
                    row.append("--")
                    continue
                value = float(model_row[metric].iloc[0])
                cell = "--" if not np.isfinite(value) else f"{value:.4f}"
                if np.isfinite(value) and np.isclose(value, best_by_metric[metric], rtol=1e-9, atol=1e-12):
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output_dir: Path,
    gt_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    test_df: pd.DataFrame,
    node_df: pd.DataFrame,
) -> None:
    """Write a UTF-8 Chinese report suitable for thesis review."""
    lines = ["# \u4ea4\u901a\u97e7\u6027\u7ed3\u679c\u5206\u6790", "", "## \u771f\u5b9e\u4ea4\u901a\u97e7\u6027\u6982\u51b5"]
    lines.append(markdown_table(gt_df, list(gt_df.columns)) if not gt_df.empty else "\u6ca1\u6709\u53ef\u7528\u7684\u771f\u5b9e\u97e7\u6027\u8bb0\u5f55\u3002")
    lines.extend(["", "## \u6a21\u578b\u4ea4\u901a\u9884\u6d4b\u7cbe\u5ea6"])
    if agg_df.empty:
        lines.append("\u6ca1\u6709\u53ef\u7528\u7684\u6a21\u578b\u9884\u6d4b\u7ed3\u679c\u3002")
    else:
        cols = ["dataset", "n_nodes", "model", "seeds", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "wape_mean", "wape_std", "tail_mae_q90_mean", "tail_mae_q90_std"]
        lines.append(markdown_table(agg_df[cols], cols))
    lines.extend(["", "## \u97e7\u6027\u6307\u6807\u9884\u6d4b\u8bef\u5dee"])
    if agg_df.empty:
        lines.append("\u6ca1\u6709\u53ef\u7528\u7684\u97e7\u6027\u6307\u6807\u9884\u6d4b\u8bef\u5dee\u8bb0\u5f55\u3002")
    else:
        cols = [
            "dataset", "n_nodes", "model",
            "performance_loss_abs_error_mean", "performance_loss_abs_error_std",
            "recovery_time_abs_error_mean", "recovery_time_abs_error_std",
            "resilience_index_abs_error_mean", "resilience_index_abs_error_std",
        ]
        lines.append(markdown_table(agg_df[cols], cols))
    lines.extend(["", "## \u7edf\u8ba1\u68c0\u9a8c"])
    lines.append(markdown_table(test_df, list(test_df.columns)) if not test_df.empty else "\u914d\u5bf9\u6837\u672c\u4e0d\u8db3\u3002")
    lines.extend(["", "## \u8282\u70b9\u8106\u5f31\u6027\u5206\u6790"])
    if node_df.empty:
        lines.append("\u6ca1\u6709\u53ef\u7528\u7684\u8282\u70b9\u8106\u5f31\u6027\u8bb0\u5f55\u3002")
    else:
        summary = node_df.groupby(["dataset", "n_nodes", "model"], as_index=False).agg(
            seeds=("seed", "nunique"),
            top5_event_mae=("top5_event_mae", "mean"),
            other_event_mae=("other_event_mae", "mean"),
            top5_minus_other=("top5_minus_other", "mean"),
        )
        lines.append("Top-5 \u8106\u5f31\u8282\u70b9\u7531\u771f\u5b9e\u4e8b\u4ef6\u7a97\u53e3\u5185\u7684\u8282\u70b9\u97e7\u6027\u7f3a\u5931\u6392\u5e8f\u5f97\u5230\u3002")
        lines.append(markdown_table(summary, list(summary.columns)))
    (output_dir / "resilience_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\resilience_analysis")
    parser.add_argument("--datasets", default="bridge,rainstorm,typhoon")
    parser.add_argument("--models", default="stat_resilience_dstsgcn_v2,dcrnn_resilience_official_adapted")
    parser.add_argument("--test-model-a", default="quality_resilience_aux_v1")
    parser.add_argument("--test-model-b", default="dcrnn_resilience_official_adapted")
    parser.add_argument("--max-nodes", type=int, default=None, help="Only analyze runs with this max_nodes value.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-event-coverage", type=float, default=0.90)
    parser.add_argument("--event-max-gap-steps", type=int, default=12)
    parser.add_argument("--event-min-steps", type=int, default=3)
    parser.add_argument("--recovery-hold-steps", type=int, default=6)
    parser.add_argument("--bootstrap-block-length", type=int, default=12)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but unavailable; using CPU.")
        device = "cpu"
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name.lower() for name in parse_csv_list(args.datasets)}
    models = [name.lower() for name in parse_csv_list(args.models)]

    records = find_runs(results_root, datasets, set(models))
    if args.max_nodes is not None:
        records = [
            record
            for record in records
            if int(record.config.get("max_nodes", -1)) == int(args.max_nodes)
        ]
    if not records:
        print("[WARNING] No completed runs found.")

    arrays_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    truth_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    gt_rows = []
    error_rows = []
    node_rows = []
    event_level_rows = []

    for record in records:
        a_key = array_cache_key(record)
        if a_key not in arrays_by_key:
            arrays_by_key[a_key] = load_run_arrays(record.config, record.dataset)
        arrays = arrays_by_key[a_key]
        t_key = truth_cache_key(record, arrays)
        if t_key not in truth_by_key:
            truth_arrays = load_canonical_resilience_arrays(record.dataset, int(arrays["raw_data"].shape[1]))
            truth_by_key[t_key] = {
                "arrays": truth_arrays,
                "truth": ground_truth_for_dataset(
                    record.dataset,
                    truth_arrays,
                    max_gap_steps=args.event_max_gap_steps,
                    min_event_steps=args.event_min_steps,
                    recovery_hold_steps=args.recovery_hold_steps,
                ),
            }
            gt_rows.append(truth_by_key[t_key]["truth"]["row"])
        truth_arrays = truth_by_key[t_key]["arrays"]
        truth = truth_by_key[t_key]["truth"]
        print(f"[INFO] Inference: dataset={record.dataset} model={record.model} seed={record.seed}")
        try:
            pred_info = infer_predictions(record, arrays, device)
        except Exception as exc:
            print(f"[WARNING] Failed inference for {record.run_dir}: {exc}")
            continue
        try:
            pred_indicators = resilience_for_prediction(
                pred_info["pred_series"],
                truth,
                train_time_end=truth_arrays["train_time_end"],
                min_event_coverage=args.min_event_coverage,
                recovery_hold_steps=args.recovery_hold_steps,
            )
        except ValueError as exc:
            print(f"[WARNING] Skipping resilience metrics for {record.run_dir}: {exc}")
            continue
        true_indicators = {
            "performance_loss": pred_indicators["aligned_true_performance_loss"],
            "recovery_time_hours": pred_indicators["aligned_true_recovery_time_hours"],
            "resilience_index": pred_indicators["aligned_true_resilience_index"],
        }
        performance_loss_error = pred_indicators["performance_loss"] - true_indicators["performance_loss"]
        recovery_time_error = pred_indicators["recovery_time_hours"] - true_indicators["recovery_time_hours"]
        resilience_index_error = pred_indicators["resilience_index"] - true_indicators["resilience_index"]
        error_rows.append(
            {
                "dataset": record.dataset,
                "n_nodes": int(arrays["raw_data"].shape[1]),
                "model": record.model,
                "seed": record.seed,
                "mae": pred_info["mae"],
                "rmse": pred_info["rmse"],
                "wape": pred_info["wape"],
                "tail_mae_q90": pred_info["tail_mae_q90"],
                "performance_loss_error": performance_loss_error,
                "performance_loss_abs_error": abs(performance_loss_error)
                if np.isfinite(performance_loss_error)
                else float("nan"),
                "recovery_time_error": recovery_time_error,
                "recovery_time_abs_error": abs(recovery_time_error)
                if np.isfinite(recovery_time_error)
                else float("nan"),
                "resilience_index_error": resilience_index_error,
                "resilience_index_abs_error": abs(resilience_index_error)
                if np.isfinite(resilience_index_error)
                else float("nan"),
                "predicted_performance_loss_hours": pred_indicators["performance_loss"],
                "predicted_recovery_time_hours": pred_indicators["recovery_time_hours"],
                "predicted_resilience_index": pred_indicators["resilience_index"],
                "true_performance_loss_hours": true_indicators["performance_loss"],
                "true_recovery_time_hours": true_indicators["recovery_time_hours"],
                "true_resilience_index": true_indicators["resilience_index"],
                "event_coverage_ratio": pred_indicators["event_coverage_ratio"],
                "recovery_coverage_ratio": pred_indicators["recovery_coverage_ratio"],
                "evaluation_start_idx": pred_indicators["evaluation_start_idx"],
                "evaluation_end_idx": pred_indicators["evaluation_end_idx"],
                "event_detection_source": truth["event_detection_source"],
                "event_count": len(truth["event_windows"]),
                "dominant_event_start": truth["event_start"],
                "dominant_event_end": truth["event_end"],
                "time_abs_errors": system_R_preserve_nan(
                    np.abs(pred_info["pred_series"] - truth_arrays["raw_data"][..., 0])
                ).tolist(),
                "run_dir": str(record.run_dir),
            }
        )
        for event_id, (segment_start, segment_end) in enumerate(truth["event_windows"], start=1):
            segment_truth = dict(truth)
            segment_truth["event_start"] = segment_start
            segment_truth["event_end"] = segment_end
            segment_truth["indicators"] = compute_resilience_indicators(
                truth["true_R_sys"],
                segment_start,
                segment_end,
                truth_arrays["train_time_end"],
                recovery_hold_steps=args.recovery_hold_steps,
            )
            try:
                segment_pred = resilience_for_prediction(
                    pred_info["pred_series"],
                    segment_truth,
                    truth_arrays["train_time_end"],
                    min_event_coverage=args.min_event_coverage,
                    recovery_hold_steps=args.recovery_hold_steps,
                )
            except ValueError:
                continue
            event_level_rows.append(
                {
                    "dataset": record.dataset,
                    "model": record.model,
                    "seed": record.seed,
                    "event_id": event_id,
                    "event_start": segment_start,
                    "event_end": segment_end,
                    "performance_loss_abs_error": abs(
                        segment_pred["performance_loss"]
                        - segment_pred["aligned_true_performance_loss"]
                    ),
                    "recovery_time_abs_error": abs(
                        segment_pred["recovery_time_hours"]
                        - segment_pred["aligned_true_recovery_time_hours"]
                    )
                    if np.isfinite(segment_pred["recovery_time_hours"])
                    and np.isfinite(segment_pred["aligned_true_recovery_time_hours"])
                    else float("nan"),
                    "resilience_index_abs_error": abs(
                        segment_pred["resilience_index"]
                        - segment_pred["aligned_true_resilience_index"]
                    ),
                    "event_coverage_ratio": segment_pred["event_coverage_ratio"],
                }
            )
        node_profile = node_error_profile(
            pred_info["pred_series"],
            truth_arrays["raw_data"][..., 0],
            truth["event_start"],
            truth["event_end"],
            truth["top5"],
        )
        node_rows.append(
            {
                "dataset": record.dataset,
                "n_nodes": int(arrays["raw_data"].shape[1]),
                "model": record.model,
                "seed": record.seed,
                "top5_nodes": ",".join(str(idx) for idx in truth["top5"]),
                "top5_event_mae": node_profile["top5_event_mae"],
                "other_event_mae": node_profile["other_event_mae"],
                "top5_minus_other": node_profile["top5_event_mae"] - node_profile["other_event_mae"],
            }
        )

    gt_df = (
        pd.DataFrame(gt_rows).drop_duplicates(subset=["dataset", "n_nodes"]).sort_values(["dataset", "n_nodes"])
        if gt_rows
        else pd.DataFrame()
    )
    error_df = (
        pd.DataFrame(error_rows).sort_values(["dataset", "n_nodes", "model", "seed"])
        if error_rows
        else pd.DataFrame()
    )
    node_df = (
        pd.DataFrame(node_rows).sort_values(["dataset", "n_nodes", "model", "seed"])
        if node_rows
        else pd.DataFrame()
    )
    event_level_df = pd.DataFrame(event_level_rows)
    agg_df = aggregate_prediction_errors(error_df)
    test_df = paired_tests(error_df, args.test_model_a, args.test_model_b)
    time_test_df = time_series_tests(
        error_df,
        args.test_model_a,
        args.test_model_b,
        block_length=args.bootstrap_block_length,
        repetitions=args.bootstrap_repetitions,
        hac_lag=11,
    )

    gt_df.to_csv(output_dir / "resilience_ground_truth.csv", index=False, encoding="utf-8-sig")
    error_csv_df = error_df.drop(columns=["time_abs_errors"], errors="ignore")
    error_csv_df.to_csv(output_dir / "resilience_prediction_errors.csv", index=False, encoding="utf-8-sig")
    agg_df.to_csv(output_dir / "resilience_prediction_error_summary.csv", index=False, encoding="utf-8-sig")
    node_df.to_csv(output_dir / "resilience_node_vulnerability.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "resilience_statistical_tests.csv", index=False, encoding="utf-8-sig")
    time_test_df.to_csv(
        output_dir / "resilience_time_series_tests.csv", index=False, encoding="utf-8-sig"
    )
    event_level_df.to_csv(
        output_dir / "resilience_event_level_errors.csv", index=False, encoding="utf-8-sig"
    )
    write_report(output_dir, gt_df, agg_df, test_df, node_df)
    if not agg_df.empty:
        write_latex_table(agg_df, output_dir / "resilience_latex_table.tex", models)
    else:
        (output_dir / "resilience_latex_table.tex").write_text("% No resilience rows available.\n", encoding="utf-8")

    print(f"Resilience analysis saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

