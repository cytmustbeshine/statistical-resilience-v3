"""Leakage-free data and artifact utilities for indirect L4 forecasting.

The module does not train a model. It prepares strict chronological splits,
variable-specific scalers, explicit matched node plans, auditable checkpoints,
and timestamp-aligned prediction archives for a later E-L4-1 runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from data import (
    StandardScaler,
    load_wide_traffic_csv,
    read_csv_with_fallback,
    split_traffic_window_indices_strict,
)
from flow_speed_resilience import match_node_variables


CHECKPOINT_METADATA_FIELDS = (
    "dataset",
    "variable",
    "node_names",
    "history",
    "horizon",
    "train_time_end_exclusive",
    "val_time_end_exclusive",
    "seed",
    "scaler",
    "timestamp_start",
    "timestamp_end",
    "input_features",
    "event_features",
)

LEGACY_MISSING_SPACE_PROTOCOL = "legacy_zero_filled"
DUAL_SPACE_MISSING_PROTOCOL = "dual_space_train_only_median"
LEGACY_SPLIT_PROTOCOL = "legacy_frozen"
PROFILE_COMPATIBLE_SPLIT_PROTOCOL = "profile_compatible_event_external"


FINAL_EVENT_EXTERNAL_SPLITS = {
    "bridge": {
        "train_time_end_exclusive": 3110,
        "val_time_end_exclusive": 4032,
    },
    "rainstorm": {
        "train_time_end_exclusive": 7257,
        "val_time_end_exclusive": 9676,
    },
    "typhoon": {
        "train_time_end_exclusive": 2592,
        "val_time_end_exclusive": 3194,
    },
}

PROFILE_COMPATIBLE_EVENT_EXTERNAL_SPLITS = {
    "bridge": {
        "train_time_end_exclusive": 3108,
        "val_time_end_exclusive": 4032,
    },
    "rainstorm": {
        "train_time_end_exclusive": 7255,
        "val_time_end_exclusive": 9676,
    },
    "typhoon": {
        "train_time_end_exclusive": 2590,
        "val_time_end_exclusive": 3194,
    },
}


def final_event_external_split(dataset: str) -> dict[str, int]:
    """Return a copy of the preregistered event-external split boundaries."""
    key = str(dataset).strip().lower()
    if key not in FINAL_EVENT_EXTERNAL_SPLITS:
        raise ValueError(f"Unknown final event-external split dataset: {dataset}")
    split = dict(FINAL_EVENT_EXTERNAL_SPLITS[key])
    if split["train_time_end_exclusive"] >= split["val_time_end_exclusive"]:
        raise RuntimeError(f"Invalid final event-external split for {key}")
    return split

def profile_compatible_event_external_split(dataset: str) -> dict[str, int]:
    """Return a copy of the frozen-profile-compatible event-external split."""
    key = str(dataset).strip().lower()
    if key not in PROFILE_COMPATIBLE_EVENT_EXTERNAL_SPLITS:
        raise ValueError(f"Unknown profile-compatible split dataset: {dataset}")
    split = dict(PROFILE_COMPATIBLE_EVENT_EXTERNAL_SPLITS[key])
    if split["train_time_end_exclusive"] >= split["val_time_end_exclusive"]:
        raise RuntimeError(f"Invalid profile-compatible event-external split for {key}")
    return split


def event_free_feature_contract() -> dict[str, object]:
    """Return the fixed input/loss restrictions for the E-L4-1 baseline."""
    return {
        "input_features": ["traffic"],
        "event_features": [],
        "weather_features": [],
        "resilience_aux_enabled": False,
        "cvar_enabled": False,
        "uncertainty_weighting_enabled": False,
        "scalar_l4_target": False,
    }


def paired_column_plan(
    columns: list[str],
    flow_suffix: str | None,
    speed_suffix: str | None,
    max_flow_nodes: int,
    excluded_columns: tuple[str, ...] = ("Time", "ID", "id"),
) -> dict[str, list[str]]:
    """Create an explicit ordered flow/speed node plan by exact base name.

    Args:
        columns: Original CSV column order.
        flow_suffix: Flow suffix, or ``None`` for a flow-only wide table.
        speed_suffix: Speed suffix, or ``None`` when unavailable.
        max_flow_nodes: Number of flow-led nodes considered before matching.
        excluded_columns: Identifier/time columns never treated as nodes.

    Returns:
        Selected flow columns plus the exact paired flow and speed subnetworks.
    """
    names = list(map(str, columns))
    if flow_suffix is None:
        flow = [name for name in names if name not in set(excluded_columns)][:max_flow_nodes]
        return {
            "selected_flow_columns": flow,
            "paired_flow_columns": [],
            "paired_speed_columns": [],
            "paired_node_names": [],
        }
    matches = match_node_variables(names, flow_suffix, speed_suffix)
    selected = matches[:max_flow_nodes]
    paired = [row for row in selected if row["speed_col"] is not None]
    return {
        "selected_flow_columns": [str(row["flow_col"]) for row in selected],
        "paired_flow_columns": [str(row["flow_col"]) for row in paired],
        "paired_speed_columns": [str(row["speed_col"]) for row in paired],
        "paired_node_names": [str(row["node"]) for row in paired],
    }


def load_ordered_univariate_series(
    csv_path: str,
    time_col: str,
    value_columns: list[str],
    value_suffix: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one traffic variable in an explicit node order with no extra features."""
    data, node_names, feature_names = load_wide_traffic_csv(
        csv_path,
        value_suffix=value_suffix,
        time_col=time_col,
        value_columns=value_columns,
        extra_feature_cols=[],
        node_feature_suffixes=[],
        add_time_features=False,
        exclude_cols=["ID", "id"],
        return_feature_names=True,
    )
    if feature_names != ["traffic"]:
        raise RuntimeError(f"Unexpected E-L4 input features: {feature_names}")
    frame = read_csv_with_fallback(csv_path)
    timestamps = pd.to_datetime(frame[time_col], errors="coerce")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    ordered_timestamps = timestamps.to_numpy()[order]
    if len(ordered_timestamps) != len(data):
        raise RuntimeError("Timestamp and traffic row counts differ")
    return data, ordered_timestamps, node_names


def load_ordered_univariate_series_physical(
    csv_path: str,
    time_col: str,
    value_columns: list[str],
    value_suffix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load ordered physical traffic values while preserving missing observations."""
    frame = read_csv_with_fallback(csv_path)
    timestamps = pd.to_datetime(frame[time_col], errors="coerce")
    order = np.argsort(timestamps.to_numpy(), kind="stable")
    ordered = frame.iloc[order]
    values = ordered[value_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    node_names = [
        str(column)[:-len(value_suffix)]
        if value_suffix and str(column).endswith(value_suffix)
        else str(column)
        for column in value_columns
    ]
    return values[..., None], timestamps.to_numpy()[order], node_names


def impute_model_inputs_train_only(
    physical_values: np.ndarray,
    train_time_end_exclusive: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create finite model inputs using per-node medians fitted only on train rows."""
    values = np.asarray(physical_values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError("physical_values must have shape [time,nodes,1]")
    end = int(train_time_end_exclusive)
    if end <= 0 or end > len(values):
        raise ValueError("invalid train_time_end_exclusive")
    train = values[:end, :, 0]
    finite_train = train[np.isfinite(train)]
    if finite_train.size == 0:
        raise ValueError("training prefix has no finite values for imputation")
    global_median = float(np.median(finite_train))
    medians = np.full(train.shape[1], global_median, dtype=float)
    for node_index in range(train.shape[1]):
        node_values = train[:, node_index]
        node_values = node_values[np.isfinite(node_values)]
        if node_values.size:
            medians[node_index] = float(np.median(node_values))
    finite = np.isfinite(values[..., 0])
    model = values[..., 0].copy()
    missing = ~finite
    model[missing] = np.take(medians, np.where(missing)[1])
    model = model[..., None]
    metadata = {
        "method": "train_only_node_median",
        "train_time_end_exclusive": end,
        "node_medians": medians.tolist(),
        "global_median": global_median,
        "raw_missing_count": int(missing.sum()),
        "model_missing_count": int((~np.isfinite(model)).sum()),
    }
    return model, metadata

def fit_train_only_scaler(
    values: np.ndarray,
    train_time_end_exclusive: int,
) -> tuple[StandardScaler, np.ndarray]:
    """Fit a variable-specific scaler on the strict raw-time training prefix."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("values must have shape [time,nodes,features]")
    end = int(train_time_end_exclusive)
    if end <= 0 or end > len(array):
        raise ValueError("invalid train_time_end_exclusive")
    scaler = StandardScaler()
    scaler.fit(array[:end])
    return scaler, scaler.transform(array)


def scaler_metadata(scaler: StandardScaler) -> dict[str, object]:
    """Serialize one fitted StandardScaler without pickle-only state."""
    if scaler.mean is None or scaler.std is None:
        raise RuntimeError("Scaler has not been fitted")
    return {
        "type": "StandardScaler",
        "mean": np.asarray(scaler.mean).tolist(),
        "std": np.asarray(scaler.std).tolist(),
    }


def scaler_from_metadata(metadata: Mapping[str, object]) -> StandardScaler:
    """Restore a StandardScaler from checkpoint metadata."""
    if metadata.get("type") != "StandardScaler":
        raise ValueError("Unsupported scaler metadata")
    scaler = StandardScaler()
    scaler.mean = np.asarray(metadata["mean"], dtype=float)
    scaler.std = np.asarray(metadata["std"], dtype=float)
    if scaler.mean.shape != scaler.std.shape:
        raise ValueError("Scaler mean/std shapes differ")
    return scaler


def strict_data_bundle(
    values: np.ndarray,
    timestamps: np.ndarray,
    history: int,
    horizon: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    train_time_end_exclusive: int | None = None,
    val_time_end_exclusive: int | None = None,
) -> dict[str, object]:
    """Build strict splits, train-only scaling and aligned target timestamps."""
    array = np.asarray(values, dtype=float)
    ts = np.asarray(timestamps)
    if len(array) != len(ts):
        raise ValueError("values and timestamps lengths differ")
    train, val, test, info = split_traffic_window_indices_strict(
        len(array),
        history,
        horizon,
        train_ratio,
        val_ratio,
        train_time_end_exclusive,
        val_time_end_exclusive,
    )
    scaler, scaled = fit_train_only_scaler(array, int(info["train_time_end_exclusive"]))
    offsets = np.arange(history, history + horizon, dtype=int)

    def target_times(indices: np.ndarray) -> np.ndarray:
        return ts[indices[:, None] + offsets[None, :]]

    return {
        "raw_values": array,
        "scaled_values": scaled,
        "scaler": scaler,
        "scaler_metadata": scaler_metadata(scaler),
        "train_indices": train,
        "val_indices": val,
        "test_indices": test,
        "train_target_timestamps": target_times(train),
        "val_target_timestamps": target_times(val),
        "test_target_timestamps": target_times(test),
        "split_info": info,
    }


def target_tensor_from_indices(
    values: np.ndarray,
    window_indices: np.ndarray,
    history: int,
    horizon: int,
) -> np.ndarray:
    """Return physical or scaled target tensors for explicit window starts."""
    array = np.asarray(values, dtype=float)
    indices = np.asarray(window_indices, dtype=int)
    if array.ndim != 3:
        raise ValueError("values must have shape [time,nodes,features]")
    offsets = np.arange(int(history), int(history) + int(horizon), dtype=int)
    target_rows = indices[:, None] + offsets[None, :]
    if target_rows.size and (target_rows.min() < 0 or target_rows.max() >= len(array)):
        raise ValueError("target window is outside the value array")
    return array[target_rows]


def prepare_dual_space_bundle(
    physical_values: np.ndarray,
    timestamps: np.ndarray,
    history: int,
    horizon: int,
    train_time_end_exclusive: int,
    val_time_end_exclusive: int,
) -> dict[str, object]:
    """Prepare NaN-preserving physical truth plus finite train-only model inputs.

    The physical array is never modified and remains the source for L4 truth and
    metric masks. The model copy is filled only with medians fitted on the
    training prefix, then scaled on the same training prefix.
    """
    physical = np.asarray(physical_values, dtype=float)
    physical_before = physical.copy()
    model_values, imputation = impute_model_inputs_train_only(
        physical,
        train_time_end_exclusive,
    )
    bundle = strict_data_bundle(
        model_values,
        timestamps,
        history,
        horizon,
        train_time_end_exclusive=train_time_end_exclusive,
        val_time_end_exclusive=val_time_end_exclusive,
    )
    return {
        "physical_values": physical,
        "model_values": model_values,
        "bundle": bundle,
        "imputation_metadata": imputation,
        "physical_valid_mask": np.isfinite(physical),
        "physical_space_missing_preserved": bool(np.array_equal(np.isnan(physical), np.isnan(physical_before))),
        "missing_space_protocol": DUAL_SPACE_MISSING_PROTOCOL,
        "split_protocol": PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
    }


def validate_checkpoint_metadata(metadata: Mapping[str, object]) -> None:
    """Validate the metadata needed for physical-space reproducibility."""
    missing = [field for field in CHECKPOINT_METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Checkpoint metadata is missing fields: {missing}")
    if list(metadata["event_features"]):
        raise ValueError("E-L4 checkpoint must not contain event features")
    if list(metadata["input_features"]) != ["traffic"]:
        raise ValueError("E-L4 checkpoint must use traffic-only input")
    scaler_from_metadata(metadata["scaler"])
    if metadata.get("missing_space_protocol") == DUAL_SPACE_MISSING_PROTOCOL:
        imputation = metadata.get("imputation")
        if not isinstance(imputation, Mapping):
            raise ValueError("Dual-space checkpoint metadata must include imputation details")
        required = {
            "method",
            "train_time_end_exclusive",
            "node_medians",
            "global_median",
            "raw_missing_count",
            "model_missing_count",
        }
        missing_imputation = sorted(required.difference(imputation))
        if missing_imputation:
            raise ValueError(f"Dual-space imputation metadata is missing fields: {missing_imputation}")
        if imputation["method"] != "train_only_node_median":
            raise ValueError("Unsupported dual-space imputation method")
        if int(imputation["train_time_end_exclusive"]) != int(metadata["train_time_end_exclusive"]):
            raise ValueError("Imputation train boundary differs from checkpoint split")
        if metadata.get("split_protocol") != PROFILE_COMPATIBLE_SPLIT_PROTOCOL:
            raise ValueError("Dual-space checkpoint must use the profile-compatible split protocol")


def save_forecast_checkpoint(
    path: str | Path,
    model_state_dict: Mapping[str, torch.Tensor],
    metadata: Mapping[str, object],
) -> None:
    """Save model parameters with scaler, node, split and timestamp metadata."""
    validate_checkpoint_metadata(metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": dict(model_state_dict), "metadata": dict(metadata)},
        destination,
    )


def load_forecast_checkpoint(path: str | Path) -> dict[str, object]:
    """Load and validate an auditable E-L4 forecast checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload or "metadata" not in payload:
        raise ValueError("Invalid forecast checkpoint payload")
    validate_checkpoint_metadata(payload["metadata"])
    return payload


def validate_prediction_arrays(
    target_timestamps: np.ndarray,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_true_physical: np.ndarray,
    y_pred_physical: np.ndarray,
    node_names: list[str],
) -> tuple[int, int, int]:
    """Validate sample/horizon/node alignment for exported predictions."""
    arrays = [
        np.asarray(y_true_scaled),
        np.asarray(y_pred_scaled),
        np.asarray(y_true_physical),
        np.asarray(y_pred_physical),
    ]
    if any(array.ndim != 4 for array in arrays):
        raise ValueError("Prediction arrays must have shape [sample,horizon,node,feature]")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("Prediction arrays have inconsistent shapes")
    samples, horizon, nodes, features = arrays[0].shape
    if features != 1 or nodes != len(node_names):
        raise ValueError("Prediction node/feature dimensions do not match metadata")
    if np.asarray(target_timestamps).shape != (samples, horizon):
        raise ValueError("target_timestamps must have shape [sample,horizon]")
    return samples, horizon, nodes


def save_prediction_archive(
    path: str | Path,
    *,
    dataset: str,
    variable: str,
    split: str,
    target_timestamps: np.ndarray,
    node_names: list[str],
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_true_physical: np.ndarray,
    y_pred_physical: np.ndarray,
    train_time_end_exclusive: int,
    val_time_end_exclusive: int,
    scaler: Mapping[str, object],
    seed: int,
    missing_space_protocol: str = LEGACY_MISSING_SPACE_PROTOCOL,
    split_protocol: str = LEGACY_SPLIT_PROTOCOL,
    imputation: Mapping[str, object] | None = None,
    physical_truth_valid_mask: np.ndarray | None = None,
    l4_valid_mask: np.ndarray | None = None,
) -> None:
    """Save aligned multi-horizon predictions in one reloadable NPZ artifact."""
    _, horizon, _ = validate_prediction_arrays(
        target_timestamps,
        y_true_scaled,
        y_pred_scaled,
        y_true_physical,
        y_pred_physical,
        node_names,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    truth_mask = (
        np.isfinite(y_true_physical)
        if physical_truth_valid_mask is None
        else np.asarray(physical_truth_valid_mask, dtype=bool)
    )
    l4_mask = truth_mask if l4_valid_mask is None else np.asarray(l4_valid_mask, dtype=bool)
    if truth_mask.shape != np.asarray(y_true_physical).shape:
        raise ValueError("physical_truth_valid_mask must match y_true_physical shape")
    if l4_mask.shape != np.asarray(y_true_physical).shape:
        raise ValueError("l4_valid_mask must match y_true_physical shape")
    if missing_space_protocol == DUAL_SPACE_MISSING_PROTOCOL:
        if imputation is None:
            raise ValueError("Dual-space prediction archives require imputation metadata")
        if int(imputation.get("train_time_end_exclusive", -1)) != int(train_time_end_exclusive):
            raise ValueError("Archive imputation boundary differs from split metadata")
    np.savez_compressed(
        destination,
        dataset=np.asarray(dataset),
        variable=np.asarray(variable),
        split=np.asarray(split),
        target_timestamps=np.asarray(target_timestamps),
        node_names=np.asarray(node_names),
        y_true_scaled=np.asarray(y_true_scaled),
        y_pred_scaled=np.asarray(y_pred_scaled),
        y_true_physical=np.asarray(y_true_physical),
        y_pred_physical=np.asarray(y_pred_physical),
        horizons=np.arange(1, horizon + 1, dtype=int),
        train_time_end_exclusive=np.asarray(train_time_end_exclusive),
        val_time_end_exclusive=np.asarray(val_time_end_exclusive),
        scaler_metadata_json=np.asarray(json.dumps(dict(scaler), sort_keys=True)),
        missing_space_protocol=np.asarray(missing_space_protocol),
        split_protocol=np.asarray(split_protocol),
        imputation_metadata_json=np.asarray(json.dumps(dict(imputation or {}), sort_keys=True)),
        physical_truth_valid_mask=truth_mask,
        l4_valid_mask=l4_mask,
        seed=np.asarray(seed),
    )


def load_prediction_archive(path: str | Path) -> dict[str, object]:
    """Load an E-L4 prediction NPZ and recover compact JSON metadata."""
    with np.load(Path(path), allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["scaler_metadata"] = json.loads(str(payload.pop("scaler_metadata_json")))
    if "imputation_metadata_json" in payload:
        payload["imputation_metadata"] = json.loads(str(payload.pop("imputation_metadata_json")))
    return payload
