"""Evaluation helpers for indirect two-dimensional L4 forecasting."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import stats

from statistical_resilience_profile import query_conditional_cdf


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return finite-value regression metrics without unsafe zero division."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {key: float("nan") for key in ("mae", "rmse", "wape", "smape", "pearson", "spearman")}
    a = truth[valid]
    b = pred[valid]
    error = b - a
    denom = np.abs(a) + np.abs(b)
    pearson = float(stats.pearsonr(a, b).statistic) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
    spearman = float(stats.spearmanr(a, b).statistic) if len(a) > 1 and np.unique(a).size > 1 and np.unique(b).size > 1 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(a)), 1e-12)),
        "smape": float(np.mean(np.divide(2.0 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 1e-12))),
        "pearson": pearson,
        "spearman": spearman,
    }


def persistence_forecast(scaled_values: np.ndarray, window_indices: np.ndarray, history: int, horizon: int) -> np.ndarray:
    """Repeat the final input observation across every forecast horizon."""
    values = np.asarray(scaled_values, dtype=float)
    indices = np.asarray(window_indices, dtype=int)
    last = values[indices + history - 1]
    return np.repeat(last[:, None, :, :], horizon, axis=1)


def _trimmed_rows(values: np.ndarray, trim_fraction: float = 0.10) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=float)
    for index, row in enumerate(np.asarray(values, dtype=float)):
        finite = np.sort(row[np.isfinite(row)])
        cut = int(len(finite) * float(np.clip(trim_fraction, 0.0, 0.49)))
        if len(finite) - 2 * cut > 0:
            finite = finite[cut:len(finite) - cut]
        if finite.size:
            result[index] = float(np.mean(finite))
    return result


def apply_frozen_lower_tail_profile(
    selected_values: np.ndarray,
    target_timestamps: np.ndarray,
    profile_bundle: Mapping[str, object],
    full_node_count: int,
    selected_node_indices: list[int],
    deficit_clip_value: float,
    probability_floor: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Apply one frozen full-node profile to forecasts for an ordered node subset."""
    values = np.asarray(selected_values, dtype=float)
    timestamps = np.asarray(target_timestamps)
    if values.ndim != 2:
        raise ValueError("selected_values must have shape [observations,nodes]")
    if len(timestamps) != len(values):
        raise ValueError("target timestamp length differs from forecast observations")
    if values.shape[1] != len(selected_node_indices):
        raise ValueError("selected node index count differs from forecast node count")
    if len(set(selected_node_indices)) != len(selected_node_indices):
        raise ValueError("selected node indices must be unique")
    if any(index < 0 or index >= full_node_count for index in selected_node_indices):
        raise ValueError("selected node index is outside the frozen profile")

    full = np.full((len(values), full_node_count), np.nan, dtype=float)
    full[:, selected_node_indices] = values
    profile = profile_bundle["profile"]
    query = query_conditional_cdf(
        full,
        profile,
        profile_bundle["ecdf"],
        timestamps=timestamps,
        probability_floor=probability_floor,
    )
    probability = np.asarray(query["cdf"], dtype=float)
    transformed = np.asarray(query["transformed_values"], dtype=float)
    median = np.asarray(query["conditional_median"], dtype=float)
    valid = np.asarray(query["valid_mask"], dtype=bool)
    deficit = np.where(valid, np.where(transformed < median, -np.log(probability), 0.0), np.nan)
    deficit = np.clip(deficit, 0.0, max(float(deficit_clip_value), 1e-6))
    return {
        "node_deficit": deficit[:, selected_node_indices],
        "system_deficit": _trimmed_rows(deficit),
        "source_level": np.asarray(query["source_level"])[:, selected_node_indices],
        "valid_mask": valid[:, selected_node_indices],
    }


def binary_state_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate one train-thresholded high-state forecast."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return {key: float("nan") for key in ("precision", "recall", "f1", "balanced_accuracy", "predicted_high_rate")}
    true_high = truth[valid] > threshold
    pred_high = pred[valid] > threshold
    tp = int(np.sum(true_high & pred_high))
    fp = int(np.sum(~true_high & pred_high))
    fn = int(np.sum(true_high & ~pred_high))
    tn = int(np.sum(~true_high & ~pred_high))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "predicted_high_rate": float(np.mean(pred_high)),
    }