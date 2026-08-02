"""Frozen two-dimensional L4 supervision for forecasting models."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_pipeline import target_tensor_from_indices


def build_frozen_l4_supervision(
    physical_values: np.ndarray,
    timestamps: np.ndarray,
    node_names: list[str],
    variable: str,
    profile: Mapping[str, object],
    conditional_ecdf: Mapping[str, object],
    train_end: int,
    history: int,
    horizon: int,
) -> dict[str, object]:
    """Build horizon-aligned node deficits from a read-only frozen L4 profile."""
    values = np.asarray(physical_values, dtype=float)
    times = np.asarray(timestamps)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError("physical_values must have shape [time,nodes,1]")
    if len(values) != len(times) or values.shape[1] != len(node_names):
        raise ValueError("physical values, timestamps and node names are not aligned")
    if variable not in {"flow", "speed"}:
        raise ValueError("variable must be flow or speed")
    frozen_end = int(profile.get("train_end_exclusive", -1))
    if frozen_end != int(train_end):
        raise ValueError("frozen profile train boundary differs from forecasting split")

    bundle = {"profile": dict(profile), "ecdf": dict(conditional_ecdf)}
    reference = compute_directional_probabilistic_deficit(
        values[..., 0], bundle, times, "lower"
    )
    deficits = np.asarray(reference["probabilistic_deficit"], dtype=float)[..., None]
    valid = np.asarray(reference["valid_mask"], dtype=bool)[..., None]
    source = np.asarray(reference["ecdf_source_level"], dtype=np.int16)[..., None]
    num_windows = max(0, len(values) - int(history) - int(horizon) + 1)
    indices = np.arange(num_windows, dtype=int)
    deficit_targets = target_tensor_from_indices(deficits, indices, history, horizon)
    valid_mask = target_tensor_from_indices(valid.astype(float), indices, history, horizon).astype(bool)
    source_level = target_tensor_from_indices(source.astype(float), indices, history, horizon).astype(np.int16)
    quantiles = reference["train_deficit_quantiles"]
    q90 = float(quantiles["q90"])
    q99 = float(quantiles["q99"])
    high = np.where(valid_mask, deficit_targets > q90, False)
    extreme = np.where(valid_mask, deficit_targets > q99, False)
    return {
        "deficit_targets": deficit_targets,
        "source_level": source_level,
        "valid_mask": valid_mask,
        "high_state_targets": high,
        "extreme_state_targets": extreme,
        "q90": q90,
        "q99": q99,
        "metadata": {
            "variable": variable,
            "dimension": "demand" if variable == "flow" else "efficiency",
            "train_end_exclusive": int(train_end),
            "history": int(history),
            "horizon": int(horizon),
            "node_names": list(node_names),
            "profile_read_only": True,
            "ecdf_read_only": True,
            "missing_filled_with_zero": False,
            "target_mode": "l4_deficit",
        },
    }
