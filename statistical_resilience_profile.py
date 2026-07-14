"""Train-only hierarchical statistical profiles for traffic resilience."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

MAD_SCALE = 1.4826

CDF_SOURCE_CELL = 0
CDF_SOURCE_NODE_DAYTYPE = 1
CDF_SOURCE_NODE = 2
CDF_SOURCE_GLOBAL = 3
CDF_SOURCE_ROBUST_PARAMETRIC = 4
CDF_SOURCE_UNAVAILABLE = 5

_TRANSFORM_MODES = {"log1p_nonnegative", "identity_robust"}


def _array(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("data must have shape [T,N]")
    return np.where(np.isfinite(x), x, np.nan)


def _transform(data: np.ndarray, transform_mode: str) -> np.ndarray:
    if transform_mode not in _TRANSFORM_MODES:
        raise ValueError(f"invalid transform_mode: {transform_mode}")
    raw = _array(data)
    if transform_mode == "identity_robust":
        return raw
    transformed = np.full_like(raw, np.nan)
    valid = raw >= 0
    transformed[valid] = np.log1p(raw[valid])
    return transformed


def _median(x: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float(default)


def _scale(x: np.ndarray, default: float = 0.0) -> float:
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float(default)
    center = np.median(values)
    return float(MAD_SCALE * np.median(np.abs(values - center)))


def _groups(n: int, timestamps, bins: int, mode: str):
    bin_id = np.arange(n) % bins
    day_type = np.zeros(n, dtype=int)
    fallback = ""
    if mode not in {"none", "weekday_weekend"}:
        raise ValueError("invalid day_type_mode")
    if timestamps is None:
        effective = "none" if mode != "none" else mode
        reason = "timestamps_missing_day_type_disabled" if mode != "none" else ""
        return bin_id, day_type, effective, reason
    parsed = pd.to_datetime(np.asarray(timestamps)[:n], errors="coerce")
    if pd.Series(parsed).notna().mean() < 0.95:
        effective = "none" if mode != "none" else mode
        reason = "timestamps_unreliable_day_type_disabled" if mode != "none" else ""
        return bin_id, day_type, effective, reason
    index = pd.DatetimeIndex(parsed)
    minutes = index.hour * 60 + index.minute
    bin_id = np.clip(np.floor(minutes.to_numpy() * bins / 1440).astype(int), 0, bins - 1)
    if mode == "weekday_weekend":
        day_type = (index.dayofweek.to_numpy() >= 5).astype(int)
    return bin_id, day_type, mode, fallback


def _fit_fixed(
    data,
    timestamps,
    bins,
    mode,
    shrinkage_m,
    neighbor,
    min_samples,
    mad_quantile,
    eps,
    transform_mode,
):
    x = _transform(data, transform_mode)
    time_steps, n_nodes = x.shape
    bin_id, day_type, effective_mode, fallback = _groups(time_steps, timestamps, bins, mode)
    n_daytypes = 2 if effective_mode == "weekday_weekend" else 1
    global_count = int(np.isfinite(x).sum())
    global_median = _median(x)
    global_scale = _scale(x, eps)

    node_count = np.sum(np.isfinite(x), axis=0)
    node_median = np.array([_median(x[:, i], global_median) for i in range(n_nodes)])
    node_scale = np.array([_scale(x[:, i], global_scale) for i in range(n_nodes)])
    nd_count = np.zeros((n_nodes, n_daytypes), dtype=int)
    nd_median = np.zeros((n_nodes, n_daytypes))
    nd_scale = np.zeros((n_nodes, n_daytypes))
    for i in range(n_nodes):
        for d in range(n_daytypes):
            values = x[day_type == d, i]
            nd_count[i, d] = int(np.isfinite(values).sum())
            nd_median[i, d] = _median(values, node_median[i])
            nd_scale[i, d] = _scale(values, node_scale[i])

    cell_count = np.zeros((bins, n_nodes, n_daytypes), dtype=int)
    expanded_count = np.zeros_like(cell_count)
    cell_median = np.full((bins, n_nodes, n_daytypes), np.nan)
    cell_scale = np.full_like(cell_median, np.nan)
    for d in range(n_daytypes):
        for b in range(bins):
            exact = (day_type == d) & (bin_id == b)
            neighbor_ids = [(b + offset) % bins for offset in range(-neighbor, neighbor + 1)]
            expanded = (day_type == d) & np.isin(bin_id, neighbor_ids)
            for i in range(n_nodes):
                values = x[exact, i]
                values = values[np.isfinite(values)]
                cell_count[b, i, d] = values.size
                if values.size < min_samples:
                    values = x[expanded, i]
                    values = values[np.isfinite(values)]
                expanded_count[b, i, d] = values.size
                if values.size:
                    cell_median[b, i, d] = np.median(values)
                    cell_scale[b, i, d] = _scale(values)

    positive = np.concatenate(
        [
            cell_scale[np.isfinite(cell_scale) & (cell_scale > 0)],
            nd_scale[np.isfinite(nd_scale) & (nd_scale > 0)],
            node_scale[np.isfinite(node_scale) & (node_scale > 0)],
        ]
    )
    scale_floor = float(np.quantile(positive, mad_quantile)) if positive.size else max(global_scale, eps)
    if not positive.size:
        fallback = ";".join(filter(None, [fallback, "no_positive_scales_global_fallback"]))
    scale_floor = max(scale_floor, eps)
    shrinkage_m = max(float(shrinkage_m), 0.0)

    nd_weight = nd_count / (nd_count + shrinkage_m) if shrinkage_m else (nd_count > 0).astype(float)
    parent_median = nd_weight * nd_median + (1 - nd_weight) * node_median[:, None]
    parent_scale = nd_weight * nd_scale + (1 - nd_weight) * node_scale[:, None]
    cell_weight = expanded_count / (expanded_count + shrinkage_m) if shrinkage_m else (expanded_count > 0).astype(float)
    location = np.empty_like(cell_median)
    scale = np.empty_like(cell_median)
    for d in range(n_daytypes):
        local_median = np.where(np.isfinite(cell_median[:, :, d]), cell_median[:, :, d], parent_median[:, d][None, :])
        local_scale = np.where(np.isfinite(cell_scale[:, :, d]), cell_scale[:, :, d], parent_scale[:, d][None, :])
        location[:, :, d] = cell_weight[:, :, d] * local_median + (1 - cell_weight[:, :, d]) * parent_median[:, d][None, :]
        scale[:, :, d] = cell_weight[:, :, d] * local_scale + (1 - cell_weight[:, :, d]) * parent_scale[:, d][None, :]

    return {
        "method": "hierarchical_robust",
        "train_only": True,
        "transform_mode": transform_mode,
        "time_of_day_bins": int(bins),
        "day_type_mode": effective_mode,
        "requested_day_type_mode": mode,
        "selected_shrinkage_m": shrinkage_m,
        "cell_count": cell_count,
        "cell_expanded_count": expanded_count,
        "cell_median_raw": cell_median,
        "cell_mad_raw": cell_scale,
        "node_daytype_count": nd_count,
        "node_daytype_median": nd_median,
        "node_daytype_mad": nd_scale,
        "node_count": node_count,
        "node_median": node_median,
        "node_mad": node_scale,
        "global_count": global_count,
        "global_median": global_median,
        "global_mad": global_scale,
        "location_shrunk": np.where(np.isfinite(location), location, global_median),
        "scale_shrunk": np.maximum(np.where(np.isfinite(scale), scale, global_scale), scale_floor),
        "scale_floor": scale_floor,
        "fallback_reason": fallback,
        "train_size": time_steps,
    }

def _refs(profile, timestamps, n):
    bin_id, day_type, _, _ = _groups(
        n,
        timestamps,
        int(profile["time_of_day_bins"]),
        str(profile["day_type_mode"]),
    )
    return (
        profile["location_shrunk"][bin_id, :, day_type],
        profile["scale_shrunk"][bin_id, :, day_type],
        bin_id,
        day_type,
    )


def select_shrinkage_by_blocked_cv(
    train_data,
    train_timestamps,
    shrinkage_candidates,
    time_of_day_bins,
    day_type_mode,
    neighborhood_bins,
    minimum_cell_samples,
    eps=1e-6,
    transform_mode="log1p_nonnegative",
):
    """Select prior sample size using leave-one-natural-day-out train-only CV."""
    raw = _array(train_data)
    timestamps = None if train_timestamps is None else np.asarray(train_timestamps)[: len(raw)]
    parsed = pd.to_datetime(timestamps, errors="coerce") if timestamps is not None else None
    if parsed is not None and pd.Series(parsed).notna().mean() >= 0.95:
        days = pd.DatetimeIndex(parsed).normalize().to_numpy()
    else:
        days = np.arange(len(raw)) // time_of_day_bins
    unique_days = pd.unique(days)
    candidates = tuple(sorted(set(float(max(0, candidate)) for candidate in shrinkage_candidates))) or (7.0,)
    common = {
        "n_valid_days": len(unique_days),
        "cv_node_count": int(raw.shape[1]),
        "transform_mode": transform_mode,
    }
    if len(unique_days) < 3:
        return {
            "selected_m": 7.0,
            "candidate_scores": {},
            "candidate_log_mae": {},
            "fold_scores": {},
            "fallback_reason": "fewer_than_3_training_days",
            **common,
        }

    folds = {str(candidate): [] for candidate in candidates}
    for day in unique_days:
        holdout = days == day
        fit_mask = ~holdout
        for candidate in candidates:
            profile = _fit_fixed(
                raw[fit_mask],
                timestamps[fit_mask] if timestamps is not None else None,
                time_of_day_bins,
                day_type_mode,
                candidate,
                neighborhood_bins,
                minimum_cell_samples,
                0.1,
                eps,
                transform_mode,
            )
            location, scale, _, _ = _refs(
                profile,
                timestamps[holdout] if timestamps is not None else None,
                int(holdout.sum()),
            )
            observed = _transform(raw[holdout], transform_mode)
            valid = np.isfinite(observed) & np.isfinite(location) & np.isfinite(scale)
            if valid.any():
                folds[str(candidate)].append(
                    {
                        "standardized_score": float(
                            np.median(np.abs(observed[valid] - location[valid]) / np.maximum(scale[valid], eps))
                        ),
                        "log_mae": float(np.mean(np.abs(observed[valid] - location[valid]))),
                    }
                )
    scores = {
        key: float(np.mean([item["standardized_score"] for item in values])) if values else np.nan
        for key, values in folds.items()
    }
    mae = {
        key: float(np.mean([item["log_mae"] for item in values])) if values else np.nan
        for key, values in folds.items()
    }
    valid_scores = [(float(key), value) for key, value in scores.items() if np.isfinite(value)]
    if not valid_scores:
        return {
            "selected_m": 7.0,
            "candidate_scores": scores,
            "candidate_log_mae": mae,
            "fold_scores": folds,
            "fallback_reason": "all_blocked_cv_folds_failed",
            **common,
        }
    best = min(value for _, value in valid_scores)
    chosen = max(candidate for candidate, value in valid_scores if abs(value - best) <= 1e-6)
    return {
        "selected_m": chosen,
        "candidate_scores": scores,
        "candidate_log_mae": mae,
        "fold_scores": folds,
        "fallback_reason": "",
        **common,
    }


def fit_hierarchical_normal_profile(
    data,
    train_end,
    timestamps=None,
    time_of_day_bins=288,
    day_type_mode="weekday_weekend",
    shrinkage_candidates=(0, 1, 3, 7, 14, 28),
    neighborhood_bins=1,
    mad_floor_quantile=0.10,
    minimum_cell_samples=3,
    eps=1e-6,
    transform_mode="log1p_nonnegative",
):
    """Fit train-only node/bin/day-type robust location and scale with shrinkage.

    Args:
        data: Traffic observations with shape ``[T, N]``.
        train_end: Exclusive end of the training segment.
        timestamps: Optional timestamps shared by all nodes.
        transform_mode: ``log1p_nonnegative`` for physical nonnegative values or
            ``identity_robust`` for standardized/signed values.

    Returns:
        Dictionary containing train-only hierarchical robust statistics.
    """
    raw = _array(data)
    end = int(np.clip(train_end, 0, len(raw)))
    train_timestamps = None if timestamps is None else np.asarray(timestamps)[:end]
    cv = select_shrinkage_by_blocked_cv(
        raw[:end],
        train_timestamps,
        shrinkage_candidates,
        time_of_day_bins,
        day_type_mode,
        neighborhood_bins,
        minimum_cell_samples,
        eps,
        transform_mode,
    )
    profile = _fit_fixed(
        raw[:end],
        train_timestamps,
        time_of_day_bins,
        day_type_mode,
        cv["selected_m"],
        neighborhood_bins,
        minimum_cell_samples,
        mad_floor_quantile,
        eps,
        transform_mode,
    )
    profile.update({"train_end_exclusive": end, "shrinkage_cv": cv})
    profile["fallback_reason"] = ";".join(
        filter(None, [profile.get("fallback_reason", ""), cv.get("fallback_reason", "")])
    )
    return profile


def fit_shrunk_conditional_ecdf(
    data,
    profile,
    train_end,
    timestamps=None,
    min_ecdf_samples=8,
    ecdf_smoothing=0.5,
):
    """Fit sorted train-only samples for all hierarchical ECDF source levels."""
    end = int(np.clip(train_end, 0, len(data)))
    x = _transform(_array(data)[:end], str(profile.get("transform_mode", "log1p_nonnegative")))
    train_timestamps = None if timestamps is None else np.asarray(timestamps)[:end]
    bin_id, day_type, _, _ = _groups(
        len(x), train_timestamps, profile["time_of_day_bins"], profile["day_type_mode"]
    )
    n_nodes = x.shape[1]
    n_daytypes = 2 if profile["day_type_mode"] == "weekday_weekend" else 1
    cells: dict[str, np.ndarray] = {}
    node_daytypes: dict[str, np.ndarray] = {}
    nodes: dict[str, np.ndarray] = {}
    for node in range(n_nodes):
        nodes[str(node)] = np.sort(x[:, node][np.isfinite(x[:, node])])
        for day in range(n_daytypes):
            values = x[day_type == day, node]
            node_daytypes[f"{node}|{day}"] = np.sort(values[np.isfinite(values)])
            for bin_value in range(profile["time_of_day_bins"]):
                values = x[(day_type == day) & (bin_id == bin_value), node]
                values = values[np.isfinite(values)]
                if values.size:
                    cells[f"{bin_value}|{node}|{day}"] = np.sort(values)
    return {
        "method": "hierarchical_shrunk_ecdf",
        "train_only": True,
        "transform_mode": str(profile.get("transform_mode", "log1p_nonnegative")),
        "selected_m": float(profile["selected_shrinkage_m"]),
        "ecdf_smoothing": float(ecdf_smoothing),
        "min_ecdf_samples": int(min_ecdf_samples),
        "cell_sorted_values": cells,
        "node_daytype_sorted_values": node_daytypes,
        "node_sorted_values": nodes,
        "global_sorted_values": np.sort(x[np.isfinite(x)]),
        "fallback_mode": "robust_parametric",
        "train_end_exclusive": end,
    }

def _cdf(sorted_values, query, smoothing):
    values = np.asarray(sorted_values, dtype=float)
    query = np.asarray(query, dtype=float)
    if not values.size:
        return np.full(query.shape, np.nan)
    return (np.searchsorted(values, query, side="right") + smoothing) / (values.size + 2 * smoothing)


def _mixed_cdf(query, samples, weights, smoothing):
    result = np.full(np.asarray(query).shape, np.nan)
    remaining = 1.0
    for values, weight in zip(samples, weights):
        probability = _cdf(values, query, smoothing)
        effective_weight = remaining if weight is None else remaining * float(np.clip(weight, 0.0, 1.0))
        if np.isfinite(probability).any():
            contribution = effective_weight * probability
            result = np.where(np.isfinite(result), result + contribution, contribution)
            remaining -= effective_weight
        if remaining <= 1e-12:
            break
    return result


def query_conditional_cdf(
    values: np.ndarray,
    profile: dict[str, object],
    conditional_ecdf: dict[str, object],
    timestamps: np.ndarray | None = None,
    probability_floor: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Query a train-only hierarchical conditional CDF with auditable sources.

    The source code records the most specific empirical distribution with enough
    training observations. Parent empirical ECDFs are preferred over the robust
    normal fallback. The robust fallback uses ``Phi((x-median)/MAD)`` only when
    every empirical level is unavailable.

    Args:
        values: Query observations with shape ``[T, N]``.
        profile: Output of :func:`fit_hierarchical_normal_profile`.
        conditional_ecdf: Output of :func:`fit_shrunk_conditional_ecdf`.
        timestamps: Query timestamps.
        probability_floor: Symmetric probability clipping floor.

    Returns:
        CDF, conditional robust references, source codes, and validity masks,
        all with shape ``[T, N]``.
    """
    if not 0 < probability_floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")
    transformed = _transform(values, str(profile.get("transform_mode", "log1p_nonnegative")))
    location, scale, bin_id, day_type = _refs(profile, timestamps, len(transformed))
    if transformed.shape[1] != location.shape[1]:
        raise ValueError("values and profile node counts differ")

    cdf = np.full_like(transformed, np.nan)
    source = np.full(transformed.shape, CDF_SOURCE_UNAVAILABLE, dtype=np.int16)
    smoothing = float(conditional_ecdf.get("ecdf_smoothing", 0.5))
    minimum = int(conditional_ecdf.get("min_ecdf_samples", 8))
    shrinkage_m = max(float(conditional_ecdf.get("selected_m", 0.0)), 0.0)
    global_values = np.asarray(conditional_ecdf.get("global_sorted_values", np.array([])), dtype=float)
    global_values = global_values[np.isfinite(global_values)]
    cells = conditional_ecdf.get("cell_sorted_values", {})
    node_daytypes = conditional_ecdf.get("node_daytype_sorted_values", {})
    nodes = conditional_ecdf.get("node_sorted_values", {})

    for node in range(transformed.shape[1]):
        node_values = np.asarray(nodes.get(str(node), np.array([])), dtype=float)
        node_weight = len(node_values) / (len(node_values) + shrinkage_m) if shrinkage_m else float(len(node_values) > 0)
        for day in np.unique(day_type):
            nd_values = np.asarray(node_daytypes.get(f"{node}|{int(day)}", np.array([])), dtype=float)
            nd_weight = len(nd_values) / (len(nd_values) + shrinkage_m) if shrinkage_m else float(len(nd_values) > 0)
            for bin_value in np.unique(bin_id[day_type == day]):
                rows = (day_type == day) & (bin_id == bin_value) & np.isfinite(transformed[:, node])
                if not rows.any():
                    continue
                query = transformed[rows, node]
                cell_values = np.asarray(
                    cells.get(f"{int(bin_value)}|{node}|{int(day)}", np.array([])), dtype=float
                )
                cell_weight = len(cell_values) / (len(cell_values) + shrinkage_m) if shrinkage_m else float(len(cell_values) > 0)
                if len(cell_values) >= minimum:
                    probability = _mixed_cdf(
                        query,
                        [cell_values, nd_values, node_values, global_values],
                        [cell_weight, nd_weight, node_weight, None],
                        smoothing,
                    )
                    level = CDF_SOURCE_CELL
                elif len(nd_values) >= minimum:
                    probability = _mixed_cdf(
                        query,
                        [nd_values, node_values, global_values],
                        [nd_weight, node_weight, None],
                        smoothing,
                    )
                    level = CDF_SOURCE_NODE_DAYTYPE
                elif len(node_values) >= minimum:
                    probability = _mixed_cdf(
                        query,
                        [node_values, global_values],
                        [node_weight, None],
                        smoothing,
                    )
                    level = CDF_SOURCE_NODE
                elif len(global_values) >= minimum:
                    probability = _cdf(global_values, query, smoothing)
                    level = CDF_SOURCE_GLOBAL
                else:
                    row_location = location[rows, node]
                    row_scale = scale[rows, node]
                    parameter_available = (
                        (int(profile.get("global_count", np.isfinite(global_values).sum())) > 0)
                        & np.isfinite(row_location)
                        & np.isfinite(row_scale)
                        & (row_scale > 0)
                    )
                    probability = np.where(
                        parameter_available,
                        ndtr((query - row_location) / row_scale),
                        np.nan,
                    )
                    level = CDF_SOURCE_ROBUST_PARAMETRIC
                finite_probability = np.isfinite(probability)
                row_indices = np.flatnonzero(rows)
                if finite_probability.any():
                    selected = row_indices[finite_probability]
                    cdf[selected, node] = probability[finite_probability]
                    source[selected, node] = level

    valid = np.isfinite(transformed) & np.isfinite(cdf)
    cdf[valid] = np.clip(cdf[valid], probability_floor, 1.0 - probability_floor)
    robust_z = np.where(valid, (transformed - location) / np.maximum(scale, 1e-12), np.nan)
    fallback = valid & (source != CDF_SOURCE_CELL)
    return {
        "cdf": cdf,
        "conditional_median": location,
        "conditional_scale": scale,
        "robust_z": robust_z,
        "transformed_values": transformed,
        "source_level": source,
        "fallback_mask": fallback,
        "valid_mask": valid,
    }


def save_conditional_ecdf_npz(path: str | Path, conditional_ecdf: dict[str, object]) -> None:
    """Save every hierarchical ECDF sample dictionary to a compressed NPZ."""
    payload: dict[str, Any] = {}
    for prefix, key in (
        ("cell", "cell_sorted_values"),
        ("node_daytype", "node_daytype_sorted_values"),
        ("node", "node_sorted_values"),
    ):
        mapping = conditional_ecdf.get(key, {})
        payload[f"{prefix}_keys"] = np.asarray(list(mapping.keys()), dtype=object)
        payload[f"{prefix}_values"] = np.asarray(
            [np.asarray(value, dtype=float) for value in mapping.values()], dtype=object
        )
    payload["global_sorted_values"] = np.asarray(conditional_ecdf.get("global_sorted_values", []), dtype=float)
    for key in (
        "method",
        "train_only",
        "transform_mode",
        "selected_m",
        "ecdf_smoothing",
        "min_ecdf_samples",
        "fallback_mode",
        "train_end_exclusive",
    ):
        payload[key] = np.asarray(conditional_ecdf.get(key))
    np.savez_compressed(path, **payload)


def load_conditional_ecdf_npz(path: str | Path) -> dict[str, object]:
    """Load a complete hierarchical ECDF saved by :func:`save_conditional_ecdf_npz`."""
    with np.load(path, allow_pickle=True) as archive:
        result: dict[str, object] = {}
        for prefix, key in (
            ("cell", "cell_sorted_values"),
            ("node_daytype", "node_daytype_sorted_values"),
            ("node", "node_sorted_values"),
        ):
            keys = archive[f"{prefix}_keys"].tolist()
            values = archive[f"{prefix}_values"].tolist()
            result[key] = {str(name): np.asarray(value, dtype=float) for name, value in zip(keys, values)}
        result["global_sorted_values"] = np.asarray(archive["global_sorted_values"], dtype=float)
        for key in ("method", "transform_mode", "fallback_mode"):
            result[key] = str(archive[key].item())
        result["train_only"] = bool(archive["train_only"].item())
        result["selected_m"] = float(archive["selected_m"].item())
        result["ecdf_smoothing"] = float(archive["ecdf_smoothing"].item())
        result["min_ecdf_samples"] = int(archive["min_ecdf_samples"].item())
        result["train_end_exclusive"] = int(archive["train_end_exclusive"].item())
    return result

def compute_probabilistic_resilience_state(
    data,
    profile,
    conditional_ecdf,
    timestamps=None,
    probability_floor=1e-4,
    deficit_clip_quantile=0.999,
    system_trim_fraction=0.10,
    eps=1e-6,
):
    """Score lower-tail surprisal under a train-only hierarchical conditional CDF."""
    query = query_conditional_cdf(
        data,
        profile,
        conditional_ecdf,
        timestamps=timestamps,
        probability_floor=probability_floor,
    )
    probability = query["cdf"]
    transformed = query["transformed_values"]
    location = query["conditional_median"]
    surprise = -np.log(probability)
    deficit = np.where(
        query["valid_mask"],
        np.where(transformed < location, surprise, 0.0),
        np.nan,
    )
    train_end = min(int(profile["train_end_exclusive"]), len(deficit))
    train_values = deficit[:train_end][np.isfinite(deficit[:train_end])]
    clip_value = (
        float(np.quantile(train_values, deficit_clip_quantile))
        if train_values.size
        else -np.log(probability_floor)
    )
    deficit = np.clip(deficit, 0.0, max(clip_value, eps))

    system = np.full(len(deficit), np.nan)
    trim = float(np.clip(system_trim_fraction, 0.0, 0.49))
    for time_index, row in enumerate(deficit):
        finite = np.sort(row[np.isfinite(row)])
        cut = int(len(finite) * trim)
        if len(finite) - 2 * cut > 0:
            finite = finite[cut : len(finite) - cut]
        if finite.size:
            system[time_index] = np.mean(finite)
    system_train = system[:train_end][np.isfinite(system[:train_end])]
    quantiles = {
        f"q{int(q * 100):02d}": float(np.quantile(system_train, q)) if system_train.size else np.nan
        for q in (0.5, 0.75, 0.9, 0.95, 0.99)
    }
    return {
        "p_lower": probability,
        "tail_surprisal": surprise,
        "probabilistic_deficit": deficit,
        "resilience_score": np.exp(-deficit),
        "robust_z": query["robust_z"],
        "transformed_values": query["transformed_values"],
        "conditional_median": query["conditional_median"],
        "conditional_scale": query["conditional_scale"],
        "system_deficit": system,
        "system_resilience": np.exp(-system),
        "ecdf_source_level": query["source_level"],
        "ecdf_fallback_mask": query["fallback_mask"],
        "valid_mask": query["valid_mask"],
        "deficit_clip_value": np.asarray(clip_value),
        "train_deficit_quantiles": quantiles,
    }