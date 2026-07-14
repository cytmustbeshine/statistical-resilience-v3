"""Traffic resilience indicators for disruption-oriented forecasting.

This module is intentionally independent from the D-STSGCN model and training
code. It only computes statistical resilience indicators from observed or
predicted traffic series.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistical_resilience_profile import (
    CDF_SOURCE_CELL,
    CDF_SOURCE_GLOBAL,
    CDF_SOURCE_NODE,
    CDF_SOURCE_NODE_DAYTYPE,
    CDF_SOURCE_ROBUST_PARAMETRIC,
    CDF_SOURCE_UNAVAILABLE,
    compute_probabilistic_resilience_state,
    fit_hierarchical_normal_profile,
    fit_shrunk_conditional_ecdf,
    load_conditional_ecdf_npz,
    query_conditional_cdf,
    save_conditional_ecdf_npz,
    select_shrinkage_by_blocked_cv,
)


EPS = 1e-6


def _finite_runs(indices: np.ndarray, max_gap_steps: int) -> list[tuple[int, int]]:
    """Group sorted indices into runs, optionally bridging short gaps."""
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx - previous > max_gap_steps + 1:
            runs.append((start, previous))
            start = idx
        previous = idx
    runs.append((start, previous))
    return runs

def _as_2d_float(data: np.ndarray) -> np.ndarray:
    """Convert input data to a finite-friendly two-dimensional float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("data must have shape [T, N] or [T].")
    return array


def _safe_nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Compute nanmean and replace all-NaN results with 0."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    counts = finite.sum(axis=axis)
    sums = safe_values.sum(axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = sums / np.maximum(counts, 1)
    result = np.where(counts == 0, 0.0, result)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_nanstd(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Compute nanstd and replace all-NaN results with 0."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    mean = _safe_nanmean(values, axis=axis)
    if axis is None:
        centered = np.where(finite, values - float(mean), 0.0)
        count = int(finite.sum())
        variance = float((centered**2).sum() / max(count, 1)) if count > 0 else 0.0
        result = np.sqrt(variance)
    else:
        expanded_mean = np.expand_dims(mean, axis=axis)
        centered = np.where(finite, values - expanded_mean, 0.0)
        counts = finite.sum(axis=axis)
        variance = (centered**2).sum(axis=axis) / np.maximum(counts, 1)
        result = np.sqrt(np.where(counts == 0, 0.0, variance))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_slice_mean(values: np.ndarray) -> float:
    """Return a finite mean for a one-dimensional slice, or NaN if unavailable."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_slice_min(values: np.ndarray) -> float:
    """Return a finite minimum for a one-dimensional slice, or NaN if unavailable."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def compute_baseline(
    data: np.ndarray,
    train_end: int,
    time_of_day_bins: int = 288,
) -> dict[str, np.ndarray]:
    """Estimate the normal traffic baseline from the training period only.

    The time-of-day baseline is computed by grouping training time steps by
    ``index % time_of_day_bins``. Empty or all-NaN bins fall back to the
    corresponding per-node training mean.

    Args:
        data: Traffic values with shape ``[T, N]``. A one-dimensional ``[T]``
            array is treated as a single-node series.
        train_end: Exclusive end index of the training period. Only
            ``data[:train_end]`` is used for baseline estimation.
        time_of_day_bins: Number of intraday periods. ``288`` corresponds to
            5-minute intervals over one day.

    Returns:
        Dictionary containing ``mean`` with shape ``[N]``, ``std`` with shape
        ``[N]``, and ``tod_mean`` with shape ``[time_of_day_bins, N]``.
    """
    array = _as_2d_float(data)
    if time_of_day_bins <= 0:
        raise ValueError("time_of_day_bins must be positive.")

    time_steps, num_nodes = array.shape
    train_end = int(np.clip(train_end, 0, time_steps))
    train_data = array[:train_end]

    if train_data.size == 0:
        mean = np.zeros(num_nodes, dtype=float)
        std = np.zeros(num_nodes, dtype=float)
    else:
        mean = _safe_nanmean(train_data, axis=0).astype(float)
        std = _safe_nanstd(train_data, axis=0).astype(float)

    tod_mean = np.tile(mean.reshape(1, -1), (time_of_day_bins, 1))
    if train_end > 0:
        train_index = np.arange(train_end)
        bins = train_index % time_of_day_bins
        frame = pd.DataFrame(train_data)
        frame["__tod_bin__"] = bins
        grouped = frame.groupby("__tod_bin__", sort=True).mean(numeric_only=True)
        for bin_idx, row in grouped.iterrows():
            values = row.to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, mean)
            tod_mean[int(bin_idx)] = values

    return {
        "mean": np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0),
        "std": np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0),
        "tod_mean": np.nan_to_num(tod_mean, nan=0.0, posinf=0.0, neginf=0.0),
    }



def compute_robust_shrunk_baseline(
    data: np.ndarray,
    train_end: int,
    time_of_day_bins: int = 288,
    shrinkage_candidates: tuple[float, ...] = (0, 1, 3, 7, 14, 28),
    eps: float = EPS,
) -> dict[str, np.ndarray | float | list[dict[str, float]] | str]:
    """Estimate a train-only empirical-Bayes seasonal baseline on log traffic.

    Per-node, per-time-bin log means are shrunk toward each node's global log
    mean. The shrinkage strength is selected by leave-one-day-out blocked
    cross-validation using only complete or partial days in the training split.

    Args:
        data: Raw non-negative traffic values with shape ``[T, N]``.
        train_end: Exclusive training-period end index.
        time_of_day_bins: Number of samples per day.
        shrinkage_candidates: Candidate prior sample sizes ``m``.
        eps: Numerical stability constant.

    Returns:
        Baseline arrays, robust scale statistics, selected shrinkage strength,
        cross-validation scores, and a fallback reason when CV is unavailable.
    """
    array = _as_2d_float(data)
    if time_of_day_bins <= 0:
        raise ValueError("time_of_day_bins must be positive.")
    train_end = int(np.clip(train_end, 0, array.shape[0]))
    train = np.clip(array[:train_end], 0.0, None)
    num_nodes = array.shape[1]
    log_train = np.log1p(train)
    finite = np.isfinite(log_train)
    safe_log = np.where(finite, log_train, 0.0)
    counts = finite.sum(axis=0)
    node_log_mean = safe_log.sum(axis=0) / np.maximum(counts, 1)
    node_median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    node_median = np.nan_to_num(node_median, nan=0.0, posinf=0.0, neginf=0.0)
    node_mad = np.nanmedian(np.abs(train - node_median.reshape(1, -1)), axis=0)
    node_mad = np.maximum(np.nan_to_num(node_mad, nan=0.0), eps)

    bins = np.arange(train_end) % time_of_day_bins
    tod_sum = np.zeros((time_of_day_bins, num_nodes), dtype=float)
    tod_count = np.zeros_like(tod_sum)
    for bin_idx in range(time_of_day_bins):
        values = log_train[bins == bin_idx]
        valid = np.isfinite(values)
        tod_sum[bin_idx] = np.where(valid, values, 0.0).sum(axis=0)
        tod_count[bin_idx] = valid.sum(axis=0)
    tod_raw = np.divide(
        tod_sum,
        np.maximum(tod_count, 1.0),
        out=np.tile(node_log_mean, (time_of_day_bins, 1)),
        where=tod_count > 0,
    )

    candidates = tuple(sorted({max(float(value), 0.0) for value in shrinkage_candidates}))
    if not candidates:
        candidates = (7.0,)
    train_days = int(np.ceil(train_end / time_of_day_bins)) if train_end else 0
    cv_scores: list[dict[str, float]] = []
    fallback_reason = ""
    selected_m = 7.0
    if train_days >= 3:
        day_ids = np.arange(train_end) // time_of_day_bins
        for candidate in candidates:
            fold_errors: list[float] = []
            for held_day in range(train_days):
                fit_mask = day_ids != held_day
                valid_mask = day_ids == held_day
                if not fit_mask.any() or not valid_mask.any():
                    continue
                fit_values = log_train[fit_mask]
                fit_bins = bins[fit_mask]
                fit_finite = np.isfinite(fit_values)
                fit_node_count = fit_finite.sum(axis=0)
                fit_node_mean = np.where(fit_finite, fit_values, 0.0).sum(axis=0) / np.maximum(
                    fit_node_count, 1
                )
                fold_sum = np.zeros((time_of_day_bins, num_nodes), dtype=float)
                fold_count = np.zeros_like(fold_sum)
                for bin_idx in np.unique(fit_bins):
                    values = fit_values[fit_bins == bin_idx]
                    valid = np.isfinite(values)
                    fold_sum[int(bin_idx)] = np.where(valid, values, 0.0).sum(axis=0)
                    fold_count[int(bin_idx)] = valid.sum(axis=0)
                fold_mean = np.divide(
                    fold_sum,
                    np.maximum(fold_count, 1.0),
                    out=np.tile(fit_node_mean, (time_of_day_bins, 1)),
                    where=fold_count > 0,
                )
                shrunk = (
                    fold_count * fold_mean + candidate * fit_node_mean.reshape(1, -1)
                ) / np.maximum(fold_count + candidate, 1.0)
                held_values = log_train[valid_mask]
                held_pred = shrunk[bins[valid_mask]]
                held_finite = np.isfinite(held_values) & np.isfinite(held_pred)
                if held_finite.any():
                    fold_errors.append(float(np.mean(np.abs(held_values[held_finite] - held_pred[held_finite]))))
            score = float(np.mean(fold_errors)) if fold_errors else float("inf")
            cv_scores.append({"m": float(candidate), "log1p_mae": score})
        finite_scores = [item for item in cv_scores if np.isfinite(item["log1p_mae"])]
        if finite_scores:
            selected_m = min(finite_scores, key=lambda item: (item["log1p_mae"], item["m"]))["m"]
        else:
            fallback_reason = "blocked CV produced no finite fold scores; using m=7"
    else:
        fallback_reason = "fewer than three training days; using m=7"

    tod_shrunk = (
        tod_count * tod_raw + selected_m * node_log_mean.reshape(1, -1)
    ) / np.maximum(tod_count + selected_m, 1.0)
    baseline_series = np.expm1(tod_shrunk[bins]) if train_end else np.empty((0, num_nodes))
    if baseline_series.size:
        log_ratio = np.log((train + eps) / (np.maximum(baseline_series, 0.0) + eps))
        finite_ratio = log_ratio[np.isfinite(log_ratio)]
        ratio_median = float(np.median(finite_ratio)) if finite_ratio.size else 0.0
        ratio_mad = float(np.median(np.abs(finite_ratio - ratio_median))) if finite_ratio.size else 1.0
    else:
        ratio_median = 0.0
        ratio_mad = 1.0
    ratio_mad = max(ratio_mad, eps)
    return {
        "mean": np.expm1(node_log_mean),
        "std": _safe_nanstd(train, axis=0) if train.size else np.zeros(num_nodes),
        "tod_mean": np.maximum(np.expm1(tod_shrunk), 0.0),
        "node_median": node_median,
        "node_mad": node_mad,
        "tod_log_mean_raw": tod_raw,
        "tod_log_mean_shrunk": tod_shrunk,
        "selected_shrinkage_m": float(selected_m),
        "shrinkage_cv_scores": cv_scores,
        "train_days": float(train_days),
        "fallback_reason": fallback_reason,
        "log_ratio_median": ratio_median,
        "log_ratio_mad": ratio_mad,
    }


def compute_log_resilience_series(
    data: np.ndarray,
    baseline: dict[str, np.ndarray | float],
    mad_clip: float = 6.0,
    eps: float = EPS,
) -> np.ndarray:
    """Compute clipped log traffic-to-seasonal-baseline ratios."""
    array = np.clip(_as_2d_float(data), 0.0, None)
    tod_mean = np.asarray(baseline["tod_mean"], dtype=float)
    bins = np.arange(array.shape[0]) % tod_mean.shape[0]
    denom = np.maximum(tod_mean[bins], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log((array + eps) / (denom + eps))
    center = float(baseline.get("log_ratio_median", 0.0))
    mad = max(float(baseline.get("log_ratio_mad", 1.0)), eps)
    radius = max(float(mad_clip), 0.0) * 1.4826 * mad
    values = np.nan_to_num(values, nan=center, posinf=center + radius, neginf=center - radius)
    return np.clip(values, center - radius, center + radius)
def compute_R_series(
    data: np.ndarray,
    baseline: dict[str, np.ndarray],
    use_tod: bool = True,
) -> np.ndarray:
    """Compute node-level performance ratios against a normal baseline.

    The performance ratio is ``R(t, i) = data(t, i) / baseline(t, i)``. Values
    are clipped to ``[0, 2]`` to reduce the influence of extreme outliers.

    Args:
        data: Traffic values with shape ``[T, N]``.
        baseline: Output dictionary from :func:`compute_baseline`.
        use_tod: If ``True``, use the time-of-day conditional mean. Otherwise
            use the global per-node mean.

    Returns:
        ``R`` array with shape ``[T, N]``.
    """
    array = _as_2d_float(data)
    time_steps, num_nodes = array.shape

    if use_tod and "tod_mean" in baseline:
        tod_mean = np.asarray(baseline["tod_mean"], dtype=float)
        if tod_mean.ndim != 2 or tod_mean.shape[1] != num_nodes:
            raise ValueError("baseline['tod_mean'] must have shape [B, N].")
        bins = np.arange(time_steps) % tod_mean.shape[0]
        denom = tod_mean[bins]
    else:
        mean = np.asarray(baseline["mean"], dtype=float).reshape(1, -1)
        if mean.shape[1] != num_nodes:
            raise ValueError("baseline['mean'] must have shape [N].")
        denom = np.repeat(mean, time_steps, axis=0)

    denom = np.maximum(np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0), EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = array / denom
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=2.0, neginf=0.0)
    return np.clip(ratio, 0.0, 2.0)


def compute_system_R(R: np.ndarray) -> np.ndarray:
    """Aggregate node-level performance ratios into a system-level series.

    Args:
        R: Node-level performance ratio with shape ``[T, N]``.

    Returns:
        System-level performance ratio ``R_sys`` with shape ``[T]``, computed
        as the finite mean over nodes at each time step.
    """
    ratio = _as_2d_float(R)
    return _safe_nanmean(ratio, axis=1).astype(float)



def detect_event_windows(
    event_signal: np.ndarray | None,
    R_sys: np.ndarray,
    event_threshold: float = 0.0,
    max_gap_steps: int = 12,
    min_event_steps: int = 3,
    explicit_event_start: int | None = None,
    explicit_event_end: int | None = None,
) -> list[tuple[int, int]]:
    """Detect separate disruption windows from metadata or observed performance.

    Explicit event metadata takes precedence. Otherwise, event-signal runs are
    segmented and only short gaps are bridged. When no event signal is usable,
    sustained system-performance deficits are used as a conservative fallback.
    """
    r_sys = np.asarray(R_sys, dtype=float).reshape(-1)
    n = r_sys.size
    if n == 0:
        return []
    max_gap_steps = max(int(max_gap_steps), 0)
    min_event_steps = max(int(min_event_steps), 1)
    if explicit_event_start is not None:
        start = int(np.clip(explicit_event_start, 0, n - 1))
        if explicit_event_end is not None:
            end = int(np.clip(explicit_event_end, start, n - 1))
            return [(start, end)]
        post = r_sys[start:]
        finite_post = np.where(np.isfinite(post), post, np.inf)
        nadir_rel = int(np.argmin(finite_post)) if np.isfinite(finite_post).any() else 0
        search_start = start + nadir_rel
        hold = 6
        end = n - 1
        for idx in range(search_start, max(search_start, n - hold + 1)):
            values = r_sys[idx : idx + hold]
            if values.size == hold and np.all(np.isfinite(values) & (values >= 0.95)):
                end = idx
                break
        return [(start, end)]

    event_indices = np.array([], dtype=int)
    if event_signal is not None:
        signal = np.asarray(event_signal, dtype=float).reshape(-1)[:n]
        event_indices = np.flatnonzero(np.isfinite(signal) & (signal > event_threshold))
    if event_indices.size == 0:
        deficit_indices = np.flatnonzero(np.isfinite(r_sys) & (r_sys < 0.90))
        runs = _finite_runs(deficit_indices, max_gap_steps)
    else:
        runs = _finite_runs(event_indices, max_gap_steps)
    return [(start, end) for start, end in runs if end - start + 1 >= min_event_steps]


def select_dominant_event_window(
    windows: list[tuple[int, int]],
    event_signal: np.ndarray | None,
    R_sys: np.ndarray,
) -> tuple[int, int] | None:
    """Select the event with greatest integrated intensity or performance loss."""
    if not windows:
        return None
    r_sys = np.asarray(R_sys, dtype=float).reshape(-1)
    signal = None if event_signal is None else np.asarray(event_signal, dtype=float).reshape(-1)
    scores: list[float] = []
    for start, end in windows:
        if signal is not None and signal.size > start:
            values = signal[start : min(end + 1, signal.size)]
            score = float(np.nansum(np.maximum(values, 0.0)))
        else:
            values = r_sys[start : min(end + 1, r_sys.size)]
            score = float(np.nansum(np.maximum(0.0, 1.0 - values)))
        scores.append(score)
    return windows[int(np.argmax(np.asarray(scores)))]
def detect_event_window(
    event_signal: np.ndarray | None,
    R_sys: np.ndarray,
    event_threshold: float = 0.0,
    R_drop_threshold: float = 0.90,
) -> tuple[int, int]:
    """Detect the disruption event window from an event signal or performance drop.

    If a nonzero event signal is available, the event window is the first to
    last index where ``event_signal > event_threshold``. Otherwise, the event
    window is inferred from the first to last time step where
    ``R_sys < R_drop_threshold``. The returned start index is padded backward by
    three steps to include a short response period.

    Args:
        event_signal: Optional event intensity series with shape ``[T]`` or a
            broadcastable array. Non-finite values are ignored.
        R_sys: System-level performance ratio with shape ``[T]``.
        event_threshold: Threshold for detecting explicit event periods.
        R_drop_threshold: Threshold for detecting implicit disruptions when no
            explicit event signal is available.

    Returns:
        Tuple ``(event_start, event_end)`` of integer indices. If no event is
        detected, both indices are ``0``.
    """
    r_sys = np.asarray(R_sys, dtype=float).reshape(-1)
    time_steps = r_sys.size
    if time_steps == 0:
        return 0, 0

    event_indices: np.ndarray
    if event_signal is not None:
        signal = np.asarray(event_signal, dtype=float).reshape(-1)
        length = min(signal.size, time_steps)
        signal = signal[:length]
        event_indices = np.flatnonzero(np.isfinite(signal) & (signal > event_threshold))
    else:
        event_indices = np.array([], dtype=int)

    if event_indices.size == 0:
        event_indices = np.flatnonzero(np.isfinite(r_sys) & (r_sys < R_drop_threshold))

    if event_indices.size == 0:
        return 0, 0

    event_start = max(int(event_indices[0]) - 3, 0)
    event_end = min(int(event_indices[-1]), time_steps - 1)
    if event_end < event_start:
        event_end = event_start
    return event_start, event_end


def compute_resilience_indicators(
    R_sys: np.ndarray,
    event_start: int,
    event_end: int,
    baseline_end: int,
    recovery_threshold: float = 0.95,
    time_step_minutes: float = 5.0,
    recovery_hold_steps: int = 6,
) -> dict[str, float | bool]:
    """Compute disruption, nadir, loss, and sustained-recovery indicators.

    Recovery is declared only after ``recovery_hold_steps`` consecutive finite
    observations meet the threshold. If the event-end observation begins such
    a run, recovery time is zero. Missing observations are never treated as
    ideal performance.
    """
    r_sys = np.asarray(R_sys, dtype=float).reshape(-1)
    time_steps = r_sys.size
    empty = {
        "pre_R_mean": float("nan"),
        "min_R": float("nan"),
        "performance_loss": float("nan"),
        "recovery_time_steps": float("nan"),
        "recovery_time_hours": float("nan"),
        "recovery_slope": float("nan"),
        "resilience_index": float("nan"),
        "nadir_idx": float("nan"),
        "recovery_idx": float("nan"),
        "observed_recovery": False,
        "event_duration_hours": float("nan"),
    }
    if time_steps == 0:
        return empty

    event_start = int(np.clip(event_start, 0, time_steps - 1))
    event_end = int(np.clip(event_end, event_start, time_steps - 1))
    baseline_end = int(np.clip(baseline_end, 0, event_start))
    event_slice = r_sys[event_start : event_end + 1]
    finite_event_mask = np.isfinite(event_slice)
    finite_event = event_slice[finite_event_mask]
    pre_R_mean = _safe_slice_mean(r_sys[baseline_end:event_start])
    min_R = float(np.min(finite_event)) if finite_event.size else float("nan")
    if finite_event.size:
        finite_positions = np.flatnonzero(finite_event_mask)
        nadir_idx = int(event_start + finite_positions[int(np.argmin(finite_event))])
        performance_loss = float(
            np.sum(np.maximum(0.0, 1.0 - finite_event)) * float(time_step_minutes) / 60.0
        )
        resilience_index = float(np.mean(finite_event))
    else:
        nadir_idx = float("nan")
        performance_loss = float("nan")
        resilience_index = float("nan")

    hold = max(int(recovery_hold_steps), 1)
    recovery_idx: int | None = None
    for idx in range(event_end, time_steps - hold + 1):
        values = r_sys[idx : idx + hold]
        if np.all(np.isfinite(values) & (values >= recovery_threshold)):
            recovery_idx = idx
            break
    observed_recovery = recovery_idx is not None
    if recovery_idx is None:
        recovery_time_steps = float("nan")
        recovery_time_hours = float("nan")
        recovery_slope = float("nan")
        recovery_idx_value = float("nan")
    else:
        recovery_time_steps = float(max(recovery_idx - event_end, 0))
        recovery_time_hours = recovery_time_steps * float(time_step_minutes) / 60.0
        recovery_idx_value = float(recovery_idx)
        slope_start = nadir_idx if isinstance(nadir_idx, int) else event_end
        recovery_values = r_sys[int(slope_start) : recovery_idx + 1]
        finite = np.isfinite(recovery_values)
        if finite.sum() >= 2:
            x_axis = np.arange(recovery_values.size, dtype=float)[finite]
            recovery_slope = float(np.polyfit(x_axis, recovery_values[finite], deg=1)[0])
        else:
            recovery_slope = float("nan")

    return {
        "pre_R_mean": float(pre_R_mean),
        "min_R": float(min_R),
        "performance_loss": performance_loss,
        "recovery_time_steps": float(recovery_time_steps),
        "recovery_time_hours": float(recovery_time_hours),
        "recovery_slope": float(recovery_slope),
        "resilience_index": float(resilience_index),
        "nadir_idx": float(nadir_idx),
        "recovery_idx": recovery_idx_value,
        "observed_recovery": observed_recovery,
        "event_duration_hours": float(event_end - event_start + 1) * float(time_step_minutes) / 60.0,
    }

def compute_node_vulnerability(
    R: np.ndarray,
    event_start: int,
    event_end: int,
) -> np.ndarray:
    """Compute node vulnerability scores during the event window.

    Args:
        R: Node-level performance ratio with shape ``[T, N]``.
        event_start: Start index of the event window, inclusive.
        event_end: End index of the event window, inclusive.

    Returns:
        Array with shape ``[N]`` where ``vulnerability_i =
        1 - mean(R[event_start:event_end, i])``. Higher values indicate more
        vulnerable nodes.
    """
    ratio = _as_2d_float(R)
    time_steps = ratio.shape[0]
    if time_steps == 0:
        return np.zeros(ratio.shape[1], dtype=float)

    event_start = int(np.clip(event_start, 0, time_steps - 1))
    event_end = int(np.clip(event_end, event_start, time_steps - 1))
    event_values = ratio[event_start : event_end + 1]
    node_mean = _safe_nanmean(event_values, axis=0)
    vulnerability = 1.0 - node_mean
    return np.nan_to_num(vulnerability, nan=0.0, posinf=0.0, neginf=0.0)


def summarize_resilience(
    data: np.ndarray,
    baseline: dict[str, np.ndarray],
    event_signal: np.ndarray | None,
    train_end: int,
    dataset_name: str,
    time_step_minutes: float = 5.0,
) -> dict[str, Any]:
    """Run the full traffic resilience indicator pipeline.

    This function computes node-level performance ratios, detects the event
    window, estimates system-level resilience indicators, and ranks the five
    most vulnerable nodes.

    Args:
        data: Traffic values with shape ``[T, N]``.
        baseline: Output dictionary from :func:`compute_baseline`.
        event_signal: Optional event intensity series. If unavailable, event
            detection falls back to drops in system performance.
        train_end: Exclusive end index of the training period.
        dataset_name: Name written to the returned summary dictionary.
        time_step_minutes: Duration represented by each time step.

    Returns:
        JSON-serializable dictionary with dataset name, resilience indicators,
        event indices, number of nodes, and ``top5_vulnerable_nodes``.
    """
    array = _as_2d_float(data)
    R = compute_R_series(array, baseline, use_tod=True)
    R_sys = compute_system_R(R)
    event_start, event_end = detect_event_window(event_signal, R_sys)
    indicators = compute_resilience_indicators(
        R_sys,
        event_start=event_start,
        event_end=event_end,
        baseline_end=int(np.clip(train_end, 0, array.shape[0])),
        time_step_minutes=time_step_minutes,
    )
    vulnerability = compute_node_vulnerability(R, event_start, event_end)
    top_k = min(5, vulnerability.size)
    top5_nodes = (
        np.argsort(-vulnerability, kind="stable")[:top_k].astype(int).tolist()
        if top_k > 0
        else []
    )

    return {
        "dataset": dataset_name,
        "pre_R_mean": indicators["pre_R_mean"],
        "min_R": indicators["min_R"],
        "performance_loss_hours": indicators["performance_loss"],
        "recovery_time_hours": indicators["recovery_time_hours"],
        "recovery_slope": indicators["recovery_slope"],
        "resilience_index": indicators["resilience_index"],
        "event_start_idx": int(event_start),
        "event_end_idx": int(event_end),
        "n_nodes": int(array.shape[1]),
        "top5_vulnerable_nodes": top5_nodes,
    }
