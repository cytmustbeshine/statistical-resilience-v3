"""Event-level traffic resilience process metrics from system deficit curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_consecutive_recovery(
    system_deficit: np.ndarray,
    start_index: int,
    threshold: float,
    consecutive_steps: int,
) -> int | None:
    """Find the first run that remains at or below a train-only threshold.

    Args:
        system_deficit: One-dimensional instantaneous system loss.
        start_index: Inclusive index from which recovery may be declared.
        threshold: Predeclared train-only recovery threshold.
        consecutive_steps: Required finite consecutive observations.

    Returns:
        Index of the first observation in the qualifying run, or ``None`` when
        the observed window cannot confirm recovery.
    """
    values = np.asarray(system_deficit, dtype=float)
    required = max(int(consecutive_steps), 1)
    run = 0
    for index in range(max(int(start_index), 0), len(values)):
        if np.isfinite(values[index]) and values[index] <= threshold:
            run += 1
            if run >= required:
                return index - required + 1
        else:
            run = 0
    return None


def compute_event_resilience_metrics(
    system_deficit: np.ndarray,
    event_start: int,
    event_end: int,
    high_threshold: float,
    extreme_threshold: float,
    recovery_threshold: float,
    consecutive_steps: int = 12,
    delta_t_hours: float = 5.0 / 60.0,
    baseline_steps: int = 288,
) -> dict[str, object]:
    """Summarize one event segment without treating time points as iid.

    Peak, mean and cumulative loss are calculated inside the segment. Recovery
    is declared only after the peak when loss stays below the predeclared
    train-only threshold for the requested consecutive run; otherwise the event
    is marked ``censored``.
    """
    values = np.asarray(system_deficit, dtype=float)
    if values.ndim != 1:
        raise ValueError("system_deficit must be one-dimensional")
    start = int(np.clip(event_start, 0, max(len(values) - 1, 0)))
    end = int(np.clip(event_end, start, max(len(values) - 1, 0)))
    event = values[start : end + 1]
    finite = np.isfinite(event)
    pre = values[max(0, start - baseline_steps) : start]
    baseline = float(np.nanmean(pre)) if np.isfinite(pre).any() else np.nan
    if not finite.any():
        return {
            "event_start": start,
            "event_end": end,
            "baseline_deficit": baseline,
            "peak_deficit": np.nan,
            "mean_deficit": np.nan,
            "cumulative_deficit": np.nan,
            "high_state_duration": np.nan,
            "extreme_state_duration": np.nan,
            "peak_time": np.nan,
            "degradation_duration": np.nan,
            "recovery_time": np.nan,
            "recovery_duration": np.nan,
            "recovery_rate": np.nan,
            "recovery_correspondence": np.nan,
            "recovery_status": "unavailable",
        }
    local_peak = int(np.nanargmax(event))
    peak_index = start + local_peak
    peak = float(event[local_peak])
    recovery_search_start = max(peak_index, end + 1)
    recovery_index = find_consecutive_recovery(
        values, recovery_search_start, recovery_threshold, consecutive_steps
    )
    if recovery_index is None:
        recovery_time = np.nan
        recovery_duration = np.nan
        recovery_rate = np.nan
        recovery_correspondence = np.nan
        recovery_status = "censored"
    else:
        recovery_time = float(recovery_index)
        recovery_duration = float(recovery_index - peak_index) * delta_t_hours
        recovery_rate = float((peak - recovery_threshold) / max(recovery_duration, 1e-12))
        recovery_correspondence = float(recovery_index - end) * delta_t_hours
        recovery_status = "observed"
    return {
        "event_start": start,
        "event_end": end,
        "baseline_deficit": baseline,
        "peak_deficit": peak,
        "mean_deficit": float(np.nanmean(event)),
        "cumulative_deficit": float(np.nansum(event) * delta_t_hours),
        "high_state_duration": float(np.sum(event > high_threshold) * delta_t_hours),
        "extreme_state_duration": float(np.sum(event > extreme_threshold) * delta_t_hours),
        "peak_time": float(peak_index),
        "degradation_duration": float(peak_index - start) * delta_t_hours,
        "recovery_time": recovery_time,
        "recovery_duration": recovery_duration,
        "recovery_rate": recovery_rate,
        "recovery_correspondence": recovery_correspondence,
        "recovery_status": recovery_status,
    }


def analyze_event_segments(
    dataset: str,
    definition: str,
    system_deficit: np.ndarray,
    segments: list[tuple[int, int]],
    train_quantiles: dict[str, float],
    delta_t_hours: float,
    recovery_quantile: str = "q75",
    consecutive_steps: int = 12,
) -> list[dict[str, object]]:
    """Analyze each event segment separately using definition-specific thresholds."""
    rows = []
    for segment_index, (start, end) in enumerate(segments, start=1):
        metrics = compute_event_resilience_metrics(
            system_deficit,
            start,
            end,
            high_threshold=float(train_quantiles["q90"]),
            extreme_threshold=float(train_quantiles["q99"]),
            recovery_threshold=float(train_quantiles[recovery_quantile]),
            consecutive_steps=consecutive_steps,
            delta_t_hours=delta_t_hours,
        )
        rows.append(
            {
                "dataset": dataset,
                "definition": definition,
                "event_segment": segment_index,
                "recovery_quantile": recovery_quantile,
                "consecutive_steps": consecutive_steps,
                **metrics,
            }
        )
    return rows


def recovery_sensitivity_rows(
    dataset: str,
    definition: str,
    system_deficit: np.ndarray,
    segments: list[tuple[int, int]],
    train_quantiles: dict[str, float],
    delta_t_hours: float,
) -> list[dict[str, object]]:
    """Evaluate the preregistered q50/q75/q90 and 6/12/24 recovery grid."""
    rows = []
    for quantile in ("q50", "q75", "q90"):
        for consecutive in (6, 12, 24):
            rows.extend(
                analyze_event_segments(
                    dataset,
                    definition,
                    system_deficit,
                    segments,
                    train_quantiles,
                    delta_t_hours,
                    recovery_quantile=quantile,
                    consecutive_steps=consecutive,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Recompute event resilience metrics from a CSV curve")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {"dataset", "definition", "system_deficit", "event_start", "event_end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()