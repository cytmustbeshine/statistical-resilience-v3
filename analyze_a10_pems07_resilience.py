"""Train-only flow-deficit analysis for the frozen A10 PEMS07 test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from flow_speed_resilience import (
    compute_directional_probabilistic_deficit,
    fit_variable_resilience_profile,
)
from l4_prediction_evaluation import binary_state_metrics


SEEDS = (42, 2024, 3407)
PEMS07_PATH = Path(r"D:\TrafficGNN\data\public\PEMS07\PEMS07.npz")
HISTORY = 12
TIME_OF_DAY_BINS = 288
ERROR_METRICS = ("deficit_mae", "deficit_rmse", "q90_tail_mae")


def aggregate_overlapping_forecasts(
    values: np.ndarray,
    window_indices: np.ndarray,
    history: int = HISTORY,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    indices = np.asarray(window_indices, dtype=int)
    if array.ndim != 4 or array.shape[-1] != 1:
        raise ValueError("forecast values must have shape [windows,horizon,nodes,1]")
    if len(array) != len(indices):
        raise ValueError("window index count differs from forecast count")
    targets = indices[:, None] + int(history) + np.arange(array.shape[1])[None, :]
    flat_targets = targets.reshape(-1)
    flat_values = array[..., 0].reshape(-1, array.shape[2])
    unique_targets, inverse = np.unique(flat_targets, return_inverse=True)
    sums = np.zeros((len(unique_targets), array.shape[2]), dtype=float)
    counts = np.zeros_like(sums)
    valid = np.isfinite(flat_values)
    for node in range(array.shape[2]):
        np.add.at(sums[:, node], inverse[valid[:, node]], flat_values[valid[:, node], node])
        np.add.at(counts[:, node], inverse[valid[:, node]], 1.0)
    aggregated = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return unique_targets, aggregated


def load_pair(root: Path, seed: int):
    pair = root / f"seed_{seed}" / "pems07" / "flow"
    candidate = np.load(pair / "a10" / "test_predictions.npz", allow_pickle=True)
    baseline = np.load(pair / "dcrnn" / "test_predictions.npz", allow_pickle=True)
    if not np.array_equal(candidate["test_indices"], baseline["test_indices"]):
        raise RuntimeError(f"test index mismatch for seed={seed}")
    if not np.array_equal(candidate["node_names"], baseline["node_names"]):
        raise RuntimeError(f"node order mismatch for seed={seed}")
    truth = np.asarray(candidate["y_true"], dtype=float)
    baseline_truth = np.asarray(baseline["y_true"], dtype=float)
    valid = np.isfinite(truth) & np.isfinite(baseline_truth)
    if truth.shape != baseline_truth.shape or not np.allclose(
        truth[valid], baseline_truth[valid], atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError(f"physical truth mismatch for seed={seed}")
    return (
        truth,
        np.asarray(candidate["y_pred"], dtype=float),
        np.asarray(baseline["y_pred"], dtype=float),
        np.asarray(candidate["test_indices"], dtype=int),
    )


def deficit_metrics(truth: np.ndarray, prediction: np.ndarray, q90: float) -> dict[str, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    error = prediction - truth
    tail = valid & (truth > q90)
    state = binary_state_metrics(truth, prediction, q90)
    return {
        "deficit_mae": float(np.mean(np.abs(error[valid]))),
        "deficit_rmse": float(np.sqrt(np.mean(np.square(error[valid])))),
        "q90_tail_mae": float(np.mean(np.abs(error[tail]))) if tail.any() else float("nan"),
        "high_state_f1": float(state["f1"]),
        "high_state_recall": float(state["recall"]),
        "predicted_high_rate": float(state["predicted_high_rate"]),
        "true_high_rate": float(np.mean(truth[valid] > q90)),
    }


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(int(block_length), n))
    starts = np.arange(max(n - block + 1, 1))
    chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
    return np.concatenate([np.arange(start, start + block) for start in chosen])[:n]


def bootstrap_comparison(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    q90: float,
    repetitions: int,
    block_length: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in (*ERROR_METRICS, "high_state_f1")}
    for _ in range(repetitions):
        sampled = moving_block_indices(len(truth), block_length, rng)
        candidate_metrics = deficit_metrics(truth[sampled], candidate[sampled], q90)
        baseline_metrics = deficit_metrics(truth[sampled], baseline[sampled], q90)
        for metric in ERROR_METRICS:
            samples[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
        samples["high_state_f1"].append(
            candidate_metrics["high_state_f1"] - baseline_metrics["high_state_f1"]
        )
    rows = []
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        lower, upper = np.quantile(array, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "difference_direction": "a10_minus_dcrnn",
                "estimate": float(np.mean(array)),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "block_length": int(block_length),
                "repetitions": int(repetitions),
            }
        )
    return rows


def lag1_dependence(residual: np.ndarray) -> dict[str, float]:
    values = np.asarray(residual, dtype=float)
    valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    if valid.sum() < 3:
        return {"lag1_pearson": float("nan"), "lag1_spearman": float("nan")}
    left, right = values[:-1][valid], values[1:][valid]
    return {
        "lag1_pearson": float(np.corrcoef(left, right)[0, 1]),
        "lag1_spearman": float(stats.spearmanr(left, right).statistic),
    }


def gate_decision(metrics: pd.DataFrame) -> dict[str, object]:
    wins = {
        metric: int((metrics[f"{metric}_ratio"] < 1.0).sum())
        for metric in ERROR_METRICS
    }
    mean_ratios = {
        metric: float(metrics[f"{metric}_ratio"].mean())
        for metric in ERROR_METRICS
    }
    mean_f1 = {
        "a10": float(metrics["a10_high_state_f1"].mean()),
        "dcrnn": float(metrics["dcrnn_high_state_f1"].mean()),
    }
    passed = bool(
        all(value >= 2 for value in wins.values())
        and all(value < 1.0 for value in mean_ratios.values())
        and mean_f1["a10"] >= mean_f1["dcrnn"]
    )
    return {
        "resilience_wins": wins,
        "resilience_mean_ratios": mean_ratios,
        "mean_high_state_f1": mean_f1,
        "resilience_gate_passed": passed,
    }


def analyze(root: Path, repetitions: int, block_length: int, seed: int):
    physical = np.asarray(np.load(PEMS07_PATH)["data"], dtype=float)[:, :41, 0]
    train_end = int(len(physical) * 0.6)
    profile = fit_variable_resilience_profile(
        physical,
        train_end,
        None,
        "flow",
        "lower",
        "log1p_nonnegative",
        time_of_day_bins=TIME_OF_DAY_BINS,
        day_type_mode="none",
    )
    truth_state = compute_directional_probabilistic_deficit(physical, profile, None, "lower")
    truth_deficit_full = np.asarray(truth_state["system_deficit"], dtype=float)
    q90 = float(truth_state["train_deficit_quantiles"]["q90"])
    rows = []
    bootstrap_rows = []
    dependence_rows = []
    common_targets = None
    for run_seed in SEEDS:
        truth, candidate, baseline, test_indices = load_pair(root, run_seed)
        targets, truth_aggregated = aggregate_overlapping_forecasts(truth, test_indices)
        candidate_targets, candidate_aggregated = aggregate_overlapping_forecasts(candidate, test_indices)
        baseline_targets, baseline_aggregated = aggregate_overlapping_forecasts(baseline, test_indices)
        if not np.array_equal(targets, candidate_targets) or not np.array_equal(targets, baseline_targets):
            raise RuntimeError(f"aggregated target mismatch for seed={run_seed}")
        if common_targets is None:
            common_targets = targets
        elif not np.array_equal(common_targets, targets):
            raise RuntimeError("test targets differ across seeds")
        if not np.allclose(truth_aggregated, physical[targets], equal_nan=True, atol=1e-8, rtol=1e-8):
            raise RuntimeError(f"aggregated truth differs from PEMS07 physical data for seed={run_seed}")
        model_deficits = {}
        for model_name, prediction in (("a10", candidate_aggregated), ("dcrnn", baseline_aggregated)):
            hybrid = physical.copy()
            hybrid[targets] = prediction
            state = compute_directional_probabilistic_deficit(hybrid, profile, None, "lower")
            model_deficits[model_name] = np.asarray(state["system_deficit"], dtype=float)[targets]
        truth_deficit = truth_deficit_full[targets]
        candidate_metrics = deficit_metrics(truth_deficit, model_deficits["a10"], q90)
        baseline_metrics = deficit_metrics(truth_deficit, model_deficits["dcrnn"], q90)
        row = {"seed": run_seed, "test_target_count": int(len(targets))}
        for metric in ERROR_METRICS:
            row[f"a10_{metric}"] = candidate_metrics[metric]
            row[f"dcrnn_{metric}"] = baseline_metrics[metric]
            row[f"{metric}_ratio"] = candidate_metrics[metric] / baseline_metrics[metric]
        for metric in ("high_state_f1", "high_state_recall", "predicted_high_rate", "true_high_rate"):
            row[f"a10_{metric}"] = candidate_metrics[metric]
            row[f"dcrnn_{metric}"] = baseline_metrics[metric]
        rows.append(row)
        for bootstrap_row in bootstrap_comparison(
            truth_deficit,
            model_deficits["a10"],
            model_deficits["dcrnn"],
            q90,
            repetitions,
            block_length,
            seed + run_seed,
        ):
            bootstrap_rows.append({"seed": run_seed, **bootstrap_row})
        for model_name in ("a10", "dcrnn"):
            dependence_rows.append(
                {
                    "seed": run_seed,
                    "model": model_name,
                    **lag1_dependence(model_deficits[model_name] - truth_deficit),
                }
            )
    metrics = pd.DataFrame(rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    dependence = pd.DataFrame(dependence_rows)
    decision = {
        "stage": "A10-PEMS07-resilience-final",
        "profile_train_only": True,
        "train_end_exclusive": train_end,
        "time_grouping": "observed_sequence_index_mod_288",
        "calendar_dates_fabricated": False,
        "day_type_mode": "none",
        "selected_shrinkage_m": float(profile["profile"]["selected_shrinkage_m"]),
        "deficit_clip_value": float(truth_state["deficit_clip_value"]),
        "train_q90": q90,
        **gate_decision(metrics),
        "final_model_selected": False,
    }
    return metrics, bootstrap, dependence, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\TrafficGNN\outputs\a10_pems07_external_confirmation")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    ordinary = json.loads((root / "a10_external_ordinary_decision.json").read_text(encoding="utf-8"))
    if not ordinary.get("statistical_gate_authorized", False):
        raise RuntimeError("ordinary A10 gate did not authorize resilience analysis")
    metrics, bootstrap, dependence, decision = analyze(
        root, args.bootstrap_repetitions, args.block_length, args.seed
    )
    metrics.to_csv(root / "a10_resilience_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(root / "a10_resilience_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    dependence.to_csv(root / "a10_resilience_residual_dependence.csv", index=False, encoding="utf-8-sig")
    (root / "a10_resilience_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    final = {
        "stage": "A10-PEMS07-complete-confirmation",
        "ordinary_metric_gate_passed": bool(ordinary["ordinary_metric_gate_passed"]),
        "resilience_gate_passed": bool(decision["resilience_gate_passed"]),
        "final_model_selected": bool(
            ordinary["ordinary_metric_gate_passed"] and decision["resilience_gate_passed"]
        ),
    }
    (root / "a10_final_decision.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**decision, **final}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
