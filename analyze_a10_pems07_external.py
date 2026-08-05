"""Final ordinary and horizon analysis for A10 PEMS07 confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_a6_outer import window_losses
from l4_prediction_evaluation import regression_metrics
from statistical_tests import moving_block_bootstrap_difference


SEEDS = (42, 2024, 3407)
METRICS = ("mae", "rmse", "smape", "wape")


def load_pair(root: Path, seed: int):
    pair = root / f"seed_{seed}" / "pems07" / "flow"
    candidate = np.load(pair / "a10" / "test_predictions.npz", allow_pickle=True)
    baseline = np.load(pair / "dcrnn" / "test_predictions.npz", allow_pickle=True)
    truth_candidate = np.asarray(candidate["y_true"], dtype=float)
    truth_baseline = np.asarray(baseline["y_true"], dtype=float)
    if truth_candidate.shape != truth_baseline.shape:
        raise RuntimeError(f"truth shape mismatch for seed={seed}")
    if not np.array_equal(candidate["test_indices"], baseline["test_indices"]):
        raise RuntimeError(f"test index mismatch for seed={seed}")
    if not np.array_equal(candidate["node_names"], baseline["node_names"]):
        raise RuntimeError(f"node order mismatch for seed={seed}")
    valid = np.isfinite(truth_candidate) & np.isfinite(truth_baseline)
    if valid.any() and not np.allclose(
        truth_candidate[valid], truth_baseline[valid], atol=1e-8, rtol=1e-8
    ):
        raise RuntimeError(f"physical truth mismatch for seed={seed}")
    return (
        truth_candidate,
        np.asarray(candidate["y_pred"], dtype=float),
        np.asarray(baseline["y_pred"], dtype=float),
    )


def horizon_mae(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = np.isfinite(truth) & np.isfinite(prediction)
    absolute = np.where(valid, np.abs(prediction - truth), 0.0)
    count = np.maximum(valid.sum(axis=(0, 2, 3)), 1)
    return absolute.sum(axis=(0, 2, 3)) / count


def analyze(root: Path, repetitions: int, block_length: int, seed: int):
    comparison_rows = []
    bootstrap_rows = []
    horizon_rows = []
    for run_seed in SEEDS:
        truth, candidate, baseline = load_pair(root, run_seed)
        candidate_metrics = regression_metrics(truth, candidate)
        baseline_metrics = regression_metrics(truth, baseline)
        row = {"seed": run_seed}
        for metric in METRICS:
            row[f"a10_{metric}"] = float(candidate_metrics[metric])
            row[f"dcrnn_{metric}"] = float(baseline_metrics[metric])
            row[f"{metric}_ratio"] = float(candidate_metrics[metric] / baseline_metrics[metric])
        comparison_rows.append(row)
        candidate_losses = window_losses(truth, candidate)
        baseline_losses = window_losses(truth, baseline)
        for metric_index, metric in enumerate(METRICS):
            result = moving_block_bootstrap_difference(
                candidate_losses[metric],
                baseline_losses[metric],
                block_length=block_length,
                repetitions=repetitions,
                seed=seed + run_seed + metric_index * 100,
            )
            bootstrap_rows.append(
                {
                    "seed": run_seed,
                    "metric": metric,
                    **result,
                    "strict_improvement": bool(result["ci_upper_95"] < 0.0),
                }
            )
        candidate_horizon = horizon_mae(truth, candidate)
        baseline_horizon = horizon_mae(truth, baseline)
        for horizon_index, (candidate_value, baseline_value) in enumerate(
            zip(candidate_horizon, baseline_horizon), start=1
        ):
            horizon_rows.append(
                {
                    "seed": run_seed,
                    "horizon": horizon_index,
                    "a10_mae": float(candidate_value),
                    "dcrnn_mae": float(baseline_value),
                    "mae_ratio": float(candidate_value / baseline_value),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    horizon = pd.DataFrame(horizon_rows)
    wins = {metric: int((comparison[f"{metric}_ratio"] < 1.0).sum()) for metric in METRICS}
    mean_ratios = {metric: float(comparison[f"{metric}_ratio"].mean()) for metric in METRICS}
    strict_count = int(bootstrap["strict_improvement"].sum())
    ordinary_gate = bool(
        all(value == 3 for value in wins.values())
        and all(value < 1.0 for value in mean_ratios.values())
        and strict_count >= 10
    )
    decision = {
        "stage": "A10-PEMS07-external-final",
        "paired_comparisons": 3,
        "bootstrap_comparisons": 12,
        "wins": wins,
        "mean_ratios": mean_ratios,
        "bootstrap_strict_improvements": strict_count,
        "horizon_improvements": int((horizon["mae_ratio"] < 1.0).sum()),
        "horizon_comparisons": int(len(horizon)),
        "ordinary_metric_gate_passed": ordinary_gate,
        "statistical_gate_authorized": ordinary_gate,
        "final_model_selected": False,
    }
    return comparison, bootstrap, horizon, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\TrafficGNN\outputs\a10_pems07_external_confirmation")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    comparison, bootstrap, horizon, decision = analyze(
        root, args.bootstrap_repetitions, args.block_length, args.seed
    )
    comparison.to_csv(root / "a10_vs_dcrnn_external_verified.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(root / "a10_external_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(root / "a10_external_horizon_mae.csv", index=False, encoding="utf-8-sig")
    (root / "a10_external_ordinary_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
