"""Final ordinary-metric analysis for A9 PEMS03 external confirmation."""
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
    pair = root / f"seed_{seed}" / "pems03" / "flow"
    a9 = np.load(pair / "a9" / "test_predictions.npz", allow_pickle=True)
    dcrnn = np.load(pair / "dcrnn" / "test_predictions.npz", allow_pickle=True)
    truth_a9 = np.asarray(a9["y_true"], dtype=float)
    truth_dcrnn = np.asarray(dcrnn["y_true"], dtype=float)
    if truth_a9.shape != truth_dcrnn.shape:
        raise RuntimeError(f"truth shape mismatch for seed={seed}")
    if not np.array_equal(np.asarray(a9["test_indices"]), np.asarray(dcrnn["test_indices"])):
        raise RuntimeError(f"test index mismatch for seed={seed}")
    if not np.array_equal(np.asarray(a9["node_names"]), np.asarray(dcrnn["node_names"])):
        raise RuntimeError(f"node order mismatch for seed={seed}")
    valid = np.isfinite(truth_a9) & np.isfinite(truth_dcrnn)
    if valid.any() and not np.allclose(truth_a9[valid], truth_dcrnn[valid], atol=1e-8, rtol=1e-8):
        raise RuntimeError(f"physical truth mismatch for seed={seed}")
    return truth_a9, np.asarray(a9["y_pred"], dtype=float), np.asarray(dcrnn["y_pred"], dtype=float)


def analyze(root: Path, repetitions: int, block_length: int, seed: int):
    comparison_rows = []
    bootstrap_rows = []
    for run_seed in SEEDS:
        truth, a9_prediction, dcrnn_prediction = load_pair(root, run_seed)
        a9_metrics = regression_metrics(truth, a9_prediction)
        dcrnn_metrics = regression_metrics(truth, dcrnn_prediction)
        row = {"seed": run_seed}
        for metric in METRICS:
            row[f"a9_{metric}"] = float(a9_metrics[metric])
            row[f"dcrnn_{metric}"] = float(dcrnn_metrics[metric])
            row[f"{metric}_ratio"] = float(a9_metrics[metric] / dcrnn_metrics[metric])
        comparison_rows.append(row)
        a9_losses = window_losses(truth, a9_prediction)
        dcrnn_losses = window_losses(truth, dcrnn_prediction)
        for metric_index, metric in enumerate(METRICS):
            result = moving_block_bootstrap_difference(
                a9_losses[metric],
                dcrnn_losses[metric],
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
    comparison = pd.DataFrame(comparison_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    wins = {metric: int((comparison[f"{metric}_ratio"] < 1.0).sum()) for metric in METRICS}
    mean_ratios = {metric: float(comparison[f"{metric}_ratio"].mean()) for metric in METRICS}
    strict_count = int(bootstrap["strict_improvement"].sum())
    ordinary_gate = bool(
        all(value == 3 for value in wins.values())
        and all(value < 1.0 for value in mean_ratios.values())
        and strict_count >= 10
    )
    decision = {
        "stage": "A9-PEMS03-external-final",
        "paired_comparisons": 3,
        "bootstrap_comparisons": 12,
        "wins": wins,
        "mean_ratios": mean_ratios,
        "bootstrap_strict_improvements": strict_count,
        "ordinary_metric_gate_passed": ordinary_gate,
        "statistical_resilience_analysis_authorized": ordinary_gate,
        "final_model_selected": False,
    }
    return comparison, bootstrap, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\TrafficGNN\outputs\a9_pems03_external_confirmation")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    comparison, bootstrap, decision = analyze(
        root, args.bootstrap_repetitions, args.block_length, args.seed
    )
    comparison.to_csv(root / "a9_vs_dcrnn_external_verified.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(root / "a9_external_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    (root / "a9_external_ordinary_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
