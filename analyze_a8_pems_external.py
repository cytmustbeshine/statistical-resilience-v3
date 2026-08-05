"""Analyze locked external PEMS predictions for A8 versus public DCRNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_a6_outer import window_losses
from l4_prediction_evaluation import regression_metrics
from statistical_tests import moving_block_bootstrap_difference


TASKS = (
    ("pems04", "flow"),
    ("pems04", "speed"),
    ("pems08", "flow"),
    ("pems08", "speed"),
)
SEEDS = (42, 2024, 3407)
METRICS = ("mae", "rmse", "smape", "wape")


def load_pair(root: Path, seed: int, dataset: str, variable: str):
    pair_root = root / f"seed_{seed}" / dataset / variable
    a8 = np.load(pair_root / "a8" / "test_predictions.npz", allow_pickle=True)
    dcrnn = np.load(pair_root / "dcrnn" / "test_predictions.npz", allow_pickle=True)
    a8_truth = np.asarray(a8["y_true"], dtype=float)
    dcrnn_truth = np.asarray(dcrnn["y_true"], dtype=float)
    if a8_truth.shape != dcrnn_truth.shape:
        raise RuntimeError(f"truth shape mismatch: {dataset}/{variable}/seed={seed}")
    common = np.isfinite(a8_truth) & np.isfinite(dcrnn_truth)
    if common.any() and not np.allclose(a8_truth[common], dcrnn_truth[common], atol=1e-8, rtol=1e-8):
        raise RuntimeError(f"physical truth mismatch: {dataset}/{variable}/seed={seed}")
    a8_indices = np.asarray(a8["test_indices"], dtype=int)
    dcrnn_indices = np.asarray(dcrnn["test_indices"], dtype=int)
    if not np.array_equal(a8_indices, dcrnn_indices):
        raise RuntimeError(f"test index mismatch: {dataset}/{variable}/seed={seed}")
    return a8_truth, np.asarray(a8["y_pred"], dtype=float), np.asarray(dcrnn["y_pred"], dtype=float)


def analyze(root: Path, repetitions: int, block_length: int, seed: int):
    comparison_rows = []
    bootstrap_rows = []
    for task_index, (dataset, variable) in enumerate(TASKS):
        for run_seed in SEEDS:
            truth, a8_prediction, dcrnn_prediction = load_pair(
                root, run_seed, dataset, variable
            )
            a8_metrics = regression_metrics(truth, a8_prediction)
            dcrnn_metrics = regression_metrics(truth, dcrnn_prediction)
            row = {"dataset": dataset, "variable": variable, "seed": run_seed}
            for metric in METRICS:
                row[f"a8_{metric}"] = float(a8_metrics[metric])
                row[f"dcrnn_{metric}"] = float(dcrnn_metrics[metric])
                row[f"{metric}_ratio"] = float(a8_metrics[metric] / dcrnn_metrics[metric])
            comparison_rows.append(row)
            a8_losses = window_losses(truth, a8_prediction)
            dcrnn_losses = window_losses(truth, dcrnn_prediction)
            for metric_index, metric in enumerate(METRICS):
                result = moving_block_bootstrap_difference(
                    a8_losses[metric],
                    dcrnn_losses[metric],
                    block_length=block_length,
                    repetitions=repetitions,
                    seed=seed + task_index * 1000 + run_seed + metric_index * 100,
                )
                bootstrap_rows.append(
                    {
                        "dataset": dataset,
                        "variable": variable,
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
    task_means = comparison.groupby(["dataset", "variable"])[
        [f"{metric}_ratio" for metric in METRICS]
    ].mean()
    mae_task_means_improved = bool((task_means["mae_ratio"] < 1.0).all())
    maximum_task_mean_ratio = float(task_means.to_numpy().max())
    strict_bootstrap = int(bootstrap["strict_improvement"].sum())
    ordinary_gate = bool(
        all(value >= 10 for value in wins.values())
        and mae_task_means_improved
        and all(value < 1.0 for value in mean_ratios.values())
        and maximum_task_mean_ratio <= 1.03
        and strict_bootstrap >= 40
    )
    decision = {
        "stage": "A8-PEMS-external-final",
        "paired_comparisons": int(len(comparison)),
        "bootstrap_comparisons": int(len(bootstrap)),
        "wins": wins,
        "mean_ratios": mean_ratios,
        "mae_all_task_means_improved": mae_task_means_improved,
        "maximum_task_mean_metric_ratio": maximum_task_mean_ratio,
        "bootstrap_strict_improvements": strict_bootstrap,
        "ordinary_metric_gate_passed": ordinary_gate,
        "statistical_resilience_analysis_authorized": ordinary_gate,
        "final_model_selected": False,
    }
    return comparison, bootstrap, task_means.reset_index(), decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=r"D:\TrafficGNN\outputs\a8_pems_external_confirmation"
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.root)
    comparison, bootstrap, task_means, decision = analyze(
        root, args.bootstrap_repetitions, args.block_length, args.seed
    )
    comparison.to_csv(root / "a8_vs_dcrnn_external_verified.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(root / "a8_external_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    task_means.to_csv(root / "a8_external_task_mean_ratios.csv", index=False, encoding="utf-8-sig")
    (root / "a8_external_final_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
