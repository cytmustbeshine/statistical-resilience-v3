"""Evaluate the locked A6 autoregressive DSTSGCN against public DCRNN."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data import TrafficWindowDataset, build_static_adjacency
from l4_prediction_evaluation import apply_frozen_lower_tail_profile, binary_state_metrics, regression_metrics
from l4_prediction_pipeline import DUAL_SPACE_MISSING_PROTOCOL, PROFILE_COMPATIBLE_SPLIT_PROTOCOL, target_tensor_from_indices
from run_a4_capacity_validation import collect_predictions, model_for_baseline, prepare_variable
from statistical_tests import moving_block_bootstrap_difference
from temporal_decoder import AutoregressiveForecastWrapper

SEEDS = (42, 2024, 3407)
TASKS = (("bridge", "flow"), ("rainstorm", "flow"), ("rainstorm", "speed"), ("typhoon", "flow"), ("typhoon", "speed"))


def checkpoint_root(seed: int) -> Path:
    if seed == 42:
        return Path(r"D:\TrafficGNN\outputs\a6_ar_decoder_full\h64_lr0.001_rs0_zi0_ar1")
    return Path(rf"D:\TrafficGNN\outputs\a6_ar_decoder_multiseed_validation\seed_{seed}\h64_lr0.001_rs0_zi0_ar1")


def window_losses(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    error = np.where(valid, y_pred - y_true, 0.0)
    absolute = np.abs(error)
    count = np.maximum(valid.sum(axis=(1, 2, 3)), 1)
    denominator = np.maximum((np.abs(y_pred) + np.abs(y_true)) / 2.0, 1e-6)
    return {
        "mae": absolute.sum(axis=(1, 2, 3)) / count,
        "rmse": np.sqrt((error ** 2).sum(axis=(1, 2, 3)) / count),
        "smape": np.where(valid, absolute / denominator, 0.0).sum(axis=(1, 2, 3)) / count,
        "wape": absolute.sum(axis=(1, 2, 3)) / np.maximum(np.abs(y_true).sum(axis=(1, 2, 3)), 1e-6),
    }


def l4_metrics(y_true: np.ndarray, y_pred: np.ndarray, q90: float) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    tail = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > q90)
    state = binary_state_metrics(y_true, y_pred, q90)
    return {
        "l4_mae": float(metrics["mae"]),
        "l4_rmse": float(metrics["rmse"]),
        "l4_tail_mae": float(np.mean(np.abs(y_pred[tail] - y_true[tail]))) if tail.any() else float("nan"),
        "high_f1": float(state["f1"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal_final")
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a6_ar_decoder_outer_evaluation")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_root = Path(args.formal_root)
    prep_args = SimpleNamespace(
        l4_dir=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4",
        source_profile_dir=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3",
        max_nodes=41,
        history=12,
        horizon=12,
        split_protocol=PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
        missing_space_protocol=DUAL_SPACE_MISSING_PROTOCOL,
    )
    traffic_rows = []
    l4_rows = []
    bootstrap_rows = []

    for seed in SEEDS:
        for dataset, variable in TASKS:
            prepared = prepare_variable(dataset, variable, prep_args)
            bundle = prepared["bundle"]
            test_indices = np.asarray(bundle["test_indices"], dtype=int)
            dataset_windows = TrafficWindowDataset(bundle["scaled_values"], 12, 12)
            loader = DataLoader(Subset(dataset_windows, test_indices.tolist()), batch_size=64, shuffle=False)
            train_end = int(bundle["split_info"]["train_time_end_exclusive"])
            adjacency = build_static_adjacency(
                bundle["scaled_values"][:train_end], prepared["model_names"], source="corr", corr_threshold=0.2
            )
            static_adj = torch.tensor(adjacency, dtype=torch.float32, device=args.device)
            base = model_for_baseline(len(prepared["model_names"]), 12, 64, 2)
            model = AutoregressiveForecastWrapper(base, hidden_dim=64, horizon=12).to(args.device)
            checkpoint = torch.load(
                checkpoint_root(seed) / dataset / variable / "best_validation_only.pt",
                map_location=args.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            y_scaled, pred_scaled = collect_predictions(model, loader, static_adj, args.device)
            y_true = target_tensor_from_indices(prepared["selected_physical_values"], test_indices, 12, 12)
            y_pred = bundle["scaler"].inverse_transform(pred_scaled)
            target_timestamps = np.asarray(prepared["timestamps"])[
                test_indices[:, None] + np.arange(12, 24)[None, :]
            ]
            baseline_archive = np.load(
                formal_root / "m0_dcrnn" / dataset / variable / f"seed_{seed}" / "traffic_predictions.npz",
                allow_pickle=True,
            )
            baseline_pred = np.asarray(baseline_archive["y_pred_physical"], dtype=float)
            baseline_true = np.asarray(baseline_archive["y_true_physical"], dtype=float)
            if target_timestamps.shape != baseline_archive["target_timestamps"].shape or not np.array_equal(
                target_timestamps.astype("datetime64[ns]"), baseline_archive["target_timestamps"].astype("datetime64[ns]")
            ):
                raise RuntimeError(f"A6/DCRNN timestamp mismatch: {dataset}/{variable}/seed={seed}")
            common = np.isfinite(y_true) & np.isfinite(baseline_true)
            if common.any() and not np.allclose(y_true[common], baseline_true[common], atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"A6/DCRNN physical truth mismatch: {dataset}/{variable}/seed={seed}")

            candidate_metrics = regression_metrics(y_true, y_pred)
            baseline_metrics = regression_metrics(y_true, baseline_pred)
            row = {"seed": seed, "dataset": dataset, "variable": variable}
            for metric in ("mae", "rmse", "smape", "wape"):
                row[f"baseline_{metric}"] = float(baseline_metrics[metric])
                row[f"a6_{metric}"] = float(candidate_metrics[metric])
                row[f"{metric}_ratio"] = float(candidate_metrics[metric] / baseline_metrics[metric])
            traffic_rows.append(row)

            candidate_losses = window_losses(y_true, y_pred)
            baseline_losses = window_losses(y_true, baseline_pred)
            for metric in ("mae", "rmse", "smape", "wape"):
                result = moving_block_bootstrap_difference(
                    candidate_losses[metric], baseline_losses[metric],
                    block_length=args.block_length,
                    repetitions=args.bootstrap_repetitions,
                    seed=seed + 100,
                )
                bootstrap_rows.append({"seed": seed, "dataset": dataset, "variable": variable, "metric": metric, **result})

            flat_times = target_timestamps.reshape(-1)
            flat_true = y_true[..., 0].reshape(-1, y_true.shape[2])
            flat_a6 = y_pred[..., 0].reshape(-1, y_pred.shape[2])
            flat_baseline = baseline_pred[..., 0].reshape(-1, baseline_pred.shape[2])
            clip = float(prepared["reference"]["deficit_clip_value"])
            q90 = float(prepared["reference"]["train_deficit_quantiles"]["q90"])
            true_l4 = apply_frozen_lower_tail_profile(
                flat_true, flat_times, prepared["profile"], prepared["full_values"].shape[1], prepared["full_indices"], clip
            )["system_deficit"]
            a6_l4 = apply_frozen_lower_tail_profile(
                flat_a6, flat_times, prepared["profile"], prepared["full_values"].shape[1], prepared["full_indices"], clip
            )["system_deficit"]
            baseline_l4 = apply_frozen_lower_tail_profile(
                flat_baseline, flat_times, prepared["profile"], prepared["full_values"].shape[1], prepared["full_indices"], clip
            )["system_deficit"]
            a6_l4_metrics = l4_metrics(true_l4, a6_l4, q90)
            baseline_l4_metrics = l4_metrics(true_l4, baseline_l4, q90)
            l4_row = {"seed": seed, "dataset": dataset, "variable": variable}
            for metric in ("l4_mae", "l4_rmse", "l4_tail_mae"):
                l4_row[f"baseline_{metric}"] = baseline_l4_metrics[metric]
                l4_row[f"a6_{metric}"] = a6_l4_metrics[metric]
                l4_row[f"{metric}_ratio"] = a6_l4_metrics[metric] / baseline_l4_metrics[metric]
            l4_row["baseline_high_f1"] = baseline_l4_metrics["high_f1"]
            l4_row["a6_high_f1"] = a6_l4_metrics["high_f1"]
            l4_row["high_f1_improved"] = a6_l4_metrics["high_f1"] > baseline_l4_metrics["high_f1"]
            l4_rows.append(l4_row)

            archive_dir = output_dir / f"seed_{seed}" / dataset / variable
            archive_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                archive_dir / "a6_predictions.npz",
                dataset=dataset,
                variable=variable,
                seed=seed,
                target_timestamps=target_timestamps,
                node_names=np.asarray(prepared["model_names"]),
                y_true_physical=y_true,
                y_pred_physical=y_pred,
                train_time_end_exclusive=train_end,
                val_time_end_exclusive=int(bundle["split_info"]["val_time_end_exclusive"]),
            )
            print(f"[A6 outer] seed={seed} {dataset}/{variable} MAE={candidate_metrics['mae']:.6f}", flush=True)

    traffic = pd.DataFrame(traffic_rows)
    l4 = pd.DataFrame(l4_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    traffic.to_csv(output_dir / "a6_traffic_vs_dcrnn.csv", index=False, encoding="utf-8-sig")
    l4.to_csv(output_dir / "a6_l4_vs_dcrnn.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "a6_traffic_block_bootstrap.csv", index=False, encoding="utf-8-sig")

    ordinary_wins = {metric: int((traffic[f"{metric}_ratio"] < 1.0).sum()) for metric in ("mae", "rmse", "smape", "wape")}
    l4_wins = {metric: int((l4[f"{metric}_ratio"] < 1.0).sum()) for metric in ("l4_mae", "l4_rmse", "l4_tail_mae")}
    l4_wins["high_f1"] = int(l4["high_f1_improved"].sum())
    decision = {
        "stage": "A6-outer-evaluation",
        "model": "pure_dstsgcn_autoregressive_decoder",
        "uses_dcrnn_predictions": False,
        "selection_used_test_metrics": False,
        "existing_test_archive_previously_inspected_in_project": True,
        "comparisons": 15,
        "ordinary_wins": ordinary_wins,
        "l4_wins": l4_wins,
        "bootstrap_strict_improvements": int((bootstrap["ci_upper_95"] < 0).sum()),
        "bootstrap_comparisons": int(len(bootstrap)),
        "all_metrics_exceeded": bool(all(value == 15 for value in ordinary_wins.values()) and all(value == 15 for value in l4_wins.values())),
        "final_model_selected": False,
    }
    (output_dir / "a6_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# A6纯DSTSGCN自回归解码外层评估",
        "",
        f"普通指标胜出计数：{ordinary_wins}。",
        f"冻结L4指标胜出计数：{l4_wins}。",
        f"moving-block bootstrap严格改善：{decision['bootstrap_strict_improvements']}/{decision['bootstrap_comparisons']}。",
        "",
        "A6不使用DCRNN预测作为输入或集成组件。由于项目此前已经查看过相同测试档案，本结果仍需新的时间外推或外部数据确认后才能作为最终论文模型。",
    ]
    (output_dir / "a6_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())