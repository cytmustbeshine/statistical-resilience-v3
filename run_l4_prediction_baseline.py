"""Run the gated E-L4-1 five-node indirect resilience forecasting smoke test."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from analyze_latent_traffic_performance import load_saved_profile
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from data import TrafficWindowDataset, build_static_adjacency
from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_evaluation import (
    apply_frozen_lower_tail_profile,
    binary_state_metrics,
    persistence_forecast,
    regression_metrics,
)
from l4_prediction_pipeline import (
    event_free_feature_contract,
    load_forecast_checkpoint,
    load_ordered_univariate_series,
    load_prediction_archive,
    paired_column_plan,
    save_forecast_checkpoint,
    save_prediction_archive,
    strict_data_bundle,
)
from model import DSTSGCN
from train import run_epoch, set_seed


@torch.no_grad()
def collect_predictions(model, loader, static_adj, device):
    model.eval()
    truth, prediction = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x, static_adj)
        truth.append(y.numpy())
        prediction.append(pred.detach().cpu().numpy())
    if not truth:
        raise RuntimeError("prediction split contains no windows")
    return np.concatenate(truth), np.concatenate(prediction)


def model_for_smoke(num_nodes, horizon, hidden_dim):
    return DSTSGCN(
        num_nodes=num_nodes,
        input_dim=1,
        output_dim=1,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_blocks=1,
        num_heads=4,
        dropout=0.1,
        graph_learner_type="lmln",
        fusion_mode="fusion",
        fusion_type="quality",
        dynamic_top_k=min(3, max(num_nodes - 1, 1)),
        quality_gate_bias=-1.0,
        matrix_hidden_dim=max(32, hidden_dim * 2),
        num_diffusion_steps=2,
        event_dim=0,
        resilience_aux=False,
        resilience_uncertainty_weighting=False,
    )


def variable_plan(dataset, variable, max_nodes):
    config = DATASETS[dataset]
    frame = ordered_frame(config)
    plan = paired_column_plan(
        list(frame.columns),
        config["flow_suffix"],
        config["speed_suffix"],
        41,
    )
    if variable == "flow":
        full_columns = plan["selected_flow_columns"]
        candidate_columns = plan["paired_flow_columns"] or full_columns
        label = "demand"
        profile_root = "l4"
    elif variable == "speed":
        full_columns = plan["paired_speed_columns"]
        candidate_columns = full_columns
        label = "speed"
        profile_root = "l3"
    else:
        raise ValueError(f"unknown variable: {variable}")
    if not candidate_columns:
        raise ValueError(f"{dataset} has no {variable} columns")
    model_columns = candidate_columns[:max_nodes]
    full_indices = [full_columns.index(column) for column in model_columns]
    return config, full_columns, model_columns, full_indices, label, profile_root


def prepare_variable(dataset, variable, args):
    config, full_columns, model_columns, full_indices, label, profile_root = variable_plan(
        dataset, variable, args.max_nodes
    )
    full_values, timestamps, full_names = load_ordered_univariate_series(
        str(config["csv"]), str(config["time_col"]), full_columns, config[f"{variable}_suffix"]
    )
    selected = full_values[:, full_indices, :]
    usable_windows = max(len(full_values) - args.history - args.horizon + 1, 0)
    frozen_train_end = min(int(usable_windows * 0.6) + args.history, len(full_values))
    frozen_val_end = min(int(usable_windows * 0.8) + args.history, len(full_values))
    bundle = strict_data_bundle(
        selected,
        timestamps,
        args.history,
        args.horizon,
        train_time_end_exclusive=frozen_train_end,
        val_time_end_exclusive=frozen_val_end,
    )
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    profile_dir = (
        Path(args.l4_dir) / dataset
        if profile_root == "l4"
        else Path(args.source_profile_dir) / dataset
    )
    profile = load_saved_profile(
        profile_dir,
        label,
        full_values[..., 0],
        train_end,
        "log1p_nonnegative",
    )
    if profile is None:
        raise RuntimeError(f"frozen profile is incompatible for {dataset}/{variable}")
    reference = compute_directional_probabilistic_deficit(
        full_values[..., 0], profile, timestamps, "lower"
    )
    return {
        "config": config,
        "full_values": full_values,
        "timestamps": timestamps,
        "full_names": full_names,
        "model_columns": model_columns,
        "model_names": [name.rsplit(str(config[f"{variable}_suffix"]), 1)[0] for name in model_columns],
        "full_indices": full_indices,
        "bundle": bundle,
        "profile": profile,
        "reference": reference,
    }


def frame_to_markdown(frame):
    """Render a compact Markdown table without the optional tabulate package."""
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def limited_indices(indices, limit):
    array = np.asarray(indices, dtype=int)
    return array if limit <= 0 else array[:limit]


def train_one(dataset, variable, args):
    prepared = prepare_variable(dataset, variable, args)
    bundle = prepared["bundle"]
    train_indices = limited_indices(bundle["train_indices"], args.max_train_windows)
    val_indices = limited_indices(bundle["val_indices"], args.max_eval_windows)
    test_indices = limited_indices(bundle["test_indices"], args.max_eval_windows)
    base_dataset = TrafficWindowDataset(bundle["scaled_values"], args.history, args.horizon)
    train_loader = DataLoader(Subset(base_dataset, train_indices.tolist()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(base_dataset, test_indices.tolist()), batch_size=args.batch_size, shuffle=False)

    split_info = bundle["split_info"]
    train_end = int(split_info["train_time_end_exclusive"])
    adjacency_np = build_static_adjacency(
        bundle["scaled_values"][:train_end], prepared["model_names"], source="corr", corr_threshold=0.2
    )
    device = args.device
    static_adj = torch.tensor(adjacency_np, dtype=torch.float32, device=device)
    set_seed(args.seed)
    model = model_for_smoke(len(prepared["model_names"]), args.horizon, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    raw_mean = float(np.asarray(bundle["scaler"].mean).reshape(-1)[0])
    raw_std = float(np.asarray(bundle["scaler"].std).reshape(-1)[0])
    best_val = float("inf")
    best_state = None
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model, train_loader, static_adj, optimizer, loss_fn,
            1e-3, 1e-4, None, None, False, 5.0, False,
            False, 0.0, 0.0, "ratio", 0.0,
            False, 0.90, 0.0, 0.90, 0.0, False,
            raw_mean, raw_std, device,
        )
        val_true_scaled, val_pred_scaled = collect_predictions(model, val_loader, static_adj, device)
        val_true = bundle["scaler"].inverse_transform(val_true_scaled)
        val_pred = bundle["scaler"].inverse_transform(val_pred_scaled)
        val_metrics = regression_metrics(val_true, val_pred)
        history_rows.append({"epoch": epoch, "train_loss": train_stats["loss"], "val_mae": val_metrics["mae"]})
        if val_metrics["mae"] < best_val:
            best_val = val_metrics["mae"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint was produced")

    run_dir = Path(args.output_dir) / "smoke" / dataset / variable
    checkpoint_path = run_dir / "best_dstsgcn.pt"
    contract = event_free_feature_contract()
    metadata = {
        "dataset": dataset,
        "variable": variable,
        "node_names": prepared["model_names"],
        "history": args.history,
        "horizon": args.horizon,
        "train_time_end_exclusive": train_end,
        "val_time_end_exclusive": int(split_info["val_time_end_exclusive"]),
        "seed": args.seed,
        "scaler": bundle["scaler_metadata"],
        "timestamp_start": str(prepared["timestamps"][0]),
        "timestamp_end": str(prepared["timestamps"][-1]),
        "input_features": contract["input_features"],
        "event_features": contract["event_features"],
        "weather_features": contract["weather_features"],
        "model_config": {"hidden_dim": args.hidden_dim, "num_blocks": 1, "graph_learner_type": "lmln", "fusion_type": "quality"},
        "smoke_limits": {"train_windows": len(train_indices), "validation_windows": len(val_indices), "test_windows": len(test_indices)},
    }
    save_forecast_checkpoint(checkpoint_path, best_state, metadata)
    loaded_checkpoint = load_forecast_checkpoint(checkpoint_path)
    model.load_state_dict(loaded_checkpoint["model_state_dict"])

    y_true_scaled, y_pred_scaled = collect_predictions(model, test_loader, static_adj, device)
    if not np.isfinite(y_pred_scaled).all():
        raise RuntimeError(f"non-finite neural prediction for {dataset}/{variable}")
    y_true = bundle["scaler"].inverse_transform(y_true_scaled)
    y_pred = bundle["scaler"].inverse_transform(y_pred_scaled)
    persistence_scaled = persistence_forecast(bundle["scaled_values"], test_indices, args.history, args.horizon)
    persistence = bundle["scaler"].inverse_transform(persistence_scaled)
    target_timestamps = np.asarray(prepared["timestamps"])[
        test_indices[:, None] + np.arange(args.history, args.history + args.horizon)[None, :]
    ]
    archive_path = run_dir / f"{variable}_predictions.npz"
    save_prediction_archive(
        archive_path,
        dataset=dataset,
        variable=variable,
        split="test_smoke",
        target_timestamps=target_timestamps,
        node_names=prepared["model_names"],
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        y_true_physical=y_true,
        y_pred_physical=y_pred,
        train_time_end_exclusive=train_end,
        val_time_end_exclusive=int(split_info["val_time_end_exclusive"]),
        scaler=bundle["scaler_metadata"],
        seed=args.seed,
    )
    archive = load_prediction_archive(archive_path)
    if archive["target_timestamps"].shape != target_timestamps.shape:
        raise RuntimeError("prediction archive timestamp roundtrip failed")

    flat_timestamps = target_timestamps.reshape(-1)
    flat_truth = y_true[..., 0].reshape(-1, y_true.shape[2])
    flat_pred = y_pred[..., 0].reshape(-1, y_pred.shape[2])
    clip_value = float(np.asarray(prepared["reference"]["deficit_clip_value"]))
    threshold = float(prepared["reference"]["train_deficit_quantiles"]["q90"])
    true_l4 = apply_frozen_lower_tail_profile(
        flat_truth, flat_timestamps, prepared["profile"], prepared["full_values"].shape[1], prepared["full_indices"], clip_value
    )
    pred_l4 = apply_frozen_lower_tail_profile(
        flat_pred, flat_timestamps, prepared["profile"], prepared["full_values"].shape[1], prepared["full_indices"], clip_value
    )
    traffic = regression_metrics(y_true, y_pred)
    persistence_metrics = regression_metrics(y_true, persistence)
    l4_metrics = regression_metrics(true_l4["system_deficit"], pred_l4["system_deficit"])
    state_metrics = binary_state_metrics(true_l4["system_deficit"], pred_l4["system_deficit"], threshold)
    finite_l4 = bool(np.isfinite(true_l4["system_deficit"]).any() and np.isfinite(pred_l4["system_deficit"]).any())
    result = {
        "dataset": dataset,
        "variable": variable,
        "dimension": "demand" if variable == "flow" else "efficiency",
        "nodes": len(prepared["model_names"]),
        "node_names": prepared["model_names"],
        "profile_node_indices": prepared["full_indices"],
        "train_windows_used": len(train_indices),
        "validation_windows_used": len(val_indices),
        "test_windows_used": len(test_indices),
        "epochs": args.epochs,
        "best_val_mae": best_val,
        "traffic_metrics": traffic,
        "persistence_metrics": persistence_metrics,
        "l4_deficit_metrics": l4_metrics,
        "l4_high_state_metrics": state_metrics,
        "l4_high_threshold_train_q90": threshold,
        "checkpoint_roundtrip": True,
        "prediction_archive_roundtrip": True,
        "target_timestamp_aligned": bool(target_timestamps.shape == y_true.shape[:2]),
        "finite_predictions": bool(np.isfinite(y_pred).all()),
        "finite_l4_postprocess": finite_l4,
        "event_free_contract": contract,
        "training_history": history_rows,
        "checkpoint": str(checkpoint_path),
        "prediction_archive": str(archive_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        for variable in ("flow", "speed"):
            print(f"[E-L4-1 smoke] {dataset}/{variable}", flush=True)
            results.append(train_one(dataset, variable, args))
            latest = results[-1]
            print(
                f"  traffic_mae={latest['traffic_metrics']['mae']:.6f} "
                f"persistence_mae={latest['persistence_metrics']['mae']:.6f} "
                f"l4_mae={latest['l4_deficit_metrics']['mae']:.6f}",
                flush=True,
            )

    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"],
            "variable": result["variable"],
            "dimension": result["dimension"],
            "nodes": result["nodes"],
            "traffic_mae": result["traffic_metrics"]["mae"],
            "traffic_rmse": result["traffic_metrics"]["rmse"],
            "persistence_mae": result["persistence_metrics"]["mae"],
            "l4_deficit_mae": result["l4_deficit_metrics"]["mae"],
            "l4_deficit_spearman": result["l4_deficit_metrics"]["spearman"],
            "high_state_precision": result["l4_high_state_metrics"]["precision"],
            "high_state_recall": result["l4_high_state_metrics"]["recall"],
            "predicted_high_rate": result["l4_high_state_metrics"]["predicted_high_rate"],
            "finite_predictions": result["finite_predictions"],
            "finite_l4_postprocess": result["finite_l4_postprocess"],
            "target_timestamp_aligned": result["target_timestamp_aligned"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "e_l4_1_smoke_metrics.csv", index=False, encoding="utf-8-sig")
    passed = bool(
        len(results) == 6
        and summary[["finite_predictions", "finite_l4_postprocess", "target_timestamp_aligned"]].all().all()
        and all(result["checkpoint_roundtrip"] and result["prediction_archive_roundtrip"] for result in results)
    )
    decision = {
        "stage": "E-L4-1-smoke",
        "smoke_passed": passed,
        "datasets": [item.strip() for item in args.datasets.split(",") if item.strip()],
        "max_nodes": args.max_nodes,
        "seed": args.seed,
        "epochs": args.epochs,
        "training_run": True,
        "full_n41_authorized": passed,
        "three_seed_authorized": False,
        "model_py_modified": False,
        "loss_definition_modified": False,
    }
    (output_dir / "e_l4_1_smoke_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E-L4-1 五节点预测 Smoke Test",
        "",
        "## 阶段结论",
        "",
        "**Smoke test 通过。**" if passed else "**Smoke test 未通过，停止后续实验。**",
        "",
        "本轮分别训练 flow 与 speed 单变量模型，只使用历史交通变量。没有使用事件、天气、occupancy、韧性辅助头、CVaR 或不确定性加权，也没有修改网络骨架和损失定义。",
        "",
        "## 配置",
        "",
        f"- 数据集：{args.datasets}",
        f"- 每个变量节点数：{args.max_nodes}",
        f"- seed：{args.seed}",
        f"- epochs：{args.epochs}",
        f"- 训练窗口上限：{args.max_train_windows}",
        f"- 验证/测试窗口上限：{args.max_eval_windows}",
        "",
        "## 结果",
        "",
        frame_to_markdown(summary),
        "",
        "## 结论边界",
        "",
        "本结果只验证小规模训练、checkpoint、反标准化、时间对齐和冻结 L4 后处理能够端到端运行，不用于论文效果结论。即使通过，也不能声称神经网络已经学会完整交通韧性。",
    ]
    (output_dir / "e_l4_1_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return decision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="rainstorm,typhoon,pems04")
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())