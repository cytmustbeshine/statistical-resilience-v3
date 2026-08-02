"""Run final frozen-L4 auxiliary and tail-risk DSTSGCN experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import build_static_adjacency
from flow_speed_resilience import trimmed_system
from l4_prediction_evaluation import (
    apply_frozen_lower_tail_profile,
    binary_state_metrics,
    persistence_forecast,
    regression_metrics,
)
from l4_prediction_pipeline import (
    DUAL_SPACE_MISSING_PROTOCOL,
    PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
    event_free_feature_contract,
    load_forecast_checkpoint,
    load_prediction_archive,
    save_forecast_checkpoint,
    save_prediction_archive,
    target_tensor_from_indices,
)
from l4_resilience_supervision import build_frozen_l4_supervision
from model import DSTSGCN
from run_l4_prediction_baseline import limited_indices, prepare_variable
from train import ResilienceAuxWindowDataset, set_seed


def empirical_cvar(losses: torch.Tensor, mask: torch.Tensor, alpha: float) -> torch.Tensor:
    values = losses[mask]
    if values.numel() == 0:
        return losses.sum() * 0.0
    tail_count = max(1, int(np.ceil((1.0 - float(alpha)) * values.numel())))
    return torch.topk(values.reshape(-1), tail_count).values.mean()


def build_model(num_nodes: int, horizon: int, hidden_dim: int, num_blocks: int) -> DSTSGCN:
    model = DSTSGCN(
        num_nodes=num_nodes,
        input_dim=1,
        output_dim=1,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
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
        resilience_aux=True,
        resilience_target_mode="ratio",
        resilience_uncertainty_weighting=False,
    )
    model.resilience_target_mode = "l4_deficit"
    return model


def unpack(batch, device):
    x, y, deficit, _ = batch
    return x.to(device), y.to(device), deficit.to(device)


def train_epoch(model, loader, adj, optimizer, variant, args, device):
    model.train()
    totals = {"loss": 0.0, "traffic_mean": 0.0, "l4_mean": 0.0, "traffic_cvar": 0.0, "l4_cvar": 0.0}
    seen = 0
    for batch in loader:
        x, y, deficit_target = unpack(batch, device)
        optimizer.zero_grad()
        pred, aux = model(x, adj, return_aux=True)
        deficit_pred = F.softplus(aux["resilience_pred"])
        traffic_element = F.smooth_l1_loss(pred, y, reduction="none")
        traffic_mask = torch.isfinite(y)
        traffic_mean = traffic_element[traffic_mask].mean()
        l4_element = F.smooth_l1_loss(
            deficit_pred,
            torch.nan_to_num(deficit_target, nan=0.0),
            reduction="none",
        )
        l4_mask = torch.isfinite(deficit_target)
        l4_mean = l4_element[l4_mask].mean() if bool(l4_mask.any()) else pred.sum() * 0.0
        traffic_tail = pred.sum() * 0.0
        l4_tail = pred.sum() * 0.0
        if variant == "m3":
            traffic_tail = empirical_cvar(traffic_element, traffic_mask, args.alpha)
            l4_tail = empirical_cvar(l4_element, l4_mask, args.alpha)
        loss = traffic_mean + args.lambda_l4 * l4_mean
        if variant == "m3":
            loss = loss + args.lambda_traffic_tail * traffic_tail + args.lambda_l4_tail * l4_tail
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        batch_size = int(x.shape[0])
        seen += batch_size
        for key, value in (
            ("loss", loss), ("traffic_mean", traffic_mean), ("l4_mean", l4_mean),
            ("traffic_cvar", traffic_tail), ("l4_cvar", l4_tail),
        ):
            totals[key] += float(value.detach().item()) * batch_size
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def collect(model, loader, adj, device):
    model.eval()
    traffic_scaled = []
    deficit_pred = []
    for batch in loader:
        x, _, _ = unpack(batch, device)
        pred, aux = model(x, adj, return_aux=True)
        traffic_scaled.append(pred.detach().cpu().numpy())
        deficit_pred.append(F.softplus(aux["resilience_pred"]).detach().cpu().numpy())
    if not traffic_scaled:
        raise RuntimeError("evaluation split contains no windows")
    return np.concatenate(traffic_scaled), np.concatenate(deficit_pred)


def system_deficit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = array.reshape(-1, array.shape[2])
    return trimmed_system(flat, 0.1).reshape(array.shape[0], array.shape[1])


def evaluate_split(model, loader, adj, device, scaler, physical_truth, deficit_truth):
    pred_scaled, deficit_pred = collect(model, loader, adj, device)
    pred_physical = scaler.inverse_transform(pred_scaled)
    traffic = regression_metrics(physical_truth, pred_physical)
    true_system = system_deficit(deficit_truth[..., 0])
    pred_system = system_deficit(deficit_pred[..., 0])
    l4 = regression_metrics(true_system, pred_system)
    return traffic, l4, pred_scaled, pred_physical, deficit_pred, true_system, pred_system


def prepare_supervision(prepared, variable, args):
    bundle = prepared["bundle"]
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    full = build_frozen_l4_supervision(
        prepared["full_values"],
        prepared["timestamps"],
        prepared["full_names"],
        variable,
        prepared["profile"]["profile"],
        prepared["profile"]["ecdf"],
        train_end,
        args.history,
        args.horizon,
    )
    selected = full["deficit_targets"][:, :, prepared["full_indices"], :]
    baselines = np.ones_like(selected, dtype=np.float32)
    return full, selected, baselines


def train_one(dataset: str, variable: str, variant: str, args) -> dict[str, object]:
    prepared = prepare_variable(dataset, variable, args)
    bundle = prepared["bundle"]
    full_supervision, deficit_targets, baselines = prepare_supervision(prepared, variable, args)
    base_dataset = ResilienceAuxWindowDataset(
        bundle["scaled_values"], deficit_targets, baselines, args.history, args.horizon
    )
    train_indices = limited_indices(bundle["train_indices"], args.max_train_windows)
    val_indices = limited_indices(bundle["val_indices"], args.max_eval_windows)
    test_indices = limited_indices(bundle["test_indices"], args.max_eval_windows)
    train_loader = DataLoader(Subset(base_dataset, train_indices.tolist()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(base_dataset, test_indices.tolist()), batch_size=args.batch_size, shuffle=False)

    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    adjacency = build_static_adjacency(
        bundle["scaled_values"][:train_end], prepared["model_names"], source="corr", corr_threshold=0.2
    )
    device = args.device
    adj = torch.tensor(adjacency, dtype=torch.float32, device=device)
    set_seed(args.seed)
    model = build_model(len(prepared["model_names"]), args.horizon, args.hidden_dim, args.num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    training_history = []
    val_truth = target_tensor_from_indices(
        prepared["selected_physical_values"], val_indices, args.history, args.horizon
    )
    val_deficit = deficit_targets[val_indices]

    for epoch in range(1, args.epochs + 1):
        stats = train_epoch(model, train_loader, adj, optimizer, variant, args, device)
        val_traffic, val_l4, *_ = evaluate_split(
            model, val_loader, adj, device, bundle["scaler"], val_truth, val_deficit
        )
        training_history.append({"epoch": epoch, **stats, "val_mae": val_traffic["mae"], "val_l4_mae": val_l4["mae"]})
        if np.isfinite(val_traffic["mae"]) and val_traffic["mae"] < best_val:
            best_val = float(val_traffic["mae"])
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(
            f"[{dataset}/{variable}/{variant}] epoch={epoch} loss={stats['loss']:.4f} "
            f"val_mae={val_traffic['mae']:.4f} val_l4_mae={val_l4['mae']:.4f}",
            flush=True,
        )
    if best_state is None:
        raise RuntimeError("no finite validation checkpoint")

    run_dir = Path(args.output_dir) / args.run_tag / variant / dataset / variable / f"seed_{args.seed}"
    checkpoint_path = run_dir / "best_dstsgcn.pt"
    contract = event_free_feature_contract()
    metadata = {
        "dataset": dataset,
        "variable": variable,
        "node_names": prepared["model_names"],
        "history": args.history,
        "horizon": args.horizon,
        "train_time_end_exclusive": train_end,
        "val_time_end_exclusive": int(bundle["split_info"]["val_time_end_exclusive"]),
        "seed": args.seed,
        "scaler": bundle["scaler_metadata"],
        "timestamp_start": str(prepared["timestamps"][0]),
        "timestamp_end": str(prepared["timestamps"][-1]),
        "input_features": contract["input_features"],
        "event_features": [],
        "weather_features": [],
        "split_protocol": PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
        "missing_space_protocol": DUAL_SPACE_MISSING_PROTOCOL,
        "imputation": prepared["imputation_metadata"],
        "raw_missing_count": int(prepared["imputation_metadata"]["raw_missing_count"]),
        "model_missing_count": int(prepared["imputation_metadata"]["model_missing_count"]),
        "checkpoint_selection_metric": "validation_physical_mae",
        "model_variant": variant,
        "resilience_target_mode": "l4_deficit",
        "loss": {
            "lambda_l4": args.lambda_l4,
            "alpha": args.alpha,
            "lambda_traffic_tail": args.lambda_traffic_tail if variant == "m3" else 0.0,
            "lambda_l4_tail": args.lambda_l4_tail if variant == "m3" else 0.0,
        },
    }
    save_forecast_checkpoint(checkpoint_path, best_state, metadata)
    loaded = load_forecast_checkpoint(checkpoint_path)
    model.load_state_dict(loaded["model_state_dict"])

    test_truth = target_tensor_from_indices(
        prepared["selected_physical_values"], test_indices, args.history, args.horizon
    )
    test_deficit = deficit_targets[test_indices]
    traffic, l4, pred_scaled, pred_physical, deficit_pred, true_system, pred_system = evaluate_split(
        model, test_loader, adj, device, bundle["scaler"], test_truth, test_deficit
    )
    persistence_scaled = persistence_forecast(bundle["scaled_values"], test_indices, args.history, args.horizon)
    persistence = bundle["scaler"].inverse_transform(persistence_scaled)
    persistence_metrics = regression_metrics(test_truth, persistence)
    target_timestamps = np.asarray(prepared["timestamps"])[
        test_indices[:, None] + np.arange(args.history, args.history + args.horizon)[None, :]
    ]
    archive_path = run_dir / "traffic_predictions.npz"
    save_prediction_archive(
        archive_path,
        dataset=dataset,
        variable=variable,
        split="test",
        target_timestamps=target_timestamps,
        node_names=prepared["model_names"],
        y_true_scaled=target_tensor_from_indices(bundle["scaled_values"], test_indices, args.history, args.horizon),
        y_pred_scaled=pred_scaled,
        y_true_physical=test_truth,
        y_pred_physical=pred_physical,
        train_time_end_exclusive=train_end,
        val_time_end_exclusive=int(bundle["split_info"]["val_time_end_exclusive"]),
        scaler=bundle["scaler_metadata"],
        seed=args.seed,
        missing_space_protocol=DUAL_SPACE_MISSING_PROTOCOL,
        split_protocol=PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
        imputation=prepared["imputation_metadata"],
        physical_truth_valid_mask=np.isfinite(test_truth),
        l4_valid_mask=np.isfinite(test_deficit),
    )
    archive = load_prediction_archive(archive_path)
    l4_path = run_dir / "l4_predictions.npz"
    np.savez_compressed(
        l4_path,
        dataset=np.asarray(dataset),
        variable=np.asarray(variable),
        model_variant=np.asarray(variant),
        target_timestamps=target_timestamps,
        node_names=np.asarray(prepared["model_names"]),
        true_node_deficit=test_deficit,
        pred_node_deficit=deficit_pred,
        true_system_deficit=true_system,
        pred_system_deficit=pred_system,
        valid_mask=np.isfinite(test_deficit),
        q90=np.asarray(full_supervision["q90"]),
        q99=np.asarray(full_supervision["q99"]),
    )
    high = binary_state_metrics(true_system, pred_system, full_supervision["q90"])
    result = {
        "dataset": dataset,
        "variable": variable,
        "dimension": "demand" if variable == "flow" else "efficiency",
        "model_variant": variant,
        "seed": args.seed,
        "nodes": len(prepared["model_names"]),
        "node_names": prepared["model_names"],
        "epochs": args.epochs,
        "train_windows": len(train_indices),
        "validation_windows": len(val_indices),
        "test_windows": len(test_indices),
        "best_val_mae": best_val,
        "traffic_metrics": traffic,
        "persistence_metrics": persistence_metrics,
        "l4_aux_metrics": l4,
        "l4_high_state_metrics": high,
        "q90": full_supervision["q90"],
        "q99": full_supervision["q99"],
        "finite_predictions": bool(np.isfinite(pred_physical).all() and np.isfinite(deficit_pred).all()),
        "l4_output_nonnegative": bool((deficit_pred >= 0).all()),
        "checkpoint_roundtrip": True,
        "archive_roundtrip": bool(archive["physical_truth_valid_mask"].shape == test_truth.shape),
        "training_history": training_history,
        "checkpoint": str(checkpoint_path),
        "traffic_archive": str(archive_path),
        "l4_archive": str(l4_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run(args) -> dict[str, object]:
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    variants = [value.strip().lower() for value in args.variants.split(",") if value.strip()]
    if any(value not in {"m2", "m3"} for value in variants):
        raise ValueError("variants must contain only m2 and m3")
    results = []
    for dataset in datasets:
        variables = ("flow",) if dataset == "bridge" else ("flow", "speed")
        for variant in variants:
            for variable in variables:
                results.append(train_one(dataset, variable, variant, args))
    rows = [{
        "dataset": row["dataset"], "variable": row["variable"], "model_variant": row["model_variant"],
        "seed": row["seed"], "traffic_mae": row["traffic_metrics"]["mae"],
        "persistence_mae": row["persistence_metrics"]["mae"], "l4_mae": row["l4_aux_metrics"]["mae"],
        "l4_spearman": row["l4_aux_metrics"]["spearman"], "high_f1": row["l4_high_state_metrics"]["f1"],
        "finite": row["finite_predictions"], "nonnegative": row["l4_output_nonnegative"],
    } for row in results]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{args.run_tag}_m2_m3_metrics.csv", index=False, encoding="utf-8-sig")
    expected = sum(1 if dataset == "bridge" else 2 for dataset in datasets) * len(variants)
    passed = bool(len(results) == expected and frame[["finite", "nonnegative"]].all().all())
    decision = {
        "stage": args.run_tag,
        "stage_passed": passed,
        "training_run": True,
        "datasets": datasets,
        "variants": variants,
        "seed": args.seed,
        "epochs": args.epochs,
        "expected_runs": expected,
        "completed_runs": len(results),
        "formal_training_authorized": False,
        "three_seed_run": False,
        "final_model_selected": False,
        "blockers": [] if passed else ["M2/M3 smoke engineering checks failed"],
    }
    (output / f"{args.run_tag}_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return decision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="rainstorm,typhoon")
    parser.add_argument("--variants", default="m2,m3")
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2b_s_smoke")
    parser.add_argument("--run-tag", default="e_l4_2b_s")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--max-eval-windows", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--lambda-l4", type=float, default=0.30)
    parser.add_argument("--lambda-traffic-tail", type=float, default=0.20)
    parser.add_argument("--lambda-l4-tail", type=float, default=0.20)
    parser.add_argument("--split-protocol", default=PROFILE_COMPATIBLE_SPLIT_PROTOCOL)
    parser.add_argument("--missing-space-protocol", default=DUAL_SPACE_MISSING_PROTOCOL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
