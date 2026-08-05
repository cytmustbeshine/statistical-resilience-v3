"""Run train/validation-only A8 statistical mixture decoder experiments."""
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

from data import TrafficWindowDataset, build_static_adjacency
from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import (
    DUAL_SPACE_MISSING_PROTOCOL,
    PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
    target_tensor_from_indices,
)
from run_a4_capacity_validation import TASKS, parse_int_list
from run_a7_curriculum_validation import DEFAULT_CONTROL_CSV, parse_tasks
from run_l4_prediction_baseline import limited_indices, model_for_baseline, prepare_variable
from statistical_mixture_decoder import StatisticalMixtureForecastWrapper
from train import masked_mean, set_seed, unpack_batch


def run_mixture_epoch(
    model: StatisticalMixtureForecastWrapper,
    loader: DataLoader,
    static_adj: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    expert_aux_weight: float,
    temporal_reg_weight: float,
    sparse_reg_weight: float,
    gradient_clip: float,
    device: str,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mixture_loss": 0.0, "expert_loss": 0.0}
    batches = 0
    for batch_data in loader:
        x, y, _, _ = unpack_batch(batch_data, device)
        optimizer.zero_grad()
        prediction, aux = model(x, static_adj, return_aux=True)
        target_mask = torch.isfinite(y)
        mixture_element = F.smooth_l1_loss(prediction, y, reduction="none")
        mixture_loss = masked_mean(mixture_element, target_mask)
        experts = aux["expert_predictions"]
        direct_loss = masked_mean(
            F.smooth_l1_loss(experts[..., 0], y, reduction="none"), target_mask
        )
        autoregressive_loss = masked_mean(
            F.smooth_l1_loss(experts[..., 1], y, reduction="none"), target_mask
        )
        expert_loss = 0.5 * (direct_loss + autoregressive_loss)
        temporal_reg = aux.get("temporal_reg", x.new_tensor(0.0))
        loss = (
            mixture_loss
            + expert_aux_weight * expert_loss
            + temporal_reg_weight * temporal_reg
        )
        if sparse_reg_weight > 0:
            loss = loss + sparse_reg_weight * torch.abs(model.graph_learner.global_residual).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["mixture_loss"] += float(mixture_loss.detach().cpu())
        totals["expert_loss"] += float(expert_loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("training loader is empty")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def collect_mixture_outputs(
    model: StatisticalMixtureForecastWrapper,
    loader: DataLoader,
    static_adj: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_batches = []
    prediction_batches = []
    weight_batches = []
    for batch_data in loader:
        x, y, _, _ = unpack_batch(batch_data, str(device))
        prediction, aux = model(x, static_adj, return_aux=True)
        true_batches.append(y.detach().cpu().numpy())
        prediction_batches.append(prediction.detach().cpu().numpy())
        weight_batches.append(aux["mixture_weights"].detach().cpu().numpy())
    if not true_batches:
        raise RuntimeError("evaluation loader is empty")
    return (
        np.concatenate(true_batches, axis=0),
        np.concatenate(prediction_batches, axis=0),
        np.concatenate(weight_batches, axis=0),
    )


def train_validation_task(dataset: str, variable: str, seed: int, args) -> dict[str, object]:
    prepared = prepare_variable(dataset, variable, args)
    bundle = prepared["bundle"]
    train_indices = limited_indices(bundle["train_indices"], args.max_train_windows)
    val_indices = limited_indices(bundle["val_indices"], args.max_val_windows)
    base_dataset = TrafficWindowDataset(bundle["scaled_values"], args.history, args.horizon)
    train_loader = DataLoader(
        Subset(base_dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    adjacency = build_static_adjacency(
        bundle["scaled_values"][:train_end],
        prepared["model_names"],
        source="corr",
        corr_threshold=0.2,
    )
    device = torch.device(args.device)
    static_adj = torch.tensor(adjacency, dtype=torch.float32, device=device)
    set_seed(seed)
    base_model = model_for_baseline(
        len(prepared["model_names"]), args.horizon, args.hidden_dim, args.num_blocks
    )
    model = StatisticalMixtureForecastWrapper(
        base_model, hidden_dim=args.hidden_dim, horizon=args.horizon
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats = run_mixture_epoch(
            model,
            train_loader,
            static_adj,
            optimizer,
            args.expert_aux_weight,
            args.temporal_reg_weight,
            args.sparse_reg_weight,
            args.gradient_clip,
            args.device,
        )
        val_true_scaled, val_pred_scaled, val_weights = collect_mixture_outputs(
            model, val_loader, static_adj, device
        )
        if prepared["missing_space_protocol"] == DUAL_SPACE_MISSING_PROTOCOL:
            val_true = target_tensor_from_indices(
                prepared["selected_physical_values"],
                val_indices,
                args.history,
                args.horizon,
            )
        else:
            val_true = bundle["scaler"].inverse_transform(val_true_scaled)
        val_pred = bundle["scaler"].inverse_transform(val_pred_scaled)
        val_mae = float(regression_metrics(val_true, val_pred)["mae"])
        mean_weights = val_weights.mean(axis=(0, 1, 2))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_mixture_loss": train_stats["mixture_loss"],
                "train_expert_loss": train_stats["expert_loss"],
                "val_mae": val_mae,
                "direct_weight": float(mean_weights[0]),
                "autoregressive_weight": float(mean_weights[1]),
                "persistence_weight": float(mean_weights[2]),
            }
        )
        if np.isfinite(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError(f"no finite validation checkpoint for {dataset}/{variable}/seed={seed}")
    model.load_state_dict(best_state)
    _, _, best_weights = collect_mixture_outputs(model, val_loader, static_adj, device)
    mean_best_weights = best_weights.mean(axis=(0, 1, 2))
    run_dir = Path(args.output_dir) / f"seed_{seed}" / dataset / variable
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "A8-validation",
        "validation_only": True,
        "test_loader_constructed": False,
        "dataset": dataset,
        "variable": variable,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "expert_aux_weight": args.expert_aux_weight,
        "gradient_clip": args.gradient_clip,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "direct_weight": float(mean_best_weights[0]),
        "autoregressive_weight": float(mean_best_weights[1]),
        "persistence_weight": float(mean_best_weights[2]),
    }
    torch.save({"model_state_dict": best_state, "metadata": metadata}, run_dir / "best_validation_only.pt")
    result = {
        **metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train_indices),
        "validation_windows": len(val_indices),
        "finite": bool(np.isfinite(best_val_mae)),
        "history": history,
    }
    (run_dir / "validation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def validation_decision(rows: pd.DataFrame, control_csv: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    controls = pd.read_csv(control_csv)
    comparisons = rows.merge(
        controls[["seed", "dataset", "variable", "m1_val_mae", "a6_val_mae"]],
        on=["seed", "dataset", "variable"],
        how="left",
        validate="one_to_one",
    )
    comparisons["a8_vs_a6_ratio"] = comparisons["best_val_mae"] / comparisons["a6_val_mae"]
    comparisons["a8_vs_m1_ratio"] = comparisons["best_val_mae"] / comparisons["m1_val_mae"]
    finite = bool(comparisons["finite"].astype(bool).all())
    seeds = set(comparisons["seed"].astype(int))
    tasks = set(zip(comparisons["dataset"], comparisons["variable"]))
    stage = "incomplete"
    accepted = False
    decision: dict[str, object] = {
        "stage": "A8-validation",
        "validation_only": True,
        "test_loader_constructed": False,
        "runs": int(len(comparisons)),
        "finite_runs": int(comparisons["finite"].astype(bool).sum()),
        "accepted": False,
        "external_confirmation_authorized": False,
    }
    if len(comparisons) == 5 and seeds == {42} and tasks == set(TASKS):
        stage = "A8-0"
        decision.update(
            {
                "a6_improved_count": int((comparisons["a8_vs_a6_ratio"] < 1.0).sum()),
                "mean_a8_vs_a6_ratio": float(comparisons["a8_vs_a6_ratio"].mean()),
                "worst_a8_vs_a6_ratio": float(comparisons["a8_vs_a6_ratio"].max()),
            }
        )
        accepted = bool(
            finite
            and decision["a6_improved_count"] >= 4
            and decision["mean_a8_vs_a6_ratio"] <= 0.98
            and decision["worst_a8_vs_a6_ratio"] <= 1.02
        )
    elif len(comparisons) == 15 and seeds == {42, 2024, 3407} and tasks == set(TASKS):
        stage = "A8-1"
        task_means = comparisons.groupby(["dataset", "variable"])["a8_vs_a6_ratio"].mean()
        maximum_expert_weight = comparisons[
            ["direct_weight", "autoregressive_weight", "persistence_weight"]
        ].max(axis=1)
        decision.update(
            {
                "a6_improved_count": int((comparisons["a8_vs_a6_ratio"] < 1.0).sum()),
                "mean_a8_vs_a6_ratio": float(comparisons["a8_vs_a6_ratio"].mean()),
                "m1_improved_count": int((comparisons["a8_vs_m1_ratio"] < 1.0).sum()),
                "worst_task_mean_a8_vs_a6_ratio": float(task_means.max()),
                "all_runs_max_expert_weight": float(maximum_expert_weight.max()),
            }
        )
        accepted = bool(
            finite
            and decision["a6_improved_count"] >= 10
            and decision["mean_a8_vs_a6_ratio"] <= 0.98
            and decision["m1_improved_count"] == 15
            and decision["worst_task_mean_a8_vs_a6_ratio"] <= 1.03
            and not bool((maximum_expert_weight > 0.95).all())
        )
    decision["protocol_stage"] = stage
    decision["accepted"] = accepted
    decision["external_confirmation_authorized"] = accepted and stage == "A8-1"
    return comparisons, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a8_statistical_mixture_validation")
    parser.add_argument("--control-csv", default=str(DEFAULT_CONTROL_CSV))
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--expert-aux-weight", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--temporal-reg-weight", type=float, default=1e-3)
    parser.add_argument("--sparse-reg-weight", type=float, default=1e-4)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split-protocol", default=PROFILE_COMPATIBLE_SPLIT_PROTOCOL)
    parser.add_argument("--missing-space-protocol", default=DUAL_SPACE_MISSING_PROTOCOL)
    args = parser.parse_args()
    if args.hidden_dim != 64 or args.lr != 0.001:
        raise ValueError("A8 protocol locks hidden_dim=64 and lr=0.001")
    if args.expert_aux_weight != 0.10 or args.gradient_clip != 5.0:
        raise ValueError("A8 protocol locks expert_aux_weight=0.10 and gradient_clip=5.0")
    tasks = parse_tasks(args.tasks)
    seeds = parse_int_list(args.seeds)
    rows = []
    for seed in seeds:
        for dataset, variable in tasks:
            print(f"[A8] seed={seed} {dataset}/{variable}", flush=True)
            rows.append(train_validation_task(dataset, variable, seed, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output_dir / "a8_validation_tasks.csv", index=False, encoding="utf-8-sig")
    comparisons, decision = validation_decision(frame, Path(args.control_csv))
    comparisons.to_csv(output_dir / "a8_validation_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "a8_validation_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
