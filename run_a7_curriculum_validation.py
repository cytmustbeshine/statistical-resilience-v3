"""Run train/validation-only A7 curriculum-decoder experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
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
from run_l4_prediction_baseline import (
    collect_predictions,
    limited_indices,
    model_for_baseline,
    prepare_variable,
)
from temporal_decoder import AutoregressiveForecastWrapper
from train import masked_mean, set_seed, unpack_batch


DEFAULT_CONTROL_CSV = Path(
    r"D:\TrafficGNN\outputs\a6_ar_decoder_multiseed_validation\a6_vs_m1_validation_corrected.csv"
)


def parse_tasks(text: str) -> list[tuple[str, str]]:
    if text.strip().lower() == "all":
        return list(TASKS)
    tasks = []
    for item in text.split(","):
        dataset, separator, variable = item.strip().partition(":")
        task = (dataset.lower(), variable.lower())
        if not separator or task not in TASKS:
            raise ValueError(f"Unsupported task: {item}")
        tasks.append(task)
    if not tasks:
        raise ValueError("At least one task is required")
    return list(dict.fromkeys(tasks))


def linear_teacher_forcing_ratio(
    global_step: int,
    total_steps: int,
    decay_fraction: float = 0.8,
) -> float:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be in (0, 1]")
    decay_steps = max(int(math.ceil(total_steps * decay_fraction)), 1)
    if global_step >= decay_steps:
        return 0.0
    return float(1.0 - global_step / decay_steps)


def run_curriculum_epoch(
    model: AutoregressiveForecastWrapper,
    loader: DataLoader,
    static_adj: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    total_steps: int,
    decay_fraction: float,
    temporal_reg_weight: float,
    sparse_reg_weight: float,
    gradient_clip: float,
    device: str,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "traffic_loss": 0.0, "teacher_forcing_ratio": 0.0}
    batches = 0
    for batch_data in loader:
        x, y, _, _ = unpack_batch(batch_data, device)
        ratio = linear_teacher_forcing_ratio(global_step, total_steps, decay_fraction)
        optimizer.zero_grad()
        prediction, aux = model(
            x,
            static_adj,
            labels=y,
            teacher_forcing_ratio=ratio,
            return_aux=True,
        )
        traffic_element = F.smooth_l1_loss(prediction, y, reduction="none")
        traffic_loss = masked_mean(traffic_element, torch.isfinite(y))
        temporal_reg = aux.get("temporal_reg", x.new_tensor(0.0))
        loss = traffic_loss + temporal_reg_weight * temporal_reg
        if sparse_reg_weight > 0:
            loss = loss + sparse_reg_weight * torch.abs(model.graph_learner.global_residual).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["traffic_loss"] += float(traffic_loss.detach().cpu())
        totals["teacher_forcing_ratio"] += ratio
        batches += 1
        global_step += 1
    if batches == 0:
        raise RuntimeError("training loader is empty")
    return {key: value / batches for key, value in totals.items()}, global_step


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
    model = AutoregressiveForecastWrapper(
        base_model, hidden_dim=args.hidden_dim, horizon=args.horizon
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_stats, global_step = run_curriculum_epoch(
            model,
            train_loader,
            static_adj,
            optimizer,
            global_step,
            total_steps,
            args.curriculum_decay_fraction,
            args.temporal_reg_weight,
            args.sparse_reg_weight,
            args.gradient_clip,
            args.device,
        )
        val_true_scaled, val_pred_scaled = collect_predictions(model, val_loader, static_adj, device)
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
        val_metrics = regression_metrics(val_true, val_pred)
        val_mae = float(val_metrics["mae"])
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_stats["loss"],
                "train_traffic_loss": train_stats["traffic_loss"],
                "mean_teacher_forcing_ratio": train_stats["teacher_forcing_ratio"],
                "val_mae": val_mae,
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
    run_dir = Path(args.output_dir) / f"seed_{seed}" / dataset / variable
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "stage": "A7-validation",
        "validation_only": True,
        "test_loader_constructed": False,
        "dataset": dataset,
        "variable": variable,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "curriculum": "linear_1_to_0",
        "curriculum_decay_fraction": args.curriculum_decay_fraction,
        "gradient_clip": args.gradient_clip,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
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
    comparisons = rows.copy()
    required_controls = {"seed", "dataset", "variable", "m1_val_mae", "a6_val_mae"}
    if control_csv.exists():
        controls = pd.read_csv(control_csv)
        missing = required_controls.difference(controls.columns)
        if missing:
            raise ValueError(f"control CSV missing columns: {sorted(missing)}")
        comparisons = comparisons.merge(
            controls[list(required_controls)],
            on=["seed", "dataset", "variable"],
            how="left",
            validate="one_to_one",
        )
        comparisons["a7_vs_a6_ratio"] = comparisons["best_val_mae"] / comparisons["a6_val_mae"]
        comparisons["a7_vs_m1_ratio"] = comparisons["best_val_mae"] / comparisons["m1_val_mae"]
    full_protocol = (
        len(comparisons) == 15
        and set(comparisons["seed"].astype(int)) == {42, 2024, 3407}
        and set(zip(comparisons["dataset"], comparisons["variable"])) == set(TASKS)
    )
    controls_complete = required_controls.difference({"seed", "dataset", "variable"}).issubset(comparisons.columns)
    finite = bool(comparisons["finite"].astype(bool).all())
    decision = {
        "stage": "A7-validation",
        "validation_only": True,
        "test_loader_constructed": False,
        "full_protocol": full_protocol,
        "controls_complete": controls_complete,
        "finite_runs": int(comparisons["finite"].astype(bool).sum()),
        "runs": int(len(comparisons)),
        "accepted": False,
        "outer_evaluation_authorized": False,
    }
    if full_protocol and controls_complete:
        task_mean = comparisons.groupby(["dataset", "variable"])["a7_vs_a6_ratio"].mean()
        decision.update(
            {
                "a6_improved_count": int((comparisons["a7_vs_a6_ratio"] < 1.0).sum()),
                "mean_a7_vs_a6_ratio": float(comparisons["a7_vs_a6_ratio"].mean()),
                "m1_improved_count": int((comparisons["a7_vs_m1_ratio"] < 1.0).sum()),
                "mean_a7_vs_m1_ratio": float(comparisons["a7_vs_m1_ratio"].mean()),
                "worst_task_mean_a7_vs_a6_ratio": float(task_mean.max()),
            }
        )
        accepted = bool(
            finite
            and decision["a6_improved_count"] >= 9
            and decision["mean_a7_vs_a6_ratio"] <= 0.99
            and decision["m1_improved_count"] >= 12
            and decision["mean_a7_vs_m1_ratio"] <= 0.97
            and decision["worst_task_mean_a7_vs_a6_ratio"] <= 1.05
        )
        decision["accepted"] = accepted
        decision["external_confirmation_authorized"] = accepted
    return comparisons, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a7_curriculum_validation")
    parser.add_argument("--control-csv", default=str(DEFAULT_CONTROL_CSV))
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="42,2024,3407")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--curriculum-decay-fraction", type=float, default=0.8)
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
        raise ValueError("A7 protocol locks hidden_dim=64 and lr=0.001")
    if args.curriculum_decay_fraction != 0.8 or args.gradient_clip != 5.0:
        raise ValueError("A7 protocol locks decay_fraction=0.8 and gradient_clip=5.0")
    tasks = parse_tasks(args.tasks)
    seeds = parse_int_list(args.seeds)
    rows = []
    for seed in seeds:
        for dataset, variable in tasks:
            print(f"[A7] seed={seed} {dataset}/{variable}", flush=True)
            rows.append(train_validation_task(dataset, variable, seed, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output_dir / "a7_validation_tasks.csv", index=False, encoding="utf-8-sig")
    comparisons, decision = validation_decision(frame, Path(args.control_csv))
    comparisons.to_csv(output_dir / "a7_validation_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "a7_validation_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
