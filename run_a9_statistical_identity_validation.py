"""Run train/validation-only A9 statistical identity residual experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import target_tensor_from_indices
from run_a4_capacity_validation import parse_int_list
from run_a8_pems_external_confirmation import PEMS_TASKS, parse_tasks, prepare_task
from statistical_identity_model import (
    StatisticalIdentityResidualForecaster,
    StatisticalIdentityWindowDataset,
    fit_shrunk_seasonal_baseline,
)
from train import set_seed


DEFAULT_CONTROL_MANIFEST = Path(
    r"D:\TrafficGNN\outputs\a8_pems_external_confirmation\training_manifest.csv"
)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return torch.abs(prediction[valid] - target[valid]).mean()


@torch.no_grad()
def collect_predictions(model, loader, adjacency: torch.Tensor, device: torch.device):
    model.eval()
    truth = []
    prediction = []
    for x, y, baseline, slot, day_of_week in loader:
        pred = model(
            x.to(device),
            adjacency,
            baseline.to(device),
            slot.to(device),
            day_of_week.to(device),
        )
        truth.append(y.numpy())
        prediction.append(pred.detach().cpu().numpy())
    if not truth:
        raise RuntimeError("validation loader is empty")
    return np.concatenate(truth), np.concatenate(prediction)


def train_task(dataset: str, variable: str, seed: int, args) -> dict[str, object]:
    prepared = prepare_task(dataset, variable, args)
    bundle = prepared["bundle"]
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    seasonal_physical, seasonal_metadata = fit_shrunk_seasonal_baseline(
        prepared["physical"], prepared["timestamps"], train_end, shrinkage=7.0
    )
    seasonal_scaled = bundle["scaler"].transform(seasonal_physical)
    dataset_windows = StatisticalIdentityWindowDataset(
        bundle["scaled_values"],
        seasonal_scaled,
        prepared["timestamps"],
        args.history,
        args.horizon,
    )
    train_loader = DataLoader(
        Subset(dataset_windows, prepared["train_indices"].tolist()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset_windows, prepared["val_indices"].tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    set_seed(seed)
    model = StatisticalIdentityResidualForecaster(
        num_nodes=len(prepared["node_names"]),
        history=args.history,
        horizon=args.horizon,
        embedding_dim=32,
        hidden_width=256,
        num_blocks=3,
        dropout=0.10,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y, baseline, slot, day_of_week in train_loader:
            x = x.to(device)
            y = y.to(device)
            baseline = baseline.to(device)
            slot = slot.to(device)
            day_of_week = day_of_week.to(device)
            optimizer.zero_grad()
            prediction = model(x, adjacency, baseline, slot, day_of_week)
            loss = masked_mae(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        val_true_scaled, val_prediction_scaled = collect_predictions(
            model, val_loader, adjacency, device
        )
        val_true = target_tensor_from_indices(
            prepared["physical"],
            prepared["val_indices"],
            args.history,
            args.horizon,
        )
        val_prediction = bundle["scaler"].inverse_transform(val_prediction_scaled)
        val_metrics = regression_metrics(val_true, val_prediction)
        val_mae = float(val_metrics["mae"])
        history.append(
            {
                "epoch": epoch,
                "train_mae_scaled": total_loss / max(batches, 1),
                "val_mae": val_mae,
                "val_rmse": float(val_metrics["rmse"]),
            }
        )
        if np.isfinite(val_mae) and val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError(f"A9 produced no finite checkpoint for {dataset}/{variable}/seed={seed}")
    run_dir = Path(args.output_dir) / f"seed_{seed}" / dataset / variable
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_a9.pt"
    metadata = {
        "stage": "A9-validation",
        "validation_only": True,
        "test_loader_constructed": False,
        "dataset": dataset,
        "variable": variable,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "seasonal_baseline_sha256": seasonal_metadata["baseline_sha256"],
        "seasonal_shrinkage": 7.0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    torch.save({"model_state_dict": best_state, "metadata": metadata}, checkpoint)
    result = {**metadata, "checkpoint": str(checkpoint), "history": history}
    (run_dir / "validation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def validation_decision(rows: pd.DataFrame, control_manifest: Path):
    controls = pd.read_csv(control_manifest)
    controls = controls.loc[controls["model"] == "official_adapted_dcrnn", [
        "dataset", "variable", "seed", "best_val_mae"
    ]].rename(columns={"best_val_mae": "dcrnn_val_mae"})
    comparison = rows.merge(
        controls,
        on=["dataset", "variable", "seed"],
        how="left",
        validate="one_to_one",
    )
    comparison["a9_vs_dcrnn_ratio"] = comparison["best_val_mae"] / comparison["dcrnn_val_mae"]
    seeds = set(comparison["seed"].astype(int))
    tasks = set(zip(comparison["dataset"], comparison["variable"]))
    finite = bool(np.isfinite(comparison["best_val_mae"]).all())
    decision = {
        "stage": "A9-validation",
        "runs": int(len(comparison)),
        "finite_runs": int(np.isfinite(comparison["best_val_mae"]).sum()),
        "validation_only": True,
        "test_loader_constructed": False,
        "accepted": False,
        "confirmation_authorized": False,
    }
    if len(comparison) == 4 and seeds == {42} and tasks == set(PEMS_TASKS):
        decision["protocol_stage"] = "A9-0"
        decision["improved_count"] = int((comparison["a9_vs_dcrnn_ratio"] < 1.0).sum())
        decision["mean_ratio"] = float(comparison["a9_vs_dcrnn_ratio"].mean())
        decision["worst_ratio"] = float(comparison["a9_vs_dcrnn_ratio"].max())
        decision["accepted"] = bool(
            finite
            and decision["improved_count"] >= 3
            and decision["mean_ratio"] <= 0.98
            and decision["worst_ratio"] <= 1.03
        )
    elif len(comparison) == 12 and seeds == {42, 2024, 3407} and tasks == set(PEMS_TASKS):
        task_means = comparison.groupby(["dataset", "variable"])["a9_vs_dcrnn_ratio"].mean()
        baseline_hash_counts = comparison.groupby(["dataset", "variable"])["seasonal_baseline_sha256"].nunique()
        decision["protocol_stage"] = "A9-1"
        decision["improved_count"] = int((comparison["a9_vs_dcrnn_ratio"] < 1.0).sum())
        decision["mean_ratio"] = float(comparison["a9_vs_dcrnn_ratio"].mean())
        decision["worst_task_mean_ratio"] = float(task_means.max())
        decision["seasonal_baseline_seed_invariant"] = bool((baseline_hash_counts == 1).all())
        decision["accepted"] = bool(
            finite
            and decision["improved_count"] >= 9
            and decision["mean_ratio"] <= 0.98
            and decision["worst_task_mean_ratio"] <= 1.03
            and decision["seasonal_baseline_seed_invariant"]
        )
        decision["confirmation_authorized"] = decision["accepted"]
    else:
        decision["protocol_stage"] = "incomplete"
    return comparison, decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a9_statistical_identity_validation")
    parser.add_argument("--control-manifest", default=str(DEFAULT_CONTROL_MANIFEST))
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_nodes != 41 or args.history != 12 or args.horizon != 12:
        raise ValueError("A9 locks 41 nodes and history/horizon 12")
    tasks = parse_tasks(args.tasks)
    seeds = parse_int_list(args.seeds)
    rows = []
    for seed in seeds:
        for dataset, variable in tasks:
            print(f"[A9] seed={seed} {dataset}/{variable}", flush=True)
            rows.append(train_task(dataset, variable, seed, args))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output_dir / "a9_validation_tasks.csv", index=False, encoding="utf-8-sig")
    comparison, decision = validation_decision(frame, Path(args.control_manifest))
    comparison.to_csv(output_dir / "a9_validation_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "a9_validation_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
