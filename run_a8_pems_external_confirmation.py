"""Train and evaluate locked A8 versus public DCRNN on external PEMS tasks."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN
from baselines.dcrnn_resilience_official_adapted.train import raw_masked_rmse
from data import TrafficWindowDataset, build_static_adjacency
from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import (
    load_ordered_univariate_series_physical,
    prepare_dual_space_bundle,
    target_tensor_from_indices,
)
from run_a4_capacity_validation import parse_int_list
from run_a8_statistical_mixture_validation import (
    collect_mixture_outputs,
    run_mixture_epoch,
)
from run_l4_prediction_baseline import limited_indices, model_for_baseline
from statistical_mixture_decoder import StatisticalMixtureForecastWrapper
from train import set_seed


PEMS_CONFIG = {
    "pems04": Path(r"D:\TrafficGNN\data\public\PEMS04\pems04.csv"),
    "pems08": Path(r"D:\TrafficGNN\data\public\PEMS08\pems08.csv"),
}
PEMS_TASKS = (
    ("pems04", "flow"),
    ("pems04", "speed"),
    ("pems08", "flow"),
    ("pems08", "speed"),
)


def parse_tasks(text: str) -> list[tuple[str, str]]:
    if text.strip().lower() == "all":
        return list(PEMS_TASKS)
    tasks = []
    for item in text.split(","):
        dataset, separator, variable = item.strip().partition(":")
        task = (dataset.lower(), variable.lower())
        if not separator or task not in PEMS_TASKS:
            raise ValueError(f"Unsupported PEMS task: {item}")
        tasks.append(task)
    if not tasks:
        raise ValueError("At least one PEMS task is required")
    return list(dict.fromkeys(tasks))


def resolve_value_columns(csv_path: Path, variable: str, max_nodes: int) -> list[str]:
    suffix = "_volume" if variable == "flow" else "_speed"
    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    matched = [str(column) for column in columns if str(column).endswith(suffix)]
    if len(matched) < max_nodes:
        raise ValueError(f"{csv_path} has only {len(matched)} columns for {suffix}")
    return matched[:max_nodes]


def prepare_task(dataset: str, variable: str, args) -> dict[str, object]:
    csv_path = PEMS_CONFIG[dataset]
    value_columns = resolve_value_columns(csv_path, variable, args.max_nodes)
    suffix = "_volume" if variable == "flow" else "_speed"
    physical, timestamps, node_names = load_ordered_univariate_series_physical(
        str(csv_path), "Time", value_columns, suffix
    )
    train_end = int(len(physical) * 0.6)
    val_end = int(len(physical) * 0.8)
    dual = prepare_dual_space_bundle(
        physical,
        timestamps,
        args.history,
        args.horizon,
        train_time_end_exclusive=train_end,
        val_time_end_exclusive=val_end,
    )
    bundle = dual["bundle"]
    train_indices = limited_indices(bundle["train_indices"], args.max_train_windows)
    val_indices = limited_indices(bundle["val_indices"], args.max_eval_windows)
    adjacency = build_static_adjacency(
        bundle["scaled_values"][:train_end],
        node_names,
        source="corr",
        corr_threshold=0.2,
    )
    return {
        "dataset": dataset,
        "variable": variable,
        "csv_path": str(csv_path),
        "physical": physical,
        "timestamps": timestamps,
        "node_names": node_names,
        "dual": dual,
        "bundle": bundle,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "adjacency": adjacency,
    }


def make_loaders(prepared: dict[str, object], args, batch_size: int, include_test: bool):
    bundle = prepared["bundle"]
    dataset = TrafficWindowDataset(bundle["scaled_values"], args.history, args.horizon)
    train_loader = DataLoader(
        Subset(dataset, prepared["train_indices"].tolist()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(dataset, prepared["val_indices"].tolist()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = None
    test_indices = None
    if include_test:
        test_indices = limited_indices(bundle["test_indices"], args.max_eval_windows)
        test_loader = DataLoader(
            Subset(dataset, test_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
        )
    return train_loader, val_loader, test_loader, test_indices


@torch.no_grad()
def collect_predictions(model, loader, adjacency: torch.Tensor, device: torch.device):
    model.eval()
    truth = []
    prediction = []
    for x, y in loader:
        pred = model(x.to(device), adjacency)
        truth.append(y.numpy())
        prediction.append(pred.detach().cpu().numpy())
    if not truth:
        raise RuntimeError("prediction loader is empty")
    return np.concatenate(truth), np.concatenate(prediction)


def physical_validation_metrics(
    prepared: dict[str, object],
    indices: np.ndarray,
    prediction_scaled: np.ndarray,
    args,
) -> dict[str, float]:
    truth = target_tensor_from_indices(
        prepared["physical"], indices, args.history, args.horizon
    )
    prediction = prepared["bundle"]["scaler"].inverse_transform(prediction_scaled)
    return regression_metrics(truth, prediction)


def train_a8(prepared: dict[str, object], seed: int, args, run_dir: Path) -> dict[str, object]:
    train_loader, val_loader, _, _ = make_loaders(
        prepared, args, args.a8_batch_size, include_test=False
    )
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    set_seed(seed)
    base_model = model_for_baseline(
        len(prepared["node_names"]), args.horizon, 64, 2
    )
    model = StatisticalMixtureForecastWrapper(base_model, 64, args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.a8_epochs + 1):
        train_stats = run_mixture_epoch(
            model,
            train_loader,
            adjacency,
            optimizer,
            0.10,
            1e-3,
            1e-4,
            5.0,
            args.device,
        )
        _, prediction_scaled, weights = collect_mixture_outputs(
            model, val_loader, adjacency, device
        )
        metrics = physical_validation_metrics(
            prepared, prepared["val_indices"], prediction_scaled, args
        )
        mean_weights = weights.mean(axis=(0, 1, 2))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "val_mae": float(metrics["mae"]),
                "val_rmse": float(metrics["rmse"]),
                "direct_weight": float(mean_weights[0]),
                "autoregressive_weight": float(mean_weights[1]),
                "persistence_weight": float(mean_weights[2]),
            }
        )
        if np.isfinite(metrics["mae"]) and metrics["mae"] < best_mae:
            best_mae = float(metrics["mae"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError("A8 produced no finite validation checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_a8.pt"
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch}, checkpoint)
    result = {
        "model": "a8_statistical_mixture",
        "dataset": prepared["dataset"],
        "variable": prepared["variable"],
        "seed": seed,
        "validation_only": True,
        "test_loader_constructed": False,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (run_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def train_dcrnn(prepared: dict[str, object], seed: int, args, run_dir: Path) -> dict[str, object]:
    train_loader, val_loader, _, _ = make_loaders(
        prepared, args, args.dcrnn_batch_size, include_test=False
    )
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    set_seed(seed)
    model = OfficialAdaptedDCRNN(
        num_nodes=len(prepared["node_names"]),
        input_dim=1,
        output_dim=1,
        rnn_units=256,
        num_rnn_layers=2,
        horizon=args.horizon,
        max_diffusion_step=1,
        cl_decay_steps=2000,
        use_curriculum_learning=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 30, 40, 50], gamma=0.1
    )
    scaler = prepared["bundle"]["scaler"]
    mean = float(np.asarray(scaler.mean).reshape(-1)[0])
    std = float(np.asarray(scaler.std).reshape(-1)[0])
    batches_seen = 0
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.dcrnn_epochs + 1):
        model.train()
        train_total = 0.0
        batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x, adjacency, labels=y, batches_seen=batches_seen)
            loss = raw_masked_rmse(prediction, y, mean, std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach().cpu())
            batches += 1
            batches_seen += 1
        _, prediction_scaled = collect_predictions(model, val_loader, adjacency, device)
        metrics = physical_validation_metrics(
            prepared, prepared["val_indices"], prediction_scaled, args
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_rmse": train_total / max(batches, 1),
                "val_mae": float(metrics["mae"]),
                "val_rmse": float(metrics["rmse"]),
                "sampling_threshold": model.sampling_threshold(batches_seen),
            }
        )
        if np.isfinite(metrics["mae"]) and metrics["mae"] < best_mae:
            best_mae = float(metrics["mae"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError("DCRNN produced no finite validation checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_dcrnn.pt"
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch}, checkpoint)
    result = {
        "model": "official_adapted_dcrnn",
        "dataset": prepared["dataset"],
        "variable": prepared["variable"],
        "seed": seed,
        "validation_only": True,
        "test_loader_constructed": False,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (run_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def train_stage(tasks: list[tuple[str, str]], seeds: list[int], args) -> int:
    for dataset, variable in tasks:
        prepared = prepare_task(dataset, variable, args)
        for seed in seeds:
            pair_dir = Path(args.output_dir) / f"seed_{seed}" / dataset / variable
            a8_result = pair_dir / "a8" / "training_result.json"
            if a8_result.exists() and (pair_dir / "a8" / "best_a8.pt").exists():
                print(f"[PEMS resume] seed={seed} {dataset}/{variable} A8 complete", flush=True)
            else:
                print(f"[PEMS train] seed={seed} {dataset}/{variable} A8", flush=True)
                train_a8(prepared, seed, args, pair_dir / "a8")
            dcrnn_result = pair_dir / "dcrnn" / "training_result.json"
            if dcrnn_result.exists() and (pair_dir / "dcrnn" / "best_dcrnn.pt").exists():
                print(f"[PEMS resume] seed={seed} {dataset}/{variable} DCRNN complete", flush=True)
            else:
                print(f"[PEMS train] seed={seed} {dataset}/{variable} DCRNN", flush=True)
                train_dcrnn(prepared, seed, args, pair_dir / "dcrnn")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in sorted(output_dir.glob("seed_*/*/*/*/training_result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.pop("history", None)
        rows.append(result)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output_dir / "training_manifest.csv", index=False, encoding="utf-8-sig")
    decision = {
        "stage": "A8-PEMS-training",
        "test_loader_constructed": False,
        "trained_runs": len(rows),
        "expected_full_runs": 24,
        "evaluation_authorized": bool(
            len(rows) == 24
            and frame["checkpoint"].map(lambda path: Path(path).exists()).all()
        ),
    }
    (output_dir / "training_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


def load_model(model_name: str, checkpoint: Path, num_nodes: int, args, device: torch.device):
    if model_name == "a8_statistical_mixture":
        base = model_for_baseline(num_nodes, args.horizon, 64, 2)
        model = StatisticalMixtureForecastWrapper(base, 64, args.horizon)
    else:
        model = OfficialAdaptedDCRNN(
            num_nodes=num_nodes,
            input_dim=1,
            output_dim=1,
            rnn_units=256,
            num_rnn_layers=2,
            horizon=args.horizon,
            max_diffusion_step=1,
            cl_decay_steps=2000,
            use_curriculum_learning=True,
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def evaluate_stage(tasks: list[tuple[str, str]], seeds: list[int], args) -> int:
    expected = len(tasks) * len(seeds) * 2
    manifest_path = Path(args.output_dir) / "training_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("training_manifest.csv is required before evaluation")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != expected or expected != 24:
        raise RuntimeError(f"external evaluation requires all 24 checkpoints, found {len(manifest)}")
    if not manifest["checkpoint"].map(lambda path: Path(path).exists()).all():
        raise RuntimeError("one or more locked checkpoints are missing")
    rows = []
    for dataset, variable in tasks:
        prepared = prepare_task(dataset, variable, args)
        device = torch.device(args.device)
        adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
        _, _, test_loader, test_indices = make_loaders(
            prepared, args, min(args.a8_batch_size, args.dcrnn_batch_size), include_test=True
        )
        physical_truth = target_tensor_from_indices(
            prepared["physical"], test_indices, args.history, args.horizon
        )
        for seed in seeds:
            for model_name, folder in (
                ("a8_statistical_mixture", "a8"),
                ("official_adapted_dcrnn", "dcrnn"),
            ):
                checkpoint = Path(args.output_dir) / f"seed_{seed}" / dataset / variable / folder / (
                    "best_a8.pt" if folder == "a8" else "best_dcrnn.pt"
                )
                model = load_model(model_name, checkpoint, len(prepared["node_names"]), args, device)
                _, prediction_scaled = collect_predictions(model, test_loader, adjacency, device)
                prediction_physical = prepared["bundle"]["scaler"].inverse_transform(prediction_scaled)
                metrics = regression_metrics(physical_truth, prediction_physical)
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "variable": variable,
                        "seed": seed,
                        **{key: float(value) for key, value in metrics.items()},
                    }
                )
                archive = Path(args.output_dir) / f"seed_{seed}" / dataset / variable / folder / "test_predictions.npz"
                np.savez_compressed(
                    archive,
                    y_true=physical_truth,
                    y_pred=prediction_physical,
                    test_indices=test_indices,
                    node_names=np.asarray(prepared["node_names"]),
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(args.output_dir) / "external_test_metrics.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot(index=["dataset", "variable", "seed"], columns="model", values=["mae", "rmse", "smape", "wape"])
    comparison_rows = []
    for index, row in pivot.iterrows():
        dataset, variable, seed = index
        item = {"dataset": dataset, "variable": variable, "seed": seed}
        for metric in ("mae", "rmse", "smape", "wape"):
            baseline = float(row[(metric, "official_adapted_dcrnn")])
            candidate = float(row[(metric, "a8_statistical_mixture")])
            item[f"dcrnn_{metric}"] = baseline
            item[f"a8_{metric}"] = candidate
            item[f"{metric}_ratio"] = candidate / baseline
        comparison_rows.append(item)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(Path(args.output_dir) / "a8_vs_dcrnn_external.csv", index=False, encoding="utf-8-sig")
    wins = {metric: int((comparison[f"{metric}_ratio"] < 1.0).sum()) for metric in ("mae", "rmse", "smape", "wape")}
    mean_ratios = {metric: float(comparison[f"{metric}_ratio"].mean()) for metric in wins}
    task_mean_max = max(
        float(comparison.groupby(["dataset", "variable"])[f"{metric}_ratio"].mean().max())
        for metric in wins
    )
    decision = {
        "stage": "A8-PEMS-external",
        "paired_comparisons": len(comparison),
        "wins": wins,
        "mean_ratios": mean_ratios,
        "maximum_task_mean_metric_ratio": task_mean_max,
        "ordinary_gate_prebootstrap": bool(
            all(value >= 10 for value in wins.values())
            and all(value < 1.0 for value in mean_ratios.values())
            and task_mean_max <= 1.03
        ),
        "bootstrap_pending": True,
        "final_model_selected": False,
    }
    (Path(args.output_dir) / "external_decision_prebootstrap.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate"], required=True)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a8_pems_external_confirmation")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="42,2024,3407")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--a8-epochs", type=int, default=20)
    parser.add_argument("--dcrnn-epochs", type=int, default=20)
    parser.add_argument("--a8-batch-size", type=int, default=64)
    parser.add_argument("--dcrnn-batch-size", type=int, default=16)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_nodes != 41 or args.history != 12 or args.horizon != 12:
        raise ValueError("external protocol locks 41 nodes and history/horizon 12")
    tasks = parse_tasks(args.tasks)
    seeds = parse_int_list(args.seeds)
    if args.stage == "train":
        return train_stage(tasks, seeds, args)
    return evaluate_stage(tasks, seeds, args)


if __name__ == "__main__":
    raise SystemExit(main())
