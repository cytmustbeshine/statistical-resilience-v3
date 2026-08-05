"""Locked A9 versus DCRNN external confirmation on PEMS03."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN
from data import TrafficWindowDataset, build_static_adjacency
from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import prepare_dual_space_bundle, target_tensor_from_indices
from run_a4_capacity_validation import parse_int_list
from run_a8_pems_external_confirmation import collect_predictions, train_dcrnn
from statistical_identity_model import (
    StatisticalIdentityResidualForecaster,
    StatisticalIdentityWindowDataset,
    fit_shrunk_seasonal_baseline,
)
from train import set_seed


PEMS03_PATH = Path(r"D:\TrafficGNN\data\public\PEMS03\PEMS03.npz")
PEMS03_MD5 = "651add9bb9eaf7f5eda2f2ee8778a182"
PEMS03_SHAPE = (26208, 358, 1)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pems03_timestamps(length: int) -> np.ndarray:
    timestamps = pd.date_range("2018-09-01 00:00:00", periods=length, freq="5min")
    if length == PEMS03_SHAPE[0] and timestamps[-1] != pd.Timestamp("2018-11-30 23:55:00"):
        raise RuntimeError("PEMS03 calendar endpoint mismatch")
    return timestamps.to_numpy()


def prepare_pems03(args) -> dict[str, object]:
    if not PEMS03_PATH.exists():
        raise FileNotFoundError(PEMS03_PATH)
    if PEMS03_PATH.stat().st_size != 15800415:
        raise RuntimeError("PEMS03 file size mismatch")
    if file_md5(PEMS03_PATH) != PEMS03_MD5:
        raise RuntimeError("PEMS03 MD5 mismatch")
    archive = np.load(PEMS03_PATH)
    physical_all = np.asarray(archive["data"], dtype=float)
    if physical_all.shape != PEMS03_SHAPE:
        raise RuntimeError(f"PEMS03 shape mismatch: {physical_all.shape}")
    physical = physical_all[:, : args.max_nodes, :]
    timestamps = pems03_timestamps(len(physical))
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
    train_indices = bundle["train_indices"] if args.max_train_windows <= 0 else bundle["train_indices"][: args.max_train_windows]
    val_indices = bundle["val_indices"] if args.max_eval_windows <= 0 else bundle["val_indices"][: args.max_eval_windows]
    node_names = [f"pems03_{index:03d}" for index in range(args.max_nodes)]
    adjacency = build_static_adjacency(
        bundle["scaled_values"][:train_end],
        node_names,
        source="corr",
        corr_threshold=0.2,
    )
    return {
        "dataset": "pems03",
        "variable": "flow",
        "physical": physical,
        "timestamps": timestamps,
        "node_names": node_names,
        "dual": dual,
        "bundle": bundle,
        "train_indices": np.asarray(train_indices, dtype=int),
        "val_indices": np.asarray(val_indices, dtype=int),
        "adjacency": adjacency,
    }


def masked_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return torch.abs(prediction[valid] - target[valid]).mean()


@torch.no_grad()
def collect_a9(model, loader, adjacency: torch.Tensor, device: torch.device):
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
        raise RuntimeError("A9 loader is empty")
    return np.concatenate(truth), np.concatenate(prediction)


def a9_loaders(prepared, args, include_test: bool):
    bundle = prepared["bundle"]
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    seasonal_physical, seasonal_metadata = fit_shrunk_seasonal_baseline(
        prepared["physical"], prepared["timestamps"], train_end, shrinkage=7.0
    )
    seasonal_scaled = bundle["scaler"].transform(seasonal_physical)
    dataset = StatisticalIdentityWindowDataset(
        bundle["scaled_values"], seasonal_scaled, prepared["timestamps"], args.history, args.horizon
    )
    train_loader = DataLoader(
        Subset(dataset, prepared["train_indices"].tolist()), batch_size=args.a9_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, prepared["val_indices"].tolist()), batch_size=args.a9_batch_size, shuffle=False
    )
    test_loader = None
    test_indices = None
    if include_test:
        test_indices = prepared["bundle"]["test_indices"]
        if args.max_eval_windows > 0:
            test_indices = test_indices[: args.max_eval_windows]
        test_loader = DataLoader(
            Subset(dataset, np.asarray(test_indices, dtype=int).tolist()),
            batch_size=args.a9_batch_size,
            shuffle=False,
        )
    return train_loader, val_loader, test_loader, np.asarray(test_indices) if test_indices is not None else None, seasonal_metadata


def train_a9(prepared, seed: int, args, run_dir: Path) -> dict[str, object]:
    train_loader, val_loader, _, _, seasonal_metadata = a9_loaders(prepared, args, include_test=False)
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    set_seed(seed)
    model = StatisticalIdentityResidualForecaster(
        num_nodes=len(prepared["node_names"]), history=args.history, horizon=args.horizon
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.a9_epochs + 1):
        model.train()
        total = 0.0
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
            total += float(loss.detach().cpu())
            batches += 1
        _, val_prediction_scaled = collect_a9(model, val_loader, adjacency, device)
        val_truth = target_tensor_from_indices(
            prepared["physical"], prepared["val_indices"], args.history, args.horizon
        )
        val_prediction = prepared["bundle"]["scaler"].inverse_transform(val_prediction_scaled)
        metrics = regression_metrics(val_truth, val_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_mae_scaled": total / max(batches, 1),
                "val_mae": float(metrics["mae"]),
                "val_rmse": float(metrics["rmse"]),
            }
        )
        if np.isfinite(metrics["mae"]) and metrics["mae"] < best_mae:
            best_mae = float(metrics["mae"])
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
    if best_state is None:
        raise RuntimeError("A9 produced no finite PEMS03 checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_a9.pt"
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch}, checkpoint)
    result = {
        "model": "a9_statistical_identity",
        "dataset": "pems03",
        "variable": "flow",
        "seed": seed,
        "validation_only": True,
        "test_loader_constructed": False,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
        "seasonal_baseline_sha256": seasonal_metadata["baseline_sha256"],
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (run_dir / "training_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def train_stage(seeds: list[int], args) -> int:
    prepared = prepare_pems03(args)
    root = Path(args.output_dir)
    for seed in seeds:
        pair_dir = root / f"seed_{seed}" / "pems03" / "flow"
        a9_result = pair_dir / "a9" / "training_result.json"
        if a9_result.exists() and (pair_dir / "a9" / "best_a9.pt").exists():
            print(f"[PEMS03 resume] seed={seed} A9 complete", flush=True)
        else:
            print(f"[PEMS03 train] seed={seed} A9", flush=True)
            train_a9(prepared, seed, args, pair_dir / "a9")
        dcrnn_result = pair_dir / "dcrnn" / "training_result.json"
        if dcrnn_result.exists() and (pair_dir / "dcrnn" / "best_dcrnn.pt").exists():
            print(f"[PEMS03 resume] seed={seed} DCRNN complete", flush=True)
        else:
            print(f"[PEMS03 train] seed={seed} DCRNN", flush=True)
            train_dcrnn(prepared, seed, args, pair_dir / "dcrnn")
    rows = []
    for path in sorted(root.glob("seed_*/pems03/flow/*/training_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.pop("history", None)
        rows.append(row)
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "training_manifest.csv", index=False, encoding="utf-8-sig")
    authorized = bool(
        len(frame) == 6
        and frame["checkpoint"].map(lambda path: Path(path).exists()).all()
        and frame["validation_only"].astype(bool).all()
        and (~frame["test_loader_constructed"].astype(bool)).all()
    )
    decision = {
        "stage": "A9-PEMS03-training",
        "trained_runs": int(len(frame)),
        "expected_runs": 6,
        "test_loader_constructed": False,
        "evaluation_authorized": authorized,
    }
    (root / "training_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


def load_a9(checkpoint: Path, num_nodes: int, args, device: torch.device):
    model = StatisticalIdentityResidualForecaster(num_nodes, args.history, args.horizon)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device)


def load_dcrnn(checkpoint: Path, num_nodes: int, args, device: torch.device):
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


def evaluate_stage(seeds: list[int], args) -> int:
    root = Path(args.output_dir)
    manifest = pd.read_csv(root / "training_manifest.csv")
    if len(manifest) != 6 or not manifest["checkpoint"].map(lambda path: Path(path).exists()).all():
        raise RuntimeError("all six PEMS03 checkpoints are required")
    prepared = prepare_pems03(args)
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    _, _, a9_test_loader, test_indices, _ = a9_loaders(prepared, args, include_test=True)
    traffic_dataset = TrafficWindowDataset(
        prepared["bundle"]["scaled_values"], args.history, args.horizon
    )
    dcrnn_test_loader = DataLoader(
        Subset(traffic_dataset, test_indices.tolist()),
        batch_size=args.dcrnn_batch_size,
        shuffle=False,
    )
    truth = target_tensor_from_indices(
        prepared["physical"], test_indices, args.history, args.horizon
    )
    rows = []
    for seed in seeds:
        pair_dir = root / f"seed_{seed}" / "pems03" / "flow"
        a9_model = load_a9(pair_dir / "a9" / "best_a9.pt", len(prepared["node_names"]), args, device)
        _, a9_scaled = collect_a9(a9_model, a9_test_loader, adjacency, device)
        dcrnn_model = load_dcrnn(pair_dir / "dcrnn" / "best_dcrnn.pt", len(prepared["node_names"]), args, device)
        _, dcrnn_scaled = collect_predictions(dcrnn_model, dcrnn_test_loader, adjacency, device)
        for model_name, folder, scaled in (
            ("a9_statistical_identity", "a9", a9_scaled),
            ("official_adapted_dcrnn", "dcrnn", dcrnn_scaled),
        ):
            prediction = prepared["bundle"]["scaler"].inverse_transform(scaled)
            metrics = regression_metrics(truth, prediction)
            rows.append({"model": model_name, "seed": seed, **{key: float(value) for key, value in metrics.items()}})
            np.savez_compressed(
                pair_dir / folder / "test_predictions.npz",
                y_true=truth,
                y_pred=prediction,
                test_indices=test_indices,
                node_names=np.asarray(prepared["node_names"]),
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "external_test_metrics.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot(index="seed", columns="model", values=["mae", "rmse", "smape", "wape"])
    comparisons = []
    for seed, row in pivot.iterrows():
        item = {"seed": int(seed)}
        for metric in ("mae", "rmse", "smape", "wape"):
            baseline = float(row[(metric, "official_adapted_dcrnn")])
            candidate = float(row[(metric, "a9_statistical_identity")])
            item[f"dcrnn_{metric}"] = baseline
            item[f"a9_{metric}"] = candidate
            item[f"{metric}_ratio"] = candidate / baseline
        comparisons.append(item)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(root / "a9_vs_dcrnn_external.csv", index=False, encoding="utf-8-sig")
    wins = {metric: int((comparison[f"{metric}_ratio"] < 1.0).sum()) for metric in ("mae", "rmse", "smape", "wape")}
    means = {metric: float(comparison[f"{metric}_ratio"].mean()) for metric in wins}
    decision = {
        "stage": "A9-PEMS03-external-prebootstrap",
        "paired_comparisons": 3,
        "wins": wins,
        "mean_ratios": means,
        "ordinary_gate_prebootstrap": bool(
            all(value == 3 for value in wins.values()) and all(value < 1.0 for value in means.values())
        ),
        "bootstrap_pending": True,
        "final_model_selected": False,
    }
    (root / "external_decision_prebootstrap.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "evaluate"], required=True)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a9_pems03_external_confirmation")
    parser.add_argument("--seeds", default="42,2024,3407")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--a9-epochs", type=int, default=50)
    parser.add_argument("--dcrnn-epochs", type=int, default=20)
    parser.add_argument("--a9-batch-size", type=int, default=64)
    parser.add_argument("--dcrnn-batch-size", type=int, default=16)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_nodes != 41 or args.history != 12 or args.horizon != 12:
        raise ValueError("PEMS03 protocol locks 41 nodes and history/horizon 12")
    seeds = parse_int_list(args.seeds)
    if args.stage == "train":
        return train_stage(seeds, args)
    return evaluate_stage(seeds, args)


if __name__ == "__main__":
    raise SystemExit(main())
