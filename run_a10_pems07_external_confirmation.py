"""Final A10 hierarchical calibration confirmation on PEMS07."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from data import TrafficWindowDataset, build_static_adjacency
from hierarchical_statistical_calibration import (
    HierarchicalCalibration,
    fit_hierarchical_calibration,
)
from l4_prediction_evaluation import regression_metrics
from l4_prediction_pipeline import prepare_dual_space_bundle, target_tensor_from_indices
from run_a4_capacity_validation import parse_int_list
from run_a8_pems_external_confirmation import collect_predictions, train_dcrnn
from run_a9_pems03_external_confirmation import load_dcrnn


PEMS07_PATH = Path(r"D:\TrafficGNN\data\public\PEMS07\PEMS07.npz")
PEMS07_MD5 = "978d3d9b85fe640a446983a34271a48d"
PEMS07_SHAPE = (28224, 883, 1)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pems07(args) -> dict[str, object]:
    if not PEMS07_PATH.exists():
        raise FileNotFoundError(PEMS07_PATH)
    if PEMS07_PATH.stat().st_size != 43705518:
        raise RuntimeError("PEMS07 file size mismatch")
    if file_md5(PEMS07_PATH) != PEMS07_MD5:
        raise RuntimeError("PEMS07 MD5 mismatch")
    physical_all = np.asarray(np.load(PEMS07_PATH)["data"], dtype=float)
    if physical_all.shape != PEMS07_SHAPE:
        raise RuntimeError(f"PEMS07 shape mismatch: {physical_all.shape}")
    physical = physical_all[:, : args.max_nodes, :]
    chronological_index = np.arange(len(physical), dtype=np.int64)
    train_end = int(len(physical) * 0.6)
    val_end = int(len(physical) * 0.8)
    dual = prepare_dual_space_bundle(
        physical,
        chronological_index,
        args.history,
        args.horizon,
        train_time_end_exclusive=train_end,
        val_time_end_exclusive=val_end,
    )
    bundle = dual["bundle"]
    train_indices = bundle["train_indices"] if args.max_train_windows <= 0 else bundle["train_indices"][: args.max_train_windows]
    val_indices = bundle["val_indices"] if args.max_eval_windows <= 0 else bundle["val_indices"][: args.max_eval_windows]
    node_names = [f"pems07_{index:03d}" for index in range(args.max_nodes)]
    adjacency = build_static_adjacency(
        bundle["scaled_values"][:train_end],
        node_names,
        source="corr",
        corr_threshold=0.2,
    )
    return {
        "dataset": "pems07",
        "variable": "flow",
        "physical": physical,
        "timestamps": chronological_index,
        "node_names": node_names,
        "dual": dual,
        "bundle": bundle,
        "train_indices": np.asarray(train_indices, dtype=int),
        "val_indices": np.asarray(val_indices, dtype=int),
        "adjacency": adjacency,
    }


def validation_loader(prepared, args):
    dataset = TrafficWindowDataset(prepared["bundle"]["scaled_values"], args.history, args.horizon)
    return DataLoader(
        Subset(dataset, prepared["val_indices"].tolist()),
        batch_size=args.dcrnn_batch_size,
        shuffle=False,
    )


def persistence_from_indices(prepared, indices: np.ndarray, args) -> np.ndarray:
    model_values = prepared["dual"]["model_values"]
    return np.stack(
        [
            model_values[index + args.history - 1 : index + args.history].repeat(
                args.horizon, axis=0
            )
            for index in indices
        ],
        axis=0,
    )


def fit_seed_calibration(prepared, seed: int, args, pair_dir: Path) -> dict[str, object]:
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    model = load_dcrnn(
        pair_dir / "dcrnn" / "best_dcrnn.pt",
        len(prepared["node_names"]),
        args,
        device,
    )
    loader = validation_loader(prepared, args)
    _, prediction_scaled = collect_predictions(model, loader, adjacency, device)
    prediction = prepared["bundle"]["scaler"].inverse_transform(prediction_scaled)
    truth = target_tensor_from_indices(
        prepared["physical"], prepared["val_indices"], args.history, args.horizon
    )
    persistence = persistence_from_indices(prepared, prepared["val_indices"], args)
    calibration = fit_hierarchical_calibration(
        truth,
        prediction,
        persistence,
        correction_shrinkage=0.25,
    )
    calibration_dir = pair_dir / "a10"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / "calibration.npz"
    calibration.save(calibration_path)
    result = {
        "model": "a10_hierarchical_calibrated_dcrnn",
        "dataset": "pems07",
        "variable": "flow",
        "seed": seed,
        "validation_only": True,
        "test_loader_constructed": False,
        "blend_weight": calibration.blend_weight,
        "correction_shrinkage": calibration.correction_shrinkage,
        "calibration_metric_ratios": calibration.calibration_metric_ratios,
        "calibration_path": str(calibration_path),
    }
    (calibration_dir / "calibration_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def train_stage(seeds: list[int], args) -> int:
    prepared = prepare_pems07(args)
    root = Path(args.output_dir)
    for seed in seeds:
        pair_dir = root / f"seed_{seed}" / "pems07" / "flow"
        dcrnn_result = pair_dir / "dcrnn" / "training_result.json"
        if dcrnn_result.exists() and (pair_dir / "dcrnn" / "best_dcrnn.pt").exists():
            print(f"[A10 resume] seed={seed} DCRNN complete", flush=True)
        else:
            print(f"[A10 train] seed={seed} DCRNN", flush=True)
            train_dcrnn(prepared, seed, args, pair_dir / "dcrnn")
        calibration_result = pair_dir / "a10" / "calibration_result.json"
        if calibration_result.exists() and (pair_dir / "a10" / "calibration.npz").exists():
            print(f"[A10 resume] seed={seed} calibration complete", flush=True)
        else:
            print(f"[A10 calibrate] seed={seed}", flush=True)
            fit_seed_calibration(prepared, seed, args, pair_dir)
    deep_rows = []
    calibration_rows = []
    for path in sorted(root.glob("seed_*/pems07/flow/dcrnn/training_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row.pop("history", None)
        deep_rows.append(row)
    for path in sorted(root.glob("seed_*/pems07/flow/a10/calibration_result.json")):
        calibration_rows.append(json.loads(path.read_text(encoding="utf-8")))
    deep = pd.DataFrame(deep_rows)
    calibration = pd.DataFrame(calibration_rows)
    root.mkdir(parents=True, exist_ok=True)
    deep.to_csv(root / "dcrnn_training_manifest.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(root / "a10_calibration_manifest.csv", index=False, encoding="utf-8-sig")
    authorized = bool(
        len(deep) == 3
        and len(calibration) == 3
        and deep["checkpoint"].map(lambda path: Path(path).exists()).all()
        and calibration["calibration_path"].map(lambda path: Path(path).exists()).all()
        and deep["validation_only"].astype(bool).all()
        and calibration["validation_only"].astype(bool).all()
        and (~deep["test_loader_constructed"].astype(bool)).all()
        and (~calibration["test_loader_constructed"].astype(bool)).all()
    )
    decision = {
        "stage": "A10-PEMS07-training-calibration",
        "dcrnn_checkpoints": int(len(deep)),
        "calibration_archives": int(len(calibration)),
        "test_loader_constructed": False,
        "evaluation_authorized": authorized,
    }
    (root / "training_calibration_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0


def evaluate_stage(seeds: list[int], args) -> int:
    root = Path(args.output_dir)
    deep = pd.read_csv(root / "dcrnn_training_manifest.csv")
    calibration_manifest = pd.read_csv(root / "a10_calibration_manifest.csv")
    if len(deep) != 3 or len(calibration_manifest) != 3:
        raise RuntimeError("three DCRNN checkpoints and calibrations are required")
    prepared = prepare_pems07(args)
    test_indices = prepared["bundle"]["test_indices"]
    if args.max_eval_windows > 0:
        test_indices = test_indices[: args.max_eval_windows]
    dataset = TrafficWindowDataset(prepared["bundle"]["scaled_values"], args.history, args.horizon)
    loader = DataLoader(
        Subset(dataset, np.asarray(test_indices, dtype=int).tolist()),
        batch_size=args.dcrnn_batch_size,
        shuffle=False,
    )
    device = torch.device(args.device)
    adjacency = torch.tensor(prepared["adjacency"], dtype=torch.float32, device=device)
    truth = target_tensor_from_indices(
        prepared["physical"], test_indices, args.history, args.horizon
    )
    persistence = persistence_from_indices(prepared, np.asarray(test_indices, dtype=int), args)
    rows = []
    for seed in seeds:
        pair_dir = root / f"seed_{seed}" / "pems07" / "flow"
        model = load_dcrnn(
            pair_dir / "dcrnn" / "best_dcrnn.pt",
            len(prepared["node_names"]),
            args,
            device,
        )
        _, prediction_scaled = collect_predictions(model, loader, adjacency, device)
        dcrnn_prediction = prepared["bundle"]["scaler"].inverse_transform(prediction_scaled)
        calibration = HierarchicalCalibration.load(pair_dir / "a10" / "calibration.npz")
        a10_prediction = calibration.apply(dcrnn_prediction, persistence)
        for model_name, folder, prediction in (
            ("official_adapted_dcrnn", "dcrnn", dcrnn_prediction),
            ("a10_hierarchical_calibrated_dcrnn", "a10", a10_prediction),
        ):
            metrics = regression_metrics(truth, prediction)
            rows.append(
                {"model": model_name, "seed": seed, **{key: float(value) for key, value in metrics.items()}}
            )
            np.savez_compressed(
                pair_dir / folder / "test_predictions.npz",
                y_true=truth,
                y_pred=prediction,
                test_indices=np.asarray(test_indices, dtype=int),
                node_names=np.asarray(prepared["node_names"]),
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "external_test_metrics.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot(index="seed", columns="model", values=["mae", "rmse", "smape", "wape"])
    comparison_rows = []
    for seed, row in pivot.iterrows():
        item = {"seed": int(seed)}
        for metric in ("mae", "rmse", "smape", "wape"):
            baseline = float(row[(metric, "official_adapted_dcrnn")])
            candidate = float(row[(metric, "a10_hierarchical_calibrated_dcrnn")])
            item[f"dcrnn_{metric}"] = baseline
            item[f"a10_{metric}"] = candidate
            item[f"{metric}_ratio"] = candidate / baseline
        comparison_rows.append(item)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(root / "a10_vs_dcrnn_external.csv", index=False, encoding="utf-8-sig")
    wins = {metric: int((comparison[f"{metric}_ratio"] < 1.0).sum()) for metric in ("mae", "rmse", "smape", "wape")}
    means = {metric: float(comparison[f"{metric}_ratio"].mean()) for metric in wins}
    decision = {
        "stage": "A10-PEMS07-external-prebootstrap",
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
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\a10_pems07_external_confirmation")
    parser.add_argument("--seeds", default="42,2024,3407")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--dcrnn-epochs", type=int, default=20)
    parser.add_argument("--dcrnn-batch-size", type=int, default=16)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-eval-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_nodes != 41 or args.history != 12 or args.horizon != 12:
        raise ValueError("A10 PEMS07 locks 41 nodes and history/horizon 12")
    seeds = parse_int_list(args.seeds)
    if args.stage == "train":
        return train_stage(seeds, args)
    return evaluate_stage(seeds, args)


if __name__ == "__main__":
    raise SystemExit(main())
