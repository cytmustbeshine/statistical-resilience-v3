"""Train the official-code-adapted DCRNN-Resilience public baseline."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data import (
    SplitScaler,
    TrafficWindowDataset,
    build_static_adjacency,
    load_wide_traffic_csv,
    read_csv_with_fallback,
    resolve_time_index,
    split_traffic_window_indices,
)
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_suffix(value: str) -> str | None:
    return None if value.strip().lower() in {"", "none", "null"} else value.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def infer_dataset_name(csv_path: str) -> str:
    stem = Path(csv_path).stem.lower()
    for name in ("rainstorm", "typhoon", "bridge", "pems04", "pems08"):
        if name in stem:
            return name
    return stem


def load_event_series(csv_path: str, time_col: str, event_col: str) -> np.ndarray | None:
    if not event_col:
        return None
    frame = read_csv_with_fallback(csv_path)
    if time_col and time_col in frame.columns:
        parsed = __import__("pandas").to_datetime(frame[time_col], errors="coerce")
        if float(parsed.notna().mean()) >= 0.95:
            frame = frame.assign(__time=parsed).sort_values("__time").drop(columns="__time").reset_index(drop=True)
        else:
            frame = frame.sort_values(time_col).reset_index(drop=True)
    if event_col not in frame.columns:
        return None
    return np.nan_to_num(frame[event_col].astype("float32").to_numpy(), nan=0.0)


def event_window_values(event_series: np.ndarray, history: int, count: int) -> np.ndarray:
    return np.asarray([event_series[min(i + history - 1, len(event_series) - 1)] for i in range(count)])


def split_dataset(
    dataset: TrafficWindowDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_mode: str = "chronological",
    event_series: np.ndarray | None = None,
    event_window_threshold: float = 0.0,
    explicit_event_index: int | None = None,
    event_test_prehistory_steps: int | None = None,
) -> tuple[Subset, Subset, Subset, dict[str, object]]:
    """Use the same fair split implementation as the proposed model."""
    train_indices, val_indices, test_indices, info = split_traffic_window_indices(
        len(dataset), history=dataset.history, train_ratio=train_ratio, val_ratio=val_ratio,
        split_mode=split_mode, event_series=event_series,
        event_window_threshold=event_window_threshold,
        explicit_event_index=explicit_event_index,
        event_test_prehistory_steps=event_test_prehistory_steps,
    )
    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, val_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
        info,
    )

def compute_metric_sums(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = torch.isfinite(target)
    safe_pred = torch.where(mask, pred, torch.zeros_like(pred))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    absolute = torch.abs(safe_pred - safe_target)
    square = (safe_pred - safe_target) ** 2
    denominator = torch.clamp((torch.abs(safe_pred) + torch.abs(safe_target)) / 2.0, min=1e-6)
    return {
        "abs_sum": float(absolute[mask].sum().item()),
        "sq_sum": float(square[mask].sum().item()),
        "smape_sum": float((absolute / denominator)[mask].sum().item()),
        "target_abs_sum": float(torch.abs(safe_target)[mask].sum().item()),
        "count": float(mask.sum().item()),
    }

def official_split(dataset: TrafficWindowDataset) -> tuple[Subset, Subset, Subset, dict[str, object]]:
    """Apply the public repository's chronological 70/10/20 split."""
    n = len(dataset)
    train_end = round(n * 0.7)
    test_count = round(n * 0.2)
    val_end = n - test_count
    train = np.arange(0, train_end)
    val = np.arange(train_end, val_end)
    test = np.arange(val_end, n)
    return Subset(dataset, train.tolist()), Subset(dataset, val.tolist()), Subset(dataset, test.tolist()), {
        "protocol": "official_70_10_20",
        "train_windows": len(train), "val_windows": len(val), "test_windows": len(test),
    }


def raw_masked_rmse(pred: torch.Tensor, target: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Match the official supervisor's inverse-scaled masked RMSE objective."""
    pred_raw = pred * std + mean
    target_raw = target * std + mean
    mask = torch.isfinite(target_raw) & (torch.abs(target_raw) > 1e-8)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return torch.sqrt(torch.mean((pred_raw[mask] - target_raw[mask]) ** 2) + 1e-12)


def evaluate(model, loader, adj, mean, std, device):
    model.eval()
    totals = {"abs_sum": 0.0, "sq_sum": 0.0, "smape_sum": 0.0, "target_abs_sum": 0.0, "count": 0.0}
    horizon_abs = None
    horizon_count = None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, adj)
            pred_raw, y_raw = pred * std + mean, y * std + mean
            batch = compute_metric_sums(pred_raw, y_raw)
            for key in totals:
                totals[key] += batch[key]
            mask = torch.isfinite(y_raw)
            errors = torch.where(mask, torch.abs(pred_raw - y_raw), torch.zeros_like(y_raw))
            current_abs = errors.sum(dim=(0, 2, 3)).cpu()
            current_count = mask.sum(dim=(0, 2, 3)).cpu()
            horizon_abs = current_abs if horizon_abs is None else horizon_abs + current_abs
            horizon_count = current_count if horizon_count is None else horizon_count + current_count
    count = max(totals["count"], 1.0)
    return {
        "mae": totals["abs_sum"] / count,
        "rmse": (totals["sq_sum"] / count) ** 0.5,
        "smape": totals["smape_sum"] / count * 100.0,
        "wape": totals["abs_sum"] / max(totals["target_abs_sum"], 1e-6) * 100.0,
        "horizon_mae": (horizon_abs / torch.clamp(horizon_count, min=1)).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--value-suffix", default="_volume")
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--history-steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--rnn-units", type=int, default=256)
    parser.add_argument("--num-rnn-layers", type=int, default=2)
    parser.add_argument("--max-diffusion-step", type=int, default=1)
    parser.add_argument("--cl-decay-steps", type=int, default=2000)
    parser.add_argument("--disable-curriculum-learning", action="store_true")
    parser.add_argument("--protocol", choices=["fair", "official"], default="fair")
    parser.add_argument("--split-mode", choices=["chronological", "event_aware", "event_aligned"], default="chronological")
    parser.add_argument("--explicit-event-time", default="")
    parser.add_argument("--event-test-prehistory-steps", type=int, default=None)
    parser.add_argument("--event-window-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--event-col", default="")
    parser.add_argument("--exclude-cols", default="ID,id")
    parser.add_argument("--extra-feature-cols", default="")
    parser.add_argument("--node-feature-suffixes", default="")
    parser.add_argument("--add-time-features", action="store_true")
    parser.add_argument("--adj-source", choices=["corr", "road", "mix"], default="corr")
    parser.add_argument("--adj-path", default="")
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or infer_dataset_name(args.csv)
    raw, nodes, feature_names = load_wide_traffic_csv(
        args.csv,
        value_suffix=parse_optional_suffix(args.value_suffix),
        time_col=args.time_col,
        max_nodes=args.max_nodes,
        extra_feature_cols=parse_csv_list(args.extra_feature_cols),
        node_feature_suffixes=parse_csv_list(args.node_feature_suffixes),
        add_time_features=args.add_time_features,
        exclude_cols=parse_csv_list(args.exclude_cols),
        return_feature_names=True,
    )
    train_time_end = int(len(raw) * (0.7 if args.protocol == "official" else 0.6))
    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(raw[:train_time_end])
    scaled = scaler.transform(raw)
    adj_np = build_static_adjacency(
        scaled[:train_time_end], nodes, source=args.adj_source,
        corr_threshold=args.corr_threshold, adj_path=args.adj_path or None,
    )
    adj = torch.tensor(adj_np, dtype=torch.float32, device=args.device)
    dataset = TrafficWindowDataset(scaled, history=args.history_steps, horizon=args.horizon, target_dim=0)
    event_series = load_event_series(args.csv, args.time_col, args.event_col)
    if args.protocol == "official":
        train_set, val_set, test_set, split_info = official_split(dataset)
    else:
        explicit_event_index = (
            resolve_time_index(args.csv, args.time_col, args.explicit_event_time)
            if args.explicit_event_time else None
        )
        train_set, val_set, test_set, split_info = split_dataset(
            dataset, split_mode=args.split_mode, event_series=event_series,
            event_window_threshold=args.event_window_threshold,
            explicit_event_index=explicit_event_index,
            event_test_prehistory_steps=args.event_test_prehistory_steps,
        )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = OfficialAdaptedDCRNN(
        num_nodes=len(nodes), input_dim=scaled.shape[-1], output_dim=1,
        rnn_units=args.rnn_units, num_rnn_layers=args.num_rnn_layers,
        horizon=args.horizon, max_diffusion_step=args.max_diffusion_step,
        cl_decay_steps=args.cl_decay_steps,
        use_curriculum_learning=not args.disable_curriculum_learning,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
    mean = float(np.asarray(scaler.traffic_scaler.mean).reshape(-1)[0])
    std = float(np.asarray(scaler.traffic_scaler.std).reshape(-1)[0])
    config = vars(args).copy()
    config.update({
        "model": "dcrnn_resilience_official_adapted",
        "model_name": "dcrnn_resilience_official_adapted",
        "canonical_model_name": "dcrnn_resilience_official_adapted",
        "dataset": dataset_name, "dataset_name": dataset_name,
        "feature_names": feature_names, "input_dim": scaled.shape[-1],
        "node_count": len(nodes), "trainable_parameters": count_trainable_parameters(model),
        "split_info": split_info,
        "source_repository": "https://github.com/Charles117/resilience_shenzhen",
        "implementation_scope": "official-code-adapted PyTorch reimplementation; private data and unpublished resilience code unavailable",
        "official_architecture": {
            "filter_type": "dual_random_walk", "masked_loss": "inverse_scaled_rmse",
            "optimizer": "adam_eps_1e-3", "lr_milestones": [20, 30, 40, 50],
        },
    })
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Loaded data: time={len(raw)}, nodes={len(nodes)}, features={scaled.shape[-1]}")
    print(f"Protocol: {args.protocol} | split={split_info}")
    print(f"Model: dcrnn_resilience_official_adapted | params={count_trainable_parameters(model)}")
    best = float("inf")
    best_path = out / "best_dcrnn_resilience_official_adapted.pt"
    batches_seen = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            pred = model(x, adj, labels=y, batches_seen=batches_seen)
            loss = raw_masked_rmse(pred, y, mean, std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batches_seen += 1
            train_sum += float(loss.item()) * x.size(0)
            seen += x.size(0)
        val = evaluate(model, val_loader, adj, mean, std, args.device)
        scheduler.step()
        threshold = model.sampling_threshold(batches_seen)
        print(f"Epoch {epoch:03d} | train_rmse={train_sum/max(seen,1):.4f} | val_mae={val['mae']:.4f} | val_rmse={val['rmse']:.4f} | sampling={threshold:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        if val["rmse"] < best:
            best = val["rmse"]
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test = evaluate(model, test_loader, adj, mean, std, args.device)
    metrics = {key: value for key, value in test.items() if key != "horizon_mae"}
    metrics["horizon_mae"] = test["horizon_mae"]
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    h = test["horizon_mae"]
    display = " ".join(f"h{i+1}={h[i]:.4f}" for i in (0, 2, 5, 11) if i < len(h))
    print(f"Test MAE={test['mae']:.4f}, RMSE={test['rmse']:.4f}, SMAPE={test['smape']:.2f}%, WAPE={test['wape']:.2f}%" + (f" | {display}" if display else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
