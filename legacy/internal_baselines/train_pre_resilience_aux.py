"""Mainline training script for thesis D-STSGCN experiments.

Only the thesis mainline models are supported through run_experiments.py:
stsgcn, dgcn, quality_v2, and quality_ood_scaled_v1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import (
    SplitScaler,
    TrafficWindowDataset,
    build_static_adjacency,
    load_wide_traffic_csv,
)
from model import DSTSGCN


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_suffix(value: str) -> str | None:
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def parse_optional_name(value: str) -> str | None:
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def resolve_feature_index(feature_name: str | None, feature_names: list[str]) -> int | None:
    if feature_name is None:
        return None
    if feature_name not in feature_names:
        raise ValueError(
            f"Feature column {feature_name!r} is not available. "
            f"Available feature names: {feature_names}"
        )
    return feature_names.index(feature_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def extract_event_signal(x: torch.Tensor, event_feature_idx: int | None) -> torch.Tensor | None:
    if event_feature_idx is None:
        return None
    return x[:, -1, :, event_feature_idx].mean(dim=1, keepdim=True)


def event_window_values(event_series: np.ndarray, history: int, num_windows: int) -> np.ndarray:
    return np.asarray(
        [event_series[min(idx + history - 1, len(event_series) - 1)] for idx in range(num_windows)],
        dtype=np.float64,
    )


def split_dataset(
    dataset: TrafficWindowDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    split_mode: str = "chronological",
    event_series: np.ndarray | None = None,
    event_window_threshold: float = 0.0,
) -> tuple[Subset, Subset, Subset, dict[str, object]]:
    """Split windows. event_aware keeps train chronological and moves events to test."""
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    indices = np.arange(n)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    event_counts = None
    warning = ""

    if split_mode not in {"chronological", "event_aware"}:
        raise ValueError(f"Unknown split mode: {split_mode}")

    if split_mode == "event_aware":
        if event_series is None:
            warning = "event-aware split requested but no event series is available; using chronological split."
        else:
            event_values = event_window_values(event_series, dataset.history, n)
            post_train = indices[train_end:]
            post_event = post_train[event_values[post_train] > event_window_threshold]
            if post_event.size == 0:
                warning = (
                    "event-aware split requested but no post-train event windows were found; "
                    "using chronological split."
                )
            else:
                test_count = len(test_indices)
                selected_events = post_event[-min(post_event.size, test_count):]
                selected_mask = np.zeros(n, dtype=bool)
                selected_mask[selected_events] = True
                remaining_slots = test_count - selected_events.size
                if remaining_slots > 0:
                    fill_candidates = post_train[~selected_mask[post_train]]
                    selected_mask[fill_candidates[-remaining_slots:]] = True
                test_indices = np.sort(post_train[selected_mask[post_train]])
                remaining_post = post_train[~selected_mask[post_train]]
                val_count = val_end - train_end
                if remaining_post.size < val_count:
                    warning = "event-aware split could not preserve validation size; using chronological split."
                    val_indices = indices[train_end:val_end]
                    test_indices = indices[val_end:]
                else:
                    val_indices = remaining_post[:val_count]

    if event_series is not None:
        event_values = event_window_values(event_series, dataset.history, n)

        def count_events(split_indices: np.ndarray) -> int:
            return int(np.sum(event_values[split_indices] > event_window_threshold))

        event_counts = {
            "train": count_events(train_indices),
            "val": count_events(val_indices),
            "test": count_events(test_indices),
        }

    info = {
        "split_mode": split_mode,
        "num_windows": int(n),
        "train_windows": int(len(train_indices)),
        "val_windows": int(len(val_indices)),
        "test_windows": int(len(test_indices)),
        "event_window_threshold": float(event_window_threshold),
        "event_window_counts": event_counts,
        "warning": warning,
    }
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
        info,
    )


@torch.no_grad()
def estimate_train_ood_profile(
    dataset: torch.utils.data.Dataset,
    ood_clip: float = 5.0,
    use_volatility: bool = True,
) -> dict[str, float]:
    """Estimate robust train-only OOD profile from scaled traffic windows."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    chunks: list[torch.Tensor] = []
    for x, _ in loader:
        chunks.append(x[..., 0].float().reshape(-1).cpu())
    if not chunks:
        return {"enabled": True, "count": 0}

    all_traffic = torch.cat(chunks)
    traffic_median = torch.median(all_traffic)
    traffic_mad = torch.median(torch.abs(all_traffic - traffic_median)).clamp_min(1e-6)
    robust_scale = 1.4826 * traffic_mad

    raw_values: list[torch.Tensor] = []
    vol_values: list[torch.Tensor] = []
    for x, _ in loader:
        traffic = x[..., 0].float()
        robust_z = torch.abs((traffic - traffic_median) / robust_scale)
        raw_ood = torch.clamp(robust_z, 0.0, float(ood_clip)).mean(dim=(1, 2))
        raw_values.append(raw_ood.cpu())
        if traffic.size(1) > 1:
            vol_values.append(torch.abs(torch.diff(traffic, dim=1)).mean(dim=(1, 2)).cpu())
        else:
            vol_values.append(torch.zeros(traffic.size(0)))

    raw_all = torch.cat(raw_values).float()
    vol_all = torch.cat(vol_values).float()
    raw_median = torch.median(raw_all)
    raw_mad = torch.median(torch.abs(raw_all - raw_median)).clamp_min(1e-6)
    vol_median = torch.median(vol_all)
    vol_mad = torch.median(torch.abs(vol_all - vol_median)).clamp_min(1e-6)
    raw_z = (raw_all - raw_median) / (1.4826 * raw_mad)
    vol_z = (vol_all - vol_median) / (1.4826 * vol_mad)
    combined = 0.7 * raw_z + 0.3 * vol_z if use_volatility else raw_z

    def q(values: torch.Tensor, p: float) -> float:
        return float(torch.quantile(values, p).item())

    return {
        "enabled": True,
        "count": int(raw_all.numel()),
        "traffic_median": float(traffic_median.item()),
        "traffic_mad": float(traffic_mad.item()),
        "raw_ood_median": float(raw_median.item()),
        "raw_ood_mad": float(raw_mad.item()),
        "raw_ood_q95": q(raw_all, 0.95),
        "raw_ood_q99": q(raw_all, 0.99),
        "volatility_median": float(vol_median.item()),
        "volatility_mad": float(vol_mad.item()),
        "combined_ood_q95": q(combined, 0.95),
        "combined_ood_q99": q(combined, 0.99),
        "ood_clip": float(ood_clip),
        "use_volatility": bool(use_volatility),
    }


def extract_ood_intensity(
    x: torch.Tensor,
    profile: dict[str, float] | None,
    ood_clip: float = 5.0,
    use_volatility: bool = True,
) -> torch.Tensor | None:
    if not profile:
        return None
    traffic = x[..., 0]
    traffic_median = x.new_tensor(float(profile.get("traffic_median", 0.0)))
    traffic_mad = x.new_tensor(max(float(profile.get("traffic_mad", 1.0)), 1e-6))
    raw_median = x.new_tensor(float(profile.get("raw_ood_median", 0.0)))
    raw_mad = x.new_tensor(max(float(profile.get("raw_ood_mad", 1.0)), 1e-6))
    vol_median = x.new_tensor(float(profile.get("volatility_median", 0.0)))
    vol_mad = x.new_tensor(max(float(profile.get("volatility_mad", 1.0)), 1e-6))
    q95 = x.new_tensor(float(profile.get("combined_ood_q95", 0.0)))
    q99 = x.new_tensor(float(profile.get("combined_ood_q99", 1.0)))

    robust_z = torch.abs((traffic - traffic_median) / (1.4826 * traffic_mad))
    raw_ood = torch.clamp(robust_z, 0.0, float(ood_clip)).mean(dim=(1, 2))
    raw_z = (raw_ood - raw_median) / (1.4826 * raw_mad)
    if use_volatility and traffic.size(1) > 1:
        volatility = torch.abs(torch.diff(traffic, dim=1)).mean(dim=(1, 2))
        vol_z = (volatility - vol_median) / (1.4826 * vol_mad)
        combined = 0.7 * raw_z + 0.3 * vol_z
    else:
        combined = raw_z
    intensity = torch.relu(combined - q95) / torch.clamp(q99 - q95, min=1e-6)
    return torch.clamp(intensity, 0.0, 1.0).unsqueeze(-1)


def compute_metric_sums(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mask = torch.isfinite(target)
    pred = torch.where(mask, pred, torch.zeros_like(pred))
    target = torch.where(mask, target, torch.zeros_like(target))
    abs_err = torch.abs(pred - target)
    sq_err = (pred - target) ** 2
    denom = torch.clamp((torch.abs(pred) + torch.abs(target)) / 2.0, min=1e-6)
    return {
        "abs_sum": float(abs_err[mask].sum().item()),
        "sq_sum": float(sq_err[mask].sum().item()),
        "smape_sum": float((abs_err / denom)[mask].sum().item()),
        "target_abs_sum": float(torch.abs(target)[mask].sum().item()),
        "count": float(mask.sum().item()),
    }


def run_epoch(
    model: DSTSGCN,
    loader: DataLoader,
    static_adj: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    temporal_reg_weight: float,
    sparse_reg_weight: float,
    event_feature_idx: int | None,
    ood_profile: dict[str, float] | None,
    ood_enabled: bool,
    ood_clip: float,
    ood_use_volatility: bool,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_temporal = 0.0
    total_ood_intensity = 0.0
    total_ood_scale = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        event_signal = extract_event_signal(x, event_feature_idx)
        ood_intensity = (
            extract_ood_intensity(x, ood_profile, ood_clip, ood_use_volatility)
            if ood_enabled
            else None
        )
        optimizer.zero_grad()
        pred, aux = model(
            x,
            static_adj,
            event_signal=event_signal,
            ood_intensity=ood_intensity,
            return_aux=True,
        )
        temporal_reg = aux.get("temporal_reg", x.new_tensor(0.0))
        loss = loss_fn(pred, y) + temporal_reg_weight * temporal_reg
        if sparse_reg_weight > 0 and hasattr(model.graph_learner, "global_residual"):
            loss = loss + sparse_reg_weight * torch.abs(model.graph_learner.global_residual).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch = x.size(0)
        total_loss += float(loss.detach().item()) * batch
        total_temporal += float(temporal_reg.detach().item()) * batch
        total_ood_intensity += float(aux.get("ood_intensity_mean", x.new_tensor(0.0)).item()) * batch
        total_ood_scale += float(aux.get("ood_scale_mean", x.new_tensor(1.0)).item()) * batch
    n = max(len(loader.dataset), 1)
    return {
        "loss": total_loss / n,
        "temporal_reg": total_temporal / n,
        "ood_intensity_mean": total_ood_intensity / n,
        "ood_scale_mean": total_ood_scale / n,
    }


@torch.no_grad()
def evaluate(
    model: DSTSGCN,
    loader: DataLoader,
    static_adj: torch.Tensor,
    event_feature_idx: int | None,
    ood_profile: dict[str, float] | None,
    ood_enabled: bool,
    ood_clip: float,
    ood_use_volatility: bool,
    device: str,
) -> dict[str, object]:
    model.eval()
    totals = {"abs_sum": 0.0, "sq_sum": 0.0, "smape_sum": 0.0, "target_abs_sum": 0.0, "count": 0.0}
    horizon_abs_sum = None
    horizon_count = None
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        event_signal = extract_event_signal(x, event_feature_idx)
        ood_intensity = (
            extract_ood_intensity(x, ood_profile, ood_clip, ood_use_volatility)
            if ood_enabled
            else None
        )
        pred = model(x, static_adj, event_signal=event_signal, ood_intensity=ood_intensity)
        batch_metrics = compute_metric_sums(pred, y)
        for key in totals:
            totals[key] += batch_metrics[key]
        mask = torch.isfinite(y)
        abs_err = torch.where(mask, torch.abs(pred - y), torch.zeros_like(y))
        batch_horizon_abs = abs_err.sum(dim=(0, 2, 3)).detach().cpu()
        batch_horizon_count = mask.sum(dim=(0, 2, 3)).detach().cpu()
        if horizon_abs_sum is None:
            horizon_abs_sum = batch_horizon_abs
            horizon_count = batch_horizon_count
        else:
            horizon_abs_sum += batch_horizon_abs
            horizon_count += batch_horizon_count
    count = max(totals["count"], 1.0)
    horizon_mae = (
        (horizon_abs_sum / torch.clamp(horizon_count, min=1)).tolist()
        if horizon_abs_sum is not None and horizon_count is not None
        else []
    )
    return {
        "mae": totals["abs_sum"] / count,
        "rmse": (totals["sq_sum"] / count) ** 0.5,
        "smape": totals["smape_sum"] / count * 100,
        "wape": totals["abs_sum"] / max(totals["target_abs_sum"], 1e-6) * 100,
        "horizon_mae": horizon_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--canonical-model-name", default="")
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--value-suffix", default="_volume")
    parser.add_argument("--exclude-cols", default="ID,id")
    parser.add_argument("--extra-feature-cols", default="")
    parser.add_argument("--node-feature-suffixes", default="")
    parser.add_argument("--add-time-features", action="store_true")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--graph-learner-type", choices=["lmln", "attention"], default="lmln")
    parser.add_argument("--fusion-mode", choices=["fusion", "static", "dynamic"], default="fusion")
    parser.add_argument("--fusion-type", choices=["quality", "quality_ood_scaled"], default="quality")
    parser.add_argument("--dynamic-top-k", type=int, default=3)
    parser.add_argument("--quality-gate-bias", type=float, default=-1.0)
    parser.add_argument("--num-diffusion-steps", type=int, default=2)
    parser.add_argument("--matrix-hidden-dim", type=int, default=256)
    parser.add_argument("--event-col", default="")
    parser.add_argument("--ood-gamma-max", type=float, default=0.35)
    parser.add_argument("--ood-clip", type=float, default=5.0)
    parser.add_argument("--ood-use-volatility", action="store_true")
    parser.add_argument("--adj-source", choices=["corr", "road", "granger", "mix"], default="corr")
    parser.add_argument("--adj-path", default="")
    parser.add_argument("--adj-from-col", default="from_node")
    parser.add_argument("--adj-to-col", default="to_node")
    parser.add_argument("--adj-weight-col", default="")
    parser.add_argument("--adj-link-id-col", default="link_id")
    parser.add_argument("--adj-undirected", action="store_true")
    parser.add_argument("--adj-geo-wkt-col", default="geo_wkt")
    parser.add_argument("--road-knn", type=int, default=3)
    parser.add_argument("--corr-threshold", type=float, default=0.2)
    parser.add_argument("--corr-method", choices=["pearson", "shrinkage", "partial"], default="pearson")
    parser.add_argument("--corr-shrinkage-lambda", type=float, default=0.1)
    parser.add_argument("--auto-corr-shrinkage", action="store_true")
    parser.add_argument("--road-weight", type=float, default=0.5)
    parser.add_argument("--granger-lag", type=int, default=3)
    parser.add_argument("--granger-p-threshold", type=float, default=0.05)
    parser.add_argument("--granger-cache-path", default="")
    parser.add_argument("--static-adj-train-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--sparse-reg-weight", type=float, default=1e-4)
    parser.add_argument("--temporal-reg-weight", type=float, default=1e-3)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["chronological", "event_aware"], default="chronological")
    parser.add_argument("--event-window-threshold", type=float, default=0.0)
    args = parser.parse_args()

    if args.fusion_type == "quality_ood_scaled" and args.fusion_mode != "fusion":
        raise ValueError("quality_ood_scaled requires --fusion-mode fusion.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, node_names, feature_names = load_wide_traffic_csv(
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
    event_col = parse_optional_name(args.event_col)
    event_feature_idx = resolve_feature_index(event_col, feature_names)
    raw_event_series = data[:, 0, event_feature_idx].copy() if event_feature_idx is not None else None

    train_time_end = int(len(data) * 0.6)
    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(data[:train_time_end])
    data = scaler.transform(data)

    adj_data = data[:train_time_end] if args.static_adj_train_only else data
    static_np = build_static_adjacency(
        adj_data,
        node_names,
        source=args.adj_source,
        corr_threshold=args.corr_threshold,
        corr_method=args.corr_method,
        corr_shrinkage_lambda=args.corr_shrinkage_lambda,
        auto_corr_shrinkage=args.auto_corr_shrinkage,
        adj_path=args.adj_path or None,
        road_weight=args.road_weight,
        granger_lag=args.granger_lag,
        granger_p_threshold=args.granger_p_threshold,
        granger_cache_path=args.granger_cache_path or None,
        from_col=args.adj_from_col,
        to_col=args.adj_to_col,
        weight_col=args.adj_weight_col or None,
        link_id_col=args.adj_link_id_col or None,
        directed=not args.adj_undirected,
        road_knn=args.road_knn,
        geo_wkt_col=args.adj_geo_wkt_col,
    )
    static_adj = torch.tensor(static_np, dtype=torch.float32, device=args.device)

    dataset = TrafficWindowDataset(data, history=args.history, horizon=args.horizon, target_dim=0)
    train_set, val_set, test_set, split_info = split_dataset(
        dataset,
        split_mode=args.split_mode,
        event_series=raw_event_series,
        event_window_threshold=args.event_window_threshold,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    ood_enabled = args.fusion_mode == "fusion" and args.fusion_type == "quality_ood_scaled"
    ood_profile = (
        estimate_train_ood_profile(train_set, args.ood_clip, args.ood_use_volatility)
        if ood_enabled
        else {"enabled": False}
    )

    model = DSTSGCN(
        num_nodes=len(node_names),
        input_dim=data.shape[-1],
        output_dim=1,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        graph_learner_type=args.graph_learner_type,
        fusion_mode=args.fusion_mode,
        fusion_type=args.fusion_type,
        dynamic_top_k=args.dynamic_top_k if args.dynamic_top_k > 0 else None,
        quality_gate_bias=args.quality_gate_bias,
        matrix_hidden_dim=args.matrix_hidden_dim,
        num_diffusion_steps=args.num_diffusion_steps,
        event_dim=1 if event_feature_idx is not None else 0,
        ood_gamma_max=args.ood_gamma_max,
    ).to(args.device)
    trainable_params = count_trainable_parameters(model)
    effective_graph_learner_type = getattr(model, "graph_learner_type", args.graph_learner_type)
    canonical_model_name = (
        args.canonical_model_name
        or args.model_name
        or (
            "quality_ood_scaled_v1"
            if args.fusion_type == "quality_ood_scaled"
            else ("stsgcn" if args.fusion_mode == "static" else "dgcn" if args.fusion_mode == "dynamic" else "quality_v2")
        )
    )
    dataset_name = args.dataset_name or Path(args.csv).stem

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "dataset": dataset_name,
            "model_name": args.model_name or canonical_model_name,
            "canonical_model_name": canonical_model_name,
            "requested_graph_learner_type": args.graph_learner_type,
            "graph_learner_type": effective_graph_learner_type,
            "split_info": split_info,
            "feature_names": feature_names,
            "node_count": len(node_names),
            "trainable_parameters": trainable_params,
            "effective_graph_learner_type": effective_graph_learner_type,
            "requested_fusion_type": args.fusion_type,
            "effective_fusion_type": args.fusion_type,
            "ood_enabled": ood_enabled,
            "ood_profile": ood_profile,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print(f"Loaded data: time={data.shape[0]}, nodes={data.shape[1]}, features={data.shape[2]}")
    print(
        f"Split mode: {args.split_mode} | train={split_info['train_windows']} "
        f"val={split_info['val_windows']} test={split_info['test_windows']}"
    )
    if split_info.get("warning"):
        print(f"[WARNING] {split_info['warning']}")
    if split_info.get("event_window_counts") is not None:
        print(f"Event window counts: {split_info['event_window_counts']}")
    print(f"Device: {args.device}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {canonical_model_name}")
    print(f"Graph learner: {effective_graph_learner_type}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Fusion type: {args.fusion_type}")
    print(f"Trainable parameters: {trainable_params}")
    if ood_enabled:
        print("[OODScaled] enabled=yes")
        print(
            "[OODScaled] "
            f"gamma_max={args.ood_gamma_max:g} ood_clip={args.ood_clip:g} "
            f"use_volatility={'yes' if args.ood_use_volatility else 'no'}"
        )
        print(
            "[OODScaled] "
            f"combined_q95={ood_profile.get('combined_ood_q95', 0.0):.6f} "
            f"combined_q99={ood_profile.get('combined_ood_q99', 0.0):.6f}"
        )
    print(f"Dynamic top-k: {args.dynamic_top_k if args.dynamic_top_k > 0 else 'disabled'}")
    print(f"Diffusion steps: {args.num_diffusion_steps}")
    print(f"Temporal regularization weight: {args.temporal_reg_weight:g}")
    print("Event column: " + (f"{event_col} (feature index {event_feature_idx})" if event_col else "disabled"))
    print(f"Adjacency source: {args.adj_source}")
    print(f"Correlation method: {args.corr_method} (threshold={args.corr_threshold:g})")
    print(f"Static adjacency train only: {'yes' if args.static_adj_train_only else 'no'}")
    print(f"Extra features: {parse_csv_list(args.extra_feature_cols) or 'none'}")
    print(f"Node feature suffixes: {parse_csv_list(args.node_feature_suffixes) or 'none'}")
    print(f"Time features: {'enabled' if args.add_time_features else 'disabled'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.HuberLoss()
    best_val = float("inf")
    best_path = output_dir / "best_dstsgcn.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            static_adj,
            optimizer,
            loss_fn,
            args.temporal_reg_weight,
            args.sparse_reg_weight,
            event_feature_idx,
            ood_profile,
            ood_enabled,
            args.ood_clip,
            args.ood_use_volatility,
            args.device,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            static_adj,
            event_feature_idx,
            ood_profile,
            ood_enabled,
            args.ood_clip,
            args.ood_use_volatility,
            args.device,
        )
        scheduler.step(float(val_metrics["mae"]))
        current_lr = optimizer.param_groups[0]["lr"]
        log_line = (
            f"Epoch {epoch:03d} | train_loss={train_stats['loss']:.4f} | "
            f"temporal_reg={train_stats['temporal_reg']:.6f} | "
            f"val_mae={val_metrics['mae']:.4f} | val_rmse={val_metrics['rmse']:.4f} | "
            f"val_smape={val_metrics['smape']:.2f}% | val_wape={val_metrics['wape']:.2f}% | "
        )
        if ood_enabled:
            log_line += (
                f"ood_intensity_mean={train_stats['ood_intensity_mean']:.4f} | "
                f"ood_scale_mean={train_stats['ood_scale_mean']:.4f} | "
            )
        log_line += f"lr={current_lr:.2e}"
        print(log_line)
        if float(val_metrics["mae"]) < best_val:
            best_val = float(val_metrics["mae"])
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_metrics = evaluate(
        model,
        test_loader,
        static_adj,
        event_feature_idx,
        ood_profile,
        ood_enabled,
        args.ood_clip,
        args.ood_use_volatility,
        args.device,
    )
    horizon_mae = test_metrics["horizon_mae"]
    display_indices = [idx for idx in (0, 2, 5, 11) if idx < len(horizon_mae)]
    horizon_display = " ".join(f"h{idx + 1}={horizon_mae[idx]:.4f}" for idx in display_indices)
    print(
        f"Test MAE={test_metrics['mae']:.4f}, "
        f"RMSE={test_metrics['rmse']:.4f}, "
        f"SMAPE={test_metrics['smape']:.2f}%, "
        f"WAPE={test_metrics['wape']:.2f}%"
        + (f" | {horizon_display}" if horizon_display else "")
    )


if __name__ == "__main__":
    main()
