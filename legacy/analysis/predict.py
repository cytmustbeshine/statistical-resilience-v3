"""Export D-STSGCN predictions and metrics to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def normalize_quantiles(quantiles: list[float] | None) -> list[float] | None:
    if not quantiles:
        return None
    cleaned = sorted(float(q) for q in quantiles)
    if any(q <= 0.0 or q >= 1.0 for q in cleaned):
        raise ValueError("--quantiles values must be between 0 and 1.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("--quantiles values must be unique.")
    return cleaned


def median_quantile_index(quantiles: list[float]) -> int:
    return min(range(len(quantiles)), key=lambda idx: abs(quantiles[idx] - 0.5))


def quantile_point_prediction(
    pred: np.ndarray,
    quantiles: list[float] | None,
) -> np.ndarray:
    if not quantiles:
        return pred
    idx = median_quantile_index(quantiles)
    return pred[..., idx : idx + 1]


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


def extract_event_signal(x: torch.Tensor, event_feature_idx: int | None) -> torch.Tensor | None:
    if event_feature_idx is None:
        return None
    return x[:, -1, :, event_feature_idx].mean(dim=1, keepdim=True)


def split_dataset(dataset: TrafficWindowDataset, train_ratio=0.6, val_ratio=0.2):
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    indices = np.arange(n)
    return (
        Subset(dataset, indices[:train_end]),
        Subset(dataset, indices[train_end:val_end]),
        Subset(dataset, indices[val_end:]),
    )


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = np.mean(np.abs(err) / denom) * 100
    smape_denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    smape = np.mean(2 * np.abs(err) / smape_denom) * 100
    wape = np.sum(np.abs(err)) / np.maximum(np.sum(np.abs(y_true)), 1e-6) * 100
    return {
        "scope": name,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "smape_percent": smape,
        "wape_percent": wape,
    }


@torch.no_grad()
def collect_predictions(model, loader, static_adj, road_adj, event_feature_idx, device):
    model.eval()
    preds = []
    trues = []
    for x, y in loader:
        x = x.to(device)
        event_signal = extract_event_signal(x, event_feature_idx)
        pred = model(x, static_adj, road_adj, event_signal=event_signal).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to wide traffic CSV.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_dstsgcn.pt.")
    parser.add_argument("--time-col", default="Time")
    parser.add_argument("--value-suffix", default="_volume")
    parser.add_argument("--exclude-cols", default="ID,id")
    parser.add_argument(
        "--extra-feature-cols",
        default="",
        help="Comma-separated global numeric feature columns, e.g. weather columns.",
    )
    parser.add_argument(
        "--node-feature-suffixes",
        default="",
        help="Comma-separated per-node feature suffixes aligned with value-suffix, e.g. _speed.",
    )
    parser.add_argument("--add-time-features", action="store_true")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--graph-learner-type", choices=["lmln", "attention"], default="lmln")
    parser.add_argument("--fusion-mode", choices=["fusion", "static", "dynamic"], default="fusion")
    parser.add_argument("--fusion-type", choices=["scalar", "node_gate", "quality"], default="quality")
    parser.add_argument(
        "--event-col",
        default="",
        help="Optional feature column used as event intensity for quality fusion.",
    )
    parser.add_argument(
        "--use-learned-static",
        action="store_true",
        help="Use learned corr/road static fusion. Requires --adj-source mix.",
    )
    parser.add_argument("--dynamic-top-k", type=int, default=0)
    parser.add_argument("--quality-gate-bias", type=float, default=-1.0)
    parser.add_argument("--num-diffusion-steps", type=int, default=2)
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=None,
        help="Quantiles used by the trained model, e.g. --quantiles 0.1 0.5 0.9.",
    )
    parser.add_argument(
        "--matrix-hidden-dim",
        type=int,
        default=256,
        help="Hidden size used by the trained LMLN matrix LSTM. Use 0 for legacy checkpoints.",
    )
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
    parser.add_argument("--road-weight", type=float, default=0.5)
    parser.add_argument("--granger-lag", type=int, default=3)
    parser.add_argument("--granger-p-threshold", type=float, default=0.05)
    parser.add_argument("--granger-cache-path", default="")
    parser.add_argument("--static-adj-train-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\predictions")
    parser.add_argument("--max-export-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.quantiles = normalize_quantiles(args.quantiles)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data, node_names, feature_names = load_wide_traffic_csv(
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

    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(raw_data[: int(len(raw_data) * 0.6)])
    data = scaler.transform(raw_data)

    if args.use_learned_static and args.adj_source != "mix":
        raise ValueError("--use-learned-static requires --adj-source mix.")

    adj_data = data[: int(len(data) * 0.6)] if args.static_adj_train_only else data
    road_adj = None
    if args.use_learned_static:
        static_adj = build_static_adjacency(
            adj_data,
            node_names,
            source="corr",
            corr_threshold=args.corr_threshold,
        )
        static_adj = torch.tensor(static_adj, dtype=torch.float32, device=args.device)
        try:
            road_adj_np = build_static_adjacency(
                adj_data,
                node_names,
                source="road",
                corr_threshold=args.corr_threshold,
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
            road_adj = torch.tensor(road_adj_np, dtype=torch.float32, device=args.device)
        except ValueError as exc:
            print(f"[WARNING] Road adjacency construction failed: {exc}")
            print("[WARNING] Falling back to correlation graph for road_adj in learned_mix.")
            road_adj = static_adj.clone()
    else:
        static_adj = build_static_adjacency(
            adj_data,
            node_names,
            source=args.adj_source,
            corr_threshold=args.corr_threshold,
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
        static_adj = torch.tensor(static_adj, dtype=torch.float32, device=args.device)

    dataset = TrafficWindowDataset(
        data,
        history=args.history,
        horizon=args.horizon,
        target_dim=0,
    )
    _, _, test_set = split_dataset(dataset)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

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
        static_adj_is_normalized=False,
        use_learned_static=args.use_learned_static,
        num_diffusion_steps=args.num_diffusion_steps,
        event_dim=1 if event_feature_idx is not None else 0,
        quantiles=args.quantiles,
    ).to(args.device)

    state_dict = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict)

    y_true_norm, y_pred_norm = collect_predictions(
        model,
        test_loader,
        static_adj,
        road_adj,
        event_feature_idx,
        args.device,
    )

    mean = float(scaler.traffic_scaler.mean.reshape(-1)[0])
    std = float(scaler.traffic_scaler.std.reshape(-1)[0])
    y_true = y_true_norm * std + mean
    y_pred = y_pred_norm * std + mean
    y_pred_point_norm = quantile_point_prediction(y_pred_norm, args.quantiles)
    y_pred_point = quantile_point_prediction(y_pred, args.quantiles)

    metrics = [metric_row("overall_original_scale", y_true, y_pred_point)]
    metrics.append(metric_row("overall_normalized", y_true_norm, y_pred_point_norm))
    for h in range(args.horizon):
        metrics.append(metric_row(f"horizon_{h + 1}", y_true[:, h], y_pred_point[:, h]))

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    rows = []
    export_samples = min(args.max_export_samples, y_true.shape[0])
    for sample_idx in range(export_samples):
        for h in range(args.horizon):
            for node_idx, node_name in enumerate(node_names):
                true_value = float(y_true[sample_idx, h, node_idx, 0])
                pred_value = float(y_pred_point[sample_idx, h, node_idx, 0])
                row = {
                    "test_sample": sample_idx,
                    "horizon_step": h + 1,
                    "minutes_ahead": (h + 1) * 5,
                    "node_idx": node_idx,
                    "node_name": node_name,
                    "true_value": true_value,
                    "pred_value": pred_value,
                    "abs_error": abs(pred_value - true_value),
                    "true_norm": float(y_true_norm[sample_idx, h, node_idx, 0]),
                    "pred_norm": float(y_pred_point_norm[sample_idx, h, node_idx, 0]),
                }
                if args.quantiles:
                    for q_idx, q in enumerate(args.quantiles):
                        label = f"q{int(round(q * 100))}"
                        row[label] = float(y_pred[sample_idx, h, node_idx, q_idx])
                        row[f"{label}_norm"] = float(
                            y_pred_norm[sample_idx, h, node_idx, q_idx]
                        )
                    row["interval_width"] = float(
                        y_pred[sample_idx, h, node_idx, -1]
                        - y_pred[sample_idx, h, node_idx, 0]
                    )
                rows.append(row)

    predictions_path = output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(pd.DataFrame(metrics).head(6).to_string(index=False))


if __name__ == "__main__":
    main()
