"""Weather and distribution-shift slice analysis for D-STSGCN runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    SplitScaler,
    TrafficWindowDataset,
    build_static_adjacency,
    load_wide_traffic_csv,
)
from model import DSTSGCN
from train import (
    extract_event_signal,
    extract_shift_signal,
    event_window_values,
    resolve_feature_index,
    split_dataset,
)
from statistical_tests import holm_bonferroni_correction, wilcoxon_paired_test


TARGET_DATASETS = {"rainstorm", "typhoon", "pems04", "pems08"}
EVENT_COLUMNS = {
    "rainstorm": "precip_accum_one_hour",
    "typhoon": "typhoon_intensity",
}
SHIFT_FUSION_TYPES = {
    "quality_stat",
    "quality_stat_reg",
    "quality_stat_residual",
    "quality_stat_safe",
    "quality_edge",
    "quality_observed",
}


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_suffix(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def bool_config(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def model_from_config(cfg: dict, data_shape: tuple[int, int, int], event_feature_idx: int | None) -> DSTSGCN:
    fusion_type = cfg.get("effective_fusion_type") or cfg["fusion_type"]
    return DSTSGCN(
        num_nodes=data_shape[1],
        input_dim=data_shape[2],
        output_dim=1,
        horizon=int(cfg["horizon"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_blocks=int(cfg["num_blocks"]),
        graph_learner_type=cfg["graph_learner_type"],
        fusion_mode=cfg["fusion_mode"],
        fusion_type=fusion_type,
        dynamic_top_k=int(cfg["dynamic_top_k"]) if int(cfg["dynamic_top_k"]) > 0 else None,
        quality_gate_bias=float(cfg["quality_gate_bias"]),
        matrix_hidden_dim=int(cfg["matrix_hidden_dim"]),
        static_adj_is_normalized=False,
        use_learned_static=bool_config(cfg["use_learned_static"]),
        num_diffusion_steps=int(cfg["num_diffusion_steps"]),
        event_dim=1 if event_feature_idx is not None else 0,
        quantiles=cfg.get("quantiles"),
        graph_residual_init=float(cfg["graph_residual_init"]),
        stat_residual_init=float(cfg["stat_gate_residual_init"]),
        stat_residual_max=float(cfg["stat_gate_residual_max"]),
    )


def load_run(run_dir: Path, device: torch.device):
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data, node_names, feature_names = load_wide_traffic_csv(
        cfg["csv"],
        value_suffix=parse_optional_suffix(cfg["value_suffix"]),
        time_col=cfg["time_col"],
        max_nodes=int(cfg["max_nodes"]),
        extra_feature_cols=parse_csv_list(cfg.get("extra_feature_cols", "")),
        node_feature_suffixes=parse_csv_list(cfg.get("node_feature_suffixes", "")),
        add_time_features=bool_config(cfg["add_time_features"]),
        exclude_cols=parse_csv_list(cfg.get("exclude_cols", "")),
        return_feature_names=True,
    )
    event_col = cfg.get("event_col") or None
    event_feature_idx = resolve_feature_index(event_col, feature_names) if event_col else None
    raw_event_series = (
        data[:, 0, event_feature_idx].copy()
        if event_feature_idx is not None
        else None
    )

    train_cut = int(len(data) * 0.6)
    scaler = SplitScaler(num_traffic_features=1)
    scaler.fit(data[:train_cut])
    data = scaler.transform(data)

    train_traffic = data[:train_cut, :, 0]
    train_mean = torch.tensor(float(np.mean(train_traffic)), dtype=torch.float32, device=device)
    train_std = torch.tensor(max(float(np.std(train_traffic)), 1e-6), dtype=torch.float32, device=device)
    train_shift_profile = cfg.get("train_shift_profile") or {}
    selected_shift_calibration = cfg.get("selected_shift_calibration")
    use_shift_standardization = bool_config(cfg.get("shift_score_standardize", False))
    if selected_shift_calibration in {"standardized", "hybrid"}:
        use_shift_standardization = True
    train_shift_mean = torch.tensor(
        float(train_shift_profile.get("mean", 0.0)),
        dtype=torch.float32,
        device=device,
    )
    train_shift_std = torch.tensor(
        max(float(train_shift_profile.get("std", 1.0)), 1e-6),
        dtype=torch.float32,
        device=device,
    )
    shift_scale_mean = (
        torch.tensor(
            max(float(train_shift_profile.get("mean", 0.0)), 1e-6),
            dtype=torch.float32,
            device=device,
        )
        if selected_shift_calibration == "hybrid"
        else None
    )
    shift_calibration_blend = cfg.get("shift_calibration_blend")
    if shift_calibration_blend is not None:
        shift_calibration_blend = float(shift_calibration_blend)

    adj_data = data[:train_cut] if bool_config(cfg["static_adj_train_only"]) else data
    static_adj_np = build_static_adjacency(
        adj_data,
        node_names,
        source=cfg["adj_source"],
        corr_threshold=float(cfg["corr_threshold"]),
        corr_method=cfg["corr_method"],
        corr_shrinkage_lambda=float(cfg["corr_shrinkage_lambda"]),
        auto_corr_shrinkage=bool_config(cfg["auto_corr_shrinkage"]),
        adj_path=cfg.get("adj_path") or None,
        road_weight=float(cfg["road_weight"]),
        granger_lag=int(cfg["granger_lag"]),
        granger_p_threshold=float(cfg["granger_p_threshold"]),
        granger_cache_path=cfg.get("granger_cache_path") or None,
        from_col=cfg["adj_from_col"],
        to_col=cfg["adj_to_col"],
        weight_col=cfg.get("adj_weight_col") or None,
        link_id_col=cfg.get("adj_link_id_col") or None,
        directed=not bool_config(cfg["adj_undirected"]),
        road_knn=int(cfg["road_knn"]),
        geo_wkt_col=cfg["adj_geo_wkt_col"],
    )
    static_adj = torch.tensor(static_adj_np, dtype=torch.float32, device=device)

    dataset = TrafficWindowDataset(
        data,
        history=int(cfg["history"]),
        horizon=int(cfg["horizon"]),
        target_dim=0,
    )
    _, _, _, test_set = split_dataset(
        dataset,
        calib_ratio=float(cfg["calib_ratio"]),
        split_mode=cfg.get("split_mode", "chronological"),
        event_series=raw_event_series,
        event_window_threshold=float(cfg.get("event_window_threshold", 0.0) or 0.0),
    )
    if raw_event_series is None:
        test_event_values = np.zeros(len(test_set), dtype=np.float32)
    else:
        all_event_values = event_window_values(
            raw_event_series,
            history=int(cfg["history"]),
            num_windows=len(dataset),
        )
        test_event_values = all_event_values[np.asarray(test_set.indices, dtype=int)]
    loader = DataLoader(test_set, batch_size=int(cfg["batch_size"]), shuffle=False)

    model = model_from_config(cfg, data.shape, event_feature_idx).to(device)
    model.load_state_dict(torch.load(run_dir / "best_dstsgcn.pt", map_location=device))
    model.eval()
    return (
        cfg,
        model,
        loader,
        static_adj,
        event_feature_idx,
        train_mean,
        train_std,
        train_shift_mean if use_shift_standardization else None,
        train_shift_std if use_shift_standardization else None,
        shift_scale_mean,
        shift_calibration_blend,
        test_event_values,
    )


@torch.no_grad()
def collect_predictions(run_dir: Path, device: torch.device) -> dict[str, np.ndarray]:
    (
        cfg,
        model,
        loader,
        static_adj,
        event_idx,
        train_mean,
        train_std,
        train_shift_mean,
        train_shift_std,
        shift_scale_mean,
        shift_calibration_blend,
        raw_events,
    ) = load_run(run_dir, device)
    effective_fusion_type = cfg.get("effective_fusion_type") or cfg["fusion_type"]
    uses_shift = effective_fusion_type in SHIFT_FUSION_TYPES
    preds = []
    targets = []
    events = []
    shifts = []
    offset = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        event_signal = extract_event_signal(x, event_idx)
        raw_shift_signal = extract_shift_signal(x, train_mean, train_std)
        shift_signal = extract_shift_signal(
            x,
            train_mean,
            train_std,
            train_shift_mean=train_shift_mean,
            train_shift_std=train_shift_std,
            shift_score_clip=float(cfg.get("shift_score_clip", 3.0) or 3.0),
            shift_scale_mean=shift_scale_mean,
            shift_calibration_blend=shift_calibration_blend,
        )
        stat_signal = shift_signal if uses_shift else None
        pred = model(
            x,
            static_adj,
            None,
            event_signal=event_signal,
            stat_signal=stat_signal,
        )
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        batch_size = x.size(0)
        events.append(raw_events[offset:offset + batch_size])
        offset += batch_size
        shifts.append(raw_shift_signal.reshape(-1).detach().cpu().numpy())
    return {
        "pred": np.concatenate(preds, axis=0),
        "target": np.concatenate(targets, axis=0),
        "event": np.concatenate(events),
        "shift": np.concatenate(shifts),
    }


def metric_values(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {"mae": math.nan, "rmse": math.nan, "smape": math.nan, "wape": math.nan}
    p = pred[mask]
    y = target[mask]
    finite = np.isfinite(p) & np.isfinite(y)
    p = p[finite]
    y = y[finite]
    if p.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "smape": math.nan, "wape": math.nan}
    err = p - y
    abs_err = np.abs(err)
    denom = np.abs(p) + np.abs(y)
    smape = np.mean(np.where(denom > 1e-6, 2.0 * abs_err / denom, 0.0)) * 100.0
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "smape": float(smape),
        "wape": float(abs_err.sum() / max(np.abs(y).sum(), 1e-6) * 100.0),
    }


def sample_abs_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.nanmean(np.abs(pred - target), axis=(1, 2, 3))


def bootstrap_ci(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if diff.size == 0:
        return math.nan, math.nan
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, diff.size, size=diff.size)
        means.append(float(diff[idx].mean()))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5])
    return float(lo), float(hi)


def paired_t_pvalue(diff: np.ndarray) -> float:
    if diff.size < 2:
        return math.nan
    sd = float(diff.std(ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(float(diff.mean())) > 1e-12 else 1.0
    t_stat = float(diff.mean()) / (sd / math.sqrt(diff.size))
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t_stat), df=diff.size - 1))
    except Exception:
        return math.nan


def slice_masks(event: np.ndarray, shift: np.ndarray, has_event: bool) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {"all_test": np.ones_like(shift, dtype=bool)}
    if shift.size > 0:
        masks["high_shift_top25"] = shift >= np.quantile(shift, 0.75)
        masks["high_shift_top10"] = shift >= np.quantile(shift, 0.90)
        masks["low_shift_bottom50"] = shift <= np.quantile(shift, 0.50)
    if has_event:
        masks["weather_nonzero"] = event > 0
        positive = event[event > 0]
        if positive.size > 0:
            masks["weather_top25"] = event >= np.quantile(positive, 0.75)
            masks["weather_top10"] = event >= np.quantile(positive, 0.90)
        else:
            masks["weather_top25"] = np.zeros_like(event, dtype=bool)
            masks["weather_top10"] = np.zeros_like(event, dtype=bool)
    return masks


def find_run(summary_rows: list[dict[str, str]], dataset: str, model: str, seed: int) -> Path | None:
    for row in summary_rows:
        if row["dataset"] == dataset and row["model"] == model and int(row["seed"]) == seed:
            raw_run_dir = row.get("run_dir") or row.get("output_dir") or ""
            run_dir = Path(raw_run_dir) if raw_run_dir else None
            if run_dir is not None and run_dir.exists():
                return run_dir
    return None


def infer_run_dir(summary_path: Path, dataset: str, model: str, seed: int, max_nodes: str, epochs: str, adj: str) -> Path:
    return summary_path.parent / f"{dataset}_{model}_seed{seed}_n{max_nodes}_e{epochs}_adj{adj}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="quality_v2")
    parser.add_argument("--shift-model", default="quality_stat_shift")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("r", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["dataset"] in TARGET_DATASETS
            and row["model"] in {args.base_model, args.shift_model}
            and row["status"] == "ok"
        ]
    keys = sorted({(row["dataset"], int(row["seed"])) for row in rows})
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, object]] = []

    for dataset, seed in keys:
        matching = {
            row["model"]: row
            for row in rows
            if row["dataset"] == dataset and int(row["seed"]) == seed
        }
        if args.base_model not in matching or args.shift_model not in matching:
            continue
        base_row = matching[args.base_model]
        shift_row = matching[args.shift_model]
        base_dir = find_run(rows, dataset, args.base_model, seed) or infer_run_dir(
            summary_path, dataset, args.base_model, seed,
            base_row["max_nodes"], base_row["epochs"], base_row["adj_source"],
        )
        shift_dir = find_run(rows, dataset, args.shift_model, seed) or infer_run_dir(
            summary_path, dataset, args.shift_model, seed,
            shift_row["max_nodes"], shift_row["epochs"], shift_row["adj_source"],
        )
        base = collect_predictions(base_dir, device)
        shift = collect_predictions(shift_dir, device)
        if base["target"].shape != shift["target"].shape:
            raise ValueError(f"Shape mismatch for {dataset} seed={seed}.")
        has_event = dataset in EVENT_COLUMNS
        masks = slice_masks(base["event"], base["shift"], has_event)
        base_sample_err = sample_abs_error(base["pred"], base["target"])
        shift_sample_err = sample_abs_error(shift["pred"], shift["target"])
        for slice_name, mask in masks.items():
            count = int(mask.sum())
            base_metrics = metric_values(base["pred"], base["target"], mask)
            shift_metrics = metric_values(shift["pred"], shift["target"], mask)
            diff = base_sample_err[mask] - shift_sample_err[mask]
            wilcoxon_result = wilcoxon_paired_test(
                shift_sample_err[mask],
                base_sample_err[mask],
            )
            ci_low, ci_high = bootstrap_ci(diff, args.bootstrap, rng)
            rel_mae = (
                (base_metrics["mae"] - shift_metrics["mae"]) / base_metrics["mae"] * 100.0
                if np.isfinite(base_metrics["mae"]) and base_metrics["mae"] > 0 else math.nan
            )
            metric_rows.append({
                "dataset": dataset,
                "seed": seed,
                "slice": slice_name,
                "count": count,
                "sample_count": count,
                "base_mae": base_metrics["mae"],
                "shift_mae": shift_metrics["mae"],
                "rel_mae_improve_pct": rel_mae,
                "base_rmse": base_metrics["rmse"],
                "shift_rmse": shift_metrics["rmse"],
                "base_smape": base_metrics["smape"],
                "shift_smape": shift_metrics["smape"],
                "base_wape": base_metrics["wape"],
                "shift_wape": shift_metrics["wape"],
                "mean_abs_error_diff": float(diff.mean()) if diff.size else math.nan,
                "diff_ci95_low": ci_low,
                "diff_ci95_high": ci_high,
                "paired_t_pvalue": paired_t_pvalue(diff),
                "wilcoxon_p": wilcoxon_result["p_value"],
                "wilcoxon_p_adj": math.nan,
                "effect_size_r": wilcoxon_result["effect_size_r"],
                "median_delta_mae": wilcoxon_result["median_diff"],
            })

    adjusted_p_values = holm_bonferroni_correction(
        [float(row["wilcoxon_p"]) for row in metric_rows]
    )
    for row, adjusted_p in zip(metric_rows, adjusted_p_values):
        row["wilcoxon_p_adj"] = adjusted_p

    metrics_path = output_dir / "weather_shift_slice_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    report_path = output_dir / "weather_shift_slice_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Weather And Shift Slice Report\n\n")
        f.write(f"Summary source: `{summary_path}`\n\n")
        for slice_name in [
            "all_test",
            "high_shift_top25",
            "high_shift_top10",
            "low_shift_bottom50",
            "weather_nonzero",
            "weather_top25",
            "weather_top10",
        ]:
            selected = [row for row in metric_rows if row["slice"] == slice_name and row["count"] > 0]
            if not selected:
                continue
            f.write(f"## {slice_name}\n\n")
            f.write(
                "| Dataset | Seeds | Samples | Base MAE | Shift MAE | "
                "Rel MAE Improve | Mean Diff CI | Min Adj p |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---|---:|\n")
            for dataset in sorted({row["dataset"] for row in selected}):
                ds_rows = [row for row in selected if row["dataset"] == dataset]
                base_mae = float(np.mean([row["base_mae"] for row in ds_rows]))
                shift_mae = float(np.mean([row["shift_mae"] for row in ds_rows]))
                rel = (base_mae - shift_mae) / base_mae * 100.0 if base_mae > 0 else math.nan
                ci_low = float(np.mean([row["diff_ci95_low"] for row in ds_rows]))
                ci_high = float(np.mean([row["diff_ci95_high"] for row in ds_rows]))
                p_adj_values = [
                    float(row["wilcoxon_p_adj"])
                    for row in ds_rows
                    if math.isfinite(float(row["wilcoxon_p_adj"]))
                ]
                min_p_adj = min(p_adj_values) if p_adj_values else math.nan
                sample_count = int(np.sum([row["sample_count"] for row in ds_rows]))
                f.write(
                    f"| {dataset} | {len(ds_rows)} | {sample_count} | {base_mae:.4f} | "
                    f"{shift_mae:.4f} | {rel:+.2f}% | [{ci_low:.5f}, {ci_high:.5f}] "
                    f"| {min_p_adj:.3g} |\n"
                )
            f.write("\n")
    significant = [
        row for row in metric_rows
        if math.isfinite(float(row["wilcoxon_p_adj"]))
        and float(row["wilcoxon_p_adj"]) < 0.05
        and math.isfinite(float(row["median_delta_mae"]))
        and float(row["median_delta_mae"]) < 0
    ]
    print("\nSignificant MAE improvements after Holm-Bonferroni correction:")
    if not significant:
        print("  none")
    else:
        print("  dataset | seed | slice | adj_p | effect_size_r | median_delta_mae")
        for row in significant:
            print(
                f"  {row['dataset']} | {row['seed']} | {row['slice']} | "
                f"{float(row['wilcoxon_p_adj']):.4g} | "
                f"{float(row['effect_size_r']):.4f} | "
                f"{float(row['median_delta_mae']):.6f}"
            )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
