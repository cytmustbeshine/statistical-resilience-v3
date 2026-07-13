"""Event-slice analysis for D-STSGCN experiment outputs."""

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
    resolve_feature_index,
    split_dataset,
)


DATASET_TYPES = {
    "rainstorm": "weather-event",
    "typhoon": "weather-event",
    "bridge": "structural-event",
    "pems04": "normal",
    "pems08": "normal",
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


def model_from_config(args: dict, data_shape: tuple[int, int, int], event_feature_idx: int | None) -> DSTSGCN:
    return DSTSGCN(
        num_nodes=data_shape[1],
        input_dim=data_shape[2],
        output_dim=1,
        horizon=int(args["horizon"]),
        hidden_dim=int(args["hidden_dim"]),
        num_blocks=int(args["num_blocks"]),
        graph_learner_type=args["graph_learner_type"],
        fusion_mode=args["fusion_mode"],
        fusion_type=args["fusion_type"],
        dynamic_top_k=int(args["dynamic_top_k"]) if int(args["dynamic_top_k"]) > 0 else None,
        quality_gate_bias=float(args["quality_gate_bias"]),
        matrix_hidden_dim=int(args["matrix_hidden_dim"]),
        static_adj_is_normalized=False,
        use_learned_static=bool(args["use_learned_static"]),
        num_diffusion_steps=int(args["num_diffusion_steps"]),
        event_dim=1 if event_feature_idx is not None else 0,
        quantiles=args.get("quantiles"),
        graph_residual_init=float(args["graph_residual_init"]),
        stat_residual_init=float(args["stat_gate_residual_init"]),
        stat_residual_max=float(args["stat_gate_residual_max"]),
    )


def load_run(run_dir: Path, device: torch.device):
    config_path = run_dir / "run_config.json"
    weight_path = run_dir / "best_dstsgcn.pt"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data, node_names, feature_names = load_wide_traffic_csv(
        cfg["csv"],
        value_suffix=parse_optional_suffix(cfg["value_suffix"]),
        time_col=cfg["time_col"],
        max_nodes=int(cfg["max_nodes"]),
        extra_feature_cols=parse_csv_list(cfg.get("extra_feature_cols", "")),
        node_feature_suffixes=parse_csv_list(cfg.get("node_feature_suffixes", "")),
        add_time_features=bool(cfg["add_time_features"]),
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

    scaler = SplitScaler(num_traffic_features=1)
    train_cut = int(len(data) * 0.6)
    scaler.fit(data[:train_cut])
    data = scaler.transform(data)

    train_traffic = data[:train_cut, :, 0]
    train_traffic_mean = torch.tensor(float(np.mean(train_traffic)), dtype=torch.float32, device=device)
    train_traffic_std = torch.tensor(max(float(np.std(train_traffic)), 1e-6), dtype=torch.float32, device=device)

    adj_data = data[:train_cut] if bool(cfg["static_adj_train_only"]) else data
    static_adj_np = build_static_adjacency(
        adj_data,
        node_names,
        source=cfg["adj_source"],
        corr_threshold=float(cfg["corr_threshold"]),
        corr_method=cfg["corr_method"],
        corr_shrinkage_lambda=float(cfg["corr_shrinkage_lambda"]),
        auto_corr_shrinkage=bool(cfg["auto_corr_shrinkage"]),
        adj_path=cfg.get("adj_path") or None,
        road_weight=float(cfg["road_weight"]),
        granger_lag=int(cfg["granger_lag"]),
        granger_p_threshold=float(cfg["granger_p_threshold"]),
        granger_cache_path=cfg.get("granger_cache_path") or None,
        from_col=cfg["adj_from_col"],
        to_col=cfg["adj_to_col"],
        weight_col=cfg.get("adj_weight_col") or None,
        link_id_col=cfg.get("adj_link_id_col") or None,
        directed=not bool(cfg["adj_undirected"]),
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
    loader = DataLoader(test_set, batch_size=int(cfg["batch_size"]), shuffle=False)

    model = model_from_config(cfg, data.shape, event_feature_idx).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return cfg, model, loader, static_adj, None, event_feature_idx, train_traffic_mean, train_traffic_std


@torch.no_grad()
def collect_errors(run_dir: Path, device: torch.device) -> dict[str, np.ndarray]:
    cfg, model, loader, static_adj, road_adj, event_idx, train_mean, train_std = load_run(run_dir, device)
    fusion_type = cfg["fusion_type"]
    uses_shift = fusion_type in {"quality_stat", "quality_stat_reg", "quality_stat_residual", "quality_stat_safe"}
    errors = []
    event_values = []
    shift_values = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        event_signal = extract_event_signal(x, event_idx)
        stat_signal = extract_shift_signal(x, train_mean, train_std) if uses_shift else None
        pred = model(
            x,
            static_adj,
            road_adj,
            event_signal=event_signal,
            stat_signal=stat_signal,
        )
        abs_err = torch.abs(pred - y).mean(dim=(1, 2, 3))
        errors.append(abs_err.detach().cpu().numpy())
        if event_signal is None:
            event_values.append(np.zeros(abs_err.numel(), dtype=np.float32))
        else:
            event_values.append(event_signal.reshape(-1).detach().cpu().numpy())
        if stat_signal is None:
            raw_shift = extract_shift_signal(x, train_mean, train_std)
            shift_values.append(raw_shift.reshape(-1).detach().cpu().numpy())
        else:
            shift_values.append(stat_signal.reshape(-1).detach().cpu().numpy())
    return {
        "error": np.concatenate(errors),
        "event": np.concatenate(event_values),
        "shift": np.concatenate(shift_values),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    boots = []
    n = values.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(float(values[idx].mean()))
    return tuple(np.percentile(boots, [2.5, 97.5]).tolist())


def paired_t_pvalue(diff: np.ndarray) -> float:
    n = diff.size
    if n < 2:
        return math.nan
    sd = float(diff.std(ddof=1))
    if sd <= 1e-12:
        return 0.0 if abs(float(diff.mean())) > 1e-12 else 1.0
    t = float(diff.mean()) / (sd / math.sqrt(n))
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        return math.nan


def summarize_pair(
    dataset: str,
    seed: int,
    base: dict[str, np.ndarray],
    shift: dict[str, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if base["error"].shape != shift["error"].shape:
        raise ValueError(f"Mismatched sample counts for {dataset} seed {seed}.")
    diff = base["error"] - shift["error"]
    event = base["event"]
    raw_shift = base["shift"]
    rows = []
    event_positive = event > 0
    event_q75 = event >= np.quantile(event, 0.75)
    event_q90 = event >= np.quantile(event, 0.90)
    shift_q75 = raw_shift >= np.quantile(raw_shift, 0.75)
    shift_q90 = raw_shift >= np.quantile(raw_shift, 0.90)
    slices = {
        "all_test": np.ones_like(diff, dtype=bool),
        "event_positive": event_positive,
        "event_top25": event_q75,
        "event_top10": event_q90,
        "shift_top25": shift_q75,
        "shift_top10": shift_q90,
    }
    for name, mask in slices.items():
        if mask.sum() == 0:
            continue
        d = diff[mask]
        b = base["error"][mask]
        s = shift["error"][mask]
        ci_low, ci_high = bootstrap_ci(d, n_boot, rng)
        rows.append(
            {
                "dataset": dataset,
                "dataset_type": DATASET_TYPES.get(dataset, ""),
                "seed": seed,
                "slice": name,
                "n_windows": int(mask.sum()),
                "base_mae": float(b.mean()),
                "shift_mae": float(s.mean()),
                "diff_base_minus_shift": float(d.mean()),
                "relative_improvement_pct": float(d.mean() / max(float(b.mean()), 1e-12) * 100.0),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "paired_t_pvalue": paired_t_pvalue(d),
                "event_mean": float(event[mask].mean()),
                "shift_score_mean": float(raw_shift[mask].mean()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=r"D:\TrafficGNN\outputs\experiments_mainline_all_n41_e8_seeds\all_plus_typhoon_seeds\experiment_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\TrafficGNN\outputs\analysis_event_slices_mainline",
    )
    parser.add_argument("--datasets", default="rainstorm,typhoon")
    parser.add_argument("--base-model", default="quality_v2")
    parser.add_argument("--compare-model", default="quality_stat_shift")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_datasets = set(parse_csv_list(args.datasets))
    device = torch.device(args.device)
    rng = np.random.default_rng(42)

    rows = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in target_datasets and r["model"] in {args.base_model, args.compare_model}:
                rows.append(r)

    by = {(r["dataset"], r["model"], int(r["seed"])): Path(r["output_dir"]) for r in rows}
    all_rows: list[dict[str, object]] = []
    for dataset in sorted(target_datasets):
        seeds = sorted(
            int(r["seed"])
            for r in rows
            if r["dataset"] == dataset and r["model"] == args.base_model
        )
        for seed in seeds:
            base_dir = by.get((dataset, args.base_model, seed))
            shift_dir = by.get((dataset, args.compare_model, seed))
            if base_dir is None or shift_dir is None:
                continue
            print(f"[Analysis] {dataset} seed={seed}")
            base = collect_errors(base_dir, device)
            shift = collect_errors(shift_dir, device)
            all_rows.extend(summarize_pair(dataset, seed, base, shift, args.bootstrap, rng))

    out_csv = output_dir / "event_slice_summary.csv"
    if all_rows:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in all_rows:
        grouped.setdefault((str(row["dataset"]), str(row["slice"])), []).append(row)

    report_lines = ["# Event Slice Analysis", ""]
    report_lines.append(
        "| dataset | slice | n(avg) | base MAE | shift MAE | rel improve | diff CI | p(avg) |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for (dataset, slice_name), items in sorted(grouped.items()):
        base_mae = np.mean([float(x["base_mae"]) for x in items])
        shift_mae = np.mean([float(x["shift_mae"]) for x in items])
        rel = np.mean([float(x["relative_improvement_pct"]) for x in items])
        ci_low = np.mean([float(x["bootstrap_ci_low"]) for x in items])
        ci_high = np.mean([float(x["bootstrap_ci_high"]) for x in items])
        pval = np.nanmean([float(x["paired_t_pvalue"]) for x in items])
        n_avg = np.mean([float(x["n_windows"]) for x in items])
        report_lines.append(
            f"| {dataset} | {slice_name} | {n_avg:.0f} | {base_mae:.4f} | "
            f"{shift_mae:.4f} | {rel:+.2f}% | [{ci_low:.5f}, {ci_high:.5f}] | {pval:.3g} |"
        )

    out_md = output_dir / "event_slice_report.md"
    out_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
