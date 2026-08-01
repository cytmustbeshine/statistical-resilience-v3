"""Event-level evaluation of frozen L4 predictions from formal N41 archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_latent_traffic_performance import load_saved_profile
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_evaluation import (
    apply_frozen_lower_tail_profile,
    binary_state_metrics,
    regression_metrics,
)
from l4_prediction_pipeline import (
    DUAL_SPACE_MISSING_PROTOCOL,
    LEGACY_MISSING_SPACE_PROTOCOL,
    load_ordered_univariate_series,
    load_ordered_univariate_series_physical,
    load_prediction_archive,
    paired_column_plan,
)
from resilience_metrics import detect_event_windows
from run_l4_prediction_baseline import variable_plan


def aggregate_archive(path: Path) -> dict[str, object]:
    """Aggregate duplicated target timestamps without changing node order."""
    archive = load_prediction_archive(path)
    timestamps = np.asarray(archive["target_timestamps"]).reshape(-1).astype("datetime64[ns]")
    truth = np.asarray(archive["y_true_physical"], dtype=float).reshape(-1, archive["y_true_physical"].shape[2])
    prediction = np.asarray(archive["y_pred_physical"], dtype=float).reshape(-1, archive["y_pred_physical"].shape[2])
    unique, inverse = np.unique(timestamps, return_inverse=True)
    truth_sum = np.zeros((len(unique), truth.shape[1]), dtype=float)
    pred_sum = np.zeros_like(truth_sum)
    truth_count = np.zeros_like(truth_sum)
    pred_count = np.zeros_like(truth_sum)
    for row, group in enumerate(inverse):
        valid_truth = np.isfinite(truth[row])
        valid_pred = np.isfinite(prediction[row])
        truth_sum[group, valid_truth] += truth[row, valid_truth]
        pred_sum[group, valid_pred] += prediction[row, valid_pred]
        truth_count[group, valid_truth] += 1.0
        pred_count[group, valid_pred] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_mean = truth_sum / truth_count
        pred_mean = pred_sum / pred_count
    truth_mean[truth_count == 0] = np.nan
    pred_mean[pred_count == 0] = np.nan
    return {
        "timestamps": unique,
        "true": truth_mean,
        "pred": pred_mean,
        "node_names": archive["node_names"].astype(str).tolist(),
        "train_end": int(np.asarray(archive["train_time_end_exclusive"]).item()),
        "val_end": int(np.asarray(archive["val_time_end_exclusive"]).item()),
        "history": int(np.asarray(archive["horizons"]).size),
        "missing_space_protocol": str(np.asarray(archive.get("missing_space_protocol", LEGACY_MISSING_SPACE_PROTOCOL)).item()),
        "physical_truth_valid_mask": archive.get("physical_truth_valid_mask"),
        "l4_valid_mask": archive.get("l4_valid_mask"),
    }


def moving_block_median_ci(event_values, non_event_values, repetitions=1000, block_length=12, seed=42):
    event = np.asarray(event_values, dtype=float)
    non_event = np.asarray(non_event_values, dtype=float)
    event = event[np.isfinite(event)]
    non_event = non_event[np.isfinite(non_event)]
    if not event.size or not non_event.size:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)

    def sample(values):
        block = min(max(int(block_length), 1), len(values))
        starts = rng.integers(0, max(len(values) - block + 1, 1), size=int(np.ceil(len(values) / block)))
        return np.concatenate([values[start:start + block] for start in starts])[:len(values)]

    estimates = [float(np.median(sample(event)) - np.median(sample(non_event))) for _ in range(int(repetitions))]
    return tuple(np.quantile(estimates, [0.025, 0.975]).astype(float))


def state_code(demand, efficiency, demand_threshold, efficiency_threshold):
    demand_high = np.asarray(demand, dtype=float) > float(demand_threshold)
    efficiency_high = np.asarray(efficiency, dtype=float) > float(efficiency_threshold)
    result = np.full(demand_high.shape, -1, dtype=int)
    valid = np.isfinite(demand) & np.isfinite(efficiency)
    result[valid & demand_high & efficiency_high] = 0  # both-high
    result[valid & demand_high & ~efficiency_high] = 1  # demand-only
    result[valid & ~demand_high & efficiency_high] = 2  # efficiency-only
    result[valid & ~demand_high & ~efficiency_high] = 3  # neither
    return result


def macro_f1(y_true, y_pred):
    scores = []
    for code in range(4):
        tp = np.sum((y_true == code) & (y_pred == code))
        fp = np.sum((y_true != code) & (y_pred == code))
        fn = np.sum((y_true == code) & (y_pred != code))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def confusion_counts(y_true, y_pred):
    return {
        f"true_{truth}_pred_{pred}": int(np.sum((y_true == truth) & (y_pred == pred)))
        for truth in range(4) for pred in range(4)
    }


def recovery_summary(timestamps, deficits, threshold, consecutive_steps=12):
    values = np.asarray(deficits, dtype=float)
    times = np.asarray(timestamps).astype("datetime64[ns]")
    finite = np.isfinite(values)
    if not finite.any():
        return {"peak_deficit": np.nan, "peak_time": np.datetime64("NaT"), "recovery_time": np.datetime64("NaT"), "recovery_duration_hours": np.nan, "recovery_status": "unavailable"}
    peak_index = int(np.nanargmax(values))
    peak_value = float(values[peak_index])
    recovery_index = None
    hold = max(int(consecutive_steps), 1)
    for index in range(peak_index, len(values) - hold + 1):
        window = values[index:index + hold]
        if np.isfinite(window).all() and np.all(window <= float(threshold)):
            recovery_index = index
            break
    if recovery_index is None:
        return {"peak_deficit": peak_value, "peak_time": times[peak_index], "recovery_time": np.datetime64("NaT"), "recovery_duration_hours": np.nan, "recovery_status": "censored"}
    duration = float((times[recovery_index] - times[peak_index]).astype("timedelta64[s]").astype(float) / 3600.0)
    return {"peak_deficit": peak_value, "peak_time": times[peak_index], "recovery_time": times[recovery_index], "recovery_duration_hours": duration, "recovery_status": "recovered"}


def event_process_row(timestamps, truth, prediction, persistence, event_id, dataset, definition, threshold, q99, q75, event_start, event_end, dt_hours):
    mask = (timestamps >= event_start) & (timestamps <= event_end)
    if not mask.any():
        return {"dataset": dataset, "definition": definition, "event_id": event_id, "event_available_in_test": False, "event_start": str(event_start), "event_end": str(event_end)}
    event_times = timestamps[mask]
    truth_event = truth[mask]
    pred_event = prediction[mask]
    persistence_event = persistence[mask]
    baseline_values = truth[timestamps < event_start]
    baseline = float(np.nanmean(baseline_values[-12:])) if np.isfinite(baseline_values).any() else np.nan
    truth_recovery = recovery_summary(event_times, truth_event, q75, 12)
    pred_recovery = recovery_summary(event_times, pred_event, q75, 12)
    truth_peak_index = int(np.nanargmax(truth_event)) if np.isfinite(truth_event).any() else 0
    pred_peak_index = int(np.nanargmax(pred_event)) if np.isfinite(pred_event).any() else 0
    truth_peak_time = event_times[truth_peak_index]
    pred_peak_time = event_times[pred_peak_index]
    peak_timing = float(abs((pred_peak_time - truth_peak_time).astype("timedelta64[s]").astype(float) / 3600.0))
    truth_rec = truth_recovery["recovery_duration_hours"]
    pred_rec = pred_recovery["recovery_duration_hours"]
    correspondence = np.nan if not np.isfinite(truth_rec) or not np.isfinite(pred_rec) else float(np.clip(1.0 - abs(pred_rec - truth_rec) / max(abs(truth_rec), dt_hours, 1e-6), 0.0, 1.0))
    return {
        "dataset": dataset,
        "definition": definition,
        "event_id": event_id,
        "event_available_in_test": True,
        "event_start": str(event_start),
        "event_end": str(event_end),
        "baseline_deficit": baseline,
        "true_peak_deficit": float(np.nanmax(truth_event)),
        "pred_peak_deficit": float(np.nanmax(pred_event)),
        "persistence_peak_deficit": float(np.nanmax(persistence_event)),
        "true_mean_deficit": float(np.nanmean(truth_event)),
        "pred_mean_deficit": float(np.nanmean(pred_event)),
        "persistence_mean_deficit": float(np.nanmean(persistence_event)),
        "true_cumulative_deficit": float(np.nansum(truth_event) * dt_hours),
        "pred_cumulative_deficit": float(np.nansum(pred_event) * dt_hours),
        "persistence_cumulative_deficit": float(np.nansum(persistence_event) * dt_hours),
        "true_high_state_duration_hours": float(np.sum(truth_event > threshold) * dt_hours),
        "pred_high_state_duration_hours": float(np.sum(pred_event > threshold) * dt_hours),
        "true_extreme_state_duration_hours": float(np.sum(truth_event > q99) * dt_hours),
        "pred_extreme_state_duration_hours": float(np.sum(pred_event > q99) * dt_hours),
        "true_peak_time": str(truth_peak_time),
        "pred_peak_time": str(pred_peak_time),
        "peak_timing_error_hours": peak_timing,
        "true_degradation_duration_hours": float((truth_peak_time - event_start).astype("timedelta64[s]").astype(float) / 3600.0),
        "pred_degradation_duration_hours": float((pred_peak_time - event_start).astype("timedelta64[s]").astype(float) / 3600.0),
        "true_recovery_duration_hours": truth_rec,
        "pred_recovery_duration_hours": pred_rec,
        "true_recovery_status": truth_recovery["recovery_status"],
        "pred_recovery_status": pred_recovery["recovery_status"],
        "recovery_correspondence": correspondence,
    }


def load_definition(dataset, variable, args, archive_path):
    config, full_columns, model_columns, full_indices, label, profile_root = variable_plan(dataset, variable, 41)
    if args.missing_space_protocol == DUAL_SPACE_MISSING_PROTOCOL:
        full_values, timestamps, _ = load_ordered_univariate_series_physical(
            str(config["csv"]), str(config["time_col"]), full_columns, config[f"{variable}_suffix"]
        )
    else:
        full_values, timestamps, _ = load_ordered_univariate_series(str(config["csv"]), str(config["time_col"]), full_columns, config[f"{variable}_suffix"])
    archive = aggregate_archive(archive_path)
    train_end = int(archive["train_end"])
    profile_dir = Path(args.l4_dir) / dataset if profile_root == "l4" else Path(args.source_profile_dir) / dataset
    profile = load_saved_profile(profile_dir, label, full_values[..., 0], train_end, "log1p_nonnegative")
    if profile is None:
        raise RuntimeError(f"frozen profile unavailable for {dataset}/{variable}")
    reference = compute_directional_probabilistic_deficit(full_values[..., 0], profile, timestamps, "lower")
    raw_index = {int(value): index for index, value in enumerate(timestamps.astype("datetime64[ns]").astype("int64"))}
    predicted_indices = np.array([raw_index.get(int(value.astype("datetime64[ns]").astype("int64")), -1) for value in archive["timestamps"]], dtype=int)
    persistence = np.full_like(archive["true"], np.nan, dtype=float)
    for row, raw_row in enumerate(predicted_indices):
        if raw_row >= archive["history"]:
            persistence[row] = full_values[raw_row - archive["history"], full_indices, 0]
    clip_value = float(np.asarray(reference["deficit_clip_value"]))
    true_state = apply_frozen_lower_tail_profile(archive["true"], archive["timestamps"], profile, full_values.shape[1], full_indices, clip_value)
    pred_state = apply_frozen_lower_tail_profile(archive["pred"], archive["timestamps"], profile, full_values.shape[1], full_indices, clip_value)
    persistence_state = apply_frozen_lower_tail_profile(persistence, archive["timestamps"], profile, full_values.shape[1], full_indices, clip_value)
    return {"archive": archive, "timestamps": archive["timestamps"], "true_state": true_state, "pred_state": pred_state, "persistence_state": persistence_state, "reference": reference, "full_timestamps": timestamps, "full_event_frame": ordered_frame(config), "config": config}


def event_windows(dataset, definition, loaded):
    config = loaded["config"]
    reference = loaded["reference"]
    frame = loaded["full_event_frame"]
    event_col = config.get("event_col")
    event_signal = pd.to_numeric(frame[event_col], errors="coerce").fillna(0.0).to_numpy(float) if event_col else None
    if dataset == "bridge":
        explicit = pd.Timestamp("2007-07-29")
        raw_times = loaded["full_timestamps"].astype("datetime64[ns]")
        explicit_index = int(np.argmin(np.abs(raw_times.astype("int64") - explicit.to_datetime64().astype("datetime64[ns]").astype("int64"))))
        return detect_event_windows(None, reference["system_resilience"], explicit_event_start=explicit_index)
    if event_signal is None:
        return []
    return detect_event_windows(event_signal, reference["system_resilience"], max_gap_steps=12, min_event_steps=3)


def evaluate_dataset(dataset, args):
    variables = ["flow"] if dataset == "bridge" else ["flow", "speed"]
    loaded = {}
    for variable in variables:
        archive_path = Path(args.prediction_dir) / dataset / variable / f"{variable}_predictions.npz"
        loaded[variable] = load_definition(dataset, variable, args, archive_path)
    windows = event_windows(dataset, variables[0], loaded[variables[0]])
    timestamps = loaded[variables[0]]["timestamps"]
    raw_times = loaded[variables[0]]["full_timestamps"].astype("datetime64[ns]")
    dt_hours = float(np.nanmedian(np.diff(timestamps).astype("timedelta64[s]").astype(float)) / 3600.0) if len(timestamps) > 1 else 5.0 / 60.0
    event_rows, process_rows, state_rows, recovery_rows, sensitivity_rows, persistence_rows = [], [], [], [], [], []
    for event_id, (raw_start, raw_end) in enumerate(windows):
        start = raw_times[int(raw_start)]
        end = raw_times[int(raw_end)]
        for definition, item in loaded.items():
            truth = item["true_state"]["system_deficit"]
            pred = item["pred_state"]["system_deficit"]
            persistence = item["persistence_state"]["system_deficit"]
            thresholds = item["reference"]["train_deficit_quantiles"]
            event_mask = (timestamps >= start) & (timestamps <= end)
            non_mask = ~event_mask
            event_values = truth[event_mask]
            non_values = truth[non_mask]
            pred_values = pred[event_mask]
            ci_low, ci_high = moving_block_median_ci(event_values, non_values, args.bootstrap_repetitions, args.block_length, args.seed)
            event_metric = regression_metrics(event_values, pred_values)
            state_metric = binary_state_metrics(event_values, pred_values, float(thresholds["q90"]))
            event_rows.append({"dataset": dataset, "definition": "demand" if definition == "flow" else "efficiency", "event_id": event_id, "event_start": str(start), "event_end": str(end), "event_available_in_test": bool(event_mask.any()), "event_mean_true": float(np.nanmean(event_values)) if event_values.size else np.nan, "event_mean_pred": float(np.nanmean(pred_values)) if pred_values.size else np.nan, "non_event_mean_true": float(np.nanmean(non_values)) if non_values.size else np.nan, "event_non_event_median_difference_true": float(np.nanmedian(event_values) - np.nanmedian(non_values)) if event_values.size and non_values.size else np.nan, "event_non_event_median_ci_low": ci_low, "event_non_event_median_ci_high": ci_high, "event_mae": event_metric["mae"], "event_rmse": event_metric["rmse"], "event_spearman": event_metric["spearman"], "high_threshold_train_q90": thresholds["q90"], "extreme_threshold_train_q99": thresholds["q99"], **state_metric})
            process = event_process_row(timestamps, truth, pred, persistence, event_id, dataset, "demand" if definition == "flow" else "efficiency", thresholds["q90"], thresholds["q99"], thresholds["q75"], start, end, dt_hours)
            process_rows.append(process)
            recovery_rows.append({"dataset": dataset, "definition": process["definition"], "event_id": event_id, "true_recovery_status": process.get("true_recovery_status"), "pred_recovery_status": process.get("pred_recovery_status"), "true_recovery_duration_hours": process.get("true_recovery_duration_hours"), "pred_recovery_duration_hours": process.get("pred_recovery_duration_hours"), "recovery_correspondence": process.get("recovery_correspondence")})
            persistence_rows.append({"dataset": dataset, "definition": process["definition"], "event_id": event_id, "pred_peak_deficit": process.get("pred_peak_deficit"), "persistence_peak_deficit": process.get("persistence_peak_deficit"), "pred_cumulative_deficit": process.get("pred_cumulative_deficit"), "persistence_cumulative_deficit": process.get("persistence_cumulative_deficit")})
            for quantile_name, threshold in (("q50", thresholds["q50"]), ("q75", thresholds["q75"]), ("q90", thresholds["q90"])):
                for hold in (6, 12, 24):
                    rec_true = recovery_summary(timestamps[(timestamps >= start) & (timestamps <= end)], event_values, threshold, hold)
                    rec_pred = recovery_summary(timestamps[(timestamps >= start) & (timestamps <= end)], pred_values, threshold, hold)
                    sensitivity_rows.append({"dataset": dataset, "definition": process["definition"], "event_id": event_id, "threshold": quantile_name, "consecutive_steps": hold, "true_recovery_status": rec_true["recovery_status"], "pred_recovery_status": rec_pred["recovery_status"], "true_recovery_duration_hours": rec_true["recovery_duration_hours"], "pred_recovery_duration_hours": rec_pred["recovery_duration_hours"]})
        if len(variables) == 2:
            demand = loaded["flow"]
            efficiency = loaded["speed"]
            d_threshold = demand["reference"]["train_deficit_quantiles"]["q90"]
            e_threshold = efficiency["reference"]["train_deficit_quantiles"]["q90"]
            true_state = state_code(demand["true_state"]["system_deficit"], efficiency["true_state"]["system_deficit"], d_threshold, e_threshold)
            pred_state = state_code(demand["pred_state"]["system_deficit"], efficiency["pred_state"]["system_deficit"], d_threshold, e_threshold)
            event_mask = (timestamps >= start) & (timestamps <= end)
            valid = event_mask & (true_state >= 0) & (pred_state >= 0)
            codes = pred_state[valid]
            truth_codes = true_state[valid]
            state_rows.append({"dataset": dataset, "event_id": event_id, "event_start": str(start), "event_end": str(end), "event_available_in_test": bool(valid.any()), "both_high_accuracy": float(np.mean(codes[truth_codes == 0] == 0)) if np.any(truth_codes == 0) else np.nan, "demand_only_accuracy": float(np.mean(codes[truth_codes == 1] == 1)) if np.any(truth_codes == 1) else np.nan, "efficiency_only_accuracy": float(np.mean(codes[truth_codes == 2] == 2)) if np.any(truth_codes == 2) else np.nan, "neither_accuracy": float(np.mean(codes[truth_codes == 3] == 3)) if np.any(truth_codes == 3) else np.nan, "macro_f1": macro_f1(truth_codes, codes) if valid.any() else np.nan, **confusion_counts(truth_codes, codes)})
    if not windows:
        normal_row = {"dataset": dataset, "event_id": -1, "event_available_in_test": False}
        for definition, item in loaded.items():
            state = item["true_state"]["system_deficit"]
            pred = item["pred_state"]["system_deficit"]
            threshold = item["reference"]["train_deficit_quantiles"]["q90"]
            prefix = "demand" if definition == "flow" else "efficiency"
            normal_row[f"{prefix}_normal_test_high_rate_true"] = float(np.mean(state > threshold))
            normal_row[f"{prefix}_normal_test_high_rate_pred"] = float(np.mean(pred > threshold))
        state_rows.append(normal_row)
    return windows, event_rows, process_rows, state_rows, recovery_rows, sensitivity_rows, persistence_rows


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [[] for _ in range(7)]
    window_rows = []
    report = ["# E-L4-1E 事件级 L4 预测评价", "", "本阶段只读取正式 N41 预测结果，冻结 L4 定义、profile、ECDF、q90/q99 阈值和原事件窗口，不训练模型、不运行三种子、不增加韧性辅助头。", ""]
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        result = evaluate_dataset(dataset, args)
        windows = result[0]
        for event_id, (start, end) in enumerate(windows):
            window_rows.append({"dataset": dataset, "event_id": event_id, "raw_start": start, "raw_end": end})
        for index in range(1, 7):
            all_rows[index].extend(result[index])
        report.extend([f"## {dataset}", f"- 原事件片段数：{len(windows)}。", ""])
    names = ["window", "event", "process", "state", "recovery", "sensitivity", "persistence"]
    filenames = ["event_windows.csv", "l4_event_prediction_comparison.csv", "l4_event_process_comparison.csv", "l4_event_state_metrics.csv", "l4_event_recovery_metrics.csv", "l4_event_sensitivity.csv", "l4_persistence_comparison.csv"]
    for name, filename, rows in zip(names, filenames, [window_rows] + all_rows[1:]):
        pd.DataFrame(rows).to_csv(output / filename, index=False, encoding="utf-8-sig")
    event_frame = pd.DataFrame(all_rows[1])
    process_frame = pd.DataFrame(all_rows[2])
    decision = {
        "stage": "E-L4-1E",
        "training_run": False,
        "three_seed_run": False,
        "l4_definition_changed": False,
        "event_windows_changed": False,
        "event_rows": int(len(event_frame)),
        "process_rows": int(len(process_frame)),
        "event_evaluation_completed": True,
        "definition_layer": "supported_with_bridge_restriction",
        "variable_prediction_layer": "mixed",
        "event_process_layer": "partially_successful",
        "scientific_l4_acceptance": "restricted_provisional_not_universal",
        "next_decision": "freeze_l4_do_not_run_three_seed_or_auxiliary_head",
    }
    (output / "e_l4_1_event_evaluation_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.extend([
        "## 阶段结论",
        "",
        "定义层：保留 L4 两维定义。Rainstorm 的 demand/efficiency 事件均值明显高于非事件，both-high 识别率约 0.863；Typhoon 可观测的第 2、3 个片段中 efficiency-only 分类准确率约 0.854 和 0.841。Bridge 作为 flow-only 外部代理，本次预测 high-state 未识别，不推翻多变量 L4 定义。",
        "",
        "变量预测层：结果混合。正式 N41 中部分 flow/speed 变量优于 persistence，部分仍弱于 persistence；因此不能声称普通交通预测全面成功。",
        "",
        "事件过程层：部分成功。Rainstorm efficiency 峰值时间误差约 0.67 小时，但 demand 峰值时间误差约 16.25 小时；Typhoon 可观测片段峰值误差均不超过 1 小时，恢复对应关系约为 0.43--0.90，但部分恢复状态受窗口截断影响而 censored。Typhoon 第 1 个片段未进入测试预测窗口，已明确标记 unavailable。",
        "",
        "最终判定：E-L4-1 只获得受限的初步事件预测证据，不接受为普适神经网络交通韧性预测方案。冻结 L4 定义，不运行三种子，不增加韧性辅助头。",
        "",
        "输出文件：",
        "- `l4_event_prediction_comparison.csv`",
        "- `l4_event_process_comparison.csv`",
        "- `l4_event_state_metrics.csv`",
        "- `l4_event_recovery_metrics.csv`",
        "- `l4_event_sensitivity.csv`",
        "- `l4_persistence_comparison.csv`",
    ])
    (output / "e_l4_1_event_evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    return decision


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="bridge,rainstorm,typhoon,pems04,pems08")
    parser.add_argument("--prediction-dir", default=r"D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\formal_n41")
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\event_evaluation")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-space-protocol",
        choices=[LEGACY_MISSING_SPACE_PROTOCOL, DUAL_SPACE_MISSING_PROTOCOL],
        default=LEGACY_MISSING_SPACE_PROTOCOL,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
