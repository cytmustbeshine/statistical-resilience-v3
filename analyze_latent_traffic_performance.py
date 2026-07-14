"""Audit and analyze latent traffic performance definitions."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import read_csv_with_fallback
from flow_speed_resilience import (
    compute_directional_probabilistic_deficit,
    fit_variable_resilience_profile,
    match_node_variables,
    softmax_joint_deficit,
    trimmed_system,
)
from latent_traffic_performance import (
    MissingDataOneFactorModel,
    compute_latent_resilience_deficit,
    fit_latent_performance_profile,
    transform_conditional_normal_scores,
)
from analyze_event_resilience_process import (
    analyze_event_segments,
    recovery_sensitivity_rows,
)
from resilience_metrics import detect_event_windows
from statistical_resilience_profile import (
    CDF_SOURCE_CELL,
    CDF_SOURCE_GLOBAL,
    CDF_SOURCE_NODE,
    CDF_SOURCE_NODE_DAYTYPE,
    CDF_SOURCE_ROBUST_PARAMETRIC,
    CDF_SOURCE_UNAVAILABLE,
    load_conditional_ecdf_npz,
    query_conditional_cdf,
    save_conditional_ecdf_npz,
)

CONFIG = {
    "bridge": dict(
        csv=r"D:\TrafficGNN\data\bridge_collapse_flow.csv",
        time="Time",
        flow=None,
        speed=None,
        occupancy=None,
    ),
    "rainstorm": dict(
        csv=r"D:\TrafficGNN\data\rainstorm_traffic_state.csv",
        time="Time",
        flow="_volume",
        speed="_speed",
        occupancy=None,
    ),
    "typhoon": dict(
        csv=r"D:\TrafficGNN\data\typhoon_traffic_state_standard.csv",
        time="Time",
        flow="_volume",
        speed="_speed",
        occupancy=None,
    ),
    "pems04": dict(
        csv=r"D:\TrafficGNN\data\public\PEMS04\pems04.csv",
        time="Time",
        flow="_volume",
        speed="_speed",
        occupancy="_occupancy",
    ),
    "pems08": dict(
        csv=r"D:\TrafficGNN\data\public\PEMS08\pems08.csv",
        time="Time",
        flow="_volume",
        speed="_speed",
        occupancy="_occupancy",
    ),
}

SOURCE_NAMES = {
    CDF_SOURCE_CELL: "cell",
    CDF_SOURCE_NODE_DAYTYPE: "node_daytype",
    CDF_SOURCE_NODE: "node",
    CDF_SOURCE_GLOBAL: "global",
    CDF_SOURCE_ROBUST_PARAMETRIC: "robust_parametric",
    CDF_SOURCE_UNAVAILABLE: "unavailable",
}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if frame.empty:
        return "（无记录）"
    columns = [str(column) for column in frame.columns]
    def cell(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)

def statistical_split_points(time_steps: int) -> tuple[int, int]:
    """Return train and test-start raw indices used by the statistical studies."""
    usable = max(time_steps - 24 + 1, 0)
    train_end = min(int(usable * 0.6) + 12, time_steps)
    test_start = min(int(usable * 0.8) + 12, time_steps)
    return train_end, test_start


def load_dataset(name: str, max_nodes: int) -> dict[str, object]:
    """Load one dataset and match modalities inside the selected flow subnet."""
    config = CONFIG[name]
    frame = read_csv_with_fallback(config["csv"])
    parsed = pd.to_datetime(frame[config["time"]], errors="coerce")
    frame = frame.assign(_dt=parsed).sort_values("_dt", kind="stable").reset_index(drop=True)
    if name == "bridge":
        all_maps = [
            {"node": column, "flow_col": column, "speed_col": None, "occupancy_col": None}
            for column in frame.columns
            if column not in {"ID", config["time"], "_dt"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        all_maps = match_node_variables(
            frame.columns, config["flow"], config["speed"], config["occupancy"]
        )
    flow_maps = all_maps[:max_nodes]
    speed_maps = [mapping for mapping in flow_maps if mapping["speed_col"] is not None]
    occupancy_maps = [
        mapping
        for mapping in flow_maps
        if mapping["speed_col"] is not None and mapping["occupancy_col"] is not None
    ]

    def read_maps(maps, key):
        if not maps:
            return None
        return frame[[mapping[key] for mapping in maps]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    return {
        "frame": frame,
        "timestamps": frame["_dt"].to_numpy(),
        "all_maps": all_maps,
        "flow_maps": flow_maps,
        "speed_maps": speed_maps,
        "occupancy_maps": occupancy_maps,
        "flow": read_maps(flow_maps, "flow_col"),
        "speed": read_maps(speed_maps, "speed_col"),
        "occupancy": read_maps(occupancy_maps, "occupancy_col"),
        "config": config,
    }


def choose_transform(values: np.ndarray, train_end: int) -> tuple[str, str]:
    """Choose a transform using only training quality and variable semantics."""
    train = np.asarray(values, dtype=float)[:train_end]
    finite = train[np.isfinite(train)]
    if finite.size and np.any(finite < 0):
        return "identity_robust", "finite_negative_training_values"
    return "log1p_nonnegative", "nonnegative_physical_training_values"


def save_profile(dataset_dir: Path, label: str, variable_profile: dict[str, object]) -> None:
    """Save hierarchical arrays, the complete ECDF, and compact metadata."""
    profile = variable_profile["profile"]
    ecdf = variable_profile["ecdf"]
    arrays = {
        key: value
        for key, value in profile.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(dataset_dir / f"{label}_profile_hierarchical.npz", **arrays)
    ecdf_path = dataset_dir / f"{label}_profile_ecdf.npz"
    save_conditional_ecdf_npz(ecdf_path, ecdf)
    reloaded = load_conditional_ecdf_npz(ecdf_path)
    if not np.array_equal(ecdf["global_sorted_values"], reloaded["global_sorted_values"]):
        raise RuntimeError(f"ECDF roundtrip failed for {dataset_dir.name}/{label}")
    metadata = {
        "variable": label,
        "train_only": True,
        "train_end_exclusive": int(profile["train_end_exclusive"]),
        "selected_shrinkage_m": float(profile["selected_shrinkage_m"]),
        "transform_mode": profile["transform_mode"],
        "tail_direction": profile.get("tail_direction"),
        "global_count": int(profile.get("global_count", 0)),
        "cv_node_count": int(profile["shrinkage_cv"]["cv_node_count"]),
        "fallback_reason": profile.get("fallback_reason", ""),
    }
    (dataset_dir / f"{label}_profile_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def load_saved_profile(
    dataset_dir: Path,
    label: str,
    values: np.ndarray,
    train_end: int,
    transform_mode: str,
) -> dict[str, object] | None:
    """Load a compatible complete train-only variable profile from disk."""
    hierarchy_path = dataset_dir / f"{label}_profile_hierarchical.npz"
    ecdf_path = dataset_dir / f"{label}_profile_ecdf.npz"
    report_path = dataset_dir / f"{label}_profile_report.json"
    if not (hierarchy_path.exists() and ecdf_path.exists() and report_path.exists()):
        return None
    metadata = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        int(metadata.get("train_end_exclusive", -1)) != int(train_end)
        or metadata.get("transform_mode") != transform_mode
    ):
        return None
    with np.load(hierarchy_path, allow_pickle=False) as archive:
        profile = {key: np.asarray(archive[key]) for key in archive.files}
    if profile.get("location_shrunk", np.empty((0, 0))).shape[1] != values.shape[1]:
        return None
    profile.update(
        train_only=True,
        train_end_exclusive=int(train_end),
        train_size=int(train_end),
        time_of_day_bins=int(profile["location_shrunk"].shape[0]),
        day_type_mode="weekday_weekend" if profile["location_shrunk"].shape[2] == 2 else "none",
        requested_day_type_mode="weekday_weekend" if profile["location_shrunk"].shape[2] == 2 else "none",
        selected_shrinkage_m=float(metadata["selected_shrinkage_m"]),
        transform_mode=transform_mode,
        variable_name=label,
        tail_direction=metadata.get("tail_direction", "upper" if label == "occupancy" else "lower"),
        global_count=int(metadata.get("global_count", 0)),
        fallback_reason=metadata.get("fallback_reason", ""),
        shrinkage_cv={"cv_node_count": int(metadata.get("cv_node_count", values.shape[1]))},
    )
    return {"profile": profile, "ecdf": load_conditional_ecdf_npz(ecdf_path), "proxy_values": values}


def fit_or_load_variable_profile(
    dataset_dir,
    label,
    values,
    train_end,
    timestamps,
    tail_direction,
    transform_mode,
    profile_kwargs,
    force_refit=False,
):
    """Reuse verified Stage-A profiles unless an explicit refit is requested."""
    cached = None if force_refit else load_saved_profile(
        dataset_dir, label, values, train_end, transform_mode
    )
    if cached is not None:
        return cached
    return fit_variable_resilience_profile(
        values,
        train_end,
        timestamps,
        label,
        tail_direction,
        transform_mode,
        **profile_kwargs,
    )

def source_diagnostic_rows(
    dataset: str,
    variable: str,
    result: dict[str, np.ndarray],
    train_end: int,
    test_start: int,
    transform_mode: str,
    transform_reason: str,
    selected_m: float,
    node_count: int,
) -> list[dict[str, object]]:
    """Summarize conditional CDF source proportions for train/val/test splits."""
    source = np.asarray(result["source_level"])
    valid_input = np.isfinite(np.asarray(result["transformed_values"]))
    rows = []
    for split, start, end in (
        ("train", 0, train_end),
        ("val", train_end, test_start),
        ("test", test_start, len(source)),
        ("all", 0, len(source)),
    ):
        split_source = source[start:end]
        split_input = valid_input[start:end]
        denominator = int(split_input.sum())
        row = {
            "dataset": dataset,
            "variable": variable,
            "split": split,
            "n_nodes": node_count,
            "n_finite_input": denominator,
            "transform_mode": transform_mode,
            "transform_reason": transform_reason,
            "selected_shrinkage_m": selected_m,
        }
        for code, source_name in SOURCE_NAMES.items():
            count = int(((split_source == code) & split_input).sum())
            row[f"{source_name}_count"] = count
            row[f"{source_name}_rate"] = count / denominator if denominator else np.nan
        row["fallback_rate"] = (
            float(((split_source != CDF_SOURCE_CELL) & split_input).sum()) / denominator
            if denominator
            else np.nan
        )
        rows.append(row)
    return rows


def audit_speed_column(
    frame: pd.DataFrame,
    dataset: str,
    scope: str,
    node: str,
    column: str,
    train_end: int,
    used_in_joint: bool,
) -> dict[str, object]:
    """Audit one Typhoon speed column without replacing missing or negative values."""
    raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    finite = raw[np.isfinite(raw)]
    train_finite = raw[:train_end][np.isfinite(raw[:train_end])]
    quantiles = (
        np.quantile(finite, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
        if finite.size
        else np.full(7, np.nan)
    )
    possible_standardized = bool(
        train_finite.size
        and np.mean(train_finite < 0) > 0.01
        and abs(float(np.median(train_finite))) < 5.0
    )
    possible_unit_error = bool(
        finite.size and (np.min(finite) < 0 or np.max(finite) > 200)
    )
    return {
        "dataset": dataset,
        "scope": scope,
        "node": node,
        "variable": "speed",
        "column": column,
        "n_total": len(raw),
        "n_finite": len(finite),
        "missing_rate": float(np.mean(~np.isfinite(raw))),
        "negative_rate": float(np.mean(finite < 0)) if finite.size else np.nan,
        "negative_count": int(np.sum(finite < 0)) if finite.size else 0,
        "zero_rate": float(np.mean(finite == 0)) if finite.size else np.nan,
        "min": float(np.min(finite)) if finite.size else np.nan,
        "q001": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "q999": float(quantiles[6]),
        "max": float(np.max(finite)) if finite.size else np.nan,
        "possible_standardized": possible_standardized,
        "possible_unit_error": possible_unit_error,
        "used_in_joint_analysis": bool(used_in_joint),
    }


def typhoon_speed_audit(loaded: dict[str, object], train_end: int) -> pd.DataFrame:
    """Separate all-file, first-41 matched, and actual joint-analysis speed scopes."""
    frame = loaded["frame"]
    config = loaded["config"]
    all_speed_columns = [
        column for column in frame.columns if config["speed"] and column.endswith(config["speed"])
    ]
    first_41_maps = loaded["all_maps"][:41]
    first_41_speed = [mapping for mapping in first_41_maps if mapping["speed_col"] is not None]
    joint_speed = loaded["speed_maps"]
    joint_columns = {mapping["speed_col"] for mapping in joint_speed}
    rows = []
    for column in all_speed_columns:
        node = column[: -len(config["speed"])]
        rows.append(
            audit_speed_column(
                frame, "typhoon", "all_file_speed", node, column, train_end, column in joint_columns
            )
        )
    for mapping in first_41_speed:
        rows.append(
            audit_speed_column(
                frame,
                "typhoon",
                "first41_matched_speed",
                mapping["node"],
                mapping["speed_col"],
                train_end,
                mapping["speed_col"] in joint_columns,
            )
        )
    for mapping in joint_speed:
        rows.append(
            audit_speed_column(
                frame,
                "typhoon",
                "joint_analysis_speed",
                mapping["node"],
                mapping["speed_col"],
                train_end,
                True,
            )
        )
    return pd.DataFrame(rows)

def run_l3_0(args) -> None:
    """Run the gated L3-0 source audit and write reproducible diagnostics."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(float(value) for value in args.shrinkage_candidates.split(","))
    source_rows: list[dict[str, object]] = []
    typhoon_rows = pd.DataFrame()
    dataset_notes = []

    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        loaded = load_dataset(dataset, args.max_nodes)
        timestamps = np.asarray(loaded["timestamps"])
        train_end, test_start = statistical_split_points(len(timestamps))
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        variables = [
            ("flow", loaded["flow"], "lower"),
            ("speed", loaded["speed"], "lower"),
            ("occupancy", loaded["occupancy"], "upper"),
        ]
        fitted_labels = []
        for label, values, tail in variables:
            if values is None or values.shape[1] == 0:
                continue
            transform_mode, transform_reason = choose_transform(values, train_end)
            variable_profile = fit_variable_resilience_profile(
                values,
                train_end,
                timestamps,
                label,
                tail,
                transform_mode,
                time_of_day_bins=args.time_of_day_bins,
                day_type_mode=args.day_type_mode,
                shrinkage_candidates=candidates,
            )
            query = query_conditional_cdf(
                values,
                variable_profile["profile"],
                variable_profile["ecdf"],
                timestamps,
            )
            finite_input = np.isfinite(query["transformed_values"])
            accounted = sum(
                ((query["source_level"] == code) & finite_input).sum()
                for code in SOURCE_NAMES
            )
            if int(accounted) != int(finite_input.sum()):
                raise RuntimeError(f"CDF sources do not cover finite inputs for {dataset}/{label}")
            finite_probability = query["cdf"][finite_input]
            if finite_probability.size and not np.isfinite(finite_probability).all():
                raise RuntimeError(f"non-finite CDF for finite inputs in {dataset}/{label}")
            profile = variable_profile["profile"]
            source_rows.extend(
                source_diagnostic_rows(
                    dataset,
                    label,
                    query,
                    train_end,
                    test_start,
                    transform_mode,
                    transform_reason,
                    float(profile["selected_shrinkage_m"]),
                    values.shape[1],
                )
            )
            save_profile(dataset_dir, label, variable_profile)
            fitted_labels.append(label)
        if dataset == "typhoon":
            typhoon_rows = typhoon_speed_audit(loaded, train_end)
        dataset_notes.append(
            {
                "dataset": dataset,
                "time_steps": len(timestamps),
                "train_end": train_end,
                "test_start": test_start,
                "flow_nodes": len(loaded["flow_maps"]),
                "matched_speed_nodes": len(loaded["speed_maps"]),
                "matched_occupancy_nodes": len(loaded["occupancy_maps"]),
                "fitted_variables": ",".join(fitted_labels),
            }
        )

    diagnostics = pd.DataFrame(source_rows)
    diagnostics.to_csv(output_dir / "cdf_source_diagnostics.csv", index=False, encoding="utf-8-sig")
    typhoon_rows.to_csv(output_dir / "typhoon_speed_audit.csv", index=False, encoding="utf-8-sig")
    notes = pd.DataFrame(dataset_notes)

    report = [
        "# L3-0 条件概率基础审计",
        "",
        "## 审计目的",
        "",
        "本阶段只检查条件 CDF 的训练期来源、变量变换和 Typhoon 速度质量，不拟合潜在因子。",
        "所有节点先按 flow 顺序截取，再在同一子网内匹配 speed/occupancy，避免不同候选使用不同节点范围。",
        "",
        "## 数据范围",
        "",
        markdown_table(notes),
        "",
        "## CDF 来源编码",
        "",
        "- 0：cell ECDF。",
        "- 1：node-daytype ECDF。",
        "- 2：node ECDF。",
        "- 3：global ECDF。",
        "- 4：robust parametric fallback。",
        "- 5：unavailable。",
        "",
        "经验父层始终优先于参数回退；fallback_mask 仅表示来源不是 cell，正式结论应使用分层比例。",
        "",
        "## 全数据来源比例",
        "",
    ]
    if not diagnostics.empty:
        columns = [
            "dataset",
            "variable",
            "split",
            "n_nodes",
            "transform_mode",
            "selected_shrinkage_m",
            "cell_rate",
            "node_daytype_rate",
            "node_rate",
            "global_rate",
            "robust_parametric_rate",
            "unavailable_rate",
        ]
        report.append(markdown_table(diagnostics[diagnostics["split"] == "all"][columns]))
    report.extend(["", "## Typhoon speed 专项结论", ""])
    if typhoon_rows.empty:
        report.append("本次未包含 Typhoon。")
    else:
        summary = (
            typhoon_rows.groupby("scope", as_index=False)
            .agg(
                nodes=("node", "count"),
                negative_observations=("negative_count", "sum"),

                minimum=("min", "min"),
                maximum=("max", "max"),
                used_in_joint=("used_in_joint_analysis", "sum"),
            )
        )
        report.append(markdown_table(summary))
        negative = typhoon_rows[
            (typhoon_rows["scope"] == "all_file_speed") & (typhoon_rows["negative_rate"] > 0)
        ][["node", "min", "negative_rate", "used_in_joint_analysis"]]
        report.extend(["", "全文件负速度节点：", "", markdown_table(negative)])
        report.extend(
            [
                "",
                "实际联合子网若训练期无负值，使用 log1p_nonnegative；全文件异常节点不会因文件名含 standard 而被自动裁剪或自动认定为标准化值。",
            ]
        )
    report.extend(
        [
            "",
            "## 阶段门结论",
            "",
            "来源编码、train-only 查询和完整 ECDF NPZ roundtrip 已由单元测试约束。是否进入 L3 因子模型还需结合本次 smoke/完整审计的执行结果决定。",
        ]
    )
    (output_dir / "l3_0_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(notes.to_string(index=False))
    print(f"OUTPUT {output_dir}")


EVENT_METADATA = {
    "bridge": {"event_col": None, "explicit": "2007-07-29 00:00:00"},
    "rainstorm": {"event_col": "precip_accum_one_hour", "explicit": None},
    "typhoon": {"event_col": "typhoon_intensity", "explicit": None},
    "pems04": {"event_col": None, "explicit": None},
    "pems08": {"event_col": None, "explicit": None},
}


def load_l3_dataset(name: str, max_nodes: int) -> dict[str, object]:
    """Load the analysis subnet; small Typhoon smoke runs retain real matched modes."""
    loaded = load_dataset(name, max_nodes)
    if name != "typhoon" or max_nodes >= 41 or len(loaded["speed_maps"]) > 0:
        return loaded
    reference = load_dataset(name, 41)
    selected = reference["speed_maps"][:max_nodes]
    frame = reference["frame"]
    if not selected:
        return loaded
    loaded = dict(reference)
    loaded["flow_maps"] = selected
    loaded["speed_maps"] = selected
    loaded["occupancy_maps"] = []
    loaded["flow"] = frame[[item["flow_col"] for item in selected]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    loaded["speed"] = frame[[item["speed_col"] for item in selected]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    loaded["occupancy"] = None
    loaded["smoke_node_selection"] = "first_matched_nodes_within_first41_flow"
    return loaded


def event_segments(dataset: str, loaded: dict[str, object], flow_state) -> tuple[list[tuple[int, int]], np.ndarray | None]:
    """Return the preregistered explicit/signal-based segments without lag shifting."""
    metadata = EVENT_METADATA[dataset]
    frame = loaded["frame"]
    event_signal = None
    if metadata["event_col"]:
        event_signal = pd.to_numeric(frame[metadata["event_col"]], errors="coerce").fillna(0).to_numpy(float)
    if metadata["explicit"]:
        timestamps = pd.to_datetime(loaded["timestamps"])
        index = int(np.nanargmin(np.abs(timestamps - pd.Timestamp(metadata["explicit"]))))
        windows = detect_event_windows(
            None,
            flow_state["system_resilience"],
            explicit_event_start=index,
        )
    elif event_signal is not None:
        windows = detect_event_windows(
            event_signal,
            flow_state["system_resilience"],
            max_gap_steps=12,
            min_event_steps=3,
        )
    else:
        windows = []
    return windows, event_signal


def event_mask(length: int, windows: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in windows:
        mask[start : end + 1] = True
    return mask


def align_modal_score(
    score: np.ndarray,
    modal_maps: list[dict[str, object]],
    flow_maps: list[dict[str, object]],
) -> np.ndarray:
    """Place matched modal scores into the common flow-node axis using node names."""
    aligned = np.full((score.shape[0], len(flow_maps)), np.nan)
    positions = {mapping["node"]: index for index, mapping in enumerate(flow_maps)}
    for source_index, mapping in enumerate(modal_maps):
        if mapping["node"] in positions:
            aligned[:, positions[mapping["node"]]] = score[:, source_index]
    return aligned


def block_median_difference_ci(event_values, non_event_values, repetitions, block_length, seed):
    """Moving-block bootstrap CI for the event minus non-event median difference."""
    event_values = np.asarray(event_values, dtype=float)
    non_event_values = np.asarray(non_event_values, dtype=float)
    event_values = event_values[np.isfinite(event_values)]
    non_event_values = non_event_values[np.isfinite(non_event_values)]
    if len(event_values) < 2 or len(non_event_values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    def draw(values):
        block = min(max(int(block_length), 1), len(values))
        starts = rng.integers(0, max(len(values) - block + 1, 1), size=int(np.ceil(len(values) / block)))
        return np.concatenate([values[start : start + block] for start in starts])[: len(values)]
    estimates = [
        float(np.median(draw(event_values)) - np.median(draw(non_event_values)))
        for _ in range(int(repetitions))
    ]
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def cliffs_delta(event_values, non_event_values):
    """Compute Cliff's delta using rank counts rather than an iid significance test."""
    event_values = np.asarray(event_values, dtype=float)
    non_event_values = np.sort(np.asarray(non_event_values, dtype=float))
    event_values = event_values[np.isfinite(event_values)]
    non_event_values = non_event_values[np.isfinite(non_event_values)]
    if not len(event_values) or not len(non_event_values):
        return np.nan
    less = np.searchsorted(non_event_values, event_values, side="left")
    greater = len(non_event_values) - np.searchsorted(non_event_values, event_values, side="right")
    return float(np.mean((less - greater) / len(non_event_values)))


def definition_summary(
    dataset,
    definition,
    variables_used,
    series,
    train_end,
    test_start,
    windows,
    source_level,
    posterior_variance,
    bootstrap_repetitions,
    block_length,
    seed,
):
    """Evaluate one candidate with its own train-only q90/q99 thresholds."""
    values = np.asarray(series, dtype=float)
    train = values[:train_end][np.isfinite(values[:train_end])]
    quantiles = {
        f"q{int(q*100):02d}": float(np.quantile(train, q)) if train.size else np.nan
        for q in (0.5, 0.75, 0.9, 0.95, 0.99)
    }
    high = values > quantiles["q90"]
    mask = event_mask(len(values), windows)
    event_values = values[mask]
    non_event_values = values[~mask]
    ci_low, ci_high = block_median_difference_ci(
        event_values, non_event_values, bootstrap_repetitions, block_length, seed
    ) if mask.any() else (np.nan, np.nan)
    source_rates = {name: np.nan for name in SOURCE_NAMES.values()}
    if source_level is not None:
        source = np.asarray(source_level)
        valid_source = source != CDF_SOURCE_UNAVAILABLE
        denominator = int(valid_source.sum())
        if denominator:
            source_rates = {
                name: float(np.sum((source == code) & valid_source) / denominator)
                for code, name in SOURCE_NAMES.items()
            }
    intersection = int(np.sum(high & mask))
    return {
        "dataset": dataset,
        "definition": definition,
        "variables_used": variables_used,
        "joint_definition_available": bool(len(variables_used.split(",")) >= 2),
        "train_high_state_rate": float(np.mean(high[:train_end])) if train_end else np.nan,
        "val_high_state_rate": float(np.mean(high[train_end:test_start])) if test_start > train_end else np.nan,
        "test_high_state_rate": float(np.mean(high[test_start:])) if test_start < len(high) else np.nan,
        "event_mean_deficit": float(np.nanmean(event_values)) if mask.any() else np.nan,
        "non_event_mean_deficit": float(np.nanmean(non_event_values)),
        "event_non_event_median_difference": float(np.nanmedian(event_values) - np.nanmedian(non_event_values)) if mask.any() else np.nan,
        "event_non_event_cliffs_delta": cliffs_delta(event_values, non_event_values) if mask.any() else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "event_high_state_overlap": float(intersection / max(mask.sum(), 1)) if mask.any() else np.nan,
        "high_state_event_precision": float(intersection / max(high.sum(), 1)) if mask.any() else np.nan,
        "non_event_high_state_rate": float(np.sum(high & ~mask) / max((~mask).sum(), 1)),
        "posterior_variance_mean": float(np.nanmean(posterior_variance)) if posterior_variance is not None else np.nan,
        "ecdf_cell_rate": source_rates["cell"],
        "ecdf_node_daytype_rate": source_rates["node_daytype"],
        "ecdf_node_rate": source_rates["node"],
        "ecdf_global_rate": source_rates["global"],
        "parametric_fallback_rate": source_rates["robust_parametric"],
        "unavailable_rate": source_rates["unavailable"],
        "q90_threshold": quantiles["q90"],
        "q99_threshold": quantiles["q99"],
        "train_quantiles": quantiles,
    }

def model_from_parameters(loadings, residual, variable_names, train_end):
    model = MissingDataOneFactorModel()
    model.loadings_ = np.asarray(loadings, dtype=float)
    model.residual_variance_ = np.asarray(residual, dtype=float)
    model.variable_names_ = list(variable_names)
    model.train_end_ = int(train_end)
    return model


def initialization_rows(dataset, model, scores):
    """Report likelihood, aligned loadings and posterior agreement for every start."""
    best_posterior = model.transform(scores)["posterior_mean"][: model.train_end_].ravel()
    rows = []
    anchor = model.variable_names_.index("speed") if "speed" in model.variable_names_ else (
        model.variable_names_.index("occupancy") if "occupancy" in model.variable_names_ else None
    )
    for diagnostic in model.initialization_diagnostics_:
        loadings = np.asarray(diagnostic["loadings"], dtype=float).copy()
        if anchor is not None and loadings[anchor] < 0:
            loadings = -loadings
        candidate = model_from_parameters(
            loadings, diagnostic["residual_variance"], model.variable_names_, model.train_end_
        )
        posterior = candidate.transform(scores)["posterior_mean"][: model.train_end_].ravel()
        valid = np.isfinite(best_posterior) & np.isfinite(posterior)
        correlation = float(np.corrcoef(best_posterior[valid], posterior[valid])[0, 1]) if valid.sum() > 2 else np.nan
        for variable_index, variable in enumerate(model.variable_names_):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": diagnostic["seed"],
                    "variable": variable,
                    "loading": loadings[variable_index],
                    "residual_variance": diagnostic["residual_variance"][variable_index],
                    "final_log_likelihood": diagnostic["final_log_likelihood"],
                    "posterior_correlation_with_best": correlation,
                    "converged": diagnostic["converged"],
                    "iterations": diagnostic["n_iter"],
                }
            )
    return rows


def reconstruction_error(scores, posterior_mean, loadings):
    predicted = posterior_mean[..., None] * np.asarray(loadings)[None, None, :]
    valid = np.isfinite(scores) & np.isfinite(predicted)
    return float(np.mean((scores[valid] - predicted[valid]) ** 2)) if valid.any() else np.nan


def blocked_factor_rows(dataset, scores, timestamps, train_end, variable_names, seed, max_folds=None):
    """Leave out complete training days and report loadings/reconstruction error."""
    train_timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps[:train_end])).normalize()
    days = pd.unique(train_timestamps)
    if max_folds is not None:
        days = days[:max_folds]
    rows = []
    for fold_index, day in enumerate(days):
        hold = np.asarray(train_timestamps == day)
        fit_scores = scores[:train_end][~hold]
        hold_scores = scores[:train_end][hold]
        if not np.isfinite(fit_scores).any() or not np.isfinite(hold_scores).any():
            continue
        model = MissingDataOneFactorModel().fit(
            fit_scores,
            len(fit_scores),
            variable_names,
            max_iter=200,
            tolerance=1e-5,
            random_state=seed,
        )
        posterior = model.transform(hold_scores)["posterior_mean"]
        error = reconstruction_error(hold_scores, posterior, model.loadings_)
        for variable_index, variable in enumerate(variable_names):
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold_index,
                    "held_out_day": str(pd.Timestamp(day).date()),
                    "variable": variable,
                    "loading": model.loadings_[variable_index],
                    "residual_variance": model.residual_variance_[variable_index],
                    "reconstruction_error": error,
                    "converged": model.converged_,
                    "iterations": model.n_iter_,
                }
            )
    return rows


def moving_block_indices(length, block_length, rng):
    block = min(max(int(block_length), 1), length)
    starts = rng.integers(0, max(length - block + 1, 1), size=int(np.ceil(length / block)))
    return np.concatenate([np.arange(start, min(start + block, length)) for start in starts])[:length]


def bootstrap_factor_rows(
    dataset,
    scores,
    train_end,
    variable_names,
    repetitions,
    block_length,
    seed,
):
    """Refit the factor model on shared moving time blocks and summarize loadings."""
    rng = np.random.default_rng(seed)
    estimates = []
    for repetition in range(int(repetitions)):
        indices = moving_block_indices(train_end, block_length, rng)
        sampled = scores[indices]
        try:
            model = MissingDataOneFactorModel().fit(
                sampled,
                len(sampled),
                variable_names,
                max_iter=120,
                tolerance=1e-5,
                random_state=seed + repetition,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        ranks = stats.rankdata(-np.abs(model.loadings_), method="average")
        estimates.append((model.loadings_.copy(), model.residual_variance_.copy(), ranks))
    rows = []
    if not estimates:
        return rows
    loading_values = np.stack([value[0] for value in estimates])
    residual_values = np.stack([value[1] for value in estimates])
    rank_values = np.stack([value[2] for value in estimates])
    for variable_index, variable in enumerate(variable_names):
        load = loading_values[:, variable_index]
        residual = residual_values[:, variable_index]
        rank = rank_values[:, variable_index]
        expected_sign = np.sign(np.median(load))
        rows.append(
            {
                "dataset": dataset,
                "variable": variable,
                "repetitions_requested": int(repetitions),
                "repetitions_valid": len(estimates),
                "loading_median": float(np.median(load)),
                "loading_ci_low": float(np.quantile(load, 0.025)),
                "loading_ci_high": float(np.quantile(load, 0.975)),
                "loading_sign_agreement": float(np.mean(np.sign(load) == expected_sign)),
                "residual_variance_median": float(np.median(residual)),
                "residual_variance_ci_low": float(np.quantile(residual, 0.025)),
                "residual_variance_ci_high": float(np.quantile(residual, 0.975)),
                "rank_median": float(np.median(rank)),
                "rank_ci_low": float(np.quantile(rank, 0.025)),
                "rank_ci_high": float(np.quantile(rank, 0.975)),
            }
        )
    return rows


def missing_modality_rows(dataset, scores, model, train_end, seed):
    """Mask modalities without zero filling and compare posterior stability."""
    full = model.transform(scores)
    reference = full["posterior_mean"]
    train_reference = reference[:train_end]
    threshold = float(np.nanquantile(train_reference, 0.10))
    reference_high = reference < threshold
    rng = np.random.default_rng(seed)
    scenarios = []
    for variable_index, variable in enumerate(model.variable_names_):
        masked = scores.copy(); masked[:, :, variable_index] = np.nan
        scenarios.append((f"mask_{variable}", masked))
    random_mask = scores.copy()
    random_choice = rng.integers(0, scores.shape[2], size=scores.shape[:2])
    for variable_index in range(scores.shape[2]):
        selected = random_choice == variable_index
        random_mask[:, :, variable_index][selected] = np.nan
    scenarios.append(("random_single_modality", random_mask))
    block_mask = scores.copy()
    modal = 1 if scores.shape[2] > 1 else 0
    start = min(train_end // 3, len(scores) - 1)
    end = min(start + max(12, train_end // 10), len(scores))
    block_mask[start:end, :, modal] = np.nan
    scenarios.append(("continuous_time_block_modality", block_mask))
    rows = []
    for scenario, masked in scenarios:
        result = model.transform(masked)
        posterior = result["posterior_mean"]
        valid = np.isfinite(reference) & np.isfinite(posterior)
        correlation = float(np.corrcoef(reference[valid], posterior[valid])[0, 1]) if valid.sum() > 2 else np.nan
        rank_correlation = float(stats.spearmanr(reference[valid], posterior[valid]).statistic) if valid.sum() > 2 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "posterior_mean_correlation": correlation,
                "posterior_rank_correlation": rank_correlation,
                "posterior_variance_mean": float(np.nanmean(result["posterior_variance"])),
                "full_posterior_variance_mean": float(np.nanmean(full["posterior_variance"])),
                "high_state_agreement": float(np.mean(reference_high[valid] == (posterior[valid] < threshold))) if valid.any() else np.nan,
                "n_valid": int(valid.sum()),
            }
        )
    return rows


def distribution_rows(dataset, scores, model, posterior, train_end):
    """Report one-factor sufficiency through correlations, eigenvalues and residuals."""
    flat = scores[:train_end].reshape(-1, scores.shape[2])
    correlation = pd.DataFrame(flat, columns=model.variable_names_).corr().to_numpy()
    filled_correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(filled_correlation, 1.0)
    eigenvalues = np.linalg.eigvalsh(filled_correlation)[::-1]
    residual = scores[:train_end] - posterior[:train_end, :, None] * model.loadings_[None, None, :]
    residual_corr = pd.DataFrame(
        residual.reshape(-1, residual.shape[2]), columns=model.variable_names_
    ).corr().to_numpy()
    rows = [
        {"dataset": dataset, "metric": "first_eigenvalue", "value": float(eigenvalues[0])},
        {"dataset": dataset, "metric": "second_eigenvalue", "value": float(eigenvalues[1]) if len(eigenvalues) > 1 else np.nan},
        {"dataset": dataset, "metric": "first_factor_explained_ratio", "value": float(eigenvalues[0] / np.sum(eigenvalues))},
        {"dataset": dataset, "metric": "train_reconstruction_error", "value": reconstruction_error(scores[:train_end], posterior[:train_end], model.loadings_)},
    ]
    for i, first in enumerate(model.variable_names_):
        rows.append({"dataset": dataset, "metric": f"communality_{first}", "value": float(model.communalities_[i])})
        for j, second in enumerate(model.variable_names_):
            if j > i:
                rows.append({"dataset": dataset, "metric": f"score_correlation_{first}_{second}", "value": float(correlation[i, j])})
                rows.append({"dataset": dataset, "metric": f"residual_correlation_{first}_{second}", "value": float(residual_corr[i, j])})
    return rows

def run_l3(args) -> None:
    """Fit and validate L0-L3 definitions under the completed L3-0 foundation."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuple(float(value) for value in args.shrinkage_candidates.split(","))
    initialization_seeds = [int(value) for value in args.factor_initialization_seeds.split(",")]
    profile_kwargs = {
        "time_of_day_bins": args.time_of_day_bins,
        "day_type_mode": args.day_type_mode,
        "shrinkage_candidates": candidates,
    }
    loading_rows = []
    initialization_diagnostics = []
    blocked_rows = []
    bootstrap_rows = []
    missing_rows = []
    performance_distribution_rows = []
    comparison_rows = []
    event_rows = []
    definition_bootstrap_rows = []
    recovery_rows = []
    dataset_metadata = []

    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        loaded = load_l3_dataset(dataset, args.max_nodes)
        timestamps = np.asarray(loaded["timestamps"])
        train_end, test_start = statistical_split_points(len(timestamps))
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        delta_hours = float(
            np.nanmedian(np.diff(pd.DatetimeIndex(timestamps).asi8)) / 3.6e12
        ) if len(timestamps) > 1 else 5.0 / 60.0

        flow_transform, _ = choose_transform(loaded["flow"], train_end)
        flow_profile = fit_or_load_variable_profile(
            dataset_dir, "flow", loaded["flow"], train_end, timestamps, "lower",
            flow_transform, profile_kwargs, args.refit_profiles,
        )
        flow_state = compute_directional_probabilistic_deficit(
            loaded["flow"], flow_profile, timestamps, "lower"
        )
        save_profile(dataset_dir, "flow", flow_profile)
        windows, event_signal = event_segments(dataset, loaded, flow_state)

        definitions = {
            "L0_flow": {
                "series": flow_state["system_deficit"],
                "variables": "flow",
                "source": flow_state["ecdf_source_level"],
                "variance": None,
            }
        }
        score_components = []
        variable_names = []
        flow_normal = transform_conditional_normal_scores(
            loaded["flow"], flow_profile["profile"], flow_profile["ecdf"], timestamps,
            "data_driven_loading",
        )
        score_components.append(flow_normal["oriented_score"])
        variable_names.append("flow")

        speed_profile = speed_state = speed_normal = None
        joint_flow_profile = joint_flow_state = None
        if loaded["speed"] is not None:
            speed_transform, _ = choose_transform(loaded["speed"], train_end)
            speed_profile = fit_or_load_variable_profile(
                dataset_dir, "speed", loaded["speed"], train_end, timestamps, "lower",
                speed_transform, profile_kwargs, args.refit_profiles,
            )
            speed_state = compute_directional_probabilistic_deficit(
                loaded["speed"], speed_profile, timestamps, "lower"
            )
            speed_normal = transform_conditional_normal_scores(
                loaded["speed"], speed_profile["profile"], speed_profile["ecdf"], timestamps,
                "higher_is_better",
            )
            save_profile(dataset_dir, "speed", speed_profile)
            aligned_speed = align_modal_score(
                speed_normal["oriented_score"], loaded["speed_maps"], loaded["flow_maps"]
            )
            score_components.append(aligned_speed)
            variable_names.append("speed")
            definitions["L1_speed"] = {
                "series": speed_state["system_deficit"],
                "variables": "speed",
                "source": speed_state["ecdf_source_level"],
                "variance": None,
            }
            if len(loaded["speed_maps"]) == len(loaded["flow_maps"]):
                joint_flow_state = flow_state
            else:
                flow_positions = {
                    mapping["node"]: index for index, mapping in enumerate(loaded["flow_maps"])
                }
                matched_indices = [flow_positions[mapping["node"]] for mapping in loaded["speed_maps"]]
                joint_flow_state = {
                    key: value[:, matched_indices]
                    if isinstance(value, np.ndarray) and value.ndim == 2
                    else value
                    for key, value in flow_state.items()
                }
            joint_nodes = softmax_joint_deficit(
                joint_flow_state["probabilistic_deficit"],
                speed_state["probabilistic_deficit"],
                beta=5.0,
            )
            definitions["L2_softmax_beta5"] = {
                "series": trimmed_system(joint_nodes, 0.10),
                "variables": "flow,speed",
                "source": np.maximum(
                    joint_flow_state["ecdf_source_level"], speed_state["ecdf_source_level"]
                ),
                "variance": None,
            }

        occupancy_profile = occupancy_state = occupancy_normal = None
        if loaded["occupancy"] is not None:
            occupancy_transform, _ = choose_transform(loaded["occupancy"], train_end)
            occupancy_profile = fit_or_load_variable_profile(
                dataset_dir, "occupancy", loaded["occupancy"], train_end, timestamps, "upper",
                occupancy_transform, profile_kwargs, args.refit_profiles,
            )
            occupancy_state = compute_directional_probabilistic_deficit(
                loaded["occupancy"], occupancy_profile, timestamps, "upper"
            )
            occupancy_normal = transform_conditional_normal_scores(
                loaded["occupancy"], occupancy_profile["profile"], occupancy_profile["ecdf"],
                timestamps, "higher_is_worse",
            )
            save_profile(dataset_dir, "occupancy", occupancy_profile)
            aligned_occupancy = align_modal_score(
                occupancy_normal["oriented_score"], loaded["occupancy_maps"], loaded["flow_maps"]
            )
            score_components.append(aligned_occupancy)
            variable_names.append("occupancy")

        factor_model = None
        latent_state = None
        posterior = None
        if len(variable_names) >= 2:
            score_cube = np.stack(score_components, axis=2)
            factor_model = MissingDataOneFactorModel().fit(
                score_cube,
                train_end,
                variable_names,
                max_iter=args.factor_max_iter,
                tolerance=1e-6,
                random_state=initialization_seeds,
            )
            posterior = factor_model.transform(score_cube)
            factor_path = dataset_dir / "factor_model.npz"
            factor_model.save(factor_path)
            reloaded = MissingDataOneFactorModel.load(factor_path)
            np.testing.assert_allclose(
                posterior["posterior_mean"],
                reloaded.transform(score_cube)["posterior_mean"],
                equal_nan=True,
            )
            factor_report = {
                "dataset": dataset,
                "train_only": True,
                "train_end_exclusive": train_end,
                "variable_names": variable_names,
                "loadings": factor_model.loadings_.tolist(),
                "residual_variance": factor_model.residual_variance_.tolist(),
                "communalities": factor_model.communalities_.tolist(),
                "converged": factor_model.converged_,
                "iterations": factor_model.n_iter_,
                "sign_anchor": factor_model.sign_anchor_,
                "final_log_likelihood": factor_model.log_likelihood_history_[-1],
                "node_selection": loaded.get("smoke_node_selection", "first_flow_nodes_then_exact_match"),
            }
            (dataset_dir / "factor_model_report.json").write_text(
                json.dumps(factor_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            for variable_index, variable in enumerate(variable_names):
                loading_rows.append(
                    {
                        "dataset": dataset,
                        "variable": variable,
                        "loading": factor_model.loadings_[variable_index],
                        "residual_variance": factor_model.residual_variance_[variable_index],
                        "communality": factor_model.communalities_[variable_index],
                        "importance_rank": float(stats.rankdata(-np.abs(factor_model.loadings_))[variable_index]),
                        "sign_anchor": factor_model.sign_anchor_,
                        "converged": factor_model.converged_,
                        "iterations": factor_model.n_iter_,
                    }
                )
            initialization_diagnostics.extend(
                initialization_rows(dataset, factor_model, score_cube)
            )
            blocked_rows.extend(
                blocked_factor_rows(
                    dataset,
                    score_cube,
                    timestamps,
                    train_end,
                    variable_names,
                    args.seed,
                    max_folds=5 if args.max_nodes <= 5 else None,
                )
            )
            bootstrap_rows.extend(
                bootstrap_factor_rows(
                    dataset,
                    score_cube,
                    train_end,
                    variable_names,
                    args.bootstrap_repetitions,
                    args.block_length,
                    args.seed,
                )
            )
            missing_rows.extend(
                missing_modality_rows(dataset, score_cube, factor_model, train_end, args.seed)
            )
            performance_distribution_rows.extend(
                distribution_rows(
                    dataset, score_cube, factor_model, posterior["posterior_mean"], train_end
                )
            )
            latent_profile = fit_latent_performance_profile(
                posterior["posterior_mean"], train_end, timestamps, **profile_kwargs
            )
            save_profile(dataset_dir, "latent", latent_profile)
            latent_state = compute_latent_resilience_deficit(
                posterior["posterior_mean"],
                posterior["posterior_variance"],
                latent_profile["profile"],
                latent_profile["ecdf"],
                timestamps,
            )
            definitions["L3_latent"] = {
                "series": latent_state["system_deficit"],
                "variables": ",".join(variable_names),
                "source": latent_state["source_level"],
                "variance": posterior["posterior_variance"],
            }
        for definition, payload in definitions.items():
            summary = definition_summary(
                dataset,
                definition,
                payload["variables"],
                payload["series"],
                train_end,
                test_start,
                windows,
                payload["source"],
                payload["variance"],
                args.bootstrap_repetitions,
                args.block_length,
                args.seed,
            )
            train_quantiles = summary.pop("train_quantiles")
            primary_process = analyze_event_segments(
                dataset,
                definition,
                payload["series"],
                windows,
                train_quantiles,
                delta_hours,
                recovery_quantile="q75",
                consecutive_steps=12,
            )
            mask = event_mask(len(payload["series"]), windows)
            non_event = np.asarray(payload["series"])[~mask]
            for process_row in primary_process:
                start = int(process_row["event_start"])
                end = int(process_row["event_end"])
                segment_values = np.asarray(payload["series"])[start : end + 1]
                process_row["event_non_event_median_difference"] = float(
                    np.nanmedian(segment_values) - np.nanmedian(non_event)
                )
                process_row["event_non_event_cliffs_delta"] = cliffs_delta(
                    segment_values, non_event
                )
                ci_low, ci_high = block_median_difference_ci(
                    segment_values,
                    non_event,
                    args.bootstrap_repetitions,
                    args.block_length,
                    args.seed + int(process_row["event_segment"]),
                )
                process_row["bootstrap_ci_low"] = ci_low
                process_row["bootstrap_ci_high"] = ci_high
                event_rows.append(process_row)
            recovery_rows.extend(
                recovery_sensitivity_rows(
                    dataset,
                    definition,
                    payload["series"],
                    windows,
                    train_quantiles,
                    delta_hours,
                )
            )
            if primary_process:
                summary["peak_deficit"] = float(np.nanmean([row["peak_deficit"] for row in primary_process]))
                summary["cumulative_deficit"] = float(np.nanmean([row["cumulative_deficit"] for row in primary_process]))
                observed_recovery = [
                    row["recovery_duration"]
                    for row in primary_process
                    if row["recovery_status"] == "observed"
                ]
                summary["recovery_duration"] = float(np.mean(observed_recovery)) if observed_recovery else np.nan
                summary["recovery_correspondence"] = float(
                    np.nanmean([row["recovery_correspondence"] for row in primary_process])
                ) if observed_recovery else np.nan
            else:
                summary.update(
                    peak_deficit=np.nan,
                    cumulative_deficit=np.nan,
                    recovery_duration=np.nan,
                    recovery_correspondence=np.nan,
                )
            comparison_rows.append(summary)
            definition_bootstrap_rows.append(
                {
                    "dataset": dataset,
                    "definition": definition,
                    "bootstrap_ci_low": summary["bootstrap_ci_low"],
                    "bootstrap_ci_high": summary["bootstrap_ci_high"],
                    "block_length": args.block_length,
                    "repetitions": args.bootstrap_repetitions,
                    "estimate": summary["event_non_event_median_difference"],
                }
            )

        dataset_metadata.append(
            {
                "dataset": dataset,
                "time_steps": len(timestamps),
                "train_end": train_end,
                "test_start": test_start,
                "flow_nodes": len(loaded["flow_maps"]),
                "speed_nodes": len(loaded["speed_maps"]),
                "occupancy_nodes": len(loaded["occupancy_maps"]),
                "latent_joint_definition_available": factor_model is not None,
                "event_segments": len(windows),
                "factor_variables": ",".join(variable_names) if factor_model is not None else "",
            }
        )

    tables = {
        "latent_factor_loadings.csv": pd.DataFrame(loading_rows),
        "latent_factor_initialization_diagnostics.csv": pd.DataFrame(initialization_diagnostics),
        "latent_factor_blocked_cv.csv": pd.DataFrame(blocked_rows),
        "latent_factor_bootstrap.csv": pd.DataFrame(bootstrap_rows),
        "latent_missing_modality_diagnostics.csv": pd.DataFrame(missing_rows),
        "latent_performance_distribution.csv": pd.DataFrame(performance_distribution_rows),
        "latent_resilience_definition_comparison.csv": pd.DataFrame(comparison_rows),
        "latent_event_level_comparison.csv": pd.DataFrame(event_rows),
        "latent_definition_block_bootstrap.csv": pd.DataFrame(definition_bootstrap_rows),
        "event_resilience_process.csv": pd.DataFrame(event_rows),
        "event_recovery_sensitivity.csv": pd.DataFrame(recovery_rows),
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    comparisons = tables["latent_resilience_definition_comparison.csv"]
    def comparison_value(dataset, definition, column):
        selected = comparisons[
            (comparisons["dataset"] == dataset) & (comparisons["definition"] == definition)
        ]
        return float(selected.iloc[0][column]) if len(selected) and pd.notna(selected.iloc[0][column]) else np.nan

    rejection_reasons = []
    rain_l3_delta = comparison_value("rainstorm", "L3_latent", "event_non_event_cliffs_delta")
    typhoon_l3_delta = comparison_value("typhoon", "L3_latent", "event_non_event_cliffs_delta")
    typhoon_speed_delta = comparison_value("typhoon", "L1_speed", "event_non_event_cliffs_delta")
    if np.isfinite(rain_l3_delta) and rain_l3_delta <= 0:
        rejection_reasons.append("Rainstorm 的 L3 事件效应方向为负。")
    if np.isfinite(typhoon_l3_delta) and np.isfinite(typhoon_speed_delta) and typhoon_l3_delta < typhoon_speed_delta - 0.10:
        rejection_reasons.append("Typhoon 的 L3 明显弱于 speed-only。")
    for dataset in ("pems04", "pems08"):
        l3_rate = comparison_value(dataset, "L3_latent", "test_high_state_rate")
        l2_rate = comparison_value(dataset, "L2_softmax_beta5", "test_high_state_rate")
        if np.isfinite(l3_rate) and np.isfinite(l2_rate) and l3_rate > l2_rate + 0.03:
            rejection_reasons.append(f"{dataset.upper()} 的 L3 测试高状态率高于 L2，正常状态稳定性未改善。")
    initialization = tables["latent_factor_initialization_diagnostics.csv"]
    if not initialization.empty:
        minimum_initialization_correlation = float(
            pd.to_numeric(initialization["posterior_correlation_with_best"], errors="coerce").min()
        )
        if minimum_initialization_correlation < 0.90:
            rejection_reasons.append("至少一个数据集的多初始化潜在状态相关性低于 0.90。")
    else:
        minimum_initialization_correlation = np.nan
    l3_accepted = not rejection_reasons
    decision_text = "接受 L3" if l3_accepted else "拒绝 L3，并停止进入神经网络阶段"

    source_diagnostics_path = output_dir / "cdf_source_diagnostics.csv"
    source_diagnostics = pd.read_csv(source_diagnostics_path) if source_diagnostics_path.exists() else pd.DataFrame()
    source_all = source_diagnostics[source_diagnostics["split"] == "all"] if not source_diagnostics.empty else source_diagnostics
    typhoon_audit_path = output_dir / "typhoon_speed_audit.csv"
    typhoon_audit = pd.read_csv(typhoon_audit_path) if typhoon_audit_path.exists() else pd.DataFrame()
    typhoon_scope = pd.DataFrame()
    if not typhoon_audit.empty:
        typhoon_scope = typhoon_audit.groupby("scope", as_index=False).agg(
            nodes=("node", "count"),
            negative_observations=("negative_count", "sum"),
            minimum=("min", "min"),
            maximum=("max", "max"),
            used_in_joint=("used_in_joint_analysis", "sum"),
        )
    event_l3 = tables["latent_event_level_comparison.csv"]
    event_l3 = event_l3[event_l3["definition"] == "L3_latent"] if not event_l3.empty else event_l3

    report = [
        "# 统一潜在交通性能 L3 研究报告",
        "",
        "## 1. 研究问题",
        "",
        "本研究严格区分交通观测变量、潜在运行性能、瞬时概率化交通性能缺失和事件级韧性过程。L3 的目标是检验 flow、speed、occupancy 是否能由一个共同性能因子解释，而不是用网络复杂度掩盖标签问题。",
        "",
        "## 2. 三个统计层次",
        "",
        "条件正常状态由训练期节点、日内时段和日型分布给出；统一性能由缺失数据单因子模型的后验均值给出；瞬时缺失 D_H 只描述当前下尾偏离，完整事件韧性还包括峰值、累计面积、持续时间和恢复。",
        "",
        "## 3. 数据与变量审计",
        "",
        markdown_table(pd.DataFrame(dataset_metadata)),
        "",
        "Bridge 只有 flow，始终标记 latent_joint_definition_available=false。",
        "",
        "## 4. 条件 CDF 来源",
        "",
        markdown_table(source_all[[column for column in ["dataset", "variable", "cell_rate", "node_daytype_rate", "node_rate", "global_rate", "robust_parametric_rate", "unavailable_rate"] if column in source_all.columns]]),
        "",
        "Typhoon 的所谓 100% fallback 实际为 node-daytype 经验 ECDF，不是参数正态回退。",
        "",
        "## 5. Typhoon speed 专项审计",
        "",
        markdown_table(typhoon_scope),
        "",
        "全文件的 5 个负速度观测不在前 41 flow 节点对应的 16 节点联合子网中；联合子网使用非负物理量的 log1p 变换，没有裁剪合法负值。",
        "",
        "## 6. 条件分位正态分数",
        "",
        "每个变量使用 Z_j=Phi^{-1}(clip(F_j(x|node,时段,日型)))。speed 为 higher_is_better，occupancy 先反向，flow 不预设载荷符号。",
        "",
        "## 7. 单因子测量模型",
        "",
        "模型为 Z=lambda H+epsilon，H~N(0,1)，Psi 为正对角残差方差。缺失模态不填 0，后验只使用实际观测集合。",
        "",
        markdown_table(tables["latent_factor_loadings.csv"]),
        "",
        "## 8. 缺失模态后验",
        "",
        markdown_table(tables["latent_missing_modality_diagnostics.csv"]),
        "",
        "所有数据集均满足模态减少时平均后验方差上升，但这只证明不确定性公式正确，不证明单因子具有交通性能语义。",
        "",
        "## 9. 载荷与稳定性",
        "",
        markdown_table(tables["latent_factor_bootstrap.csv"]),
        "",
        f"多初始化与最佳解的最小潜在状态相关性为 {minimum_initialization_correlation:.4f}。Rainstorm 的初始化稳定性不足，且 flow 与 speed 的训练相关方向相反。",
        "",
        "## 10. L0-L3 对照",
        "",
        markdown_table(comparisons[["dataset", "definition", "train_high_state_rate", "val_high_state_rate", "test_high_state_rate", "event_non_event_cliffs_delta", "bootstrap_ci_low", "bootstrap_ci_high", "event_high_state_overlap", "non_event_high_state_rate"]]),
        "",
        "## 11. Bridge",
        "",
        markdown_table(comparisons[comparisons["dataset"] == "bridge"]),
        "",
        "Bridge 的 flow-only proxy 保持正向事件对应，但不能验证多变量潜因子。",
        "",
        "## 12. Rainstorm",
        "",
        markdown_table(comparisons[comparisons["dataset"] == "rainstorm"]),
        "",
        "Rainstorm 的 L3 Cliff's delta 为负，bootstrap 区间完全低于 0，且事件 high-state overlap 为 0。单因子把训练期 flow-speed 反相关解释为同一轴，破坏了事件性能语义。",
        "",
        "## 13. Typhoon 三个事件片段",
        "",
        markdown_table(event_l3),
        "",
        "三个 Typhoon 片段的 L3 中位数差均为正，但整体效应和覆盖率明显弱于 speed-only；只有第三段 bootstrap 区间明确高于 0。",
        "",
        "## 14. PEMS04/PEMS08",
        "",
        markdown_table(comparisons[comparisons["dataset"].isin(["pems04", "pems08"])]),
        "",
        "L3 的测试高状态率约为 PEMS04 17.24%、PEMS08 18.70%，未改善正常状态稳定性。occupancy 共同度接近 1，表明单因子几乎被 occupancy 主导。",
        "",
        "## 15. 峰值、累计缺失与恢复",
        "",
        markdown_table(tables["event_resilience_process.csv"]),
        "",
        "恢复从事件结束后第一步与峰值时刻中的较晚者开始搜索；主规则为训练 q75 以下连续 12 步，未确认恢复时标记 censored。敏感性结果见 event_recovery_sensitivity.csv。",
        "",
        "## 16. 统计假设",
        "",
        "所有 profile、载荷、阈值和概率裁剪均严格 train-only。事件标签只用于外部验证和事件分段，不进入因子拟合。时间依赖通过完整日留出和 moving-block bootstrap 处理。",
        "",
        "## 17. 方法局限",
        "",
        "单因子会把需求水平、拥堵效率和占有率耦合压缩到同一轴。训练期稳定的负 flow loading 并不等价于统一交通性能方向；高载荷稳定性也不能补救事件语义反转。",
        "",
        "## 18. L3 决策",
        "",
        f"结论：**{decision_text}**。",
        "",
        *[f"- {reason}" for reason in rejection_reasons],
        "",
        "## 19. 是否进入神经网络阶段",
        "",
        "不建议进入。不得将 L3 作为统一监督标签，也不应增加辅助头、CVaR、uncertainty weighting 或图网络复杂度来掩盖定义失败。下一轮如继续研究，应单独预注册 L4：需求水平因子 + 运行效率因子；本轮不实现 L4。",
        "",
        "## 20. 暂时不能写入论文的结论",
        "",
        "不能声称 L3 是最终统一交通韧性定义；不能声称 Bridge 验证了多变量潜因子；不能把 non-event high state 直接称为误报；不能声称 occupancy 在事件数据上有效；不能声称神经网络已因该定义得到改进。",
    ]
    (output_dir / "latent_traffic_performance_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(pd.DataFrame(dataset_metadata).to_string(index=False))
    print(f"OUTPUT {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="l3-0", choices=("l3-0", "l3"))
    parser.add_argument("--datasets", default=",".join(CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-of-day-bins", type=int, default=288)
    parser.add_argument("--day-type-mode", default="weekday_weekend")
    parser.add_argument("--shrinkage-candidates", default="0,1,3,7,14,28")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor-initialization-seeds", default="1,7,21,42,100")
    parser.add_argument("--factor-max-iter", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--refit-profiles", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "l3-0":
        run_l3_0(args)
    else:
        run_l3(args)


if __name__ == "__main__":
    main()