"""Audit and analyze latent traffic performance definitions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import read_csv_with_fallback
from flow_speed_resilience import fit_variable_resilience_profile, match_node_variables
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="l3-0", choices=("l3-0",))
    parser.add_argument("--datasets", default=",".join(CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-of-day-bins", type=int, default=288)
    parser.add_argument("--day-type-mode", default="weekday_weekend")
    parser.add_argument("--shrinkage-candidates", default="0,1,3,7,14,28")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_l3_0(args)


if __name__ == "__main__":
    main()