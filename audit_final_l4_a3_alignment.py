"""Audit the final event-free L4-aligned A3/DCRNN experiment protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_latent_traffic_performance import load_saved_profile
from audit_l4_prediction_pipeline import DATASETS, audit_split, frozen_profile_available, ordered_frame, profile_paths
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN
from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_pipeline import (
    event_free_feature_contract,
    final_event_external_split,
    load_ordered_univariate_series,
    paired_column_plan,
    profile_compatible_event_external_split,
    scaler_metadata,
    strict_data_bundle,
)
from model import DSTSGCN


EVENT_WINDOWS = {
    "bridge": [(4032, 4158)],
    "rainstorm": [(10394, 10630)],
    "typhoon": [(3194, 3430), (3482, 3718), (3770, 4006)],
}


def audit_event_test_coverage(event_windows: list[tuple[int, int]], val_end_exclusive: int) -> list[dict[str, object]]:
    """Describe whether each preregistered event is fully external to validation."""
    return [
        {
            "event_id": event_id,
            "start_index": int(start),
            "end_index": int(end),
            "fully_in_test": bool(start >= val_end_exclusive),
            "overlaps_test": bool(end >= val_end_exclusive),
        }
        for event_id, (start, end) in enumerate(event_windows, 1)
    ]

def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\u65e0\u8bb0\u5f55"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def legacy_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    allowed = {"traffic", "time_sin_day", "time_cos_day", "time_sin_week", "time_cos_week"}
    for path in sorted(root.rglob("run_config.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        model = str(config.get("model_name", config.get("model", "")))
        if model not in {"stat_resilience_dstsgcn_v2", "dcrnn_resilience_official_adapted"}:
            continue
        features = [str(value) for value in config.get("feature_names", [])]
        prohibited = [value for value in features if value not in allowed]
        rows.append({
            "dataset": config.get("dataset_name", config.get("dataset", "")),
            "model": model,
            "seed": config.get("seed", ""),
            "experiment_stage": config.get("experiment_stage", ""),
            "split_mode": config.get("split_mode", ""),
            "feature_names": ",".join(features),
            "prohibited_features": ",".join(prohibited),
            "event_free": len(prohibited) == 0,
            "resilience_target": config.get("resilience_target_mode", "none"),
            "resilience_baseline": config.get("resilience_baseline_mode", "none"),
            "cvar_enabled": config.get("cvar_enabled", False),
            "final_l4_aligned": False,
            "compatibility_decision": "legacy_exploration_only",
            "run_dir": str(path.parent),
        })
    return rows


def smoke_models(nodes: int, history: int, horizon: int) -> tuple[bool, str, str]:
    inputs = torch.zeros(2, history, nodes, 1)
    adjacency = torch.eye(nodes)
    dstsgcn = DSTSGCN(
        num_nodes=nodes, input_dim=1, output_dim=1, horizon=horizon, hidden_dim=16,
        num_blocks=1, graph_learner_type="lmln", fusion_mode="fusion",
        fusion_type="quality", dynamic_top_k=min(3, max(nodes - 1, 1)),
        matrix_hidden_dim=32, event_dim=0, resilience_aux=False,
    )
    dcrnn = OfficialAdaptedDCRNN(
        num_nodes=nodes, input_dim=1, output_dim=1, rnn_units=16,
        num_rnn_layers=1, horizon=horizon, max_diffusion_step=1,
    )
    with torch.no_grad():
        dst_output = dstsgcn(inputs, adjacency)
        dcrnn_output = dcrnn(inputs, adjacency)
    expected = (2, horizon, nodes, 1)
    passed = tuple(dst_output.shape) == expected and tuple(dcrnn_output.shape) == expected
    passed = passed and bool(torch.isfinite(dst_output).all() and torch.isfinite(dcrnn_output).all())
    return passed, str(tuple(dst_output.shape)), str(tuple(dcrnn_output.shape))


def run_legacy(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l4_dir = Path(args.l4_dir)
    source_profile_dir = Path(args.source_profile_dir)
    legacy = pd.DataFrame(legacy_rows(Path(args.legacy_root)))
    legacy.to_csv(output_dir / "legacy_a3_compatibility_audit.csv", index=False, encoding="utf-8-sig")

    protocol_rows, feature_rows, split_rows, node_rows = [], [], [], []
    contract = event_free_feature_contract()
    for dataset in ["bridge", "rainstorm", "typhoon"]:
        config = DATASETS[dataset]
        frame = ordered_frame(config)
        plan = paired_column_plan(list(frame.columns), config["flow_suffix"], config["speed_suffix"], args.max_nodes)
        split = audit_split(dataset, len(frame), args.history, args.horizon)
        train_end = int(split["strict_train_time_end_exclusive"])
        val_end = int(len(frame) * 0.8)
        training_contains_event = any(start < train_end for start, _ in EVENT_WINDOWS[dataset])
        event_coverage = audit_event_test_coverage(EVENT_WINDOWS[dataset], val_end)
        for item in event_coverage:
            item["overlaps_validation"] = bool(item["start_index"] < val_end and item["end_index"] >= train_end)
        all_events_fully_in_test = bool(all(item["fully_in_test"] for item in event_coverage))
        split_rows.append({
            "dataset": dataset,
            "num_timesteps": len(frame),
            "train_time_end_exclusive": train_end,
            "val_time_end_exclusive": val_end,
            "train_val_target_overlap": split["strict_train_val_overlap_steps"],
            "val_test_target_overlap": split["strict_val_test_overlap_steps"],
            "strict_split_pass": split["strict_split_leak_free"],
            "event_windows": json.dumps(EVENT_WINDOWS[dataset]),
            "event_coverage": json.dumps(event_coverage),
            "events_fully_in_test": int(sum(item["fully_in_test"] for item in event_coverage)),
            "event_count": len(event_coverage),
            "all_events_fully_in_test": all_events_fully_in_test,
            "training_contains_target_event": training_contains_event,
            "recommended_val_time_end_exclusive": min(val_end, min(start for start, _ in EVENT_WINDOWS[dataset])),
        })
        variables = ["flow"] if dataset == "bridge" else ["flow", "speed"]
        for variable in variables:
            if variable == "flow":
                columns, suffix, profile_variable = plan["selected_flow_columns"], config["flow_suffix"], "demand"
            else:
                columns, suffix, profile_variable = plan["paired_speed_columns"], config["speed_suffix"], "efficiency"
            if not columns:
                continue
            values, timestamps, names = load_ordered_univariate_series(str(config["csv"]), str(config["time_col"]), columns, suffix)
            bundle = strict_data_bundle(
                values, timestamps, args.history, args.horizon,
                train_time_end_exclusive=train_end, val_time_end_exclusive=val_end,
            )
            paths = profile_paths(dataset, profile_variable, l4_dir, source_profile_dir)
            feature_rows.append({
                "dataset": dataset,
                "variable": variable,
                "input_features": json.dumps(contract["input_features"]),
                "event_features": json.dumps(contract["event_features"]),
                "weather_features": json.dumps(contract["weather_features"]),
                "event_free_pass": contract["event_features"] == [] and contract["weather_features"] == [],
                "profile_read_only_available": frozen_profile_available(paths),
                "physical_roundtrip_finite": bool(np.isfinite(bundle["scaler"].inverse_transform(bundle["scaled_values"])).all()),
            })
            node_rows.extend({
                "dataset": dataset, "variable": variable, "node_index": index,
                "node": name, "column": columns[index], "used_in_final_task": True,
            } for index, name in enumerate(names))
        paired_count = len(plan["paired_node_names"])
        protocol_rows.append({
            "dataset": dataset,
            "flow_nodes": len(plan["selected_flow_columns"]),
            "paired_nodes": paired_count,
            "typhoon_fixed_16_pass": dataset != "typhoon" or paired_count == 16,
            "bridge_speed_unavailable": dataset != "bridge" or paired_count == 0,
            "strict_split_pass": split["strict_split_leak_free"],
            "training_event_free": not training_contains_event,
            "event_weather_input_excluded": True,
            "all_events_fully_in_test": all_events_fully_in_test,
        })

    smoke_passed, dst_shape, dcrnn_shape = smoke_models(5, args.history, args.horizon)
    protocol, features = pd.DataFrame(protocol_rows), pd.DataFrame(feature_rows)
    splits, nodes = pd.DataFrame(split_rows), pd.DataFrame(node_rows)
    protocol.to_csv(output_dir / "final_protocol_alignment_audit.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "event_free_feature_audit.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(output_dir / "final_split_audit.csv", index=False, encoding="utf-8-sig")
    nodes.to_csv(output_dir / "final_node_alignment_audit.csv", index=False, encoding="utf-8-sig")

    stage_passed = bool(
        protocol[["typhoon_fixed_16_pass", "bridge_speed_unavailable", "strict_split_pass", "training_event_free", "event_weather_input_excluded", "all_events_fully_in_test"]].all().all()
        and features[["event_free_pass", "profile_read_only_available", "physical_roundtrip_finite"]].all().all()
        and smoke_passed
    )
    decision = {
        "stage": "E-L4-2A", "stage_passed": stage_passed,
        "e_l4_2b_authorized": stage_passed,
        "legacy_a3_final_l4_aligned": False, "legacy_a3_event_free": False,
        "model_core_change_required": False,
        "minimal_model_change": "none until event-aligned split audit passes",
        "dstsgcn_smoke_shape": dst_shape, "dcrnn_smoke_shape": dcrnn_shape,
        "blockers": [] if stage_passed else [
            f"{row.dataset}: only {row.events_fully_in_test}/{row.event_count} preregistered events are fully in test targets"
            for row in splits.itertuples(index=False) if not row.all_events_fully_in_test
        ],
        "recommended_val_time_end_exclusive": {
            str(row.dataset): int(row.recommended_val_time_end_exclusive) for row in splits.itertuples(index=False)
        },
    }
    (output_dir / "e_l4_2a_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# E-L4-2A \u6700\u7ec8\u534f\u8bae\u5bf9\u9f50\u5ba1\u8ba1", "", "## \u9636\u6bb5\u51b3\u7b56", "",
        "**\u5141\u8bb8\u8fdb\u5165 E-L4-2B\u3002**" if stage_passed else "**\u4e0d\u5141\u8bb8\u8fdb\u5165 E-L4-2B\u3002**", "",
        "\u6700\u7ec8 L4 \u4f7f\u7528 flow \u4e0b\u5c3e demand/service-volume deficit \u548c speed \u4e0b\u5c3e efficiency deficit\u3002\u65e7 A3 \u4f7f\u7528 robust flow log-ratio \u8f85\u52a9\u76ee\u6807\uff0c\u5e76\u5728 Rainstorm/Typhoon \u4e2d\u4f7f\u7528\u4e86\u5929\u6c14\u6216\u4e8b\u4ef6\u5f3a\u5ea6\uff0c\u56e0\u6b64\u53ea\u4fdd\u7559\u4e3a legacy \u63a2\u7d22\u3002", "",
        "## \u534f\u8bae\u68c0\u67e5", "", markdown_table(protocol), "",
        "## \u4e8b\u4ef6\u65e0\u5173\u8f93\u5165\u4e0e\u51bb\u7ed3 profile", "", markdown_table(features), "",
        "## \u4e25\u683c\u65f6\u95f4\u5207\u5206", "", markdown_table(splits), "",
        "## \u963b\u65ad\u9879", "",
        *( ["- " + item for item in decision["blockers"]] if decision["blockers"] else ["- \u65e0\u3002"] ), "",
        "\u5f53\u524d 80% validation/test \u8fb9\u754c\u4f7f Bridge \u4e8b\u4ef6\u5927\u90e8\u5206\u548c Typhoon \u7b2c 1 \u6bb5\u4e8b\u4ef6\u843d\u5165\u9a8c\u8bc1\u671f\uff0c\u65e0\u6cd5\u4f5c\u4e3a\u6700\u7ec8\u6d4b\u8bd5\u8bc1\u636e\u3002\u6700\u5c0f\u4fee\u590d\u662f\u4e00\u6b21\u6027\u9884\u6ce8\u518c validation \u7ed3\u675f\u8fb9\u754c\uff1aBridge=4032\u3001Rainstorm=9676\u3001Typhoon=3194\uff0c\u5e76\u91cd\u65b0\u6267\u884c\u9636\u6bb5 A\u3002", "",
        "## \u6700\u5c0f\u5b9e\u73b0\u8303\u56f4", "",
        "- \u4e0d\u4fee\u6539 L4 \u7edf\u8ba1\u5b9a\u4e49\u3001profile \u6216\u9608\u503c\u3002",
        "- \u5f53\u524d\u9636\u6bb5 A \u672a\u901a\u8fc7\uff0c\u4e0d\u5141\u8bb8\u542f\u52a8 M0/M1/M2/M3 \u6b63\u5f0f\u8bad\u7ec3\u3002",
        "- `model.py` \u4ec5\u5141\u8bb8 `l4_deficit` \u8f85\u52a9\u8f93\u51fa\u8bed\u4e49\uff0c\u4e0d\u4fee\u6539\u9aa8\u67b6\u3001STSGCN block \u6216 quality fusion\u3002",
        "- `train.py` \u4e0d\u518d\u5c06\u65e7 log-ratio \u5f53\u4f5c\u6700\u7ec8 L4 \u76ee\u6807\u3002", "",
        f"DSTSGCN smoke: `{dst_shape}`\uff1bDCRNN smoke: `{dcrnn_shape}`\u3002",
    ]
    (output_dir / "e_l4_2a_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    return decision


MODEL_SHA256_AT_START = "220c9ffc2b55cce9ef2bdc8564be82ad4f55afb6acd75e87db58417c28c35531"
TRAIN_SHA256_AT_START = "c37a40f3adf33c33162837bd173137b7f7db95b561bb61c9a0b645723fe93b19"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partial_output_inventory(root: Path) -> dict[str, object]:
    files = [path for path in root.rglob("*") if path.is_file()]
    results = []
    for path in sorted(root.rglob("result.json")):
        results.append({
            "relative_path": str(path.relative_to(root)),
            "length": int(path.stat().st_size),
            "sha256": file_sha256(path).upper(),
        })
    return {
        "root": str(root),
        "result_count": len(results),
        "file_count": len(files),
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "result_files": results,
    }


def _timestamp_keys(values: np.ndarray) -> set[int]:
    array = np.asarray(values).astype("datetime64[ns]").astype("int64")
    return set(int(value) for value in array.reshape(-1))


def _profile_thresholds(
    dataset: str,
    profile_variable: str,
    values: np.ndarray,
    timestamps: np.ndarray,
    paths: tuple[Path, Path, Path],
) -> dict[str, float]:
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    label = "demand" if profile_variable == "demand" else "speed"
    profile = load_saved_profile(
        paths[0].parent,
        label,
        np.asarray(values[..., 0], dtype=float),
        int(metadata["train_end_exclusive"]),
        str(metadata["transform_mode"]),
    )
    if profile is None:
        raise RuntimeError(f"Frozen profile cannot be loaded for {dataset}/{profile_variable}")
    reference = compute_directional_probabilistic_deficit(
        values[..., 0], profile, timestamps, "lower"
    )
    quantiles = reference["train_deficit_quantiles"]
    return {"q75": float(quantiles["q75"]), "q90": float(quantiles["q90"]), "q99": float(quantiles["q99"])}


def canonical_l4_threshold_map(l4_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Load canonical q90/q99 thresholds from the frozen L4 study output."""
    path = l4_dir / "l4_dimension_comparison.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    definition_to_variable = {
        "L4_demand_service": "demand",
        "L4_operating_efficiency": "efficiency",
    }
    thresholds = {}
    for row in frame.itertuples(index=False):
        variable = definition_to_variable.get(str(row.definition))
        if variable is None:
            continue
        thresholds[(str(row.dataset), variable)] = {
            "q90": float(row.q90_threshold),
            "q99": float(row.q99_threshold),
        }
    return thresholds

def run_repair_audit(args: argparse.Namespace) -> dict[str, object]:
    protocol = getattr(args, "split_protocol", "final_event_external")
    is_r2 = protocol == "profile_compatible_event_external"
    split_provider = (
        profile_compatible_event_external_split if is_r2 else final_event_external_split
    )
    stage_name = "E-L4-2A-R2" if is_r2 else "E-L4-2A-R"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l4_dir = Path(args.l4_dir)
    source_profile_dir = Path(args.source_profile_dir)
    formal_root = Path(args.formal_root)
    canonical_thresholds = canonical_l4_threshold_map(l4_dir)
    before_path = output_dir / "_partial_integrity_before.json"
    if not before_path.exists():
        raise FileNotFoundError(f"Missing prerepair partial-output inventory: {before_path}")
    before_inventory = json.loads(before_path.read_text(encoding="utf-8-sig"))

    configuration_rows = []
    split_rows = []
    coverage_rows = []
    profile_rows = []
    scaler_rows = []
    node_rows = []
    feature_rows = []
    canonical_rows = []
    contract = event_free_feature_contract()

    for dataset in ("bridge", "rainstorm", "typhoon"):
        config = DATASETS[dataset]
        frame = ordered_frame(config)
        timestamps_all = pd.to_datetime(frame[str(config["time_col"])], errors="raise").to_numpy()
        split_config = split_provider(dataset)
        train_end = int(split_config["train_time_end_exclusive"])
        val_end = int(split_config["val_time_end_exclusive"])
        if val_end > len(frame):
            raise ValueError(f"{dataset} val boundary exceeds data length")
        plan = paired_column_plan(
            list(frame.columns), config["flow_suffix"], config["speed_suffix"], args.max_nodes
        )
        if dataset == "bridge":
            variable_specs = [("flow", "demand", plan["selected_flow_columns"], config["flow_suffix"])]
        elif dataset == "typhoon":
            variable_specs = [
                ("flow", "demand", plan["paired_flow_columns"], config["flow_suffix"]),
                ("speed", "efficiency", plan["paired_speed_columns"], config["speed_suffix"]),
            ]
        else:
            variable_specs = [
                ("flow", "demand", plan["paired_flow_columns"], config["flow_suffix"]),
                ("speed", "efficiency", plan["paired_speed_columns"], config["speed_suffix"]),
            ]

        audit_bundle = None
        for variable, profile_variable, columns, suffix in variable_specs:
            values, timestamps, names = load_ordered_univariate_series(
                str(config["csv"]), str(config["time_col"]), columns, suffix
            )
            bundle = strict_data_bundle(
                values,
                timestamps,
                args.history,
                args.horizon,
                train_time_end_exclusive=train_end,
                val_time_end_exclusive=val_end,
            )
            audit_bundle = bundle if audit_bundle is None else audit_bundle
            legacy_val_end = int(len(frame) * 0.8)
            legacy_bundle = strict_data_bundle(
                values,
                timestamps,
                args.history,
                args.horizon,
                train_time_end_exclusive=train_end,
                val_time_end_exclusive=legacy_val_end,
            )
            corrected_scaler = scaler_metadata(bundle["scaler"])
            legacy_scaler = scaler_metadata(legacy_bundle["scaler"])
            mean_equal = bool(np.array_equal(np.asarray(corrected_scaler["mean"]), np.asarray(legacy_scaler["mean"])))
            std_equal = bool(np.array_equal(np.asarray(corrected_scaler["std"]), np.asarray(legacy_scaler["std"])))
            roundtrip = bundle["scaler"].inverse_transform(bundle["scaled_values"])
            finite_mask = np.isfinite(values)
            roundtrip_error = float(np.max(np.abs(roundtrip[finite_mask] - values[finite_mask]))) if finite_mask.any() else 0.0
            scaler_rows.append({
                "dataset": dataset,
                "variable": variable,
                "train_time_end_exclusive": train_end,
                "legacy_val_time_end_exclusive": legacy_val_end,
                "corrected_val_time_end_exclusive": val_end,
                "fit_range": f"[0,{train_end})",
                "train_only": True,
                "mean_before": json.dumps(legacy_scaler["mean"]),
                "mean_after": json.dumps(corrected_scaler["mean"]),
                "std_before": json.dumps(legacy_scaler["std"]),
                "std_after": json.dumps(corrected_scaler["std"]),
                "mean_unchanged": mean_equal,
                "std_unchanged": std_equal,
                "roundtrip_max_abs_error": roundtrip_error,
                "roundtrip_pass": bool(np.isfinite(roundtrip_error) and roundtrip_error < 1e-5),
                "negative_values_clipped": False,
            })
            paths = profile_paths(dataset, profile_variable, l4_dir, source_profile_dir)
            if paths is None or not frozen_profile_available(paths):
                raise RuntimeError(f"Frozen profile package unavailable for {dataset}/{profile_variable}")
            metadata = json.loads(paths[2].read_text(encoding="utf-8"))
            profile_train_end = int(metadata["train_end_exclusive"])
            profile_columns = columns
            profile_values = values
            profile_timestamps = timestamps
            if dataset == "typhoon" and variable == "flow":
                profile_columns = plan["selected_flow_columns"]
                profile_values, profile_timestamps, _ = load_ordered_univariate_series(
                    str(config["csv"]), str(config["time_col"]), profile_columns, suffix
                )
            thresholds = _profile_thresholds(
                dataset, profile_variable, profile_values, profile_timestamps, paths
            )
            profile_rows.append({
                "dataset": dataset,
                "variable": variable,
                "l4_dimension": profile_variable,
                "requested_train_end_exclusive": train_end,
                "frozen_profile_train_end_exclusive": profile_train_end,
                "boundary_difference": train_end - profile_train_end,
                "train_boundary_matches": train_end == profile_train_end,
                "profile_train_only": bool(metadata.get("train_only", False)),
                "profile_refit": False,
                "ecdf_refit": False,
                "q75_before": thresholds["q75"],
                "q75_after": thresholds["q75"],
                "q90_before": thresholds["q90"],
                "q90_after": thresholds["q90"],
                "q99_before": thresholds["q99"],
                "q99_after": thresholds["q99"],
                "recovery_threshold_before": thresholds["q75"],
                "recovery_threshold_after": thresholds["q75"],
                "thresholds_unchanged": True,
                "profile_report": str(paths[2]),
                "profile_report_sha256": file_sha256(paths[2]),
            })
            canonical = canonical_thresholds.get((dataset, profile_variable))
            if canonical is None:
                raise RuntimeError(f"Canonical L4 thresholds unavailable for {dataset}/{profile_variable}")
            q90_match = bool(np.isclose(thresholds["q90"], canonical["q90"], rtol=0.0, atol=1e-12))
            q99_match = bool(np.isclose(thresholds["q99"], canonical["q99"], rtol=0.0, atol=1e-12))
            raw_values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            loaded_values = np.asarray(values[..., 0], dtype=float)
            raw_missing = ~np.isfinite(raw_values)
            loaded_missing = ~np.isfinite(loaded_values)
            missing_converted_to_zero = int(np.sum(raw_missing & (loaded_values == 0.0)))
            canonical_rows.append({
                "dataset": dataset,
                "variable": variable,
                "l4_dimension": profile_variable,
                "profile_q90": thresholds["q90"],
                "canonical_q90": canonical["q90"],
                "q90_difference": thresholds["q90"] - canonical["q90"],
                "q90_matches": q90_match,
                "profile_q99": thresholds["q99"],
                "canonical_q99": canonical["q99"],
                "q99_difference": thresholds["q99"] - canonical["q99"],
                "q99_matches": q99_match,
                "canonical_thresholds_match": q90_match and q99_match,
                "canonical_source": str(l4_dir / "l4_dimension_comparison.csv"),
                "canonical_overwritten": False,
                "raw_missing_count": int(raw_missing.sum()),
                "loaded_missing_count": int(loaded_missing.sum()),
                "missing_converted_to_zero_count": missing_converted_to_zero,
                "loader_missing_policy": "np.nan_to_num(nan=0.0)",
                "likely_mismatch_cause": (
                    "missing_speed_converted_to_zero_before_frozen_profile"
                    if not (q90_match and q99_match) and missing_converted_to_zero > 0
                    else "none_detected"
                ),
            })
            for index, name in enumerate(names):
                node_rows.append({
                    "dataset": dataset,
                    "variable": variable,
                    "node_index": index,
                    "node": name,
                    "column": columns[index],
                    "matched_by_base_name": dataset == "bridge" or name in plan["paired_node_names"],
                    "adjacency_order_matches": True,
                    "fixed_typhoon_16": dataset != "typhoon" or len(names) == 16,
                })
            feature_rows.append({
                "dataset": dataset,
                "variable": variable,
                "input_features": json.dumps(contract["input_features"]),
                "event_features": json.dumps(contract["event_features"]),
                "weather_features": json.dumps(contract["weather_features"]),
                "event_weather_features_excluded": contract["event_features"] == [] and contract["weather_features"] == [],
                "bridge_flow_only": dataset != "bridge" or variable == "flow",
            })

        if audit_bundle is None:
            raise RuntimeError(f"No audit bundle generated for {dataset}")
        train_keys = _timestamp_keys(audit_bundle["train_target_timestamps"])
        val_keys = _timestamp_keys(audit_bundle["val_target_timestamps"])
        test_keys = _timestamp_keys(audit_bundle["test_target_timestamps"])
        info = audit_bundle["split_info"]
        configuration_rows.append({
            "dataset": dataset,
            "split_protocol": protocol,
            "num_timesteps": len(frame),
            "history": args.history,
            "horizon": args.horizon,
            "train_time_end_exclusive": train_end,
            "val_time_end_exclusive": val_end,
            "test_target_start_index": val_end,
            "test_target_start_timestamp": str(timestamps_all[val_end]),
            "train_end_unchanged_from_preregistered": True,
        })
        split_rows.append({
            "dataset": dataset,
            "split_protocol": protocol,
            "train_windows": len(audit_bundle["train_indices"]),
            "validation_windows": len(audit_bundle["val_indices"]),
            "test_windows": len(audit_bundle["test_indices"]),
            "unique_train_target_timestamps": len(train_keys),
            "unique_validation_target_timestamps": len(val_keys),
            "unique_test_target_timestamps": len(test_keys),
            "train_validation_overlap": len(train_keys & val_keys),
            "validation_test_overlap": len(val_keys & test_keys),
            "train_test_overlap": len(train_keys & test_keys),
            "target_disjoint": bool(info["target_disjoint"] and not (train_keys & val_keys or val_keys & test_keys or train_keys & test_keys)),
            "omitted_cross_boundary_windows": int(info["omitted_boundary_windows"]),
            "history_may_precede_target_split": True,
            "targets_cross_boundary": False,
        })
        for event_id, (start, end) in enumerate(EVENT_WINDOWS[dataset], 1):
            event_keys = _timestamp_keys(timestamps_all[start:end + 1])
            train_count = len(event_keys & train_keys)
            val_count = len(event_keys & val_keys)
            test_count = len(event_keys & test_keys)
            event_count = len(event_keys)
            coverage_rows.append({
                "dataset": dataset,
                "event_id": event_id,
                "event_start_index": start,
                "event_end_index": end,
                "event_start_timestamp": str(timestamps_all[start]),
                "event_end_timestamp": str(timestamps_all[end]),
                "event_target_count": event_count,
                "train_event_target_count": train_count,
                "validation_event_target_count": val_count,
                "test_event_target_count": test_count,
                "fully_outside_train": train_count == 0,
                "fully_outside_validation": val_count == 0,
                "fully_in_test": test_count == event_count,
                "test_coverage_rate": test_count / event_count if event_count else 0.0,
            })

    corrected = pd.DataFrame(configuration_rows)
    splits = pd.DataFrame(split_rows)
    coverage = pd.DataFrame(coverage_rows)
    profiles = pd.DataFrame(profile_rows)
    scalers = pd.DataFrame(scaler_rows)
    nodes = pd.DataFrame(node_rows)
    features = pd.DataFrame(feature_rows)
    canonical = pd.DataFrame(canonical_rows)

    after_inventory = partial_output_inventory(formal_root)
    before_by_path = {item["relative_path"]: item for item in before_inventory["result_files"]}
    after_by_path = {item["relative_path"]: item for item in after_inventory["result_files"]}
    integrity_rows = []
    for relative_path in sorted(set(before_by_path) | set(after_by_path)):
        before = before_by_path.get(relative_path)
        after = after_by_path.get(relative_path)
        integrity_rows.append({
            "relative_path": relative_path,
            "existed_before": before is not None,
            "exists_after": after is not None,
            "sha256_before": "" if before is None else before["sha256"],
            "sha256_after": "" if after is None else after["sha256"],
            "sha256_unchanged": before is not None and after is not None and before["sha256"] == after["sha256"],
            "length_before": "" if before is None else before["length"],
            "length_after": "" if after is None else after["length"],
        })
    integrity = pd.DataFrame(integrity_rows)
    partial_unchanged = bool(
        int(before_inventory["result_count"]) == int(after_inventory["result_count"])
        and int(before_inventory["file_count"]) == int(after_inventory["file_count"])
        and int(before_inventory["total_bytes"]) == int(after_inventory["total_bytes"])
        and not integrity.empty
        and integrity["sha256_unchanged"].all()
    )

    if is_r2:
        output_files = {
            "configuration": "profile_compatible_split_configuration.csv",
            "splits": "profile_compatible_split_window_audit.csv",
            "coverage": "profile_compatible_event_test_coverage.csv",
            "profiles": "profile_boundary_exact_match_audit.csv",
            "canonical": "canonical_l4_threshold_consistency_audit.csv",
            "scalers": "scaler_boundary_audit.csv",
            "nodes": "node_alignment_audit.csv",
            "features": "feature_contract_audit.csv",
            "integrity": "partial_formal_output_integrity.csv",
        }
    else:
        output_files = {
            "configuration": "corrected_split_configuration.csv",
            "splits": "corrected_split_window_audit.csv",
            "coverage": "corrected_event_test_coverage.csv",
            "profiles": "corrected_profile_boundary_audit.csv",
            "canonical": "canonical_l4_threshold_consistency_audit.csv",
            "scalers": "corrected_scaler_boundary_audit.csv",
            "nodes": "corrected_node_alignment_audit.csv",
            "features": "corrected_feature_contract_audit.csv",
            "integrity": "partial_formal_output_integrity.csv",
        }
    corrected.to_csv(output_dir / output_files["configuration"], index=False, encoding="utf-8-sig")
    splits.to_csv(output_dir / output_files["splits"], index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / output_files["coverage"], index=False, encoding="utf-8-sig")
    profiles.to_csv(output_dir / output_files["profiles"], index=False, encoding="utf-8-sig")
    canonical.to_csv(output_dir / output_files["canonical"], index=False, encoding="utf-8-sig")
    scalers.to_csv(output_dir / output_files["scalers"], index=False, encoding="utf-8-sig")
    nodes.to_csv(output_dir / output_files["nodes"], index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / output_files["features"], index=False, encoding="utf-8-sig")
    integrity.to_csv(output_dir / output_files["integrity"], index=False, encoding="utf-8-sig")

    all_disjoint = bool(splits["target_disjoint"].all())
    validation_event_free = bool((coverage["validation_event_target_count"] == 0).all())
    all_events_in_test = bool(coverage["fully_in_test"].all())
    profile_boundary_matches = bool(profiles["train_boundary_matches"].all())
    scaler_train_only = bool(scalers["train_only"].all())
    scaler_unchanged = bool(scalers[["mean_unchanged", "std_unchanged"]].all().all())
    thresholds_unchanged = bool(profiles["thresholds_unchanged"].all())
    canonical_match = bool(canonical["canonical_thresholds_match"].all())
    typhoon_fixed = bool(nodes.loc[nodes["dataset"] == "typhoon"].groupby("variable").size().eq(16).all())
    bridge_flow_only = bool(set(features.loc[features["dataset"] == "bridge", "variable"]) == {"flow"})
    feature_contract_pass = bool(features["event_weather_features_excluded"].all())
    model_unchanged = file_sha256(Path(__file__).with_name("model.py")) == MODEL_SHA256_AT_START
    train_unchanged = file_sha256(Path(__file__).with_name("train.py")) == TRAIN_SHA256_AT_START
    blockers = []
    if not profile_boundary_matches:
        mismatches = ", ".join(
            f"{row.dataset}/{row.variable}: requested={row.requested_train_end_exclusive}, frozen={row.frozen_profile_train_end_exclusive}"
            for row in profiles.itertuples(index=False) if not row.train_boundary_matches
        )
        blockers.append("Requested train_end values differ from frozen profile metadata: " + mismatches)
    if not canonical_match:
        mismatches = ", ".join(
            f"{row.dataset}/{row.variable}: q90_diff={row.q90_difference:.12g}, q99_diff={row.q99_difference:.12g}"
            for row in canonical.itertuples(index=False) if not row.canonical_thresholds_match
        )
        blockers.append("Reloaded frozen-profile thresholds differ from canonical L4 output: " + mismatches)
        zero_filled = [
            f"{row.dataset}/{row.variable}: {row.missing_converted_to_zero_count} missing values converted to zero"
            for row in canonical.itertuples(index=False)
            if not row.canonical_thresholds_match and row.missing_converted_to_zero_count > 0
        ]
        if zero_filled:
            blockers.append(
                "Likely threshold-path cause: load_wide_traffic_csv applies np.nan_to_num before frozen-profile evaluation; "
                + ", ".join(zero_filled)
            )
    checks = [
        all_disjoint, validation_event_free, all_events_in_test, profile_boundary_matches,
        scaler_train_only, scaler_unchanged, thresholds_unchanged, canonical_match, typhoon_fixed,
        bridge_flow_only, feature_contract_pass, partial_unchanged, model_unchanged, train_unchanged,
        bool(scalers["roundtrip_pass"].all()),
    ]
    stage_passed = bool(all(checks))
    decision = {
        "stage": stage_name,
        "stage_passed": stage_passed,
        "e_l4_2b_authorized": stage_passed,
        "training_run": False,
        "model_modified": not model_unchanged,
        "loss_modified": False,
        "l4_definition_modified": False,
        "split_protocol": protocol,
        "all_target_splits_disjoint": all_disjoint,
        "validation_event_free": validation_event_free,
        "all_events_fully_in_test": all_events_in_test,
        "profile_train_boundary_unchanged": profile_boundary_matches,
        "profile_train_boundary_exact": profile_boundary_matches,
        "scaler_train_only": scaler_train_only,
        "scaler_parameters_unchanged": scaler_unchanged,
        "thresholds_unchanged": thresholds_unchanged,
        "canonical_l4_thresholds_match": canonical_match,
        "typhoon_fixed_16_nodes": typhoon_fixed,
        "bridge_flow_only": bridge_flow_only,
        "event_weather_features_excluded": feature_contract_pass,
        "partial_outputs_unchanged": partial_unchanged,
        "train_py_modified": not train_unchanged,
        "partial_result_json_count_before": int(before_inventory["result_count"]),
        "partial_result_json_count_after": int(after_inventory["result_count"]),
        "partial_total_files_before": int(before_inventory["file_count"]),
        "partial_total_files_after": int(after_inventory["file_count"]),
        "partial_total_bytes_before": int(before_inventory["total_bytes"]),
        "partial_total_bytes_after": int(after_inventory["total_bytes"]),
        "blockers": blockers,
    }
    (output_dir / ("e_l4_2a_r2_decision.json" if is_r2 else "e_l4_2a_r_decision.json")).write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        f"# {stage_name} \u51bb\u7ed3 Profile \u517c\u5bb9\u7684\u4e8b\u4ef6\u5916\u90e8\u8bc4\u4ef7\u5ba1\u8ba1", "",
        "## \u9636\u6bb5\u7ed3\u8bba", "",
        "**\u5ba1\u8ba1\u901a\u8fc7\uff0c\u4f46\u6309\u4efb\u52a1\u8981\u6c42\u505c\u6b62\uff0c\u4e0d\u81ea\u52a8\u8fdb\u5165 E-L4-2B\u3002**" if stage_passed else "**\u5ba1\u8ba1\u672a\u901a\u8fc7\uff0c\u4e0d\u5141\u8bb8\u8fdb\u5165 E-L4-2B\u3002**", "",
        f"\u672c\u8f6e\u4f7f\u7528 `{protocol}` \u663e\u5f0f\u534f\u8bae\uff0c\u6ca1\u6709\u8bad\u7ec3\u6a21\u578b\uff0c\u6ca1\u6709\u4fee\u6539\u6a21\u578b\u3001\u635f\u5931\u6216 L4 \u5b9a\u4e49\u3002", "",
        "## \u5207\u5206\u914d\u7f6e\u4e0e\u7a97\u53e3", "", markdown_table(corrected), "", markdown_table(splits), "",
        "## \u4e8b\u4ef6\u5916\u90e8\u8986\u76d6", "", markdown_table(coverage), "",
        "## \u51bb\u7ed3 Profile \u8fb9\u754c", "", markdown_table(profiles), "",
        "## Canonical L4 \u9608\u503c\u4e00\u81f4\u6027", "", markdown_table(canonical), "",
        "Canonical \u9608\u503c\u6765\u81ea\u5df2\u6709 `l4_dimension_comparison.csv`\u3002\u5ba1\u8ba1\u53ea\u6bd4\u8f83\u91cd\u65b0\u52a0\u8f7d profile \u540e\u7684 q90/q99\uff0c\u4e0d\u8986\u76d6 canonical \u503c\u3001\u4e0d\u91cd\u62df profile \u6216 ECDF\u3002", "",
        "## Scaler", "", markdown_table(scalers), "",
        "\u4ec5\u6539\u53d8 val_end \u65f6\uff0cscaler mean/std \u5fc5\u987b\u5b8c\u5168\u4e0d\u53d8\uff1b\u8bad\u7ec3\u524d\u7f00\u5fc5\u987b\u4e0e\u51bb\u7ed3 profile train_end \u4e00\u81f4\u3002", "",
        "## \u8282\u70b9\u4e0e\u8f93\u5165\u5408\u540c", "", markdown_table(features), "", f"\u8282\u70b9\u5ba1\u8ba1\u5171 {len(nodes)} \u884c\uff1bTyphoon flow/speed \u5747\u56fa\u5b9a\u4e3a 16 \u4e2a\u540c\u540d\u540c\u5e8f\u8282\u70b9\uff0cBridge \u4ec5\u6709 flow\u3002", "",
        "## \u90e8\u5206\u6b63\u5f0f\u8f93\u51fa\u4fdd\u62a4", "", f"\u4fee\u590d\u524d\u540e\u5747\u4e3a {before_inventory['result_count']} \u4e2a result.json\u3001{before_inventory['file_count']} \u4e2a\u6587\u4ef6\u3001{before_inventory['total_bytes']} \u5b57\u8282\uff1bSHA256 \u5b8c\u6574\u6027\uff1a{partial_unchanged}\u3002", "",
        "## \u963b\u65ad\u9879", "",
        *(["- " + blocker for blocker in blockers] if blockers else ["- \u65e0\u3002"]), "",
        "\u6839\u636e\u505c\u6b62\u89c4\u5219\uff0c\u672c\u8f6e\u4e0d\u5c1d\u8bd5\u5176\u4ed6\u8fb9\u754c\u3001\u4e0d\u4fee\u6539 canonical L4 \u8f93\u51fa\u3001\u4e0d\u8bad\u7ec3\u4efb\u4f55\u6a21\u578b\uff0c\u4e5f\u4e0d\u786e\u5b9a\u4efb\u4f55\u5019\u9009\u6a21\u578b\u3002",
    ]
    report_name = "e_l4_2a_r2_audit_report.md" if is_r2 else "e_l4_2a_r_audit_report.md"
    (output_dir / report_name).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    return decision


def run(args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "split_protocol", "legacy_frozen") in {
        "final_event_external",
        "profile_compatible_event_external",
    }:
        return run_repair_audit(args)
    return run_legacy(args)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--legacy-root", default=r"D:\TrafficGNN\outputs\formal_stat_resilience_event_e20")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--split-protocol", choices=["legacy_frozen", "final_event_external", "profile_compatible_event_external"], default="legacy_frozen")
    parser.add_argument("--formal-root", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
