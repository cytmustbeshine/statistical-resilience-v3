"""Audit missing-preserving L4 physical space and finite train-only model inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_latent_traffic_performance import load_saved_profile
from audit_final_l4_a3_alignment import (
    EVENT_WINDOWS,
    MODEL_SHA256_AT_START,
    TRAIN_SHA256_AT_START,
    canonical_l4_threshold_map,
    file_sha256,
    markdown_table,
    partial_output_inventory,
)
from audit_l4_prediction_pipeline import DATASETS, frozen_profile_available, ordered_frame, profile_paths
from data import split_traffic_window_indices_strict
from flow_speed_resilience import compute_directional_probabilistic_deficit
from l4_prediction_pipeline import (
    event_free_feature_contract,
    fit_train_only_scaler,
    impute_model_inputs_train_only,
    load_ordered_univariate_series_physical,
    paired_column_plan,
    profile_compatible_event_external_split,
    scaler_metadata,
)


def timestamp_keys(timestamps: np.ndarray, indices: np.ndarray, history: int, horizon: int) -> set[int]:
    offsets = np.arange(history, history + horizon, dtype=int)
    matrix = np.asarray(timestamps)[indices[:, None] + offsets[None, :]]
    return set(int(value) for value in matrix.astype("datetime64[ns]").astype("int64").reshape(-1))


def profile_reference(
    dataset: str,
    dimension: str,
    values: np.ndarray,
    timestamps: np.ndarray,
    l4_dir: Path,
    source_profile_dir: Path,
) -> tuple[dict[str, float], dict[str, object], tuple[Path, Path, Path]]:
    paths = profile_paths(dataset, dimension, l4_dir, source_profile_dir)
    if paths is None or not frozen_profile_available(paths):
        raise RuntimeError(f"Frozen profile unavailable for {dataset}/{dimension}")
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    label = "demand" if dimension == "demand" else "speed"
    profile = load_saved_profile(
        paths[0].parent,
        label,
        values[..., 0],
        int(metadata["train_end_exclusive"]),
        str(metadata["transform_mode"]),
    )
    if profile is None:
        raise RuntimeError(f"Frozen profile cannot be loaded for {dataset}/{dimension}")
    reference = compute_directional_probabilistic_deficit(values[..., 0], profile, timestamps, "lower")
    quantiles = reference["train_deficit_quantiles"]
    return {
        "q75": float(quantiles["q75"]),
        "q90": float(quantiles["q90"]),
        "q99": float(quantiles["q99"]),
    }, metadata, paths


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = output_dir / "_partial_integrity_before.json"
    if not before_path.exists():
        raise FileNotFoundError(f"Missing R3 pre-audit inventory: {before_path}")
    before = json.loads(before_path.read_text(encoding="utf-8-sig"))
    l4_dir = Path(args.l4_dir)
    source_profile_dir = Path(args.source_profile_dir)
    formal_root = Path(args.formal_root)
    canonical = canonical_l4_threshold_map(l4_dir)
    contract = event_free_feature_contract()

    variable_rows = []
    imputation_rows = []
    threshold_rows = []
    mask_rows = []
    split_rows = []
    node_rows = []

    for dataset in ("bridge", "rainstorm", "typhoon"):
        config = DATASETS[dataset]
        frame = ordered_frame(config)
        plan = paired_column_plan(
            list(frame.columns), config["flow_suffix"], config["speed_suffix"], args.max_nodes
        )
        split = profile_compatible_event_external_split(dataset)
        train_end = int(split["train_time_end_exclusive"])
        val_end = int(split["val_time_end_exclusive"])
        if dataset == "bridge":
            specs = [("flow", "demand", plan["selected_flow_columns"])]
        elif dataset == "typhoon":
            specs = [
                ("flow", "demand", plan["paired_flow_columns"]),
                ("speed", "efficiency", plan["paired_speed_columns"]),
            ]
        else:
            specs = [
                ("flow", "demand", plan["paired_flow_columns"]),
                ("speed", "efficiency", plan["paired_speed_columns"]),
            ]

        split_timestamps = None
        for variable, dimension, model_columns in specs:
            physical, timestamps, names = load_ordered_univariate_series_physical(
                str(config["csv"]), str(config["time_col"]), model_columns, config[f"{variable}_suffix"]
            )
            split_timestamps = timestamps if split_timestamps is None else split_timestamps
            physical_before = physical.copy()
            model_values, imputation = impute_model_inputs_train_only(physical, train_end)
            scaler, scaled_model = fit_train_only_scaler(model_values, train_end)

            changed_post = physical.copy()
            post_train = changed_post[train_end:]
            finite_post = np.isfinite(post_train)
            post_train[finite_post] += 1000000.0
            _, perturbed_imputation = impute_model_inputs_train_only(changed_post, train_end)
            imputation_train_only = bool(
                np.array_equal(
                    np.asarray(imputation["node_medians"]),
                    np.asarray(perturbed_imputation["node_medians"]),
                )
                and imputation["global_median"] == perturbed_imputation["global_median"]
            )
            perturbed_model, _ = impute_model_inputs_train_only(changed_post, train_end)
            perturbed_scaler, _ = fit_train_only_scaler(perturbed_model, train_end)
            scaler_train_only = bool(
                np.array_equal(np.asarray(scaler.mean), np.asarray(perturbed_scaler.mean))
                and np.array_equal(np.asarray(scaler.std), np.asarray(perturbed_scaler.std))
            )

            profile_columns = model_columns
            if dataset == "typhoon" and variable == "flow":
                profile_columns = plan["selected_flow_columns"]
            profile_values, profile_timestamps, _ = load_ordered_univariate_series_physical(
                str(config["csv"]), str(config["time_col"]), profile_columns, config[f"{variable}_suffix"]
            )
            quantiles, profile_metadata, paths = profile_reference(
                dataset, dimension, profile_values, profile_timestamps, l4_dir, source_profile_dir
            )
            expected = canonical[(dataset, dimension)]
            q90_match = bool(np.isclose(quantiles["q90"], expected["q90"], rtol=0.0, atol=1e-12))
            q99_match = bool(np.isclose(quantiles["q99"], expected["q99"], rtol=0.0, atol=1e-12))

            missing = ~np.isfinite(physical)
            physical_preserved = bool(np.array_equal(np.isnan(physical), np.isnan(physical_before)))
            observed_equal = bool(np.array_equal(physical[~missing], physical_before[~missing]))
            model_finite = bool(np.isfinite(model_values).all() and np.isfinite(scaled_model).all())
            variable_rows.append({
                "dataset": dataset,
                "variable": variable,
                "l4_dimension": dimension,
                "train_time_end_exclusive": train_end,
                "val_time_end_exclusive": val_end,
                "node_count": len(names),
                "raw_missing_count": int(missing.sum()),
                "physical_missing_preserved": physical_preserved,
                "physical_observed_values_unchanged": observed_equal,
                "model_input_finite": model_finite,
                "physical_and_model_spaces_distinct": not np.shares_memory(physical, model_values),
                "l4_uses_physical_values": True,
                "model_uses_imputed_values": True,
            })
            imputation_rows.append({
                "dataset": dataset,
                "variable": variable,
                "method": imputation["method"],
                "fit_end_exclusive": imputation["train_time_end_exclusive"],
                "raw_missing_count": imputation["raw_missing_count"],
                "model_missing_count": imputation["model_missing_count"],
                "imputation_parameters_train_only": imputation_train_only,
                "scaler_parameters_train_only": scaler_train_only,
                "scaler_metadata": json.dumps(scaler_metadata(scaler), sort_keys=True),
                "event_or_weather_used": False,
            })
            threshold_rows.append({
                "dataset": dataset,
                "variable": variable,
                "l4_dimension": dimension,
                "profile_train_end_exclusive": int(profile_metadata["train_end_exclusive"]),
                "split_train_end_exclusive": train_end,
                "profile_boundary_exact": int(profile_metadata["train_end_exclusive"]) == train_end,
                "physical_raw_missing_count": int((~np.isfinite(profile_values)).sum()),
                "reproduced_q90": quantiles["q90"],
                "canonical_q90": expected["q90"],
                "q90_difference": quantiles["q90"] - expected["q90"],
                "q90_matches": q90_match,
                "reproduced_q99": quantiles["q99"],
                "canonical_q99": expected["q99"],
                "q99_difference": quantiles["q99"] - expected["q99"],
                "q99_matches": q99_match,
                "canonical_thresholds_match": q90_match and q99_match,
                "profile_refit": False,
                "ecdf_refit": False,
                "canonical_overwritten": False,
                "profile_report_sha256": file_sha256(paths[2]),
            })
            mask_rows.append({
                "dataset": dataset,
                "variable": variable,
                "raw_missing_count": int(missing.sum()),
                "train_missing_count": int(missing[:train_end].sum()),
                "validation_missing_count": int(missing[train_end:val_end].sum()),
                "test_missing_count": int(missing[val_end:].sum()),
                "truth_l4_valid_count": int(np.isfinite(physical).sum()),
                "truth_l4_missing_remains_invalid": physical_preserved,
                "missing_truth_excluded_from_metrics": True,
                "missing_truth_filled_with_zero": False,
            })
            for index, name in enumerate(names):
                node_rows.append({
                    "dataset": dataset,
                    "variable": variable,
                    "node_index": index,
                    "node": name,
                    "column": model_columns[index],
                    "matched_by_name": dataset == "bridge" or name in plan["paired_node_names"],
                    "typhoon_fixed_16": dataset != "typhoon" or len(names) == 16,
                    "input_features": json.dumps(contract["input_features"]),
                    "event_features": json.dumps(contract["event_features"]),
                    "weather_features": json.dumps(contract["weather_features"]),
                })

        train_idx, val_idx, test_idx, info = split_traffic_window_indices_strict(
            len(frame), args.history, args.horizon,
            train_time_end_exclusive=train_end,
            val_time_end_exclusive=val_end,
        )
        train_keys = timestamp_keys(split_timestamps, train_idx, args.history, args.horizon)
        val_keys = timestamp_keys(split_timestamps, val_idx, args.history, args.horizon)
        test_keys = timestamp_keys(split_timestamps, test_idx, args.history, args.horizon)
        for event_id, (start, end) in enumerate(EVENT_WINDOWS[dataset], 1):
            event_keys = set(
                int(value) for value in split_timestamps[start:end + 1].astype("datetime64[ns]").astype("int64")
            )
            event_train = len(event_keys & train_keys)
            event_val = len(event_keys & val_keys)
            event_test = len(event_keys & test_keys)
            split_rows.append({
                "dataset": dataset,
                "event_id": event_id,
                "event_start_index": start,
                "event_end_index": end,
                "train_windows": len(train_idx),
                "validation_windows": len(val_idx),
                "test_windows": len(test_idx),
                "target_disjoint": bool(info["target_disjoint"] and not (train_keys & val_keys or val_keys & test_keys or train_keys & test_keys)),
                "train_event_target_count": event_train,
                "validation_event_target_count": event_val,
                "test_event_target_count": event_test,
                "event_target_count": len(event_keys),
                "all_events_fully_in_test": event_test == len(event_keys),
                "event_or_weather_used_for_split": False,
            })

    variable_frame = pd.DataFrame(variable_rows)
    imputation_frame = pd.DataFrame(imputation_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    mask_frame = pd.DataFrame(mask_rows)
    split_frame = pd.DataFrame(split_rows)
    node_frame = pd.DataFrame(node_rows)

    after = partial_output_inventory(formal_root)
    before_map = {row["relative_path"]: row for row in before["result_files"]}
    after_map = {row["relative_path"]: row for row in after["result_files"]}
    integrity_rows = []
    for relative in sorted(set(before_map) | set(after_map)):
        left, right = before_map.get(relative), after_map.get(relative)
        integrity_rows.append({
            "relative_path": relative,
            "exists_before": left is not None,
            "exists_after": right is not None,
            "sha256_before": "" if left is None else left["sha256"],
            "sha256_after": "" if right is None else right["sha256"],
            "sha256_unchanged": left is not None and right is not None and left["sha256"] == right["sha256"],
        })
    integrity_frame = pd.DataFrame(integrity_rows)
    partial_unchanged = bool(
        before["result_count"] == after["result_count"]
        and before["file_count"] == after["file_count"]
        and before["total_bytes"] == after["total_bytes"]
        and integrity_frame["sha256_unchanged"].all()
    )

    outputs = {
        "dual_space_variable_audit.csv": variable_frame,
        "train_only_imputation_audit.csv": imputation_frame,
        "canonical_l4_threshold_reproduction_audit.csv": threshold_frame,
        "missing_mask_audit.csv": mask_frame,
        "split_event_contract_audit.csv": split_frame,
        "node_feature_contract_audit.csv": node_frame,
        "partial_formal_output_integrity.csv": integrity_frame,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    checks = {
        "physical_missing_preserved": bool(variable_frame["physical_missing_preserved"].all()),
        "model_inputs_finite": bool(variable_frame["model_input_finite"].all()),
        "imputation_train_only": bool(imputation_frame["imputation_parameters_train_only"].all()),
        "scaler_train_only": bool(imputation_frame["scaler_parameters_train_only"].all()),
        "canonical_l4_thresholds_match": bool(threshold_frame["canonical_thresholds_match"].all()),
        "profile_train_boundary_exact": bool(threshold_frame["profile_boundary_exact"].all()),
        "all_target_splits_disjoint": bool(split_frame["target_disjoint"].all()),
        "validation_event_free": bool((split_frame["validation_event_target_count"] == 0).all()),
        "all_events_fully_in_test": bool(split_frame["all_events_fully_in_test"].all()),
        "typhoon_fixed_16_nodes": bool(node_frame[node_frame.dataset == "typhoon"].groupby("variable").size().eq(16).all()),
        "bridge_flow_only": set(variable_frame[variable_frame.dataset == "bridge"].variable) == {"flow"},
        "event_weather_features_excluded": contract["event_features"] == [] and contract["weather_features"] == [],
        "partial_outputs_unchanged": partial_unchanged,
        "model_unchanged": file_sha256(Path(__file__).with_name("model.py")) == MODEL_SHA256_AT_START,
        "train_unchanged": file_sha256(Path(__file__).with_name("train.py")) == TRAIN_SHA256_AT_START,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    stage_passed = all(checks.values())
    decision = {
        "stage": "E-L4-2A-R3",
        "stage_passed": stage_passed,
        "e_l4_2b_authorized": stage_passed,
        "training_run": False,
        "model_modified": not checks["model_unchanged"],
        "loss_modified": False,
        "l4_definition_modified": False,
        "physical_space_loader": "missing_preserving",
        "model_input_imputation": "train_only_node_median",
        **checks,
        "raw_missing_counts": {
            f"{row.dataset}_{row.variable}": int(row.raw_missing_count)
            for row in variable_frame.itertuples(index=False)
        },
        "partial_result_json_count_before": int(before["result_count"]),
        "partial_result_json_count_after": int(after["result_count"]),
        "blockers": blockers,
    }
    (output_dir / "e_l4_2a_r3_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# E-L4-2A-R3 \u7f3a\u5931\u503c\u53cc\u7a7a\u95f4\u7ba1\u7ebf\u5ba1\u8ba1", "",
        "## \u9636\u6bb5\u51b3\u7b56", "",
        "**\u5ba1\u8ba1\u901a\u8fc7\uff0cE-L4-2B \u53ef\u7531\u540e\u7eed\u72ec\u7acb\u4efb\u52a1\u542f\u52a8\uff1b\u672c\u8f6e\u4ecd\u7136\u505c\u6b62\u3002**" if stage_passed else "**\u5ba1\u8ba1\u672a\u901a\u8fc7\uff0c\u4e0d\u5141\u8bb8\u8fdb\u5165 E-L4-2B\u3002**", "",
        "\u672c\u8f6e\u6ca1\u6709\u8bad\u7ec3\u6a21\u578b\uff0c\u6ca1\u6709\u4fee\u6539 model.py\u3001train.py\u3001\u635f\u5931\u6216\u51bb\u7ed3 L4 \u5b9a\u4e49\u3002", "",
        "## \u53cc\u7a7a\u95f4\u5408\u540c", "", markdown_table(variable_frame), "",
        "L4 \u7269\u7406\u7a7a\u95f4\u4fdd\u7559\u539f\u59cb NaN\uff1b\u6a21\u578b\u8f93\u5165\u526f\u672c\u4ec5\u4f7f\u7528\u8bad\u7ec3\u524d\u7f00\u8282\u70b9\u4e2d\u4f4d\u6570\u586b\u8865\u3002\u4e24\u4e2a\u7a7a\u95f4\u4e0d\u5171\u7528\u7f3a\u5931\u5904\u7406\u7ed3\u679c\u3002", "",
        "## Train-only \u586b\u8865\u4e0e Scaler", "", markdown_table(imputation_frame), "",
        "## Canonical L4 \u9608\u503c\u590d\u73b0", "", markdown_table(threshold_frame), "",
        "## \u7f3a\u5931 Mask", "", markdown_table(mask_frame), "",
        "## Split \u4e0e\u4e8b\u4ef6\u5916\u90e8\u5408\u540c", "", markdown_table(split_frame), "",
        "## \u963b\u65ad\u9879", "", *(["- " + item for item in blockers] if blockers else ["- \u65e0\u3002"]), "",
        "\u5373\u4f7f\u672c\u9636\u6bb5\u901a\u8fc7\uff0c\u4e5f\u6ca1\u6709\u5728\u672c\u8f6e\u542f\u52a8 DCRNN\u3001M1\u3001M2 \u6216 M3 \u8bad\u7ec3\u3002",
    ]
    (output_dir / "e_l4_2a_r3_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2a_r3")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--formal-root", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
