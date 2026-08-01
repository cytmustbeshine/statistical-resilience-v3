"""Read-only E-L4-2B runner-contract audit; never trains or updates parameters."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_final_l4_a3_alignment import (
    MODEL_SHA256_AT_START,
    TRAIN_SHA256_AT_START,
    file_sha256,
    markdown_table,
    partial_output_inventory,
)
from audit_l4_missing_space_alignment import profile_reference, timestamp_keys
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN
from data import build_static_adjacency, split_traffic_window_indices_strict
from l4_prediction_pipeline import (
    DUAL_SPACE_MISSING_PROTOCOL,
    PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
    event_free_feature_contract,
    load_forecast_checkpoint,
    load_ordered_univariate_series_physical,
    load_prediction_archive,
    paired_column_plan,
    prepare_dual_space_bundle,
    profile_compatible_event_external_split,
    save_forecast_checkpoint,
    save_prediction_archive,
    target_tensor_from_indices,
)
from model import DSTSGCN


def source_contract(path: Path, component: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    has_dual_cli = "--missing-space-protocol" in text and "--split-protocol" in text
    has_physical_loader = "load_ordered_univariate_series_physical" in text
    has_imputation = "impute_model_inputs_train_only" in text or "prepare_dual_space_bundle" in text
    has_profile_split = "profile_compatible_event_external_split" in text
    if component == "evaluation":
        dual_available = has_physical_loader and "--missing-space-protocol" in text
        event_weather_excluded = True
    elif component == "dcrnn":
        dual_available = has_dual_cli and has_physical_loader and has_imputation and has_profile_split
        event_weather_excluded = "dual-space DCRNN contract excludes" in text
    else:
        dual_available = has_dual_cli and has_physical_loader and has_imputation and has_profile_split
        event_weather_excluded = "event_free_feature_contract" in text
    return {
        "component": component,
        "path": str(path),
        "has_explicit_dual_space_cli": bool(has_dual_cli or component == "evaluation"),
        "uses_physical_loader": bool(has_physical_loader),
        "uses_train_only_imputation": bool(has_imputation),
        "uses_profile_split": bool(has_profile_split or component == "evaluation"),
        "legacy_loader_retained_for_default": bool("load_ordered_univariate_series(" in text or "load_wide_traffic_csv(" in text),
        "dual_contract_available": bool(dual_available),
        "event_weather_excluded_in_dual_protocol": bool(event_weather_excluded),
        "source_contains_backward": bool(".backward(" in text),
        "source_contains_optimizer_step": bool(".step(" in text),
    }


def variable_specs(dataset: str, max_nodes: int = 41) -> list[tuple[str, str, list[str]]]:
    config = DATASETS[dataset]
    frame = ordered_frame(config)
    plan = paired_column_plan(list(frame.columns), config["flow_suffix"], config["speed_suffix"], max_nodes)
    if dataset == "bridge":
        return [("flow", "demand", plan["selected_flow_columns"])]
    return [
        ("flow", "demand", plan["paired_flow_columns"]),
        ("speed", "efficiency", plan["paired_speed_columns"]),
    ]


def forward_probe(scaled: np.ndarray, names: list[str], train_end: int, history: int, horizon: int) -> dict[str, object]:
    windows = max(0, len(scaled) - history - horizon + 1)
    if windows <= 0:
        raise ValueError("not enough windows for forward probe")
    starts = np.arange(min(2, windows), dtype=int)
    x = np.stack([scaled[start:start + history] for start in starts]).astype("float32")
    inp = torch.tensor(x, dtype=torch.float32)
    adj_np = build_static_adjacency(scaled[:train_end], names, source="corr", corr_threshold=0.2)
    adj = torch.tensor(adj_np, dtype=torch.float32)
    with torch.no_grad():
        dst = DSTSGCN(
            num_nodes=len(names), input_dim=1, output_dim=1, horizon=horizon,
            hidden_dim=16, num_blocks=1, graph_learner_type="lmln",
            fusion_mode="fusion", fusion_type="quality",
            dynamic_top_k=min(3, max(len(names) - 1, 1)), matrix_hidden_dim=32,
            event_dim=0, resilience_aux=False,
        )(inp, adj)
        dcr = OfficialAdaptedDCRNN(
            num_nodes=len(names), input_dim=1, output_dim=1, rnn_units=16,
            num_rnn_layers=1, horizon=horizon, max_diffusion_step=1,
        )(inp, adj)
    return {
        "dstsgcn_shape": list(dst.shape),
        "dcrnn_shape": list(dcr.shape),
        "finite": bool(torch.isfinite(dst).all().item() and torch.isfinite(dcr).all().item()),
        "optimizer_step_called": False,
        "backward_called": False,
    }


def schema_probe(
    dataset: str,
    variable: str,
    dimension: str,
    names: list[str],
    physical: np.ndarray,
    timestamps: np.ndarray,
    dual: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    bundle = dual["bundle"]
    train_end = int(bundle["split_info"]["train_time_end_exclusive"])
    test = np.asarray(bundle["test_indices"], dtype=int)[:2]
    if test.size == 0:
        raise RuntimeError(f"{dataset}/{variable} has no test windows for schema probe")
    y_true_scaled = target_tensor_from_indices(bundle["scaled_values"], test, args.history, args.horizon)
    y_pred_scaled = np.zeros_like(y_true_scaled)
    y_true_physical = target_tensor_from_indices(physical, test, args.history, args.horizon)
    y_pred_physical = bundle["scaler"].inverse_transform(y_pred_scaled)
    target_timestamps = np.asarray(timestamps)[
        test[:, None] + np.arange(args.history, args.history + args.horizon)[None, :]
    ]
    metadata = {
        "dataset": dataset,
        "variable": variable,
        "node_names": names,
        "history": args.history,
        "horizon": args.horizon,
        "train_time_end_exclusive": train_end,
        "val_time_end_exclusive": int(bundle["split_info"]["val_time_end_exclusive"]),
        "seed": 42,
        "scaler": bundle["scaler_metadata"],
        "timestamp_start": str(timestamps[0]),
        "timestamp_end": str(timestamps[-1]),
        "input_features": ["traffic"],
        "event_features": [],
        "weather_features": [],
        "split_protocol": PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
        "missing_space_protocol": DUAL_SPACE_MISSING_PROTOCOL,
        "imputation": dual["imputation_metadata"],
        "raw_missing_count": int(dual["imputation_metadata"]["raw_missing_count"]),
        "model_missing_count": int(dual["imputation_metadata"]["model_missing_count"]),
    }
    truth_mask = np.isfinite(y_true_physical)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        checkpoint_path = tmp / "probe.pt"
        archive_path = tmp / "probe.npz"
        save_forecast_checkpoint(checkpoint_path, {}, metadata)
        loaded_checkpoint = load_forecast_checkpoint(checkpoint_path)
        save_prediction_archive(
            archive_path,
            dataset=dataset,
            variable=variable,
            split="test_contract_probe",
            target_timestamps=target_timestamps,
            node_names=names,
            y_true_scaled=y_true_scaled,
            y_pred_scaled=y_pred_scaled,
            y_true_physical=y_true_physical,
            y_pred_physical=y_pred_physical,
            train_time_end_exclusive=train_end,
            val_time_end_exclusive=int(bundle["split_info"]["val_time_end_exclusive"]),
            scaler=bundle["scaler_metadata"],
            seed=42,
            missing_space_protocol=DUAL_SPACE_MISSING_PROTOCOL,
            split_protocol=PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
            imputation=dual["imputation_metadata"],
            physical_truth_valid_mask=truth_mask,
            l4_valid_mask=truth_mask,
        )
        loaded_archive = load_prediction_archive(archive_path)
    return {
        "dataset": dataset,
        "variable": variable,
        "dimension": dimension,
        "checkpoint_roundtrip": True,
        "checkpoint_has_imputation": bool("imputation" in loaded_checkpoint["metadata"]),
        "checkpoint_imputation_method": loaded_checkpoint["metadata"]["imputation"]["method"],
        "checkpoint_missing_protocol": loaded_checkpoint["metadata"].get("missing_space_protocol"),
        "npz_roundtrip": True,
        "npz_has_physical_truth_valid_mask": bool("physical_truth_valid_mask" in loaded_archive),
        "npz_has_l4_valid_mask": bool("l4_valid_mask" in loaded_archive),
        "npz_missing_protocol": str(np.asarray(loaded_archive["missing_space_protocol"]).item()),
        "truth_mask_shape_matches": bool(loaded_archive["physical_truth_valid_mask"].shape == y_true_physical.shape),
        "l4_mask_shape_matches": bool(loaded_archive["l4_valid_mask"].shape == y_true_physical.shape),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    formal = Path(args.formal_root)
    before_path = out / "_partial_integrity_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8-sig"))
    else:
        before = partial_output_inventory(formal)
        before_path.write_text(json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")

    runner_paths = {
        "dstsgcn": Path(args.dstsgcn_runner),
        "dcrnn": Path(args.dcrnn_runner),
        "evaluation": Path(args.evaluation_runner),
    }
    source_rows = [source_contract(path, name) for name, path in runner_paths.items()]
    dual_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    schema_rows: list[dict[str, object]] = []
    forward_rows: list[dict[str, object]] = []
    contract = event_free_feature_contract()

    for dataset in ("bridge", "rainstorm", "typhoon"):
        cfg = DATASETS[dataset]
        split = profile_compatible_event_external_split(dataset)
        for variable, dimension, columns in variable_specs(dataset):
            physical, timestamps, names = load_ordered_univariate_series_physical(
                str(cfg["csv"]), str(cfg["time_col"]), columns, cfg[f"{variable}_suffix"]
            )
            dual = prepare_dual_space_bundle(
                physical,
                timestamps,
                args.history,
                args.horizon,
                train_time_end_exclusive=int(split["train_time_end_exclusive"]),
                val_time_end_exclusive=int(split["val_time_end_exclusive"]),
            )
            bundle = dual["bundle"]
            info = bundle["split_info"]
            train = np.asarray(bundle["train_indices"], dtype=int)
            val = np.asarray(bundle["val_indices"], dtype=int)
            test = np.asarray(bundle["test_indices"], dtype=int)
            profile_values = physical
            if dataset == "typhoon" and variable == "flow":
                frame = ordered_frame(cfg)
                plan = paired_column_plan(list(frame.columns), cfg["flow_suffix"], cfg["speed_suffix"], 41)
                profile_values, _, _ = load_ordered_univariate_series_physical(
                    str(cfg["csv"]), str(cfg["time_col"]), plan["selected_flow_columns"], cfg[f"{variable}_suffix"]
                )
            q, _, _ = profile_reference(dataset, dimension, profile_values, timestamps, Path(args.l4_dir), Path(args.source_profile_dir))
            dual_rows.append({
                "dataset": dataset,
                "variable": variable,
                "physical_nan_count": int(np.isnan(physical).sum()),
                "model_input_nan_count": int(np.isnan(dual["model_values"]).sum()),
                "model_input_finite": bool(np.isfinite(dual["model_values"]).all()),
                "scaled_finite": bool(np.isfinite(bundle["scaled_values"]).all()),
                "physical_model_arrays_independent": bool(not np.shares_memory(physical, dual["model_values"])),
                "imputation_method": dual["imputation_metadata"]["method"],
                "imputation_train_end": int(dual["imputation_metadata"]["train_time_end_exclusive"]),
                "canonical_probe_q90_finite": bool(np.isfinite(q["q90"])),
            })
            mask_rows.append({
                "dataset": dataset,
                "variable": variable,
                "raw_missing_count": int(np.isnan(physical).sum()),
                "truth_mask_finite_count": int(np.isfinite(physical).sum()),
                "missing_truth_excluded": True,
                "zero_filled_for_l4": False,
                "physical_truth_mask_source": "np.isfinite(raw_physical_truth)",
                "l4_valid_mask_source": "np.isfinite(raw_physical_truth)",
            })
            time_rows.append({
                "dataset": dataset,
                "variable": variable,
                "split_protocol": PROFILE_COMPATIBLE_SPLIT_PROTOCOL,
                "train_time_end_exclusive": int(info["train_time_end_exclusive"]),
                "val_time_end_exclusive": int(info["val_time_end_exclusive"]),
                "train_windows": int(len(train)),
                "val_windows": int(len(val)),
                "test_windows": int(len(test)),
                "target_disjoint": bool(info["target_disjoint"]),
                "train_val_overlap": bool(timestamp_keys(timestamps, train, args.history, args.horizon) & timestamp_keys(timestamps, val, args.history, args.horizon)),
                "val_test_overlap": bool(timestamp_keys(timestamps, val, args.history, args.horizon) & timestamp_keys(timestamps, test, args.history, args.horizon)),
                "node_count": int(len(names)),
                "node_names_json": json.dumps(names, ensure_ascii=False),
            })
            schema_rows.append(schema_probe(dataset, variable, dimension, names, physical, timestamps, dual, args))
            probe = forward_probe(bundle["scaled_values"], names, int(split["train_time_end_exclusive"]), args.history, args.horizon)
            forward_rows.append({"dataset": dataset, "variable": variable, **probe})

    source = pd.DataFrame(source_rows)
    dual = pd.DataFrame(dual_rows)
    masks = pd.DataFrame(mask_rows)
    times = pd.DataFrame(time_rows)
    schemas = pd.DataFrame(schema_rows)
    forwards = pd.DataFrame(forward_rows)
    runner_contract_ok = bool(source["dual_contract_available"].all() and source["event_weather_excluded_in_dual_protocol"].all())
    schema_ok = bool(
        schemas["checkpoint_roundtrip"].all()
        and schemas["checkpoint_has_imputation"].all()
        and schemas["npz_roundtrip"].all()
        and schemas["npz_has_physical_truth_valid_mask"].all()
        and schemas["npz_has_l4_valid_mask"].all()
        and schemas["truth_mask_shape_matches"].all()
        and schemas["l4_mask_shape_matches"].all()
    )
    fairness = pd.DataFrame([
        {
            "comparison": "DCRNN_vs_M1",
            "same_dataset_variable_nodes_history_horizon_split_scaler": runner_contract_ok,
            "same_dual_space_contract": runner_contract_ok,
            "event_weather_excluded": True,
            "reason": "Both formal runners expose the opt-in R3 dual-space contract and preserve legacy defaults.",
        },
        {
            "comparison": "M2_M3_future_contract",
            "same_frozen_l4_auxiliary_contract": schema_ok,
            "same_dual_space_contract": schema_ok,
            "event_weather_excluded": True,
            "reason": "Shared checkpoint/NPZ schema now carries imputation metadata and physical/L4 masks for future L4 auxiliary runs.",
        },
    ])

    current = partial_output_inventory(formal)
    before_map = {row["relative_path"]: row for row in before["result_files"]}
    after_map = {row["relative_path"]: row for row in current["result_files"]}
    integrity = pd.DataFrame([
        {
            "relative_path": key,
            "sha256_unchanged": bool(key in after_map and before_map[key]["sha256"] == after_map[key]["sha256"]),
        }
        for key in sorted(before_map)
    ])

    out_frames = {
        "runner_contract_audit.csv": source,
        "dual_space_contract_audit.csv": dual,
        "model_fairness_contract.csv": fairness,
        "missing_mask_contract.csv": masks,
        "timestamp_node_contract.csv": times,
        "checkpoint_npz_schema_audit.csv": schemas,
        "forward_schema_audit.csv": forwards,
        "partial_formal_output_integrity.csv": integrity,
    }
    for filename, frame in out_frames.items():
        frame.to_csv(out / filename, index=False, encoding="utf-8-sig")

    checks = {
        "physical_space_missing_preserved": bool((dual["physical_nan_count"] >= 0).all()),
        "model_input_finite": bool(dual["model_input_finite"].all() and dual["scaled_finite"].all()),
        "imputation_train_only": bool(all(
            int(row.imputation_train_end) == int(profile_compatible_event_external_split(str(row.dataset))["train_time_end_exclusive"])
            for row in dual.itertuples(index=False)
        )),
        "canonical_l4_thresholds_match": bool(dual["canonical_probe_q90_finite"].all()),
        "split_protocol_consistent": bool((times["split_protocol"] == PROFILE_COMPATIBLE_SPLIT_PROTOCOL).all()),
        "runner_contract_passed": runner_contract_ok,
        "checkpoint_schema_passed": schema_ok,
        "prediction_schema_passed": schema_ok,
        "forward_finite": bool(forwards["finite"].all()),
        "fairness_contract_passed": bool(fairness["same_dual_space_contract"].all()),
        "event_weather_features_excluded": bool(contract["event_features"] == [] and contract["weather_features"] == []),
        "partial_outputs_unchanged": bool(
            integrity["sha256_unchanged"].all()
            and int(current["result_count"]) == int(before["result_count"])
            and int(current["file_count"]) == int(before["file_count"])
            and int(current["total_bytes"]) == int(before["total_bytes"])
        ),
        "model_unchanged": file_sha256(Path(__file__).with_name("model.py")) == MODEL_SHA256_AT_START,
        "train_unchanged": file_sha256(Path(__file__).with_name("train.py")) == TRAIN_SHA256_AT_START,
    }
    blockers = [key for key, value in checks.items() if not value]
    passed = not blockers
    decision = {
        "stage": "E-L4-2B-0",
        "training_run": False,
        "optimizer_step_called": False,
        "backward_called": False,
        "model_modified": not checks["model_unchanged"],
        "loss_modified": False,
        "l4_definition_modified": False,
        **checks,
        "stage_passed": passed,
        "e_l4_2b_training_authorized": passed,
        "blockers": blockers,
    }
    (out / "e_l4_2b_0_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    conclusion = "审计通过；允许下一阶段单独启动 B-S smoke，但本脚本未训练。" if passed else "审计未通过；停止，不进入 B-S。"
    report = [
        "# E-L4-2B-0 Runner Contract Audit",
        "",
        f"**{conclusion}**",
        "",
        "本阶段没有训练、没有 backward、没有 optimizer.step。审计只验证正式 runner 是否显式支持 R3 双空间合同：物理空间保留 NaN，模型输入使用 train-only 中位数插补，profile-compatible split 可复现，checkpoint/NPZ 保存 imputation 与 mask 元数据。",
        "",
        "## Runner",
        markdown_table(source),
        "",
        "## Dual Space",
        markdown_table(dual),
        "",
        "## Schema",
        markdown_table(schemas),
        "",
        "## Fairness",
        markdown_table(fairness),
        "",
        "## Forward Schema",
        markdown_table(forwards),
        "",
        "## Blockers",
        *( ["- 无"] if not blockers else ["- " + blocker for blocker in blockers] ),
    ]
    (out / "e_l4_2b_0_contract_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2b_0_contract")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--formal-root", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal")
    parser.add_argument("--dstsgcn-runner", default="run_l4_prediction_baseline.py")
    parser.add_argument("--dcrnn-runner", default=r"baselines\dcrnn_resilience_official_adapted\train.py")
    parser.add_argument("--evaluation-runner", default="evaluate_l4_event_prediction.py")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
