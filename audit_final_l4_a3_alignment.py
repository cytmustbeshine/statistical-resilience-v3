"""Audit the final event-free L4-aligned A3/DCRNN experiment protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_l4_prediction_pipeline import DATASETS, audit_split, frozen_profile_available, ordered_frame, profile_paths
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN
from l4_prediction_pipeline import event_free_feature_contract, load_ordered_univariate_series, paired_column_plan, strict_data_bundle
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


def run(args: argparse.Namespace) -> dict[str, object]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=r"D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3")
    parser.add_argument("--l4-dir", default=r"D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4")
    parser.add_argument("--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3")
    parser.add_argument("--legacy-root", default=r"D:\TrafficGNN\outputs\formal_stat_resilience_event_e20")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
