"""Analyze preregistered L4 demand and operating-efficiency resilience dimensions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_event_resilience_process import analyze_event_segments, recovery_sensitivity_rows
from analyze_latent_traffic_performance import (
    align_modal_score,
    blocked_factor_rows,
    bootstrap_factor_rows,
    choose_transform,
    definition_summary,
    event_segments,
    initialization_rows,
    load_l3_dataset,
    load_saved_profile,
    markdown_table,
    missing_modality_rows,
    save_profile,
    statistical_split_points,
)
from flow_speed_resilience import compute_directional_probabilistic_deficit
from latent_traffic_performance import (
    compute_latent_resilience_deficit,
    fit_latent_performance_profile,
    transform_conditional_normal_scores,
)
from two_factor_traffic_resilience import DemandEfficiencyModel, two_dimension_event_states


def load_base_profile(source_dir, dataset, label, values, train_end, transform_mode):
    """Load the verified Stage-A profile and fail rather than silently refit."""
    profile = load_saved_profile(
        Path(source_dir) / dataset, label, values, train_end, transform_mode
    )
    if profile is None:
        raise FileNotFoundError(f"compatible source profile missing for {dataset}/{label}")
    return profile


def time_step_hours(timestamps):
    if len(timestamps) < 2:
        return 5.0 / 60.0
    return float(np.nanmedian(np.diff(pd.DatetimeIndex(timestamps).asi8)) / 3.6e12)


def event_quadrant_rows(dataset, states, windows):
    rows = []
    if not states.get("available", False):
        return rows
    for event_index, (start, end) in enumerate(windows, start=1):
        valid = states["valid_mask"][start : end + 1]
        denominator = max(int(valid.sum()), 1)
        rows.append(
            {
                "dataset": dataset,
                "event_segment": event_index,
                "event_start": start,
                "event_end": end,
                "both_high_rate": float(states["both_high"][start : end + 1].sum() / denominator),
                "demand_only_high_rate": float(states["demand_only_high"][start : end + 1].sum() / denominator),
                "efficiency_only_high_rate": float(states["efficiency_only_high"][start : end + 1].sum() / denominator),
                "neither_high_rate": float(states["neither_high"][start : end + 1].sum() / denominator),
            }
        )
    return rows


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.factor_initialization_seeds.split(",")]
    profile_kwargs = {
        "time_of_day_bins": args.time_of_day_bins,
        "day_type_mode": args.day_type_mode,
        "shrinkage_candidates": tuple(float(value) for value in args.shrinkage_candidates.split(",")),
    }
    comparison_rows = []
    event_rows = []
    quadrant_rows = []
    loading_rows = []
    initialization_diagnostics = []
    blocked_rows = []
    bootstrap_rows = []
    missing_rows = []
    recovery_rows = []
    metadata_rows = []

    for dataset in [value.strip() for value in args.datasets.split(",") if value.strip()]:
        loaded = load_l3_dataset(dataset, args.max_nodes)
        timestamps = np.asarray(loaded["timestamps"])
        train_end, test_start = statistical_split_points(len(timestamps))
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        delta_hours = time_step_hours(timestamps)

        flow_transform, _ = choose_transform(loaded["flow"], train_end)
        flow_profile = load_base_profile(
            args.source_profile_dir, dataset, "flow", loaded["flow"], train_end, flow_transform
        )
        flow_state = compute_directional_probabilistic_deficit(
            loaded["flow"], flow_profile, timestamps, "lower"
        )
        flow_normal = transform_conditional_normal_scores(
            loaded["flow"], flow_profile["profile"], flow_profile["ecdf"], timestamps,
            "data_driven_loading",
        )
        save_profile(dataset_dir, "demand", flow_profile)

        speed_profile = speed_state = speed_normal = None
        aligned_speed = None
        if loaded["speed"] is not None:
            speed_transform, _ = choose_transform(loaded["speed"], train_end)
            speed_profile = load_base_profile(
                args.source_profile_dir, dataset, "speed", loaded["speed"], train_end,
                speed_transform,
            )
            speed_state = compute_directional_probabilistic_deficit(
                loaded["speed"], speed_profile, timestamps, "lower"
            )
            speed_normal = transform_conditional_normal_scores(
                loaded["speed"], speed_profile["profile"], speed_profile["ecdf"], timestamps,
                "higher_is_better",
            )
            aligned_speed = align_modal_score(
                speed_normal["oriented_score"], loaded["speed_maps"], loaded["flow_maps"]
            )

        occupancy_profile = occupancy_normal = None
        aligned_occupancy = None
        if loaded["occupancy"] is not None:
            occupancy_transform, _ = choose_transform(loaded["occupancy"], train_end)
            occupancy_profile = load_base_profile(
                args.source_profile_dir, dataset, "occupancy", loaded["occupancy"], train_end,
                occupancy_transform,
            )
            occupancy_normal = transform_conditional_normal_scores(
                loaded["occupancy"], occupancy_profile["profile"], occupancy_profile["ecdf"],
                timestamps, "higher_is_worse",
            )
            aligned_occupancy = align_modal_score(
                occupancy_normal["oriented_score"], loaded["occupancy_maps"], loaded["flow_maps"]
            )

        l4_model = DemandEfficiencyModel().fit(
            flow_normal["oriented_score"], aligned_speed, aligned_occupancy, train_end,
            initialization_seeds=seeds, max_iter=args.factor_max_iter,
        )
        transformed = l4_model.transform(
            flow_normal["oriented_score"], aligned_speed, aligned_occupancy
        )
        model_path = dataset_dir / "demand_efficiency_model.npz"
        l4_model.save(model_path)
        reloaded = DemandEfficiencyModel.load(model_path)
        np.testing.assert_allclose(
            transformed["efficiency_score"],
            reloaded.transform(flow_normal["oriented_score"], aligned_speed, aligned_occupancy)["efficiency_score"],
            equal_nan=True,
        )
        model_report = {
            "dataset": dataset,
            "train_only": True,
            "train_end_exclusive": train_end,
            "efficiency_mode": l4_model.efficiency_mode_,
            "efficiency_variables": l4_model.variable_names_,
            "flow_excluded_from_efficiency": True,
            "scalar_joint_score_prohibited": True,
        }
        if l4_model.efficiency_factor_ is not None:
            model_report.update(
                loadings=l4_model.efficiency_factor_.loadings_.tolist(),
                residual_variance=l4_model.efficiency_factor_.residual_variance_.tolist(),
                communalities=l4_model.efficiency_factor_.communalities_.tolist(),
                converged=l4_model.efficiency_factor_.converged_,
                sign_anchor=l4_model.efficiency_factor_.sign_anchor_,
            )
        (dataset_dir / "demand_efficiency_model_report.json").write_text(
            json.dumps(model_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        efficiency_state = None
        efficiency_source = None
        if l4_model.efficiency_mode_ == "speed_observed":
            efficiency_state = speed_state
            efficiency_source = speed_state["ecdf_source_level"]
        elif l4_model.efficiency_mode_ == "speed_occupancy_factor":
            efficiency_profile = fit_latent_performance_profile(
                transformed["efficiency_score"], train_end, timestamps, **profile_kwargs
            )
            save_profile(dataset_dir, "efficiency", efficiency_profile)
            efficiency_state = compute_latent_resilience_deficit(
                transformed["efficiency_score"],
                transformed["efficiency_posterior_variance"],
                efficiency_profile["profile"],
                efficiency_profile["ecdf"],
                timestamps,
            )
            efficiency_source = efficiency_state["source_level"]
            factor = l4_model.efficiency_factor_
            score_cube = np.stack([aligned_speed, aligned_occupancy], axis=2)
            for variable_index, variable in enumerate(factor.variable_names_):
                loading_rows.append(
                    {
                        "dataset": dataset,
                        "variable": variable,
                        "loading": factor.loadings_[variable_index],
                        "residual_variance": factor.residual_variance_[variable_index],
                        "communality": factor.communalities_[variable_index],
                        "importance_rank": float(stats.rankdata(-np.abs(factor.loadings_))[variable_index]),
                    }
                )
            initialization_diagnostics.extend(
                initialization_rows(dataset, factor, score_cube)
            )
            blocked_rows.extend(
                blocked_factor_rows(
                    dataset, score_cube, timestamps, train_end, factor.variable_names_, args.seed,
                    max_folds=5 if args.max_nodes <= 5 else None,
                )
            )
            bootstrap_rows.extend(
                bootstrap_factor_rows(
                    dataset, score_cube, train_end, factor.variable_names_,
                    args.bootstrap_repetitions, args.block_length, args.seed,
                )
            )
            missing_rows.extend(
                missing_modality_rows(dataset, score_cube, factor, train_end, args.seed)
            )

        windows, _ = event_segments(dataset, loaded, flow_state)
        definitions = {
            "L4_demand_service": {
                "series": flow_state["system_deficit"],
                "variables": "flow",
                "source": flow_state["ecdf_source_level"],
                "variance": None,
            }
        }
        if efficiency_state is not None:
            definitions["L4_operating_efficiency"] = {
                "series": efficiency_state["system_deficit"],
                "variables": "speed" if l4_model.efficiency_mode_ == "speed_observed" else "speed,occupancy",
                "source": efficiency_source,
                "variance": transformed["efficiency_posterior_variance"],
            }

        dimension_summaries = {}
        for definition, payload in definitions.items():
            summary = definition_summary(
                dataset, definition, payload["variables"], payload["series"], train_end,
                test_start, windows, payload["source"], payload["variance"],
                args.bootstrap_repetitions, args.block_length, args.seed,
            )
            train_quantiles = summary.pop("train_quantiles")
            dimension_summaries[definition] = (summary, train_quantiles)
            comparison_rows.append(summary)
            process = analyze_event_segments(
                dataset, definition, payload["series"], windows, train_quantiles,
                delta_hours, recovery_quantile="q75", consecutive_steps=12,
            )
            event_mask_values = np.zeros(len(payload["series"]), dtype=bool)
            for start, end in windows:
                event_mask_values[start : end + 1] = True
            non_event = np.asarray(payload["series"])[~event_mask_values]
            for row in process:
                start = int(row["event_start"]); end = int(row["event_end"])
                event_values = np.asarray(payload["series"])[start : end + 1]
                row["event_non_event_median_difference"] = float(
                    np.nanmedian(event_values) - np.nanmedian(non_event)
                )
                row["event_non_event_cliffs_delta"] = float(
                    definition_summary(
                        dataset, definition, payload["variables"], payload["series"], train_end,
                        test_start, [(start, end)], payload["source"], payload["variance"],
                        args.bootstrap_repetitions, args.block_length, args.seed + int(row["event_segment"]),
                    )["event_non_event_cliffs_delta"]
                )
                event_rows.append(row)
            recovery_rows.extend(
                recovery_sensitivity_rows(
                    dataset, definition, payload["series"], windows, train_quantiles, delta_hours
                )
            )

        if efficiency_state is not None:
            demand_summary, demand_quantiles = dimension_summaries["L4_demand_service"]
            efficiency_summary, efficiency_quantiles = dimension_summaries["L4_operating_efficiency"]
            states = two_dimension_event_states(
                flow_state["system_deficit"], efficiency_state["system_deficit"],
                demand_quantiles["q90"], efficiency_quantiles["q90"],
            )
            quadrant_rows.extend(event_quadrant_rows(dataset, states, windows))

        metadata_rows.append(
            {
                "dataset": dataset,
                "flow_nodes": len(loaded["flow_maps"]),
                "speed_nodes": len(loaded["speed_maps"]),
                "occupancy_nodes": len(loaded["occupancy_maps"]),
                "efficiency_mode": l4_model.efficiency_mode_,
                "event_segments": len(windows),
                "train_end": train_end,
                "test_start": test_start,
            }
        )
    tables = {
        "l4_dimension_comparison.csv": pd.DataFrame(comparison_rows),
        "l4_event_level.csv": pd.DataFrame(event_rows),
        "l4_event_quadrants.csv": pd.DataFrame(quadrant_rows),
        "l4_efficiency_loadings.csv": pd.DataFrame(loading_rows),
        "l4_efficiency_initializations.csv": pd.DataFrame(initialization_diagnostics),
        "l4_efficiency_blocked_cv.csv": pd.DataFrame(blocked_rows),
        "l4_efficiency_bootstrap.csv": pd.DataFrame(bootstrap_rows),
        "l4_missing_modality.csv": pd.DataFrame(missing_rows),
        "l4_recovery_sensitivity.csv": pd.DataFrame(recovery_rows),
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    historical_path = Path(args.source_profile_dir) / "latent_resilience_definition_comparison.csv"
    historical = pd.read_csv(historical_path) if historical_path.exists() else pd.DataFrame()
    current = tables["l4_dimension_comparison.csv"]
    combined = pd.concat([historical, current], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "l0_l4_comparison.csv", index=False, encoding="utf-8-sig")

    def value(dataset, definition, column):
        selected = current[(current["dataset"] == dataset) & (current["definition"] == definition)]
        return float(selected.iloc[0][column]) if len(selected) and pd.notna(selected.iloc[0][column]) else np.nan

    reasons = []
    rain_efficiency = value("rainstorm", "L4_operating_efficiency", "event_non_event_cliffs_delta")
    if not np.isfinite(rain_efficiency) or rain_efficiency <= 0:
        reasons.append("Rainstorm 运行效率维度未呈现正向事件效应。")
    typhoon_events = tables["l4_event_level.csv"]
    typhoon_events = typhoon_events[
        (typhoon_events["dataset"] == "typhoon")
        & (typhoon_events["definition"] == "L4_operating_efficiency")
    ] if not typhoon_events.empty else typhoon_events
    positive_typhoon = int(
        np.sum(pd.to_numeric(typhoon_events.get("event_non_event_cliffs_delta", pd.Series(dtype=float)), errors="coerce") > 0)
    )
    if positive_typhoon < 2:
        reasons.append("少于两个 Typhoon 片段具有正向效率效应。")

    occupancy_restrictions = []
    for dataset in ("pems04", "pems08"):
        l4_rate = value(dataset, "L4_operating_efficiency", "test_high_state_rate")
        historical_speed = historical[
            (historical.get("dataset") == dataset) & (historical.get("definition") == "L1_speed")
        ] if not historical.empty else pd.DataFrame()
        speed_rate = float(historical_speed.iloc[0]["test_high_state_rate"]) if len(historical_speed) else np.nan
        if np.isfinite(l4_rate) and np.isfinite(speed_rate) and l4_rate > speed_rate + 0.03:
            occupancy_restrictions.append(
                f"{dataset.upper()} 的 speed+occupancy 效率扩展高状态率高于 speed-only。"
            )
    full_l4_accepted = not reasons and not occupancy_restrictions
    vector_framework_supported = not reasons
    decision = (
        "接受完整 L4 两维框架（包括 occupancy 效率扩展）"
        if full_l4_accepted
        else "仅支持需求/效率分离框架，拒绝或限制当前 occupancy 扩展"
        if vector_framework_supported
        else "拒绝当前 L4 两维定义"
    )

    report = [
        "# L4 需求—运行效率双维交通韧性报告",
        "",
        "## 1. 预注册定义",
        "",
        "L4 不再寻找统一标量。flow 仅进入需求/服务量维度；speed 与反向 occupancy 仅进入运行效率维度。两个维度分别建立 train-only 阈值、缺失曲线和恢复过程。",
        "",
        "## 2. 数据和效率模式",
        "",
        markdown_table(pd.DataFrame(metadata_rows)),
        "",
        "## 3. 数学解释",
        "",
        "需求维度使用 flow 条件分位状态，其下尾表示异常低服务量或需求，但不自动解释为效率下降。效率维度在仅有 speed 时直接使用 speed 条件状态；PEMS 使用 speed 与方向反转 occupancy 的缺失数据单因子。flow 被禁止进入效率因子。",
        "",
        "## 4. 两维结果",
        "",
        markdown_table(current[["dataset", "definition", "train_high_state_rate", "val_high_state_rate", "test_high_state_rate", "event_non_event_cliffs_delta", "bootstrap_ci_low", "bootstrap_ci_high", "event_high_state_overlap", "non_event_high_state_rate"]]),
        "",
        "## 5. 事件四象限",
        "",
        markdown_table(tables["l4_event_quadrants.csv"]),
        "",
        "both-high 表示服务量与效率同时异常；demand-only 表示主要是服务量/需求下降；efficiency-only 表示流量未明显下降但通行效率下降；neither 表示两个维度均未越过各自训练 q90。",
        "",
        "## 6. Typhoon 三段",
        "",
        markdown_table(typhoon_events),
        "",
        f"三个片段中有 {positive_typhoon} 个效率效应方向为正。事件窗口和 lag 没有为了 L4 调整。",
        "",
        "## 7. PEMS efficiency 扩展",
        "",
        markdown_table(tables["l4_efficiency_loadings.csv"]),
        "",
        markdown_table(tables["l4_efficiency_bootstrap.csv"]),
        "",
        "## 8. 缺失模态与不确定性",
        "",
        markdown_table(tables["l4_missing_modality.csv"]),
        "",
        "## 9. 事件峰值、累计缺失和恢复",
        "",
        markdown_table(tables["l4_event_level.csv"]),
        "",
        "恢复使用训练 q75 以下连续 12 步，且只能从事件结束后开始确认；完整敏感性结果保存在 l4_recovery_sensitivity.csv。",
        "",
        "## 10. 决策",
        "",
        f"结论：**{decision}**。",
        "",
        *[f"- {reason}" for reason in reasons],
        *[f"- {reason}" for reason in occupancy_restrictions],
        "",
        "L4 的接受只表示二维统计描述具有解释性，不授权把两个维度再次加权成一个标签，也不授权进入神经网络训练。",
        "",
        "## 11. 下一阶段限制",
        "",
        "在决定向量监督、多任务学习或事件过程损失之前，继续冻结 DGCN-STSGCN、训练损失、CVaR、uncertainty weighting 和 conformal prediction。",
    ]
    (output_dir / "l4_two_dimension_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    decision_metadata = {
        "decision": decision,
        "vector_framework_supported": vector_framework_supported,
        "full_l4_accepted": full_l4_accepted,
        "positive_typhoon_segments": positive_typhoon,
        "rejection_reasons": reasons,
        "occupancy_restrictions": occupancy_restrictions,
        "neural_network_authorized": False,
    }
    (output_dir / "l4_decision.json").write_text(
        json.dumps(decision_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(metadata_rows).to_string(index=False))
    print(json.dumps(decision_metadata, ensure_ascii=False))
    print(f"OUTPUT {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="bridge,rainstorm,typhoon,pems04,pems08")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-profile-dir", default=r"D:\TrafficGNN\outputs\latent_traffic_performance_l3"
    )
    parser.add_argument("--time-of-day-bins", type=int, default=288)
    parser.add_argument("--day-type-mode", default="weekday_weekend")
    parser.add_argument("--shrinkage-candidates", default="0,1,3,7,14,28")
    parser.add_argument("--max-nodes", type=int, default=41)
    parser.add_argument("--factor-initialization-seeds", default="1,7,21,42,100")
    parser.add_argument("--factor-max-iter", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()