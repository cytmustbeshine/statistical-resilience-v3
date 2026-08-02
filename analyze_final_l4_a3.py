"""Analyze the final three-seed frozen-L4 M0-M3 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_latent_traffic_performance import load_saved_profile
from audit_final_l4_a3_alignment import EVENT_WINDOWS
from l4_prediction_evaluation import apply_frozen_lower_tail_profile, binary_state_metrics, regression_metrics
from l4_prediction_pipeline import load_ordered_univariate_series_physical, load_prediction_archive, profile_compatible_event_external_split
from run_l4_prediction_baseline import variable_plan


SEEDS = (42, 2024, 3407)
MODELS = ("m0", "m1", "m2", "m3")


def profile_context(dataset: str, variable: str, args, cache):
    key = (dataset, variable)
    if key in cache:
        return cache[key]
    config, full_columns, model_columns, full_indices, label, root_kind = variable_plan(dataset, variable, 41)
    values, timestamps, _ = load_ordered_univariate_series_physical(
        str(config["csv"]), str(config["time_col"]), full_columns, config[f"{variable}_suffix"]
    )
    train_end = profile_compatible_event_external_split(dataset)["train_time_end_exclusive"]
    root = Path(args.l4_dir) / dataset if root_kind == "l4" else Path(args.source_profile_dir) / dataset
    profile = load_saved_profile(root, label, values[..., 0], train_end, "log1p_nonnegative")
    if profile is None:
        raise RuntimeError(f"frozen profile unavailable: {dataset}/{variable}")
    from flow_speed_resilience import compute_directional_probabilistic_deficit
    reference = compute_directional_probabilistic_deficit(values[..., 0], profile, timestamps, "lower")
    cache[key] = {
        "config": config, "values": values, "timestamps": timestamps, "profile": profile,
        "full_indices": full_indices, "model_columns": model_columns,
        "q75": float(reference["train_deficit_quantiles"]["q75"]),
        "q90": float(reference["train_deficit_quantiles"]["q90"]),
        "q99": float(reference["train_deficit_quantiles"]["q99"]),
        "clip": float(reference["deficit_clip_value"]),
    }
    return cache[key]


def aggregate_unique(timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps).reshape(-1).astype("datetime64[ns]")
    array = np.asarray(values, dtype=float).reshape(len(times), -1)
    unique, inverse = np.unique(times, return_inverse=True)
    sums = np.zeros((len(unique), array.shape[1]), dtype=float)
    counts = np.zeros_like(sums)
    for row, group in enumerate(inverse):
        valid = np.isfinite(array[row])
        sums[group, valid] += array[row, valid]
        counts[group, valid] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts
    mean[counts == 0] = np.nan
    return unique, mean


def derived_l4(archive, context):
    truth = np.asarray(archive["y_true_physical"], dtype=float)
    pred = np.asarray(archive["y_pred_physical"], dtype=float)
    shape = truth.shape[:2]
    times = np.asarray(archive["target_timestamps"]).reshape(-1)
    true_state = apply_frozen_lower_tail_profile(
        truth[..., 0].reshape(-1, truth.shape[2]), times, context["profile"],
        context["values"].shape[1], context["full_indices"], context["clip"],
    )
    pred_state = apply_frozen_lower_tail_profile(
        pred[..., 0].reshape(-1, pred.shape[2]), times, context["profile"],
        context["values"].shape[1], context["full_indices"], context["clip"],
    )
    return true_state["system_deficit"].reshape(shape), pred_state["system_deficit"].reshape(shape)


def record_paths(root: Path):
    rows = []
    for seed in SEEDS:
        for dataset in ("bridge", "rainstorm", "typhoon"):
            variables = ("flow",) if dataset == "bridge" else ("flow", "speed")
            for variable in variables:
                rows.append(("m0", dataset, variable, seed,
                    root / "m0_dcrnn" / dataset / variable / f"seed_{seed}" / "traffic_predictions.npz", None))
                m1_result = root / "m1_dstsgcn" / f"seed_{seed}" / dataset / variable / "smoke_result.json"
                m1 = json.loads(m1_result.read_text(encoding="utf-8"))
                rows.append(("m1", dataset, variable, seed, Path(m1["prediction_archive"]), None))
                for model in ("m2", "m3"):
                    result_path = root / "m2_m3_dstsgcn" / f"seed_{seed}" / model / dataset / variable / f"seed_{seed}" / "result.json"
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    rows.append((model, dataset, variable, seed, Path(result["traffic_archive"]), Path(result["l4_archive"])))
    return rows


def binary_extreme(truth, pred, threshold):
    valid = np.isfinite(truth) & np.isfinite(pred)
    a = truth[valid] > threshold; b = pred[valid] > threshold
    tp = int(np.sum(a & b)); fp = int(np.sum(~a & b)); fn = int(np.sum(a & ~b))
    return {
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "predicted_rate": float(np.mean(b)) if b.size else np.nan,
    }


def recovery_duration(times, values, threshold, hold=12):
    values = np.asarray(values, float); times = np.asarray(times).astype("datetime64[ns]")
    if not np.isfinite(values).any(): return np.nan, "unavailable"
    peak = int(np.nanargmax(values))
    for index in range(peak, len(values) - hold + 1):
        window = values[index:index + hold]
        if np.isfinite(window).all() and np.all(window <= threshold):
            hours = float((times[index] - times[peak]).astype("timedelta64[s]").astype(float) / 3600.0)
            return hours, "recovered"
    return np.nan, "censored"


def event_rows(series, contexts):
    rows = []
    for (model, dataset, variable, seed), values in series.items():
        context = contexts[(dataset, variable)]
        raw_times = context["timestamps"].astype("datetime64[ns]")
        dt = np.median(np.diff(raw_times).astype("timedelta64[s]").astype(float)) / 3600.0
        for event_id, (start, end) in enumerate(EVENT_WINDOWS[dataset]):
            event_start, event_end = raw_times[start], raw_times[end]
            mask = (values["timestamps"] >= event_start) & (values["timestamps"] <= event_end)
            if not mask.any():
                rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,"event_id":event_id,"available":False})
                continue
            times = values["timestamps"][mask]; truth = values["true_l4"][mask]; pred = values["pred_l4"][mask]
            true_peak = int(np.nanargmax(truth)); pred_peak = int(np.nanargmax(pred))
            true_rec, true_status = recovery_duration(times, truth, context["q75"])
            pred_rec, pred_status = recovery_duration(times, pred, context["q75"])
            rows.append({
                "model":model,"dataset":dataset,"variable":variable,"seed":seed,"event_id":event_id,"available":True,
                "true_peak_deficit":float(np.nanmax(truth)),"pred_peak_deficit":float(np.nanmax(pred)),
                "peak_deficit_error":float(abs(np.nanmax(pred)-np.nanmax(truth))),
                "peak_timing_error_hours":float(abs((times[pred_peak]-times[true_peak]).astype("timedelta64[s]").astype(float))/3600.0),
                "true_cumulative_deficit":float(np.nansum(truth)*dt),"pred_cumulative_deficit":float(np.nansum(pred)*dt),
                "cumulative_deficit_error":float(abs(np.nansum(pred)-np.nansum(truth))*dt),
                "true_high_duration_hours":float(np.sum(truth>context["q90"])*dt),"pred_high_duration_hours":float(np.sum(pred>context["q90"])*dt),
                "true_extreme_duration_hours":float(np.sum(truth>context["q99"])*dt),"pred_extreme_duration_hours":float(np.sum(pred>context["q99"])*dt),
                "true_recovery_duration_hours":true_rec,"pred_recovery_duration_hours":pred_rec,
                "true_recovery_status":true_status,"pred_recovery_status":pred_status,
            })
    return rows


def moving_block_indices(n, block, rng):
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(start, min(start + block, n)) for start in starts])[:n]


def bootstrap_rows(series, repetitions, block_length, seed):
    rows=[]; rng=np.random.default_rng(seed)
    for dataset in ("bridge","rainstorm","typhoon"):
        variables=("flow",) if dataset=="bridge" else ("flow","speed")
        for variable in variables:
            for run_seed in SEEDS:
                candidate=series[("m3",dataset,variable,run_seed)]
                for baseline in ("m0","m1","m2"):
                    other=series[(baseline,dataset,variable,run_seed)]
                    common, ia, ib=np.intersect1d(candidate["timestamps"],other["timestamps"],return_indices=True)
                    c={key:np.asarray(candidate[key])[ia] for key in ("traffic_abs","traffic_sq","true_l4","pred_l4")}
                    b={key:np.asarray(other[key])[ib] for key in ("traffic_abs","traffic_sq","true_l4","pred_l4")}
                    n=len(common); estimates={name:[] for name in ("traffic_mae","traffic_rmse","l4_mae","l4_tail_mae","high_f1")}
                    q90=candidate["q90"]
                    for _ in range(repetitions):
                        idx=moving_block_indices(n,min(block_length,n),rng)
                        estimates["traffic_mae"].append(float(np.nanmean(c["traffic_abs"][idx])-np.nanmean(b["traffic_abs"][idx])))
                        estimates["traffic_rmse"].append(float(np.sqrt(np.nanmean(c["traffic_sq"][idx]))-np.sqrt(np.nanmean(b["traffic_sq"][idx]))))
                        estimates["l4_mae"].append(float(np.nanmean(np.abs(c["pred_l4"][idx]-c["true_l4"][idx]))-np.nanmean(np.abs(b["pred_l4"][idx]-b["true_l4"][idx]))))
                        tail=np.isfinite(c["true_l4"][idx])&(c["true_l4"][idx]>q90)
                        estimates["l4_tail_mae"].append(float(np.nanmean(np.abs(c["pred_l4"][idx][tail]-c["true_l4"][idx][tail]))-np.nanmean(np.abs(b["pred_l4"][idx][tail]-b["true_l4"][idx][tail]))) if tail.any() else np.nan)
                        cm=binary_state_metrics(c["true_l4"][idx],c["pred_l4"][idx],q90)["f1"]
                        bm=binary_state_metrics(b["true_l4"][idx],b["pred_l4"][idx],q90)["f1"]
                        estimates["high_f1"].append(float(cm-bm))
                    for metric, samples in estimates.items():
                        arr=np.asarray(samples,float); arr=arr[np.isfinite(arr)]
                        point=float(np.mean(arr)) if arr.size else np.nan
                        lo,hi=(np.quantile(arr,[.025,.975]) if arr.size else (np.nan,np.nan))
                        rows.append({"candidate":"m3","baseline":baseline,"dataset":dataset,"variable":variable,"seed":run_seed,"metric":metric,"difference":point,"ci_low":float(lo),"ci_high":float(hi),"block_length":block_length,"repetitions":repetitions})
    return rows


def event_bootstrap_rows(series, contexts, repetitions, block_length, seed):
    """Bootstrap paired event-process errors without treating time points as iid."""
    rows = []
    rng = np.random.default_rng(seed + 7919)
    dt_hours = 5.0 / 60.0
    for dataset in ("bridge", "rainstorm", "typhoon"):
        variables = ("flow",) if dataset == "bridge" else ("flow", "speed")
        for variable in variables:
            context = contexts[(dataset, variable)]
            raw_times = context["timestamps"].astype("datetime64[ns]")
            for run_seed in SEEDS:
                candidate = series[("m3", dataset, variable, run_seed)]
                for baseline in ("m0", "m1", "m2"):
                    other = series[(baseline, dataset, variable, run_seed)]
                    common, ia, ib = np.intersect1d(
                        candidate["timestamps"], other["timestamps"], return_indices=True
                    )
                    for event_id, (start, end) in enumerate(EVENT_WINDOWS[dataset]):
                        start_time, end_time = raw_times[start], raw_times[end]
                        mask = (common >= start_time) & (common <= end_time)
                        if not mask.any():
                            continue
                        ct = candidate["true_l4"][ia[mask]]
                        cp = candidate["pred_l4"][ia[mask]]
                        bp = other["pred_l4"][ib[mask]]
                        truth = candidate["true_l4"][ia[mask]]
                        n = len(truth)
                        estimates = {"peak_deficit_error": [], "cumulative_deficit_error": []}
                        for _ in range(repetitions):
                            indices = moving_block_indices(n, min(block_length, n), rng)
                            t = ct[indices]
                            c = cp[indices]
                            b = bp[indices]
                            valid_c = np.isfinite(t) & np.isfinite(c)
                            valid_b = np.isfinite(t) & np.isfinite(b)
                            if valid_c.any() and valid_b.any():
                                estimates["peak_deficit_error"].append(
                                    abs(np.nanmax(c[valid_c]) - np.nanmax(t[valid_c]))
                                    - abs(np.nanmax(b[valid_b]) - np.nanmax(t[valid_b]))
                                )
                                estimates["cumulative_deficit_error"].append(
                                    abs(np.nansum(c[valid_c]) - np.nansum(t[valid_c])) * dt_hours
                                    - abs(np.nansum(b[valid_b]) - np.nansum(t[valid_b])) * dt_hours
                                )
                        for metric, values in estimates.items():
                            values = np.asarray(values, dtype=float)
                            values = values[np.isfinite(values)]
                            lo, hi = np.quantile(values, [0.025, 0.975])
                            rows.append({
                                "candidate": "m3", "baseline": baseline,
                                "dataset": dataset, "variable": variable,
                                "seed": run_seed, "event_id": event_id,
                                "metric": metric, "difference": float(np.mean(values)),
                                "ci_low": float(lo), "ci_high": float(hi),
                                "block_length": block_length, "repetitions": repetitions,
                            })
    return rows


def macro_f1_from_codes(true, pred):
    scores = []
    for state in range(4):
        tp = np.sum((true == state) & (pred == state))
        fp = np.sum((true != state) & (pred == state))
        fn = np.sum((true == state) & (pred != state))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def run(args):
    root=Path(args.formal_root); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    contexts={}; traffic_rows=[]; horizon_rows=[]; continuous_rows=[]; tail_rows=[]; high_rows=[]; extreme_rows=[]; series={}
    for model,dataset,variable,seed,traffic_path,l4_path in record_paths(root):
        context=profile_context(dataset,variable,args,contexts)
        archive=load_prediction_archive(traffic_path)
        truth=np.asarray(archive["y_true_physical"],float); pred=np.asarray(archive["y_pred_physical"],float); times=np.asarray(archive["target_timestamps"])
        traffic=regression_metrics(truth,pred)
        # Final L4 evaluation must be the same deterministic postprocessing for
        # every model: apply the frozen profile to physical traffic forecasts.
        # M2/M3 auxiliary-head archives are training diagnostics, not the final
        # indirect L4 prediction, and must not be used for a fairness comparison.
        true_l4,pred_l4=derived_l4(archive,context)
        l4=regression_metrics(true_l4,pred_l4); high=binary_state_metrics(true_l4,pred_l4,context["q90"]); extreme=binary_extreme(true_l4,pred_l4,context["q99"])
        tail90=np.isfinite(true_l4)&np.isfinite(pred_l4)&(true_l4>context["q90"]); tail99=np.isfinite(true_l4)&np.isfinite(pred_l4)&(true_l4>context["q99"])
        traffic_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,**traffic})
        continuous_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,**l4})
        tail_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,"q90":context["q90"],"q99":context["q99"],"l4_tail_mae_q90":float(np.mean(np.abs(pred_l4[tail90]-true_l4[tail90]))) if tail90.any() else np.nan,"l4_extreme_mae_q99":float(np.mean(np.abs(pred_l4[tail99]-true_l4[tail99]))) if tail99.any() else np.nan,"q90_count":int(tail90.sum()),"q99_count":int(tail99.sum())})
        high_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,**high})
        extreme_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,**extreme})
        for horizon in range(truth.shape[1]):
            horizon_rows.append({"model":model,"dataset":dataset,"variable":variable,"seed":seed,"horizon":horizon+1,"traffic_mae":regression_metrics(truth[:,horizon],pred[:,horizon])["mae"],"l4_mae":regression_metrics(true_l4[:,horizon],pred_l4[:,horizon])["mae"]})
        unique, traffic_abs=aggregate_unique(times,np.nanmean(np.abs(pred-truth),axis=(2,3)))
        _, traffic_sq=aggregate_unique(times,np.nanmean((pred-truth)**2,axis=(2,3)))
        _, true_unique=aggregate_unique(times,true_l4); _, pred_unique=aggregate_unique(times,pred_l4)
        series[(model,dataset,variable,seed)]={"timestamps":unique,"traffic_abs":traffic_abs[:,0],"traffic_sq":traffic_sq[:,0],"true_l4":true_unique[:,0],"pred_l4":pred_unique[:,0],"q90":context["q90"],"q99":context["q99"]}

    event=pd.DataFrame(event_rows(series,contexts))
    bootstrap=pd.DataFrame(
        bootstrap_rows(series,args.bootstrap_repetitions,args.block_length,args.seed)
        + event_bootstrap_rows(series,contexts,args.bootstrap_repetitions,args.block_length,args.seed)
    )
    traffic=pd.DataFrame(traffic_rows); continuous=pd.DataFrame(continuous_rows); tails=pd.DataFrame(tail_rows); high=pd.DataFrame(high_rows); extreme=pd.DataFrame(extreme_rows); horizon=pd.DataFrame(horizon_rows)
    confusion=[]; four_state_metrics=[]
    for model in MODELS:
        for dataset in ("rainstorm","typhoon"):
            for seed in SEEDS:
                d=series[(model,dataset,"flow",seed)]; e=series[(model,dataset,"speed",seed)]
                common,di,ei=np.intersect1d(d["timestamps"],e["timestamps"],return_indices=True)
                true=np.where((d["true_l4"][di]>d["q90"])&(e["true_l4"][ei]>e["q90"]),0,np.where(d["true_l4"][di]>d["q90"],1,np.where(e["true_l4"][ei]>e["q90"],2,3)))
                pred=np.where((d["pred_l4"][di]>d["q90"])&(e["pred_l4"][ei]>e["q90"]),0,np.where(d["pred_l4"][di]>d["q90"],1,np.where(e["pred_l4"][ei]>e["q90"],2,3)))
                valid=np.isfinite(d["true_l4"][di])&np.isfinite(e["true_l4"][ei])&np.isfinite(d["pred_l4"][di])&np.isfinite(e["pred_l4"][ei])
                for a in range(4):
                    for b in range(4): confusion.append({"model":model,"dataset":dataset,"seed":seed,"true_state":a,"pred_state":b,"count":int(np.sum((true==a)&(pred==b)))})
                four_state_metrics.append({"model":model,"dataset":dataset,"seed":seed,"macro_f1":macro_f1_from_codes(true[valid],pred[valid]) if valid.any() else np.nan})
    confusion=pd.DataFrame(confusion)
    four_state_metrics=pd.DataFrame(four_state_metrics)
    ablation=traffic.merge(continuous,on=["model","dataset","variable","seed"],suffixes=("_traffic","_l4")).merge(tails,on=["model","dataset","variable","seed"])
    stability=ablation.groupby(["model","dataset","variable"]).agg({"mae_traffic":["mean","std"],"mae_l4":["mean","std"],"l4_tail_mae_q90":["mean","std"],"l4_extreme_mae_q99":["mean","std"]}).reset_index(); stability.columns=['_'.join(c).strip('_') for c in stability.columns]
    files={"traffic_prediction_metrics.csv":traffic,"traffic_horizon_metrics.csv":horizon,"l4_continuous_metrics.csv":continuous,"l4_tail_metrics.csv":tails,"l4_high_state_metrics.csv":high,"l4_extreme_state_metrics.csv":extreme,"l4_four_state_confusion.csv":confusion,"l4_four_state_metrics.csv":four_state_metrics,"l4_event_process_metrics.csv":event,"l4_model_ablation.csv":ablation,"l4_block_bootstrap_comparison.csv":bootstrap,"l4_seed_stability.csv":stability}
    for name,frame in files.items(): frame.to_csv(output/name,index=False,encoding='utf-8-sig')
    pivot=ablation.pivot_table(index=['dataset','variable','seed'],columns='model',values=['mae_traffic','mae_l4','l4_tail_mae_q90'])
    m3_better_m0=int((pivot['mae_traffic']['m3']<pivot['mae_traffic']['m0']).sum()); m2_better_m1=int((pivot['mae_l4']['m2']<pivot['mae_l4']['m1']).sum()); m3_tail_better_m2=int((pivot['l4_tail_mae_q90']['m3']<pivot['l4_tail_mae_q90']['m2']).sum())
    nondegenerate=int(((high[high.model=='m3'].predicted_high_rate>0)&(high[high.model=='m3'].predicted_high_rate<1)).sum())
    bootstrap_positive=int(((bootstrap.metric.isin(['traffic_mae','l4_mae','l4_tail_mae','peak_deficit_error','cumulative_deficit_error']))&(bootstrap.ci_high<0)).sum())
    accept_aux=m2_better_m1>=8; accept_cvar=m3_tail_better_m2>=8; accept_m3_dcrnn=m3_better_m0>=8
    decision={"stage":"E-L4-2B","training_completed":True,"seeds":list(SEEDS),"models":list(MODELS),"formal_runs":60,"final_l4_evaluation":"frozen_profile_postprocess_from_physical_traffic_predictions","auxiliary_l4_archives_used_for_final_metrics":False,"m3_traffic_better_than_m0_count":m3_better_m0,"m2_l4_better_than_m1_count":m2_better_m1,"m3_tail_better_than_m2_count":m3_tail_better_m2,"comparisons_total":15,"m3_nondegenerate_high_state_count":nondegenerate,"bootstrap_significant_improvements":bootstrap_positive,"accept_l4_auxiliary_supervision":accept_aux,"accept_cvar_tail_optimization":accept_cvar,"accept_a3_l4_over_dcrnn":accept_m3_dcrnn,"final_model_selected":False,"e_l4_2b_accepted":bool(accept_aux and accept_cvar and accept_m3_dcrnn),"three_seed_completed":True,"blockers":[]}
    if not decision['e_l4_2b_accepted']: decision['blockers'].append('Scientific acceptance criteria are not jointly satisfied.')
    (output/'final_l4_a3_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=stability[['model','dataset','variable','mae_traffic_mean','mae_l4_mean','l4_tail_mae_q90_mean']]
    report=['# E-L4-2B 最终三种子分析','',f"正式60个模型任务全部完成。最终联合接受：{'是' if decision['e_l4_2b_accepted'] else '否'}。",'',summary.to_string(index=False),'','## 消融判定',f"- M2 L4 MAE 优于 M1：{m2_better_m1}/15。",f"- M3 q90 tail MAE 优于 M2：{m3_tail_better_m2}/15。",f"- M3 traffic MAE 优于 DCRNN：{m3_better_m0}/15。",f"- M3 high-state 非退化：{nondegenerate}/15。",f"- bootstrap 显著改善项：{bootstrap_positive}。",'',f"L4辅助监督：{'接受' if accept_aux else '拒绝'}。",f"CVaR尾部优化：{'接受' if accept_cvar else '拒绝'}。",f"A3-L4优于DCRNN：{'接受' if accept_m3_dcrnn else '拒绝'}。",'',"本结论不表示神经网络已经学习完整交通韧性；Bridge 仍仅为 flow 代理，demand 与 efficiency 始终分开。"]
    (output/'final_l4_a3_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(decision,ensure_ascii=False))


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--formal-root',default=r'D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal_final'); p.add_argument('--output-dir',default=r'D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\analysis_final'); p.add_argument('--l4-dir',default=r'D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4'); p.add_argument('--source-profile-dir',default=r'D:\TrafficGNN\outputs\latent_traffic_performance_l3'); p.add_argument('--bootstrap-repetitions',type=int,default=2000); p.add_argument('--block-length',type=int,default=12); p.add_argument('--seed',type=int,default=42); return p.parse_args()
if __name__=='__main__': run(parse_args())
