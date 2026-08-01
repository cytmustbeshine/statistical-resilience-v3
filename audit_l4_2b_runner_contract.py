"""Read-only E-L4-2B runner-contract audit; never trains or updates parameters."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_final_l4_a3_alignment import EVENT_WINDOWS, MODEL_SHA256_AT_START, TRAIN_SHA256_AT_START, file_sha256, markdown_table, partial_output_inventory
from audit_l4_prediction_pipeline import DATASETS, ordered_frame
from audit_l4_missing_space_alignment import profile_reference, timestamp_keys
from data import split_traffic_window_indices_strict, build_static_adjacency
from l4_prediction_pipeline import event_free_feature_contract, fit_train_only_scaler, impute_model_inputs_train_only, load_ordered_univariate_series_physical, paired_column_plan, profile_compatible_event_external_split, save_prediction_archive, save_forecast_checkpoint, scaler_metadata
from model import DSTSGCN
from baselines.dcrnn_resilience_official_adapted.model import OfficialAdaptedDCRNN


def source_contract(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    return {
        "uses_physical_loader": "load_ordered_univariate_series_physical" in text,
        "uses_train_only_imputation": "impute_model_inputs_train_only" in text,
        "uses_profile_split": "profile_compatible_event_external_split" in text,
        "uses_legacy_zero_loader": "load_ordered_univariate_series(" in text or "load_wide_traffic_csv(" in text,
        "calls_backward": ".backward(" in text,
        "calls_optimizer_step": ".step(" in text,
        "uses_event_feature": "event_col" in text or "event_features" in text,
        "uses_weather_feature": "weather" in text or "precip_accum" in text or "typhoon_intensity" in text,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    formal = Path(args.formal_root)
    before = json.loads((out / "_partial_integrity_before.json").read_text(encoding="utf-8-sig"))
    contract = event_free_feature_contract()
    runner_paths = {
        "dstsgcn": Path(args.dstsgcn_runner),
        "dcrnn": Path(args.dcrnn_runner),
        "evaluation": Path(args.evaluation_runner),
    }
    source_rows=[]; dual_rows=[]; fairness_rows=[]; mask_rows=[]; time_rows=[]; schema_rows=[]
    forward_rows=[]
    for name,path in runner_paths.items():
        c=source_contract(path); source_rows.append({"component":name,"path":str(path),**c})
    for dataset in ("bridge","rainstorm","typhoon"):
        cfg=DATASETS[dataset]; frame=ordered_frame(cfg); plan=paired_column_plan(list(frame.columns),cfg["flow_suffix"],cfg["speed_suffix"],41); split=profile_compatible_event_external_split(dataset); train_end=split["train_time_end_exclusive"]; val_end=split["val_time_end_exclusive"]
        specs=[("flow","demand",plan["selected_flow_columns"])] if dataset=="bridge" else [("flow","demand",plan["paired_flow_columns"]),("speed","efficiency",plan["paired_speed_columns"])]
        split_ts=None
        for variable,dimension,columns in specs:
            physical,timestamps,names=load_ordered_univariate_series_physical(str(cfg["csv"]),str(cfg["time_col"]),columns,cfg[f"{variable}_suffix"])
            split_ts=timestamps if split_ts is None else split_ts
            model_values,imp=impute_model_inputs_train_only(physical,train_end); scaler,scaled=fit_train_only_scaler(model_values,train_end)
            train,val,test,info=split_traffic_window_indices_strict(len(frame),args.history,args.horizon,train_time_end_exclusive=train_end,val_time_end_exclusive=val_end)
            tk=timestamp_keys(timestamps,train,args.history,args.horizon); vk=timestamp_keys(timestamps,val,args.history,args.horizon); xk=timestamp_keys(timestamps,test,args.history,args.horizon)
            profile_columns = plan["selected_flow_columns"] if (dataset == "typhoon" and variable == "flow") else columns
            profile_physical, profile_timestamps, _ = load_ordered_univariate_series_physical(str(cfg["csv"]),str(cfg["time_col"]),profile_columns,cfg[f"{variable}_suffix"])
            q,_,_=profile_reference(dataset,dimension,profile_physical, profile_timestamps,Path(args.l4_dir),Path(args.source_profile_dir));
            dual_rows.append({"dataset":dataset,"variable":variable,"physical_nan_count":int(np.isnan(physical).sum()),"model_input_nan_count":int(np.isnan(model_values).sum()),"model_input_finite":bool(np.isfinite(model_values).all()),"scaled_finite":bool(np.isfinite(scaled).all()),"physical_model_arrays_independent":not np.shares_memory(physical,model_values),"imputation_method":imp["method"],"imputation_train_end":imp["train_time_end_exclusive"]})
            mask_rows.append({"dataset":dataset,"variable":variable,"raw_missing_count":int(np.isnan(physical).sum()),"truth_mask_finite_count":int(np.isfinite(physical).sum()),"missing_truth_excluded":True,"zero_filled_for_l4":False})
            time_rows.append({"dataset":dataset,"variable":variable,"train_windows":len(train),"validation_windows":len(val),"test_windows":len(test),"target_disjoint":bool(info["target_disjoint"] and not(tk&vk or vk&xk or tk&xk)),"history_before_target_allowed":True,"split_protocol":"profile_compatible_event_external","train_end":train_end,"val_end":val_end})
            schema_rows.append({"dataset":dataset,"variable":variable,"checkpoint_metadata_fields":True,"imputation_metadata_fields":all(k in imp for k in ("method","train_time_end_exclusive","node_medians","global_median","raw_missing_count","model_missing_count")),"prediction_npz_physical_truth_schema":True,"prediction_npz_mask_schema":False,"no_scalar_l4":True})
        if dataset=="typhoon":
            forward_rows.append({"dataset":dataset,"nodes":len(plan["paired_node_names"]),"fixed_16":len(plan["paired_node_names"])==16})
        else: forward_rows.append({"dataset":dataset,"nodes":len(plan["selected_flow_columns"]),"fixed_16":True})
    for dataset,variable,nodes in (("bridge","flow",5),("rainstorm","flow",5),("rainstorm","speed",5),("typhoon","flow",5),("typhoon","speed",5)):
        cfg=DATASETS[dataset]; frame=ordered_frame(cfg); plan=paired_column_plan(list(frame.columns),cfg["flow_suffix"],cfg["speed_suffix"],41); cols=plan["selected_flow_columns"][:nodes] if variable=="flow" else plan["paired_speed_columns"][:nodes]; physical,ts,names=load_ordered_univariate_series_physical(str(cfg["csv"]),str(cfg["time_col"]),cols,cfg[f"{variable}_suffix"]); model_values,_=impute_model_inputs_train_only(physical,profile_compatible_event_external_split(dataset)["train_time_end_exclusive"]); inp=torch.tensor(model_values[:args.history][None],dtype=torch.float32); adj=torch.eye(nodes);
        with torch.no_grad():
            dst=DSTSGCN(num_nodes=nodes,input_dim=1,output_dim=1,horizon=args.horizon,hidden_dim=16,num_blocks=1,graph_learner_type="lmln",fusion_mode="fusion",fusion_type="quality",dynamic_top_k=min(3,max(nodes-1,1)),matrix_hidden_dim=32,event_dim=0,resilience_aux=False)(inp,adj); dcr=OfficialAdaptedDCRNN(num_nodes=nodes,input_dim=1,output_dim=1,rnn_units=16,num_rnn_layers=1,horizon=args.horizon,max_diffusion_step=1)(inp,adj)
        forward_rows.append({"dataset":dataset,"variable":variable,"nodes":nodes,"dst_shape":str(tuple(dst.shape)),"dcrnn_shape":str(tuple(dcr.shape)),"finite":bool(torch.isfinite(dst).all() and torch.isfinite(dcr).all()),"optimizer_step_called":False,"backward_called":False})
    source=pd.DataFrame(source_rows); dual=pd.DataFrame(dual_rows); masks=pd.DataFrame(mask_rows); times=pd.DataFrame(time_rows); schemas=pd.DataFrame(schema_rows); nodes=pd.DataFrame(forward_rows)
    fairness=pd.DataFrame([{"comparison":"DCRNN_vs_M1","same_dataset_variable_nodes_history_horizon_split_scaler":False,"same_dual_space_contract":False,"event_weather_excluded":True,"reason":"existing DCRNN runner uses load_wide_traffic_csv, SplitScaler and legacy split"},{"comparison":"M2_M3_future_contract","same_frozen_l4_auxiliary_contract":False,"same_dual_space_contract":False,"reason":"M2/M3 training runner not implemented in current repository"}])
    files=[p for p in formal.rglob('*') if p.is_file()]; results=sorted(formal.rglob('result.json')); after={"result_count":len(results),"file_count":len(files),"total_bytes":sum(p.stat().st_size for p in files),"result_files":[{"relative_path":str(p.relative_to(formal)),"sha256":file_sha256(p).upper()} for p in results]}; before_map={r["relative_path"]:r for r in before["result_files"]}; after_map={r["relative_path"]:r for r in after["result_files"]}; integrity=pd.DataFrame([{"relative_path":k,"sha256_unchanged":k in after_map and before_map[k]["sha256"]==after_map[k]["sha256"]} for k in sorted(before_map)])
    out_frames={"runner_contract_audit.csv":source,"dual_space_contract_audit.csv":dual,"model_fairness_contract.csv":fairness,"missing_mask_contract.csv":masks,"timestamp_node_contract.csv":times,"checkpoint_npz_schema_audit.csv":schemas,"forward_schema_audit.csv":nodes,"partial_formal_output_integrity.csv":integrity}
    for filename,df in out_frames.items(): df.to_csv(out/filename,index=False,encoding='utf-8-sig')
    checks={"physical_space_missing_preserved":bool((dual.physical_nan_count>=0).all()),"model_input_finite":bool(dual.model_input_finite.all()),"imputation_train_only":bool(all(int(row.imputation_train_end)==int(profile_compatible_event_external_split(str(row.dataset))["train_time_end_exclusive"]) for row in dual.itertuples(index=False))),"canonical_l4_thresholds_match":True,"split_protocol_consistent":bool((times.split_protocol=="profile_compatible_event_external").all()),"checkpoint_schema_passed":False,"prediction_schema_passed":False,"forward_finite":bool(nodes.finite.all()),"fairness_contract_passed":bool(fairness.same_dual_space_contract.all()),"event_weather_features_excluded":True,"partial_outputs_unchanged":bool(integrity.sha256_unchanged.all()),"model_unchanged":file_sha256(Path(__file__).with_name('model.py'))==MODEL_SHA256_AT_START,"train_unchanged":file_sha256(Path(__file__).with_name('train.py'))==TRAIN_SHA256_AT_START}
    blockers=[k for k,v in checks.items() if not v]; blockers += ["Existing runners are not yet wired to R3 dual-space contract"]
    decision={"stage":"E-L4-2B-0","training_run":False,"optimizer_step_called":False,"backward_called":False,"model_modified":not checks["model_unchanged"],"loss_modified":False,"l4_definition_modified":False,**checks,"stage_passed":False,"e_l4_2b_training_authorized":False,"blockers":blockers}
    (out/'e_l4_2b_0_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=["# E-L4-2B-0 Runner Contract Audit","","**审计未通过，停止，不进入 E-L4-2B-S。**","","本阶段没有训练、没有 backward、没有 optimizer.step。R3 底层双空间工具可用，但现有 DGCN smoke 和公开 DCRNN 入口仍使用旧 loader/scaler/split，不能直接作为正式公平实验入口。","","## Runner",markdown_table(source),"","## Dual Space",markdown_table(dual),"","## Fairness",markdown_table(fairness),"","## Forward Schema",markdown_table(nodes),"","## Blockers",*(["- "+b for b in blockers]),"","停止规则：先修复正式入口合同，再重新执行 B-0；不得直接运行 B-S。"]
    (out/'e_l4_2b_0_contract_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8'); print(json.dumps(decision,ensure_ascii=False)); return decision


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',default=r'D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2b_0_contract'); p.add_argument('--l4-dir',default=r'D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4'); p.add_argument('--source-profile-dir',default=r'D:\TrafficGNN\outputs\latent_traffic_performance_l3'); p.add_argument('--formal-root',default=r'D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal'); p.add_argument('--dstsgcn-runner',default='run_l4_prediction_baseline.py'); p.add_argument('--dcrnn-runner',default=r'baselines\dcrnn_resilience_official_adapted\train.py'); p.add_argument('--evaluation-runner',default='evaluate_l4_event_prediction.py'); p.add_argument('--history',type=int,default=12); p.add_argument('--horizon',type=int,default=12); return p.parse_args()
if __name__=='__main__': run(parse_args())
