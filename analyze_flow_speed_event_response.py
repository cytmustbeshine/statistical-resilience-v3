"""Analyze flow-speed event response and compare statistical resilience definitions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import read_csv_with_fallback
from resilience_metrics import detect_event_windows
from flow_speed_resilience import match_node_variables,fit_variable_resilience_profile,compute_directional_probabilistic_deficit,quadrant_rates,softmax_joint_deficit,fisher_empirical_joint,trimmed_system,lag_curve,bootstrap_optimal_lag
CFG={'bridge':dict(csv=r'D:\TrafficGNN\data\bridge_collapse_flow.csv',time='Time',flow=None,speed=None,occ=None,event=None,explicit='2007-07-29 00:00:00'),'rainstorm':dict(csv=r'D:\TrafficGNN\data\rainstorm_traffic_state.csv',time='Time',flow='_volume',speed='_speed',occ=None,event='precip_accum_one_hour',explicit=None),'typhoon':dict(csv=r'D:\TrafficGNN\data\typhoon_traffic_state_standard.csv',time='Time',flow='_volume',speed='_speed',occ=None,event='typhoon_intensity',explicit=None),'pems04':dict(csv=r'D:\TrafficGNN\data\public\PEMS04\pems04.csv',time='Time',flow='_volume',speed='_speed',occ='_occupancy',event=None,explicit=None),'pems08':dict(csv=r'D:\TrafficGNN\data\public\PEMS08\pems08.csv',time='Time',flow='_volume',speed='_speed',occ='_occupancy',event=None,explicit=None)}
def load(name,max_nodes):
 c=CFG[name];df=read_csv_with_fallback(c['csv']);df=df.assign(_dt=pd.to_datetime(df[c['time']],errors='coerce')).sort_values('_dt',kind='stable').reset_index(drop=True)
 if name=='bridge':
  all_maps=[dict(node=z,flow_col=z,speed_col=None,occupancy_col=None) for z in df if z not in {'ID',c['time'],'_dt'} and pd.api.types.is_numeric_dtype(df[z])]
 else:
  all_maps=match_node_variables(df.columns,c['flow'],c['speed'],c['occ'])
 flow_maps=all_maps[:max_nodes];joint_maps=[m for m in all_maps if m['speed_col'] is not None][:max_nodes];occ_maps=[m for m in all_maps if m['speed_col'] is not None and m['occupancy_col'] is not None][:max_nodes]
 def read_maps(maps,key):
  return df[[m[key] for m in maps]].apply(pd.to_numeric,errors='coerce').to_numpy(float) if maps else None
 flow=read_maps(flow_maps,'flow_col');joint_flow=read_maps(joint_maps,'flow_col');speed=read_maps(joint_maps,'speed_col');occ=read_maps(occ_maps,'occupancy_col');event=pd.to_numeric(df[c['event']],errors='coerce').fillna(0).to_numpy(float) if c['event'] else None
 return df['_dt'].to_numpy(),flow,[m['node'] for m in flow_maps],joint_flow,[m['node'] for m in joint_maps],speed,occ,event,c
def aggregate_candidate(node_values):return trimmed_system(node_values,.1)
def masks(T,windows):
 m=np.zeros(T,bool)
 for a,b in windows:m[a:b+1]=True
 return m
def cliff(a,b):
 a=np.asarray(a);a=a[np.isfinite(a)];b=np.asarray(b);b=b[np.isfinite(b)]
 if not len(a) or not len(b):return np.nan
 return float(np.mean([np.mean(x>b)-np.mean(x<b) for x in a]))
def block_ci(a,b,reps,block,seed):
 a=np.asarray(a);a=a[np.isfinite(a)];b=np.asarray(b);b=b[np.isfinite(b)]
 if len(a)<2 or len(b)<2:return (np.nan,np.nan)
 rng=np.random.default_rng(seed);vals=[]
 def draw(x):
  bl=min(block,len(x));starts=rng.integers(0,max(len(x)-bl+1,1),size=int(np.ceil(len(x)/bl)));return np.concatenate([x[s:s+bl] for s in starts])[:len(x)]
 for _ in range(reps):vals.append(np.median(draw(a))-np.median(draw(b)))
 return tuple(map(float,np.quantile(vals,[.025,.975])))
def evaluate(name,label,series,train_end,eventmask,windows,reps,block,seed,meta):
 q90=float(np.nanquantile(series[:train_end],.9));high=series>q90;ev=series[eventmask];non=series[~eventmask];ci=block_ci(ev,non,reps,block,seed);inter=np.sum(high&eventmask)
 return dict(dataset=name,definition=label,train_high_state_rate=float(np.nanmean(high[:train_end])),val_high_state_rate=float(np.nanmean(high[train_end:int(len(series)*.8)])),test_high_state_rate=float(np.nanmean(high[int(len(series)*.8):])),event_mean_deficit=float(np.nanmean(ev)) if len(ev) else np.nan,non_event_mean_deficit=float(np.nanmean(non)),event_non_event_median_difference=float(np.nanmedian(ev)-np.nanmedian(non)) if len(ev) else np.nan,event_non_event_cliffs_delta=cliff(ev,non),event_non_event_ci_low=ci[0],event_non_event_ci_high=ci[1],event_high_state_overlap=float(inter/max(eventmask.sum(),1)),high_state_event_precision=float(inter/max(high.sum(),1)),non_event_high_state_rate=float(np.sum(high&~eventmask)/max((~eventmask).sum(),1)),**meta)
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',default=','.join(CFG));p.add_argument('--output-dir',required=True);p.add_argument('--time-of-day-bins',type=int,default=288);p.add_argument('--day-type-mode',default='weekday_weekend');p.add_argument('--shrinkage-candidates',default='0,1,3,7,14,28');p.add_argument('--max-nodes',type=int,default=41);p.add_argument('--seed',type=int,default=42);p.add_argument('--lag-max-steps',type=int,default=288);p.add_argument('--bootstrap-repetitions',type=int,default=1000);p.add_argument('--block-length',type=int,default=12);p.add_argument('--softmax-betas',default='1,2,5,10');a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);cand=tuple(map(float,a.shrinkage_candidates.split(',')));betas=list(map(float,a.softmax_betas.split(',')));quadrants=[];qsummary=[];lagrows=[];lagboots=[];comparisons=[];eventrows=[];bootrows=[];sensitivity=[];occupancyrows=[];report=['# Flow-Speed 统计韧性定义研究',''];
 for name in a.datasets.split(','):
  name=name.strip();ts,flow,fn,joint_flow,sn,speed,occ,event,c=load(name,a.max_nodes);T=len(ts);train_end=min(int((T-24+1)*.6)+12,T);kwargs=dict(time_of_day_bins=a.time_of_day_bins,day_type_mode=a.day_type_mode,shrinkage_candidates=cand);fp=fit_variable_resilience_profile(flow,train_end,ts,'flow','lower','log1p_nonnegative',**kwargs);fs=compute_directional_probabilistic_deficit(flow,fp,ts,'lower');sp=ss=jfp=jfs=None
  if speed is not None:
   jfp=fit_variable_resilience_profile(joint_flow,train_end,ts,'joint_flow','lower','log1p_nonnegative',**kwargs);jfs=compute_directional_probabilistic_deficit(joint_flow,jfp,ts,'lower');sp=fit_variable_resilience_profile(speed,train_end,ts,'speed','lower','log1p_nonnegative',**kwargs);ss=compute_directional_probabilistic_deficit(speed,sp,ts,'lower')
  op=os=None
  if occ is not None:op=fit_variable_resilience_profile(occ,train_end,ts,'occupancy','upper','log1p_nonnegative',**kwargs);os=compute_directional_probabilistic_deficit(occ,op,ts,'upper')
  if c['explicit']:
   idx=int(np.nanargmin(np.abs(pd.to_datetime(ts)-pd.Timestamp(c['explicit']))));windows=detect_event_windows(None,fs['system_resilience'],explicit_event_start=idx)
  elif event is not None:windows=detect_event_windows(event,fs['system_resilience'],max_gap_steps=12,min_event_steps=3)
  else:windows=[]
  eventmask=masks(T,windows);joint_available=ss is not None
  q=quadrant_rates(jfs['p_lower'] if jfs else fs['p_lower'],ss['p_lower'] if ss else None,.1)
  if q['available']:
   for t in range(T):quadrants.append(dict(dataset=name,time_idx=t,timestamp=str(pd.Timestamp(ts[t])),q1_rate=q['q1_rate'][t],q2_rate=q['q2_rate'][t],q3_rate=q['q3_rate'][t],q4_rate=q['q4_rate'][t]))
   for eid,(x,y) in enumerate(windows):qsummary.append(dict(dataset=name,event_id=eid,event_start=x,event_end=y,q1_low_flow_low_speed_rate=float(np.nanmean(q['q1_rate'][x:y+1])),q2_low_flow_normal_speed_rate=float(np.nanmean(q['q2_rate'][x:y+1])),q3_normal_flow_low_speed_rate=float(np.nanmean(q['q3_rate'][x:y+1])),q4_normal_flow_normal_speed_rate=float(np.nanmean(q['q4_rate'][x:y+1])),flow_deficit_mean=float(np.nanmean(fs['system_deficit'][x:y+1])),speed_deficit_mean=float(np.nanmean(ss['system_deficit'][x:y+1])),flow_deficit_peak=float(np.nanmax(fs['system_deficit'][x:y+1])),speed_deficit_peak=float(np.nanmax(ss['system_deficit'][x:y+1]))))
  series={'P0_flow':fs['system_deficit']};meta={'variables_used':'flow','joint_definition_available':joint_available,'recovery_correspondence':np.nan,'ecdf_fallback_rate':float(np.mean(fs['ecdf_fallback_mask'])),'selected_flow_m':fp['profile']['selected_shrinkage_m'],'selected_speed_m':sp['profile']['selected_shrinkage_m'] if sp else np.nan,'weight_flow':1.0,'weight_speed':0.0,'softmax_beta':np.nan,'train_calibration_error':abs(np.mean(fs['system_deficit'][:train_end]>fs['train_deficit_quantiles']['q90'])-.1)}
  comparisons.append(evaluate(name,'P0_flow',series['P0_flow'],train_end,eventmask,windows,min(a.bootstrap_repetitions,2000),a.block_length,a.seed,meta.copy()))
  if ss:
   # Compare only matched node count by order returned from suffix matching.
   dq=jfs['probabilistic_deficit'];dv=ss['probabilistic_deficit'];flow_score=fp['profile']['shrinkage_cv']['candidate_scores'].get(str(fp['profile']['selected_shrinkage_m']),1);speed_score=sp['profile']['shrinkage_cv']['candidate_scores'].get(str(sp['profile']['selected_shrinkage_m']),1);iwq=1/max(flow_score,1e-6);iwv=1/max(speed_score,1e-6);wq=iwq/(iwq+iwv);wv=1-wq
   node_defs={'P1_speed':dv,'P2a_equal':.5*dq+.5*dv,'P2b_stability':wq*dq+wv*dv,'P3a_max':np.maximum(dq,dv),'P3b_softmax_beta5':softmax_joint_deficit(dq,dv,5)}
   for label,nv in node_defs.items():
    ser=aggregate_candidate(nv);wf,ws=(0,1) if label=='P1_speed' else ((.5,.5) if label=='P2a_equal' else ((wq,wv) if label=='P2b_stability' else (np.nan,np.nan)));m=meta.copy();m.update(variables_used='speed' if label=='P1_speed' else 'flow,speed',weight_flow=wf,weight_speed=ws,softmax_beta=5 if 'softmax' in label else np.nan,ecdf_fallback_rate=float(np.mean(ss['ecdf_fallback_mask'])) if label=='P1_speed' else float(np.mean(fs['ecdf_fallback_mask']|ss['ecdf_fallback_mask'][:,:fs['ecdf_fallback_mask'].shape[1]])) if fs['ecdf_fallback_mask'].shape==ss['ecdf_fallback_mask'].shape else np.nan);comparisons.append(evaluate(name,label,ser,train_end,eventmask,windows,min(a.bootstrap_repetitions,2000),a.block_length,a.seed,m));series[label]=ser
   fisher=fisher_empirical_joint(jfs['p_lower'],ss['p_lower'],train_end);series['P4_fisher_empirical']=fisher;mm=meta.copy();mm.update(variables_used='flow,speed',weight_flow=np.nan,weight_speed=np.nan);comparisons.append(evaluate(name,'P4_fisher_empirical',fisher,train_end,eventmask,windows,min(a.bootstrap_repetitions,2000),a.block_length,a.seed,mm))
   for beta in betas:
    ser=aggregate_candidate(softmax_joint_deficit(dq,dv,beta));row=evaluate(name,f'P3b_beta{beta:g}',ser,train_end,eventmask,windows,min(a.bootstrap_repetitions,500),a.block_length,a.seed,meta.copy());row['softmax_beta']=beta;sensitivity.append(row)
   if event is not None:
    responses={'flow_deficit':jfs['system_deficit'],'speed_deficit':ss['system_deficit'],'joint_beta5':series['P3b_softmax_beta5'],'system_flow':np.nanmean(joint_flow,axis=1),'system_speed':np.nanmean(speed,axis=1)};lags=np.arange(-a.lag_max_steps,a.lag_max_steps+1)
    for metric,response in responses.items():
     curve=lag_curve(event,response,lags)
     for lag,rho,nv in curve:lagrows.append(dict(dataset=name,metric=metric,lag_steps=lag,lag_hours=lag*5/60,spearman_correlation=rho,n_valid=nv))
     boot=bootstrap_optimal_lag(event,response,lags,a.block_length,a.bootstrap_repetitions,a.seed);boot.update(dataset=name,metric=metric,optimal_lag_hours_median=boot.get('optimal_lag_median',np.nan)*5/60);lagboots.append(boot)
  if os:
   occ_defs=[('flow',fs['system_deficit']),('speed',ss['system_deficit']),('occupancy',os['system_deficit']),('max_flow_speed',aggregate_candidate(np.maximum(fs['probabilistic_deficit'],ss['probabilistic_deficit']))),('max_speed_occupancy',aggregate_candidate(np.maximum(ss['probabilistic_deficit'],os['probabilistic_deficit']))),('max_all',aggregate_candidate(np.maximum(np.maximum(fs['probabilistic_deficit'],ss['probabilistic_deficit']),os['probabilistic_deficit'])))]
   for label,ser in occ_defs:
    q90=np.nanquantile(ser[:train_end],.9);occupancyrows.append(dict(dataset=name,definition=label,train_high_rate=float(np.mean(ser[:train_end]>q90)),test_high_rate=float(np.mean(ser[int(T*.8):]>q90))))
  for eid,(x,y) in enumerate(windows):
   for label,ser in series.items():eventrows.append(dict(dataset=name,event_id=eid,definition=label,event_start=x,event_end=y,event_median=float(np.nanmedian(ser[x:y+1])),event_mean=float(np.nanmean(ser[x:y+1])),event_peak=float(np.nanmax(ser[x:y+1]))))
  ds=out/name;ds.mkdir(exist_ok=True)
  for label,vp,st in [('flow',fp,fs),('speed',sp,ss),('occupancy',op,os)]:
   if vp is None:continue
   np.savez_compressed(ds/f'{label}_profile_hierarchical.npz',location_shrunk=vp['profile']['location_shrunk'],scale_shrunk=vp['profile']['scale_shrunk']);np.savez_compressed(ds/f'{label}_profile_ecdf.npz',global_sorted_values=vp['ecdf']['global_sorted_values']);(ds/f'{label}_profile_report.json').write_text(json.dumps(dict(variable=label,train_only=True,selected_m=vp['profile']['selected_shrinkage_m'],transform=vp['profile']['transform_mode'],tail=vp['profile']['tail_direction'],fallback=vp['profile']['fallback_reason']),ensure_ascii=False,indent=2),encoding='utf-8')
  report += [f'## {name}',f'- flow节点 {flow.shape[1]}，匹配子网节点 {0 if speed is None else speed.shape[1]}，联合可用={joint_available}。',f'- 事件片段 {len(windows)} 个。','']
 for fn,rows in [('flow_speed_quadrants.csv',quadrants),('flow_speed_quadrant_summary.csv',qsummary),('event_response_lag.csv',lagrows),('event_response_lag_bootstrap.csv',lagboots),('resilience_performance_definition_comparison.csv',comparisons),('definition_event_level_comparison.csv',eventrows),('definition_block_bootstrap.csv',[{k:r.get(k) for k in ('dataset','definition','event_non_event_ci_low','event_non_event_ci_high')} for r in comparisons]),('definition_sensitivity_beta.csv',sensitivity),('occupancy_extension_comparison.csv',occupancyrows)]:pd.DataFrame(rows).to_csv(out/fn,index=False,encoding='utf-8-sig')
 (out/'flow_speed_resilience_report.md').write_text('\n'.join(report),encoding='utf-8');print(pd.DataFrame(comparisons)[['dataset','definition','event_non_event_cliffs_delta','event_high_state_overlap','non_event_high_state_rate']].to_string(index=False));print('OUTPUT',out)
if __name__=='__main__':main()