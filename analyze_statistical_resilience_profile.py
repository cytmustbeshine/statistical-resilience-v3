"""Standalone diagnostics for train-only statistical traffic resilience profiles."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import read_csv_with_fallback,resolve_time_index
from resilience_metrics import fit_hierarchical_normal_profile,fit_shrunk_conditional_ecdf,compute_probabilistic_resilience_state,detect_event_windows,select_dominant_event_window,compute_baseline,compute_R_series

CONFIG={
'bridge':dict(csv=r'D:\TrafficGNN\data\bridge_collapse_flow.csv',time='Time',suffix=None,event=None,split='event_aligned',explicit='2007-07-29 00:00:00'),
'rainstorm':dict(csv=r'D:\TrafficGNN\data\rainstorm_traffic_state.csv',time='Time',suffix='_volume',event='precip_accum_one_hour',split='event_aware',explicit=None),
'typhoon':dict(csv=r'D:\TrafficGNN\data\typhoon_traffic_state_standard.csv',time='Time',suffix='_volume',event='typhoon_intensity',split='event_aware',explicit=None),
'pems04':dict(csv=r'D:\TrafficGNN\data\public\PEMS04\pems04.csv',time='Time',suffix='_volume',event=None,split='chronological',explicit=None),
'pems08':dict(csv=r'D:\TrafficGNN\data\public\PEMS08\pems08.csv',time='Time',suffix='_volume',event=None,split='chronological',explicit=None)}

def load(name,max_nodes):
 c=CONFIG[name];df=read_csv_with_fallback(c['csv']);parsed=pd.to_datetime(df[c['time']],errors='coerce');df=df.assign(_dt=parsed).sort_values('_dt',kind='stable').reset_index(drop=True)
 cols=[z for z in df.columns if z not in {c['time'],'_dt','ID'} and pd.api.types.is_numeric_dtype(df[z]) and (not c['suffix'] or z.endswith(c['suffix']))][:max_nodes]
 data=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float);event=pd.to_numeric(df[c['event']],errors='coerce').fillna(0).to_numpy() if c['event'] and c['event'] in df else None
 return data,df['_dt'].to_numpy(),event,cols,c

def state_rates(system,q):
 v=np.asarray(system);valid=np.isfinite(v);n=max(valid.sum(),1)
 return dict(normal=float(np.sum(valid&(v<=q['q75']))/n),moderate=float(np.sum(valid&(v>q['q75'])&(v<=q['q90']))/n),high=float(np.sum(valid&(v>q['q90'])&(v<=q['q95']))/n),extreme=float(np.sum(valid&(v>q['q95']))/n))

def overlap(system,q,windows):
 high=np.isfinite(system)&(system>q['q90']);mask=np.zeros(len(system),bool)
 for a,b in windows:mask[max(0,a):min(len(mask),b+1)]=True
 inter=np.sum(mask&high);return dict(event_high_state_overlap=float(inter/max(mask.sum(),1)),event_detection_recall=float(inter/max(mask.sum(),1)),high_state_event_precision=float(inter/max(high.sum(),1)),non_event_high_state_rate=float(np.sum((~mask)&high)/max(np.sum(~mask),1)))

def effect(a,b):
 a=np.asarray(a);a=a[np.isfinite(a)];b=np.asarray(b);b=b[np.isfinite(b)]
 if not len(a) or not len(b):return np.nan
 # rank-biserial/Cliff delta: P(a>b)-P(a<b)
 return float(np.mean([np.mean(x>b)-np.mean(x<b) for x in a]))

def bootstrap_nodes(deficit,window,reps=1000,block=12,seed=42):
 a,b=window;v=deficit[a:b+1];T,N=v.shape
 if T<2:return []
 rng=np.random.default_rng(seed);bl=max(1,min(block,T));ranks=np.full((reps,N),np.nan)
 for r in range(reps):
  starts=rng.integers(0,max(T-bl+1,1),size=int(np.ceil(T/bl)));idx=np.concatenate([np.arange(s,min(s+bl,T)) for s in starts])[:T];score=np.nanmean(v[idx],axis=0);order=np.argsort(-np.nan_to_num(score,nan=-np.inf));ranks[r,order]=np.arange(1,N+1)
 return [dict(node=i,top5_probability=float(np.mean(ranks[:,i]<=5)),rank_median=float(np.nanmedian(ranks[:,i])),rank_ci_low=float(np.nanquantile(ranks[:,i],.025)),rank_ci_high=float(np.nanquantile(ranks[:,i],.975))) for i in range(N)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',default='bridge,rainstorm,typhoon,pems04,pems08');p.add_argument('--output-dir',required=True);p.add_argument('--time-of-day-bins',type=int,default=288);p.add_argument('--day-type-mode',default='weekday_weekend');p.add_argument('--shrinkage-candidates',default='0,1,3,7,14,28');p.add_argument('--max-nodes',type=int,default=41);p.add_argument('--seed',type=int,default=42);args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);candidates=tuple(float(x) for x in args.shrinkage_candidates.split(','));summary=[];methods=[];events=[];eventlevels=[];nodes=[];boots=[];dists=[];autos=[];cvs=[];report=['# 统计韧性 Profile 诊断报告','']
 for name in args.datasets.split(','):
  name=name.strip().lower();data,ts,event,node_names,c=load(name,args.max_nodes);T,N=data.shape;train_end=int((T-24+1)*.6)+12;train_end=min(max(train_end,1),T);profile=fit_hierarchical_normal_profile(data,train_end,ts,args.time_of_day_bins,args.day_type_mode,candidates);ecdf=fit_shrunk_conditional_ecdf(data,profile,train_end,ts);state=compute_probabilistic_resilience_state(data,profile,ecdf,ts);q=state['train_deficit_quantiles'];system=state['system_deficit']
  if c['explicit']:
   target=pd.Timestamp(c['explicit']);idx=int(np.nanargmin(np.abs(pd.to_datetime(ts)-target)));windows=detect_event_windows(None,state['system_resilience'],explicit_event_start=idx)
  elif event is not None:windows=detect_event_windows(event,state['system_resilience'],max_gap_steps=12,min_event_steps=3)
  else:windows=[]
  dominant=select_dominant_event_window(windows,event,state['system_resilience']) if windows else None;ov=overlap(system,q,windows);rates=state_rates(system,q);cell=profile['cell_count'];zero=np.mean(np.isfinite(profile['cell_mad_raw'])&(profile['cell_mad_raw']==0));empty=np.mean(cell==0);fallback=np.mean(state['ecdf_fallback_mask'])
  summary.append(dict(dataset=name,rows=T,nodes=N,time_start=str(pd.Timestamp(ts[0])),time_end=str(pd.Timestamp(ts[-1])),monotonic=bool(pd.DatetimeIndex(ts).is_monotonic_increasing),train_end=train_end,selected_shrinkage_m=profile['selected_shrinkage_m'],scale_floor=profile['scale_floor'],empty_cell_rate=empty,zero_mad_cell_rate=zero,ecdf_fallback_rate=fallback,**{f'train_{k}_rate':v for k,v in state_rates(system[:train_end],q).items()},test_high_rate=float(np.mean(system[int(T*.8):]>q['q90'])),event_count=len(windows),**ov))
  for key,val in profile['shrinkage_cv']['candidate_scores'].items():cvs.append(dict(dataset=name,shrinkage_m=key,standardized_score=val,log_mae=profile['shrinkage_cv']['candidate_log_mae'].get(key,np.nan)))
  raw=np.where(np.isfinite(data)&(data>=0),data,np.nan);log=np.log1p(raw);rz=state['robust_z']
  for label,arr in [('raw',raw),('log1p',log),('robust_z',rz)]:
   v=arr[np.isfinite(arr)];dists.append(dict(dataset=name,variable=label,mean=np.mean(v),std=np.std(v),median=np.median(v),mad=np.median(np.abs(v-np.median(v))),skew=stats.skew(v),excess_kurtosis=stats.kurtosis(v),q01=np.quantile(v,.01),q05=np.quantile(v,.05),q50=np.quantile(v,.5),q95=np.quantile(v,.95),q99=np.quantile(v,.99)))
  for lag in (1,12,288):
   vals=[]
   if lag<T:
    for i in range(N):
     a=rz[:-lag,i];b=rz[lag:,i];ok=np.isfinite(a)&np.isfinite(b)
     if ok.sum()>2:vals.append(np.corrcoef(a[ok],b[ok])[0,1])
   autos.append(dict(dataset=name,lag=lag,median=np.nanmedian(vals) if vals else np.nan,q25=np.nanquantile(vals,.25) if vals else np.nan,q75=np.nanquantile(vals,.75) if vals else np.nan))
  eventmask=np.zeros(T,bool)
  for eid,(a,b) in enumerate(windows):eventmask[a:b+1]=True;eventlevels.append(dict(dataset=name,event_id=eid,event_start=a,event_end=b,peak_deficit=float(np.nanmax(system[a:b+1])),mean_deficit=float(np.nanmean(system[a:b+1])),high_overlap=float(np.mean(system[a:b+1]>q['q90']))))
  base=compute_baseline(raw,train_end,args.time_of_day_bins);r0=compute_R_series(raw,{"mean":base["mean"],"tod_mean":base["tod_mean"]},use_tod=False);r1=compute_R_series(raw,base,use_tod=True)
  def aggregate(z):
   out=np.full(T,np.nan)
   for tt in range(T):
    vv=np.sort(z[tt][np.isfinite(z[tt])]);cut=int(len(vv)*.1);vv=vv[cut:len(vv)-cut] if len(vv)-2*cut>0 else vv
    if len(vv):out[tt]=np.mean(vv)
   return out
  method_series={"M0_global_ratio":aggregate(np.maximum(0,1-r0)),"M1_tod_ratio":aggregate(np.maximum(0,1-r1)),"M2_tod_robust_z":aggregate(np.maximum(0,-rz)),"M3_hierarchical_parametric":aggregate(np.maximum(0,-rz)),"M4_hierarchical_ecdf":system}
  for method,series in method_series.items():
   tq={f"q{int(z*100):02d}":float(np.nanquantile(series[:train_end],z)) for z in (.75,.9,.95)};mov=overlap(series,tq,windows);ev=series[eventmask];non=series[~eventmask];delta=effect(ev,non)
   methods.append(dict(dataset=name,method=method,blocked_cv_standardized_score=profile['shrinkage_cv']['candidate_scores'].get(str(profile['selected_shrinkage_m']),np.nan) if method.startswith('M3') or method.startswith('M4') else np.nan,blocked_cv_log_mae=profile['shrinkage_cv']['candidate_log_mae'].get(str(profile['selected_shrinkage_m']),np.nan) if method.startswith('M3') or method.startswith('M4') else np.nan,train_high_state_rate=np.mean(series[:train_end]>tq['q90']),val_high_state_rate=np.mean(series[train_end:int(T*.8)]>tq['q90']),test_high_state_rate=np.mean(series[int(T*.8):]>tq['q90']),event_mean_deficit=np.nanmean(ev) if len(ev) else np.nan,non_event_mean_deficit=np.nanmean(non),event_non_event_effect_size=delta,event_high_state_overlap=mov['event_high_state_overlap'],high_state_event_precision=mov['high_state_event_precision'],non_event_high_state_rate=mov['non_event_high_state_rate'],bridge_pre_post_effect_size=delta if name=='bridge' else np.nan,ecdf_fallback_rate=fallback if method.startswith('M4') else 0.0))
  if dominant:
   a,b=dominant;v=np.nanmean(state['probabilistic_deficit'][a:b+1],axis=0);order=np.argsort(-np.nan_to_num(v,nan=-np.inf));
   for rank,i in enumerate(order,1):nodes.append(dict(dataset=name,node_index=int(i),node_name=node_names[i],vulnerability=float(v[i]),rank=rank,event_start=a,event_end=b))
   for row in bootstrap_nodes(state['probabilistic_deficit'],dominant,1000,12,args.seed):row.update(dataset=name,node_name=node_names[row['node']]);boots.append(row)
  ds=out/name;ds.mkdir(exist_ok=True);np.savez_compressed(ds/'normal_profile_hierarchical.npz',location_shrunk=profile['location_shrunk'],scale_shrunk=profile['scale_shrunk'],cell_count=profile['cell_count'],node_median=profile['node_median']);np.savez_compressed(ds/'normal_profile_ecdf.npz',global_sorted_values=ecdf['global_sorted_values'],cell_keys=np.asarray(list(ecdf['cell_sorted_values']),dtype=object),cell_values=np.asarray(list(ecdf['cell_sorted_values'].values()),dtype=object))
  meta=dict(dataset=name,train_only=True,train_start_idx=0,train_end_idx=train_end,train_start_time=str(pd.Timestamp(ts[0])),train_end_time=str(pd.Timestamp(ts[train_end-1])),time_of_day_bins=args.time_of_day_bins,day_type_mode=profile['day_type_mode'],selected_shrinkage_m=profile['selected_shrinkage_m'],shrinkage_cv_scores=profile['shrinkage_cv']['candidate_scores'],n_train_days=profile['shrinkage_cv']['n_valid_days'],minimum_cell_samples=3,min_ecdf_samples=8,ecdf_smoothing=.5,scale_floor=profile['scale_floor'],deficit_clip_value=float(state['deficit_clip_value']),train_deficit_quantiles=q,fallback_reason=profile['fallback_reason'],ecdf_fallback_rate=fallback,zero_mad_cell_rate=zero,empty_cell_rate=empty);(ds/'normal_profile_report.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
  report += [f'## {name}',f'- 时间：{ts[0]} 至 {ts[-1]}，节点 {N}，训练截止索引 {train_end}。',f'- 收缩参数 m={profile["selected_shrinkage_m"]}，空 cell={empty:.2%}，ECDF 回退={fallback:.2%}。',f'- 事件片段 {len(windows)} 个，事件覆盖 high-state={ov["event_high_state_overlap"]:.2%}。','']
 for filename,rows in [('statistical_profile_summary.csv',summary),('profile_method_comparison.csv',methods),('event_state_overlap.csv',[{k:v for k,v in r.items() if k.startswith('dataset') or 'event_' in k or 'high_' in k or 'non_event' in k} for r in summary]),('event_level_statistical_resilience.csv',eventlevels),('node_vulnerability.csv',nodes),('node_vulnerability_bootstrap.csv',boots),('distribution_diagnostics.csv',dists),('autocorrelation_diagnostics.csv',autos),('shrinkage_cv_diagnostics.csv',cvs)]:pd.DataFrame(rows).to_csv(out/filename,index=False,encoding='utf-8-sig')
 (out/'statistical_resilience_profile_report.md').write_text('\n'.join(report),encoding='utf-8')
 print(pd.DataFrame(summary).to_string(index=False));print('OUTPUT',out)
if __name__=='__main__':main()