"""Audit traffic resilience variables without replacing missing values by zero."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import read_csv_with_fallback
from flow_speed_resilience import match_node_variables
CONFIG={
'bridge':dict(csv=r'D:\TrafficGNN\data\bridge_collapse_flow.csv',time='Time',flow=None,speed=None,occ=None),
'rainstorm':dict(csv=r'D:\TrafficGNN\data\rainstorm_traffic_state.csv',time='Time',flow='_volume',speed='_speed',occ=None),
'typhoon':dict(csv=r'D:\TrafficGNN\data\typhoon_traffic_state_standard.csv',time='Time',flow='_volume',speed='_speed',occ=None),
'pems04':dict(csv=r'D:\TrafficGNN\data\public\PEMS04\pems04.csv',time='Time',flow='_volume',speed='_speed',occ='_occupancy'),
'pems08':dict(csv=r'D:\TrafficGNN\data\public\PEMS08\pems08.csv',time='Time',flow='_volume',speed='_speed',occ='_occupancy')}
def longest(mask):
 best=cur=0
 for z in mask:cur=cur+1 if z else 0;best=max(best,cur)
 return best
def main():
 p=argparse.ArgumentParser();p.add_argument('--datasets',default=','.join(CONFIG));p.add_argument('--output-dir',required=True);p.add_argument('--max-nodes',type=int,default=41);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);matches=[];audit=[];report=['# 交通韧性变量审计','']
 for name in a.datasets.split(','):
  name=name.strip();c=CONFIG[name];df=read_csv_with_fallback(c['csv']);parsed=pd.to_datetime(df[c['time']],errors='coerce');df=df.assign(_dt=parsed).sort_values('_dt',kind='stable').reset_index(drop=True)
  if name=='bridge':
   cols=[z for z in df if z not in {'ID',c['time'],'_dt'} and pd.api.types.is_numeric_dtype(df[z])];mapping=[{'node':z,'flow_col':z,'speed_col':None,'occupancy_col':None} for z in cols]
  else:mapping=match_node_variables(df.columns,c['flow'],c['speed'],c['occ'])
  total=len(mapping);mapping=mapping[:a.max_nodes];speed_match=np.mean([m['speed_col'] is not None for m in mapping]);occ_match=np.mean([m['occupancy_col'] is not None for m in mapping]);report += [f'## {name}',f'- flow节点总数 {total}，分析 {len(mapping)}；speed匹配率 {speed_match:.2%}，occupancy匹配率 {occ_match:.2%}。','']
  for m in mapping:
   matches.append(dict(dataset=name,**m,speed_matched=m['speed_col'] is not None,occupancy_matched=m['occupancy_col'] is not None))
   for kind,key in [('flow','flow_col'),('speed','speed_col'),('occupancy','occupancy_col')]:
    col=m[key]
    if not col:continue
    raw=pd.to_numeric(df[col],errors='coerce').to_numpy(float);finite=raw[np.isfinite(raw)];standard=('standard' in c['csv'].lower() and np.any(finite<0)) or (finite.size and np.mean(finite<0)>.01);constant=np.unique(finite).size<=1 if finite.size else True;transform='identity_robust' if standard or np.any(finite<0) else 'log1p_nonnegative'
    row=dict(dataset=name,node=m['node'],variable=kind,column=col,dtype=str(df[col].dtype),valid_count=len(finite),missing_rate=float(np.mean(~np.isfinite(raw))),zero_rate=float(np.mean(finite==0)) if finite.size else np.nan,negative_rate=float(np.mean(finite<0)) if finite.size else np.nan,minimum=float(np.min(finite)) if finite.size else np.nan,q01=float(np.quantile(finite,.01)) if finite.size else np.nan,q05=float(np.quantile(finite,.05)) if finite.size else np.nan,median=float(np.median(finite)) if finite.size else np.nan,mean=float(np.mean(finite)) if finite.size else np.nan,q95=float(np.quantile(finite,.95)) if finite.size else np.nan,q99=float(np.quantile(finite,.99)) if finite.size else np.nan,maximum=float(np.max(finite)) if finite.size else np.nan,unique_count=int(np.unique(finite).size),longest_zero_run=longest(np.isfinite(raw)&(raw==0)),longest_missing_run=longest(~np.isfinite(raw)),possibly_standardized=bool(standard),recommended_transform=transform,unit_warning='physical_unit_unverified',constant_warning=bool(constant));audit.append(row)
 pd.DataFrame(matches).to_csv(out/'node_variable_matching.csv',index=False,encoding='utf-8-sig');pd.DataFrame(audit).to_csv(out/'traffic_variable_audit.csv',index=False,encoding='utf-8-sig');(out/'traffic_variable_audit.md').write_text('\n'.join(report),encoding='utf-8');print(pd.DataFrame(matches).groupby('dataset')[['speed_matched','occupancy_matched']].mean());print('OUTPUT',out)
if __name__=='__main__':main()