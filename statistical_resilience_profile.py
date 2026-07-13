"""Train-only hierarchical statistical profiles for traffic resilience."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

MAD_SCALE = 1.4826

def _array(data):
    x=np.asarray(data,dtype=float)
    if x.ndim==1:x=x[:,None]
    if x.ndim!=2:raise ValueError("data must have shape [T,N]")
    return np.where(np.isfinite(x)&(x>=0),x,np.nan)

def _median(x, default=0.0):
    v=np.asarray(x,dtype=float);v=v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float(default)

def _scale(x, default=0.0):
    v=np.asarray(x,dtype=float);v=v[np.isfinite(v)]
    if not v.size:return float(default)
    c=np.median(v);return float(MAD_SCALE*np.median(np.abs(v-c)))

def _groups(n,timestamps,bins,mode):
    bid=np.arange(n)%bins;dt=np.zeros(n,dtype=int);fallback=""
    if mode not in {"none","weekday_weekend"}:raise ValueError("invalid day_type_mode")
    if timestamps is None:
        return bid,dt,"none" if mode!="none" else mode,"timestamps_missing_day_type_disabled" if mode!="none" else ""
    parsed=pd.to_datetime(np.asarray(timestamps)[:n],errors="coerce")
    if pd.Series(parsed).notna().mean()<.95:
        return bid,dt,"none" if mode!="none" else mode,"timestamps_unreliable_day_type_disabled" if mode!="none" else ""
    idx=pd.DatetimeIndex(parsed);minutes=idx.hour*60+idx.minute
    bid=np.clip(np.floor(minutes.to_numpy()*bins/1440).astype(int),0,bins-1)
    if mode=="weekday_weekend":dt=(idx.dayofweek.to_numpy()>=5).astype(int)
    return bid,dt,mode,fallback

def _fit_fixed(data,timestamps,bins,mode,m,neighbor,min_samples,mad_q,eps):
    raw=_array(data);x=np.log1p(raw);T,N=x.shape
    bid,dt,mode,fallback=_groups(T,timestamps,bins,mode);D=2 if mode=="weekday_weekend" else 1
    gm=_median(x);gs=_scale(x,eps)
    nc=np.sum(np.isfinite(x),axis=0);nm=np.array([_median(x[:,i],gm) for i in range(N)]);ns=np.array([_scale(x[:,i],gs) for i in range(N)])
    ndc=np.zeros((N,D),int);ndm=np.zeros((N,D));nds=np.zeros((N,D))
    for i in range(N):
        for d in range(D):
            v=x[dt==d,i];ndc[i,d]=np.isfinite(v).sum();ndm[i,d]=_median(v,nm[i]);nds[i,d]=_scale(v,ns[i])
    cc=np.zeros((bins,N,D),int);ec=np.zeros_like(cc);cm=np.full((bins,N,D),np.nan);cs=np.full_like(cm,np.nan)
    for d in range(D):
        for b in range(bins):
            exact=(dt==d)&(bid==b);ids=[(b+k)%bins for k in range(-neighbor,neighbor+1)];expanded=(dt==d)&np.isin(bid,ids)
            for i in range(N):
                v=x[exact,i];v=v[np.isfinite(v)];cc[b,i,d]=v.size
                if v.size<min_samples:v=x[expanded,i];v=v[np.isfinite(v)]
                ec[b,i,d]=v.size
                if v.size:cm[b,i,d]=np.median(v);cs[b,i,d]=_scale(v)
    pos=np.concatenate([cs[np.isfinite(cs)&(cs>0)],nds[np.isfinite(nds)&(nds>0)],ns[np.isfinite(ns)&(ns>0)]])
    floor=float(np.quantile(pos,mad_q)) if pos.size else max(gs,eps)
    if not pos.size:fallback=";".join(filter(None,[fallback,"no_positive_scales_global_fallback"]))
    floor=max(floor,eps);m=max(float(m),0)
    wd=ndc/(ndc+m) if m else (ndc>0).astype(float);pm=wd*ndm+(1-wd)*nm[:,None];ps=wd*nds+(1-wd)*ns[:,None]
    w=ec/(ec+m) if m else (ec>0).astype(float);loc=np.empty_like(cm);scale=np.empty_like(cm)
    for d in range(D):
        l=np.where(np.isfinite(cm[:,:,d]),cm[:,:,d],pm[:,d][None,:]);s=np.where(np.isfinite(cs[:,:,d]),cs[:,:,d],ps[:,d][None,:])
        loc[:,:,d]=w[:,:,d]*l+(1-w[:,:,d])*pm[:,d][None,:];scale[:,:,d]=w[:,:,d]*s+(1-w[:,:,d])*ps[:,d][None,:]
    return {"method":"hierarchical_robust","train_only":True,"time_of_day_bins":bins,"day_type_mode":mode,"requested_day_type_mode":mode,"selected_shrinkage_m":m,"cell_count":cc,"cell_expanded_count":ec,"cell_median_raw":cm,"cell_mad_raw":cs,"node_daytype_count":ndc,"node_daytype_median":ndm,"node_daytype_mad":nds,"node_count":nc,"node_median":nm,"node_mad":ns,"global_median":gm,"global_mad":gs,"location_shrunk":np.where(np.isfinite(loc),loc,gm),"scale_shrunk":np.maximum(np.where(np.isfinite(scale),scale,gs),floor),"scale_floor":floor,"fallback_reason":fallback,"train_size":T}

def _refs(profile,timestamps,n):
    b,d,_,_=_groups(n,timestamps,int(profile["time_of_day_bins"]),str(profile["day_type_mode"]));return profile["location_shrunk"][b,:,d],profile["scale_shrunk"][b,:,d],b,d

def select_shrinkage_by_blocked_cv(train_data,train_timestamps,shrinkage_candidates,time_of_day_bins,day_type_mode,neighborhood_bins,minimum_cell_samples,eps=1e-6):
    """Select prior sample size using leave-one-natural-day-out train-only CV."""
    x=_array(train_data);x=x[:,:min(x.shape[1],8)];ts=None if train_timestamps is None else np.asarray(train_timestamps)[:len(x)]
    parsed=pd.to_datetime(ts,errors="coerce") if ts is not None else None
    days=pd.DatetimeIndex(parsed).normalize().to_numpy() if parsed is not None and pd.Series(parsed).notna().mean()>=.95 else np.arange(len(x))//time_of_day_bins
    unique=pd.unique(days);cand=tuple(sorted(set(float(max(0,c)) for c in shrinkage_candidates))) or (7.,)
    if len(unique)<3:return {"selected_m":7.,"candidate_scores":{},"candidate_log_mae":{},"fold_scores":{},"n_valid_days":len(unique),"fallback_reason":"fewer_than_3_training_days","cv_node_count":int(x.shape[1])}
    folds={str(c):[] for c in cand}
    for day in unique:
        hold=days==day;fit=~hold
        for c in cand:
            p=_fit_fixed(x[fit],ts[fit] if ts is not None else None,time_of_day_bins,day_type_mode,c,neighborhood_bins,minimum_cell_samples,.1,eps)
            mu,sc,_,_=_refs(p,ts[hold] if ts is not None else None,hold.sum());obs=np.log1p(x[hold]);valid=np.isfinite(obs)&np.isfinite(mu)&np.isfinite(sc)
            if valid.any():folds[str(c)].append({"standardized_score":float(np.median(np.abs(obs[valid]-mu[valid])/np.maximum(sc[valid],eps))),"log_mae":float(np.mean(np.abs(obs[valid]-mu[valid])))})
    scores={k:float(np.mean([z["standardized_score"] for z in v])) if v else np.nan for k,v in folds.items()};mae={k:float(np.mean([z["log_mae"] for z in v])) if v else np.nan for k,v in folds.items()}
    valid=[(float(k),v) for k,v in scores.items() if np.isfinite(v)]
    if not valid:return {"selected_m":7.,"candidate_scores":scores,"candidate_log_mae":mae,"fold_scores":folds,"n_valid_days":len(unique),"fallback_reason":"all_blocked_cv_folds_failed","cv_node_count":int(x.shape[1])}
    best=min(v for _,v in valid);chosen=max(k for k,v in valid if abs(v-best)<=1e-6)
    return {"selected_m":chosen,"candidate_scores":scores,"candidate_log_mae":mae,"fold_scores":folds,"n_valid_days":len(unique),"fallback_reason":"","cv_node_count":int(x.shape[1])}

def fit_hierarchical_normal_profile(data,train_end,timestamps=None,time_of_day_bins=288,day_type_mode="weekday_weekend",shrinkage_candidates=(0,1,3,7,14,28),neighborhood_bins=1,mad_floor_quantile=.10,minimum_cell_samples=3,eps=1e-6):
    """Fit train-only node/bin/day-type robust location and scale with shrinkage."""
    x=_array(data);end=int(np.clip(train_end,0,len(x)));ts=None if timestamps is None else np.asarray(timestamps)[:end]
    cv=select_shrinkage_by_blocked_cv(x[:end],ts,shrinkage_candidates,time_of_day_bins,day_type_mode,neighborhood_bins,minimum_cell_samples,eps)
    p=_fit_fixed(x[:end],ts,time_of_day_bins,day_type_mode,cv["selected_m"],neighborhood_bins,minimum_cell_samples,mad_floor_quantile,eps);p.update({"train_end_exclusive":end,"shrinkage_cv":cv})
    p["fallback_reason"]=";".join(filter(None,[p.get("fallback_reason",""),cv.get("fallback_reason","")]))
    return p

def fit_shrunk_conditional_ecdf(data,profile,train_end,timestamps=None,min_ecdf_samples=8,ecdf_smoothing=.5):
    """Fit sorted train-only samples for hierarchical shrunk conditional ECDFs."""
    x=np.log1p(_array(data)[:train_end]);ts=None if timestamps is None else np.asarray(timestamps)[:train_end];b,d,_,_=_groups(len(x),ts,profile["time_of_day_bins"],profile["day_type_mode"]);N=x.shape[1];D=2 if profile["day_type_mode"]=="weekday_weekend" else 1
    cell={};nd={};node={}
    for i in range(N):
        node[str(i)]=np.sort(x[:,i][np.isfinite(x[:,i])])
        for j in range(D):nd[f"{i}|{j}"]=np.sort(x[d==j,i][np.isfinite(x[d==j,i])])
        for j in range(D):
            for k in range(profile["time_of_day_bins"]):
                v=x[(d==j)&(b==k),i];v=v[np.isfinite(v)]
                if v.size:cell[f"{k}|{i}|{j}"]=np.sort(v)
    return {"method":"hierarchical_shrunk_ecdf","train_only":True,"selected_m":profile["selected_shrinkage_m"],"ecdf_smoothing":float(ecdf_smoothing),"min_ecdf_samples":int(min_ecdf_samples),"cell_sorted_values":cell,"node_daytype_sorted_values":nd,"node_sorted_values":node,"global_sorted_values":np.sort(x[np.isfinite(x)]),"fallback_mode":"robust_parametric","train_end_exclusive":int(train_end)}

def _cdf(v,q,a):
    v=np.asarray(v,float);return np.full(np.asarray(q).shape,np.nan) if not v.size else (np.searchsorted(v,q,side="right")+a)/(v.size+2*a)

def compute_probabilistic_resilience_state(data,profile,conditional_ecdf,timestamps=None,probability_floor=1e-4,deficit_clip_quantile=.999,system_trim_fraction=.10,eps=1e-6):
    """Score lower-tail surprisal under a train-only hierarchical shrunk ECDF."""
    raw=_array(data);x=np.log1p(raw);mu,sc,b,d=_refs(profile,timestamps,len(x));T,N=x.shape;p=np.full_like(x,np.nan);fallback=np.zeros_like(x,bool);m=conditional_ecdf["selected_m"];a=conditional_ecdf["ecdf_smoothing"];minimum=conditional_ecdf["min_ecdf_samples"];glob=conditional_ecdf["global_sorted_values"]
    for i in range(N):
        nv=conditional_ecdf["node_sorted_values"].get(str(i),np.array([]));wn=len(nv)/(len(nv)+m) if m else float(len(nv)>0)
        for j in np.unique(d):
            nd=conditional_ecdf["node_daytype_sorted_values"].get(f"{i}|{int(j)}",np.array([]));wd=len(nd)/(len(nd)+m) if m else float(len(nd)>0)
            for k in np.unique(b[d==j]):
                rows=(d==j)&(b==k)&np.isfinite(x[:,i]);q=x[rows,i]
                if not q.size:continue
                gc=_cdf(glob,q,a);nc=_cdf(nv,q,a);nc=np.where(np.isfinite(nc),wn*nc+(1-wn)*gc,gc);dc=_cdf(nd,q,a);parent=np.where(np.isfinite(dc),wd*dc+(1-wd)*nc,nc);cv=conditional_ecdf["cell_sorted_values"].get(f"{int(k)}|{i}|{int(j)}",np.array([]));wc=len(cv)/(len(cv)+m) if m else float(len(cv)>0)
                p[rows,i]=wc*_cdf(cv,q,a)+(1-wc)*parent if len(cv)>=minimum else parent;fallback[rows,i]=len(cv)<minimum
    p=np.clip(p,probability_floor,1);surprise=-np.log(p);deficit=np.where(np.isfinite(x),np.where(x<mu,surprise,0.0),np.nan);end=min(profile["train_end_exclusive"],len(x));tv=deficit[:end][np.isfinite(deficit[:end])];clip=float(np.quantile(tv,deficit_clip_quantile)) if tv.size else -np.log(probability_floor);deficit=np.clip(deficit,0,max(clip,eps));system=np.full(T,np.nan);trim=np.clip(system_trim_fraction,0,.49)
    for t in range(T):
        v=np.sort(deficit[t][np.isfinite(deficit[t])]);cut=int(len(v)*trim);v=v[cut:len(v)-cut] if len(v)-2*cut>0 else v
        if len(v):system[t]=np.mean(v)
    system_train=system[:end][np.isfinite(system[:end])];qs={f"q{int(q*100):02d}":float(np.quantile(system_train,q)) if system_train.size else np.nan for q in (.5,.75,.9,.95,.99)}
    return {"p_lower":p,"tail_surprisal":surprise,"probabilistic_deficit":deficit,"resilience_score":np.exp(-deficit),"robust_z":(x-mu)/np.maximum(sc,eps),"system_deficit":system,"system_resilience":np.exp(-system),"ecdf_fallback_mask":fallback,"deficit_clip_value":np.asarray(clip),"train_deficit_quantiles":qs}