"""Flow-speed statistical resilience utilities built on train-only profiles."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import stats
from statistical_resilience_profile import fit_hierarchical_normal_profile,fit_shrunk_conditional_ecdf,compute_probabilistic_resilience_state

def match_node_variables(columns,flow_suffix,speed_suffix=None,occupancy_suffix=None):
    """Match flow, speed and occupancy columns by exact node base suffix."""
    cols=list(map(str,columns));flows={c[:-len(flow_suffix)]:c for c in cols if flow_suffix and c.endswith(flow_suffix)}
    speeds={c[:-len(speed_suffix)]:c for c in cols if speed_suffix and c.endswith(speed_suffix)} if speed_suffix else {}
    occs={c[:-len(occupancy_suffix)]:c for c in cols if occupancy_suffix and c.endswith(occupancy_suffix)} if occupancy_suffix else {}
    return [{"node":b,"flow_col":c,"speed_col":speeds.get(b),"occupancy_col":occs.get(b)} for b,c in flows.items()]

def fit_variable_resilience_profile(values,train_end,timestamps,variable_name,tail_direction,transform_mode,**kwargs):
    """Fit a train-only variable profile while recording direction and transform.

    ``log1p_nonnegative`` delegates to the existing hierarchical ECDF estimator.
    ``identity_robust`` applies a train-only positive offset before delegation;
    the monotone shift preserves ranks and ECDF probabilities, including valid
    negative standardized observations.
    """
    x=np.asarray(values,float);end=int(np.clip(train_end,0,len(x)))
    if transform_mode not in {"log1p_nonnegative","identity_robust"}:raise ValueError("invalid transform_mode")
    if tail_direction not in {"lower","upper","two_sided"}:raise ValueError("invalid tail_direction")
    offset=0.0
    if transform_mode=="identity_robust":
        finite=x[:end][np.isfinite(x[:end])];offset=max(0.0,-float(np.min(finite))+1.0) if finite.size else 1.0
        proxy=np.where(np.isfinite(x),np.expm1(np.clip(x+offset,-50,50)),np.nan)
    else:
        proxy=x
    profile=fit_hierarchical_normal_profile(proxy,end,timestamps,**kwargs);ecdf=fit_shrunk_conditional_ecdf(proxy,profile,end,timestamps)
    profile.update(variable_name=variable_name,tail_direction=tail_direction,transform_mode=transform_mode,identity_offset=offset)
    return {"profile":profile,"ecdf":ecdf,"proxy_values":proxy}

def compute_directional_probabilistic_deficit(values,variable_profile,timestamps,tail_direction=None,probability_floor=1e-4):
    """Compute lower-, upper-, or two-sided train-calibrated probabilistic deficit."""
    x=np.asarray(values,float);p=variable_profile["profile"];direction=tail_direction or p["tail_direction"];offset=float(p.get("identity_offset",0))
    proxy=np.where(np.isfinite(x),np.expm1(np.clip(x+offset,-50,50)),np.nan) if p["transform_mode"]=="identity_robust" else x
    base=compute_probabilistic_resilience_state(proxy,p,variable_profile["ecdf"],timestamps,probability_floor=probability_floor)
    lower=base["p_lower"];mu=np.asarray(base["robust_z"])*0 # shape only
    transformed=np.log1p(np.where(np.isfinite(proxy)&(proxy>=0),proxy,np.nan));loc=transformed-base["robust_z"]*1 # overwritten below
    # Median gating is equivalent to p <= 0.5 for continuous ECDFs.
    if direction=="lower":prob=lower;active=lower<=.5
    elif direction=="upper":prob=1-lower;active=lower>=.5
    else:prob=2*np.minimum(lower,1-lower);active=np.isfinite(lower)
    prob=np.clip(prob,probability_floor,1);d=np.where(np.isfinite(x),np.where(active,-np.log(prob),0.0),np.nan)
    end=min(int(p["train_end_exclusive"]),len(d));tv=d[:end][np.isfinite(d[:end])];clip=float(np.quantile(tv,.999)) if tv.size else -np.log(probability_floor);d=np.clip(d,0,max(clip,1e-6))
    system=trimmed_system(d,.1);train=system[:end][np.isfinite(system[:end])];qs={f"q{int(q*100):02d}":float(np.quantile(train,q)) if train.size else np.nan for q in (.5,.75,.9,.95,.99)}
    return {**base,"probability":prob,"probabilistic_deficit":d,"system_deficit":system,"system_resilience":np.exp(-system),"train_deficit_quantiles":qs,"deficit_clip_value":clip,"tail_direction":direction}

def trimmed_system(values,trim=.1):
    """Aggregate node deficits with a finite-value symmetric trimmed mean."""
    x=np.asarray(values,float);out=np.full(len(x),np.nan)
    for t,row in enumerate(x):
        v=np.sort(row[np.isfinite(row)]);cut=int(len(v)*trim);v=v[cut:len(v)-cut] if len(v)-2*cut>0 else v
        if len(v):out[t]=np.mean(v)
    return out

def quadrant_rates(flow_probability,speed_probability,threshold=.10):
    """Return mutually exclusive node and system Q1-Q4 flow-speed states."""
    if speed_probability is None:return {"available":False}
    pf=np.asarray(flow_probability,float);ps=np.asarray(speed_probability,float);valid=np.isfinite(pf)&np.isfinite(ps);lf=pf<threshold;ls=ps<threshold
    masks=[valid&lf&ls,valid&lf&~ls,valid&~lf&ls,valid&~lf&~ls];rates=[]
    den=np.maximum(valid.sum(axis=1),1)
    for m in masks:rates.append(m.sum(axis=1)/den)
    return {"available":True,"q1_rate":rates[0],"q2_rate":rates[1],"q3_rate":rates[2],"q4_rate":rates[3],"valid_mask":valid}

def softmax_joint_deficit(flow_deficit,speed_deficit,beta=5.0):
    """Stable non-compensatory smooth maximum with zero origin."""
    a=np.asarray(flow_deficit,float);b=np.asarray(speed_deficit,float);stack=np.stack([beta*a,beta*b]);m=np.nanmax(stack,axis=0);out=(m+np.log(np.exp(stack[0]-m)+np.exp(stack[1]-m)))/beta-np.log(2)/beta
    return np.maximum(out,0)

def fisher_empirical_joint(flow_probability,speed_probability,train_end,probability_floor=1e-4):
    """Calibrate Fisher-style dependence score by its train empirical CDF, not chi-square."""
    p=np.clip(np.asarray(flow_probability,float),probability_floor,1);q=np.clip(np.asarray(speed_probability,float),probability_floor,1);score=-2*(np.log(p)+np.log(q));system=trimmed_system(score,.1);train=np.sort(system[:train_end][np.isfinite(system[:train_end])]);rank=np.searchsorted(train,system,side="right");upper=np.clip(1-(rank+.5)/(len(train)+1) if len(train) else np.nan,probability_floor,1);return -np.log(upper)

def lag_curve(event,response,lags,min_samples=30):
    """Spearman curve where positive lag means response occurs after event."""
    e=np.asarray(event,float);r=np.asarray(response,float);rows=[]
    for lag in lags:
        if lag>0:a,b=e[:-lag],r[lag:]
        elif lag<0:a,b=e[-lag:],r[:lag]
        else:a,b=e,r
        ok=np.isfinite(a)&np.isfinite(b);rho=float(stats.spearmanr(a[ok],b[ok]).statistic) if ok.sum()>=min_samples and np.unique(a[ok]).size>1 and np.unique(b[ok]).size>1 else np.nan;rows.append((int(lag),rho,int(ok.sum())))
    return rows

def bootstrap_optimal_lag(event_signal,response,lag_range,block_length=12,repetitions=1000,seed=42):
    """Moving-block bootstrap optimal Spearman lag using shared block indices.

    Each replicate rank-transforms the jointly resampled event and response
    once, then evaluates Pearson correlation of the ranks at every lag. This is
    algebraically Spearman correlation and avoids repeated scipy rank work.
    """
    e=np.asarray(event_signal,float);r=np.asarray(response,float);n=len(e);rng=np.random.default_rng(seed);bl=max(1,min(block_length,n));lags_grid=np.asarray(list(lag_range),int);best_lags=[];best_corr=[]
    for _ in range(repetitions):
        starts=rng.integers(0,max(n-bl+1,1),size=int(np.ceil(n/bl)));idx=np.concatenate([np.arange(s,min(s+bl,n)) for s in starts])[:n];er=stats.rankdata(e[idx]);rr=stats.rankdata(r[idx]);chosen_lag=0;chosen_corr=np.nan
        for lag in lags_grid:
            if lag>0:a,b=er[:-lag],rr[lag:]
            elif lag<0:a,b=er[-lag:],rr[:lag]
            else:a,b=er,rr
            if len(a)<30:continue
            sa=np.std(a);sb=np.std(b)
            if sa==0 or sb==0:continue
            corr=float(np.mean((a-np.mean(a))*(b-np.mean(b)))/(sa*sb))
            if not np.isfinite(chosen_corr) or abs(corr)>abs(chosen_corr):chosen_corr=corr;chosen_lag=int(lag)
        if np.isfinite(chosen_corr):best_lags.append(chosen_lag);best_corr.append(chosen_corr)
    if not best_lags:return {"optimal_lag_median":np.nan,"stable":False}
    l=np.asarray(best_lags);c=np.asarray(best_corr);ci=np.quantile(l,[.025,.975]);sign=max(np.mean(l>0),np.mean(l<0),np.mean(l==0));return {"optimal_lag_median":float(np.median(l)),"optimal_lag_ci_low":float(ci[0]),"optimal_lag_ci_high":float(ci[1]),"correlation_median":float(np.median(c)),"correlation_ci_low":float(np.quantile(c,.025)),"correlation_ci_high":float(np.quantile(c,.975)),"positive_lag_rate":float(np.mean(l>0)),"zero_lag_rate":float(np.mean(l==0)),"negative_lag_rate":float(np.mean(l<0)),"stable":bool(ci[1]-ci[0]<=72 and sign>=.8)}