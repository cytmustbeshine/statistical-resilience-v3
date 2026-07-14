"""Flow-speed statistical resilience utilities built on train-only profiles."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import stats
from statistical_resilience_profile import (
    compute_probabilistic_resilience_state,
    fit_hierarchical_normal_profile,
    fit_shrunk_conditional_ecdf,
)

def match_node_variables(columns,flow_suffix,speed_suffix=None,occupancy_suffix=None):
    """Match flow, speed and occupancy columns by exact node base suffix."""
    cols=list(map(str,columns));flows={c[:-len(flow_suffix)]:c for c in cols if flow_suffix and c.endswith(flow_suffix)}
    speeds={c[:-len(speed_suffix)]:c for c in cols if speed_suffix and c.endswith(speed_suffix)} if speed_suffix else {}
    occs={c[:-len(occupancy_suffix)]:c for c in cols if occupancy_suffix and c.endswith(occupancy_suffix)} if occupancy_suffix else {}
    return [{"node":b,"flow_col":c,"speed_col":speeds.get(b),"occupancy_col":occs.get(b)} for b,c in flows.items()]

def fit_variable_resilience_profile(values,train_end,timestamps,variable_name,tail_direction,transform_mode,**kwargs):
    """Fit one train-only profile through the shared hierarchical implementation.

    Args:
        values: Variable observations with shape ``[T, N]``.
        train_end: Exclusive training cutoff.
        timestamps: Timestamps shared by all variables.
        variable_name: Name stored in profile metadata.
        tail_direction: ``lower``, ``upper`` or ``two_sided``.
        transform_mode: ``log1p_nonnegative`` or ``identity_robust``.

    Returns:
        Dictionary containing the hierarchical profile and conditional ECDF.
    """
    x=np.asarray(values,float);end=int(np.clip(train_end,0,len(x)))
    if transform_mode not in {"log1p_nonnegative","identity_robust"}:raise ValueError("invalid transform_mode")
    if tail_direction not in {"lower","upper","two_sided"}:raise ValueError("invalid tail_direction")
    profile=fit_hierarchical_normal_profile(
        x,end,timestamps,transform_mode=transform_mode,**kwargs
    )
    ecdf=fit_shrunk_conditional_ecdf(x,profile,end,timestamps)
    profile.update(
        variable_name=variable_name,
        tail_direction=tail_direction,
        transform_mode=transform_mode,
        identity_offset=0.0,
    )
    return {"profile":profile,"ecdf":ecdf,"proxy_values":x}


def compute_directional_probabilistic_deficit(values,variable_profile,timestamps,tail_direction=None,probability_floor=1e-4):
    """Compute a direction-consistent train-calibrated probabilistic deficit.

    Lower-tail loss is active below the conditional median, upper-tail loss is
    active above it, and two-sided loss uses twice the smaller tail probability.
    Missing or semantically unavailable observations remain NaN.
    """
    x=np.asarray(values,float);profile=variable_profile["profile"]
    direction=tail_direction or profile["tail_direction"]
    base=compute_probabilistic_resilience_state(
        x,profile,variable_profile["ecdf"],timestamps,probability_floor=probability_floor
    )
    lower=base["p_lower"]
    transformed=np.asarray(base["transformed_values"],float)
    median=np.asarray(base["conditional_median"],float)
    valid=np.asarray(base["valid_mask"],bool)
    if direction=="lower":
        probability=lower;active=transformed<median
    elif direction=="upper":
        probability=1-lower;active=transformed>median
    elif direction=="two_sided":
        probability=2*np.minimum(lower,1-lower);active=valid
    else:
        raise ValueError("invalid tail_direction")
    probability=np.where(valid,np.clip(probability,probability_floor,1),np.nan)
    deficit=np.where(valid,np.where(active,-np.log(probability),0.0),np.nan)
    end=min(int(profile["train_end_exclusive"]),len(deficit))
    train_values=deficit[:end][np.isfinite(deficit[:end])]
    clip=float(np.quantile(train_values,.999)) if train_values.size else -np.log(probability_floor)
    deficit=np.clip(deficit,0,max(clip,1e-6))
    system=trimmed_system(deficit,.1)
    train=system[:end][np.isfinite(system[:end])]
    quantiles={f"q{int(q*100):02d}":float(np.quantile(train,q)) if train.size else np.nan for q in (.5,.75,.9,.95,.99)}
    return {
        **base,
        "probability":probability,
        "probabilistic_deficit":deficit,
        "system_deficit":system,
        "system_resilience":np.exp(-system),
        "train_deficit_quantiles":quantiles,
        "deficit_clip_value":clip,
        "tail_direction":direction,
    }

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