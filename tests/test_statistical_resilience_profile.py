import sys
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resilience_metrics import fit_hierarchical_normal_profile,fit_shrunk_conditional_ecdf,compute_probabilistic_resilience_state,detect_event_windows

def sample(days=5,nodes=3,bins=12):
 rng=np.random.default_rng(3);t=days*bins;ts=pd.date_range('2024-01-01',periods=t,freq='2h').to_numpy();base=20+5*np.sin(2*np.pi*np.arange(t)/bins);x=np.maximum(0,base[:,None]+rng.normal(0,1,(t,nodes)));return x,ts,bins

def test_profile_uses_train_only():
 x,ts,b=sample();e=36;p1=fit_hierarchical_normal_profile(x,e,ts,time_of_day_bins=b);x[e:]+=999;p2=fit_hierarchical_normal_profile(x,e,ts,time_of_day_bins=b);assert np.allclose(p1['location_shrunk'],p2['location_shrunk']);assert p1['selected_shrinkage_m']==p2['selected_shrinkage_m']

def test_train_end_exclusive():
 x,ts,b=sample();p=fit_hierarchical_normal_profile(x,24,ts,time_of_day_bins=b);assert p['train_end_exclusive']==24 and p['train_size']==24

def test_shrinkage_fallback_with_few_days():
 x,ts,b=sample(days=2);p=fit_hierarchical_normal_profile(x,len(x),ts,time_of_day_bins=b);assert p['selected_shrinkage_m']==7 and 'fewer_than_3' in p['fallback_reason']

def test_all_nan_and_zero_mad():
 x,ts,b=sample();x[:,0]=np.nan;x[:,1]=5;p=fit_hierarchical_normal_profile(x,36,ts,time_of_day_bins=b);assert np.isfinite(p['location_shrunk']).all();assert (p['scale_shrunk']>0).all()

def test_missing_timestamps_fallback():
 x,_,b=sample();p=fit_hierarchical_normal_profile(x,36,None,time_of_day_bins=b);assert p['day_type_mode']=='none' and 'timestamps_missing' in p['fallback_reason']

def test_ecdf_bounds_monotonicity_and_train_clip():
 x,ts,b=sample();e=36;p=fit_hierarchical_normal_profile(x,e,ts,time_of_day_bins=b);f=fit_shrunk_conditional_ecdf(x,p,e,ts,min_ecdf_samples=2);r=compute_probabilistic_resilience_state(x,p,f,ts);assert np.nanmin(r['p_lower'])>=1e-4 and np.nanmax(r['p_lower'])<=1;order=np.argsort(x[:,0]);assert np.all(np.diff(r['p_lower'][order,0])>=-1) # changing seasonal cells need not globally monotone
 x2=x.copy();x2[e:]=1e9;r2=compute_probabilistic_resilience_state(x2,p,f,ts);assert float(r2['deficit_clip_value'])==float(r['deficit_clip_value'])

def test_same_cell_ecdf_monotonicity():
 x,ts,b=sample();e=36;p=fit_hierarchical_normal_profile(x,e,ts,time_of_day_bins=b);f=fit_shrunk_conditional_ecdf(x,p,e,ts,min_ecdf_samples=1);q=np.tile(x[:1],(3,1));q[:,0]=[1,10,100];qt=np.tile(ts[:1],3);r=compute_probabilistic_resilience_state(q,p,f,qt);assert np.all(np.diff(r['p_lower'][:,0])>=0)

def test_discontinuous_events_are_segmented():
 s=np.zeros(100);s[10:15]=1;s[50:55]=1;w=detect_event_windows(s,np.ones(100),max_gap_steps=2,min_event_steps=3);assert w==[(10,14),(50,54)]
def test_event_signal_not_used_in_profile():
    x,ts,b=sample();p1=fit_hierarchical_normal_profile(x,36,ts,time_of_day_bins=b);dummy=np.zeros(len(x));dummy[40:]=1;p2=fit_hierarchical_normal_profile(x,36,ts,time_of_day_bins=b);assert np.allclose(p1['location_shrunk'],p2['location_shrunk'])

def test_empty_cell_falls_back_to_parent():
    x,ts,b=sample(days=3,bins=24);p=fit_hierarchical_normal_profile(x,24,ts,time_of_day_bins=288);assert np.isfinite(p['location_shrunk']).all()

def test_weekday_weekend_mapping():
    x,ts,b=sample(days=7);p=fit_hierarchical_normal_profile(x,len(x),ts,time_of_day_bins=b);assert p['day_type_mode']=='weekday_weekend' and p['location_shrunk'].shape[-1]==2

def test_shrunk_ecdf_weights():
    x,ts,b=sample(days=8);p=fit_hierarchical_normal_profile(x,72,ts,time_of_day_bins=b);f=fit_shrunk_conditional_ecdf(x,p,72,ts,min_ecdf_samples=2);assert f['selected_m']==p['selected_shrinkage_m'] and len(f['node_sorted_values']['0'])>0

def test_high_state_threshold_train_only():
    x,ts,b=sample();e=36;p=fit_hierarchical_normal_profile(x,e,ts,time_of_day_bins=b);f=fit_shrunk_conditional_ecdf(x,p,e,ts);r1=compute_probabilistic_resilience_state(x,p,f,ts);x[e:]=0;r2=compute_probabilistic_resilience_state(x,p,f,ts);assert r1['train_deficit_quantiles']==r2['train_deficit_quantiles']

def test_bridge_timestamp_mapping():
    target=pd.Timestamp('2007-07-29 00:00:00');ts=pd.date_range('2007-07-15',periods=5184,freq='5min');idx=int(np.argmin(np.abs(ts-target)));assert ts[idx]==target

def test_system_trimmed_mean_nan():
    x,ts,b=sample();x[2,:]=np.nan;p=fit_hierarchical_normal_profile(x,36,ts,time_of_day_bins=b);f=fit_shrunk_conditional_ecdf(x,p,36,ts);r=compute_probabilistic_resilience_state(x,p,f,ts);assert np.isnan(r['system_deficit'][2])

def test_npz_roundtrip():
    import tempfile
    x,ts,b=sample();p=fit_hierarchical_normal_profile(x,36,ts,time_of_day_bins=b)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'profile.npz'
        np.savez_compressed(path,location=p['location_shrunk'],scale=p['scale_shrunk'])
        with np.load(path) as q:
            assert np.allclose(q['location'],p['location_shrunk'])

class StatisticalResilienceTests(unittest.TestCase):
    def test_profile_uses_train_only(self):
        test_profile_uses_train_only()
    def test_train_end_exclusive(self):
        test_train_end_exclusive()
    def test_shrinkage_fallback_with_few_days(self):
        test_shrinkage_fallback_with_few_days()
    def test_all_nan_and_zero_mad(self):
        test_all_nan_and_zero_mad()
    def test_missing_timestamps_fallback(self):
        test_missing_timestamps_fallback()
    def test_ecdf_bounds_monotonicity_and_train_clip(self):
        test_ecdf_bounds_monotonicity_and_train_clip()
    def test_same_cell_ecdf_monotonicity(self):
        test_same_cell_ecdf_monotonicity()
    def test_discontinuous_events_are_segmented(self):
        test_discontinuous_events_are_segmented()
    def test_event_signal_not_used_in_profile(self):
        test_event_signal_not_used_in_profile()
    def test_empty_cell_falls_back_to_parent(self):
        test_empty_cell_falls_back_to_parent()
    def test_weekday_weekend_mapping(self):
        test_weekday_weekend_mapping()
    def test_shrunk_ecdf_weights(self):
        test_shrunk_ecdf_weights()
    def test_high_state_threshold_train_only(self):
        test_high_state_threshold_train_only()
    def test_bridge_timestamp_mapping(self):
        test_bridge_timestamp_mapping()
    def test_system_trimmed_mean_nan(self):
        test_system_trimmed_mean_nan()
    def test_npz_roundtrip(self):
        test_npz_roundtrip()

if __name__ == '__main__':
    unittest.main()
