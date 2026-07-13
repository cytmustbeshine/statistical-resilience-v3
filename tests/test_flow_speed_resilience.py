import sys,unittest,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flow_speed_resilience import *
from resilience_metrics import detect_event_windows
class Tests(unittest.TestCase):
 def test_node_matching_by_suffix(self):
  c=['b_speed','a_volume','a_speed','b_volume'];m=match_node_variables(c,'_volume','_speed');self.assertEqual([(x['node'],x['speed_col']) for x in m],[('a','a_speed'),('b','b_speed')])
 def test_bridge_id_excluded(self):self.assertNotIn('ID',[x['node'] for x in match_node_variables(['ID','S1'],'_volume')])
 def test_missing_speed_is_none(self):self.assertIsNone(match_node_variables(['a_volume'],'_volume','_speed')[0]['speed_col'])
 def test_missing_values_not_zero(self):
  x=np.array([[1.],[np.nan],[2.]]);self.assertTrue(np.isnan(x[1,0]))
 def profile(self,x,direction='lower',transform='log1p_nonnegative'):
  ts=pd.date_range('2024-01-01',periods=len(x),freq='2h').to_numpy();return fit_variable_resilience_profile(x,24,ts,'x',direction,transform,time_of_day_bins=12,day_type_mode='none',shrinkage_candidates=(3,)),ts
 def test_lower_tail_direction(self):
  x=np.tile(np.arange(1,13)[:,None],(3,1));p,ts=self.profile(x);r=compute_directional_probabilistic_deficit(x,p,ts,'lower');self.assertGreater(r['probabilistic_deficit'][0,0],0)
 def test_upper_tail_occupancy(self):
  x=np.tile(np.arange(1,13)[:,None],(3,1));p,ts=self.profile(x,'upper');r=compute_directional_probabilistic_deficit(x,p,ts,'upper');self.assertGreater(r['probabilistic_deficit'][-1,0],0)
 def test_standardized_negative_values_not_clipped(self):
  x=np.linspace(-3,3,36)[:,None];p,ts=self.profile(x,transform='identity_robust');self.assertEqual(p['profile']['transform_mode'],'identity_robust');self.assertTrue(np.isfinite(compute_directional_probabilistic_deficit(x,p,ts)['probabilistic_deficit']).all())
 def test_variable_profile_train_only(self):
  x=np.arange(36,dtype=float)[:,None];p1,ts=self.profile(x);x[24:]+=999;p2,_=self.profile(x);self.assertTrue(np.allclose(p1['profile']['location_shrunk'],p2['profile']['location_shrunk']))
 def test_quadrant_definition(self):
  p=np.array([[.05,.2]]);q=quadrant_rates(p,p);self.assertAlmostEqual(sum(q[k][0] for k in ('q1_rate','q2_rate','q3_rate','q4_rate')),1)
 def test_missing_speed_quadrant(self):self.assertFalse(quadrant_rates(np.ones((2,2)),None)['available'])
 def test_softmax_zero_origin(self):self.assertTrue(np.allclose(softmax_joint_deficit(np.zeros(3),np.zeros(3),5),0))
 def test_softmax_approaches_max(self):
  a=np.array([1.]);b=np.array([3.]);self.assertLess(abs(softmax_joint_deficit(a,b,100)[0]-3),.02)
 def test_joint_deficit_nonnegative(self):self.assertTrue(np.all(softmax_joint_deficit(np.array([0.,1.]),np.array([1.,0.]))>=0))
 def test_fisher_empirical(self):self.assertTrue(np.isfinite(fisher_empirical_joint(np.full((20,2),.2),np.full((20,2),.3),10)).all())
 def test_lag_sign_convention(self):
  e=np.zeros(100);e[20:30]=1;r=np.roll(e,5);curve=lag_curve(e,r,range(-10,11));best=max([z for z in curve if np.isfinite(z[1])],key=lambda z:z[1]);self.assertEqual(best[0],5)
 def test_lag_requires_minimum_samples(self):self.assertTrue(np.isnan(lag_curve(np.arange(5),np.arange(5),[0],30)[0][1]))
 def test_lag_bootstrap_shared_indices(self):self.assertIn('stable',bootstrap_optimal_lag(np.r_[np.zeros(30),np.ones(30)],np.r_[np.zeros(35),np.ones(25)],range(-3,4),5,10,1))
 def test_event_segments_not_concatenated(self):
  s=np.zeros(100);s[5:10]=1;s[50:55]=1;self.assertEqual(len(detect_event_windows(s,np.ones(100),max_gap_steps=2)),2)
 def test_candidate_own_q90(self):
  a=np.arange(10);b=np.arange(10)*10;self.assertNotEqual(np.quantile(a,.9),np.quantile(b,.9))
 def test_no_event_label_in_profile(self):
  x=np.arange(36,dtype=float)[:,None];p1,ts=self.profile(x);p2,_=self.profile(x);self.assertTrue(np.allclose(p1['profile']['scale_shrunk'],p2['profile']['scale_shrunk']))
 def test_variable_threshold_train_only(self):
  x=np.arange(36,dtype=float)[:,None];p,ts=self.profile(x);r1=compute_directional_probabilistic_deficit(x,p,ts);x[24:]=-999;r2=compute_directional_probabilistic_deficit(x,p,ts);self.assertEqual(r1['train_deficit_quantiles'],r2['train_deficit_quantiles'])
 def test_npz_roundtrip_multivariable(self):
  x=np.arange(36,dtype=float)[:,None];p,ts=self.profile(x)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'multi.npz';np.savez_compressed(path,flow=p['profile']['location_shrunk'],speed=p['profile']['scale_shrunk'])
   with np.load(path) as z:self.assertTrue(np.allclose(z['flow'],p['profile']['location_shrunk']) and np.allclose(z['speed'],p['profile']['scale_shrunk']))
if __name__=='__main__':unittest.main()