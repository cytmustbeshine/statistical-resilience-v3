# EXPERIMENT LOG

## Statistical profile N41
Output: D:\TrafficGNN\outputs\statistical_resilience_profile_n41
Result: hierarchical shrinkage improved standardized blocked-CV scores on all five datasets; residuals are skewed, heavy-tailed, and strongly autocorrelated.

## Flow-speed resilience analysis
Output: D:\TrafficGNN\outputs\flow_speed_resilience_analysis
Bridge P0 flow delta 0.7078, CI [0.7490, 3.0547].
Rainstorm P0 flow delta 0.9688; P1 speed 0.9460; P3b 0.9937 but increased non-event high-state.
Typhoon P0 flow delta -0.0833, CI crosses zero; P1 speed delta 0.5437, CI [0.4226, 0.6820]; P3b delta 0.3453.
No neural-network experiment was run in these two statistical stages.
## L3-0 auditable conditional CDF (2026-07-14)
Output: D:\TrafficGNN\outputs\latent_traffic_performance_l3
Commit: 7b2c675
Result: all source levels are explicit and queryable. N41 source audit found Bridge flow 57.176% cell and 42.824% node-daytype; Typhoon flow and speed 100% node-daytype; robust-parametric and unavailable rates were 0 for finite analyzed observations. Typhoon all-file speed range was -19.73 to 250 with five negative observations, none in the 16-node matched analysis subnet, whose range was 0 to 141.12.

## L3 latent traffic performance N41 (2026-07-15)
Output: D:\TrafficGNN\outputs\latent_traffic_performance_l3
Configuration: bridge,rainstorm,typhoon,pems04,pems08; max_nodes=41; initialization seeds 1,7,21,42,100; moving-block bootstrap=1000; block_length=12.
Engineering: compilation passed; existing unittest suites 16 and 22 passed; L3 suite 40 passed; pytest is not installed in D:\soft\Python310, so unittest fallback was actually executed. Factor NPZ and JSON roundtrips passed. Missing-modal posterior variance increased when information was removed.
Results:
- Bridge L0 flow delta 0.6906; median-difference CI [0.6261, 2.9667]. Bridge remained flow-only.
- Rainstorm L3 delta -0.8850; median-difference CI [-0.5700, -0.4967]; event overlap 0. Flow loading -0.5758, speed loading 0.1676.
- Typhoon L3 delta 0.2346; overall median-difference CI [0.1362, 0.6909]; speed-only delta 0.5514. All three L3 segment median differences were positive, but only segment 3 had a bootstrap interval fully above zero.
- PEMS04 L3 test high-state 17.24% versus L2 6.73%.
- PEMS08 L3 test high-state 18.70% versus L2 15.15%.
- Rainstorm minimum initialization posterior correlation with the best solution was 0.3898.
Decision: reject L3 and do not run neural-network experiments.
## L4 demand-efficiency N41 study (2026-07-15)
Protocol commit: e01dadc
Output: D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4
Configuration: five datasets, max_nodes=41, factor seeds 1,7,21,42,100, moving-block bootstrap=1000, block_length=12.
Engineering: compilation passed; 15 new L4 tests passed; all previous 79 tests continued to pass (94 unittest total). Five-node/100-bootstrap smoke and N41/1000-bootstrap diagnostics completed. No neural-network experiment was run.
Results:
- Bridge demand/service proxy retained; efficiency unavailable and not fabricated.
- Rainstorm efficiency delta 0.9463. Main event quadrant: both-high 92.83%, demand-only 7.17%.
- Typhoon efficiency segment deltas: 0.5274, 0.5296, 0.4051. Efficiency-only quadrant rates: 40.08%, 40.51%, 34.60%.
- PEMS04 efficiency factor loadings: speed 0.3198, reversed occupancy 0.6348; test high-state 15.77% versus speed-only 7.78%.
- PEMS08 efficiency factor loadings: speed 0.4766, reversed occupancy 0.6366; test high-state 17.13% versus speed-only 12.74%.
Decision: support the two-dimension demand/efficiency framework, restrict the occupancy extension, prohibit scalar recombination, and keep neural integration frozen.
## E-L4-0 prediction-pipeline audit (2026-07-15)
Output: D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline
Configuration: five datasets, max_nodes=41, history=12, horizon=12, no training.
Artifacts: prediction_pipeline_audit.csv, dataset_prediction_capability.csv, node_alignment_audit.csv, split_window_audit.csv, scaler_roundtrip_audit.csv, e_l4_0_audit_report.md and e_l4_0_decision.json.
Results:
- all five datasets have 11 overlapping target timestamps at both adjacent split boundaries under the current split implementation;
- the current scaler cutoff includes 2 validation target timestamps on every audited dataset;
- strict raw-time target-boundary splits reduce target overlap to zero;
- Bridge supports flow only; Rainstorm and PEMS selected nodes match flow/speed; Typhoon has 16 matched speed nodes among the first 41 flow nodes but the current loader cannot explicitly select the corresponding paired subnetwork;
- frozen demand and core speed profiles are available read-only;
- inference-only DSTSGCN smoke produced finite [2,12,5,1] output from [2,12,5,1] input;
- 26 new pipeline tests and all previous 94 tests passed (120 unittest total); pytest is not installed.
Decision: reject entry to E-L4-1. No neural-network training was run.
