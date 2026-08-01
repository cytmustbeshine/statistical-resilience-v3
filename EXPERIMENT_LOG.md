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

## E-L4-0R prediction-pipeline repair (2026-07-15)
Environment: Windows PowerShell; Python `D:\soft\Python310\python.exe`; branch `main`; baseline commit `31e5fc0`.

Scope: repaired only split assignment, train-only scaler fitting, explicit ordered node selection, checkpoint metadata, aligned multi-horizon prediction export and the event-free feature contract. No network training was run and `model.py`/training-loss mathematics were not modified.

Verification commands/results:
- `python -m py_compile data.py l4_prediction_pipeline.py audit_l4_prediction_pipeline.py tests/test_l4_prediction_pipeline.py`: passed.
- `python tests/test_statistical_resilience_profile.py`: 16 passed.
- `python tests/test_flow_speed_resilience.py`: 22 passed.
- `python tests/test_latent_traffic_performance.py`: 41 passed.
- `python tests/test_two_factor_traffic_resilience.py`: 15 passed.
- `python tests/test_l4_prediction_pipeline.py`: 35 passed.
- Total unittest: 129 passed.
- `python -m pytest --version`: unavailable (`No module named pytest`); this is not reported as a pytest pass.
- Five-dataset N41 audit command: `python audit_l4_prediction_pipeline.py --output-dir D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline --max-nodes 41 --history 12 --horizon 12`.
- Audit decision: `stage_a_passed=true`, `e_l4_1_authorized=true`, `blockers=[]`, `model_py_modified=false`, `training_run=false`.

Outputs refreshed in `D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline`: prediction pipeline audit, dataset capability, node alignment, split-window, scaler roundtrip, UTF-8 Chinese report and decision JSON. Outputs and temporary smoke artifacts are not committed.
## E-L4-1 five-node smoke training (2026-07-27)
Command: `D:\soft\Python310\python.exe run_l4_prediction_baseline.py --datasets rainstorm,typhoon,pems04 --max-nodes 5 --seed 42 --epochs 2 --max-train-windows 512 --max-eval-windows 256 --device cuda`.

Environment: PyTorch 2.8.0+cu128; NVIDIA GeForce RTX 4060; event/weather features disabled; occupancy excluded; no resilience head, CVaR or uncertainty weighting. Flow and speed were trained independently.

A pre-training frozen-profile compatibility check initially stopped Rainstorm before optimizer execution because strict raw-row boundaries differed from the frozen profile boundary by two steps. The pipeline was repaired to accept explicit frozen raw-time boundaries while retaining target-disjoint windows. Verified profile/scaler boundaries: Rainstorm 7255, Typhoon 2590, PEMS04 10193. No profile was refitted.

Smoke traffic MAE versus persistence MAE:
- Rainstorm flow: 111.9790 vs 40.9081; speed: 6.8794 vs 5.0503.
- Typhoon flow: 34.2156 vs 6.1199; speed: 5.1989 vs 3.9152.
- PEMS04 flow: 79.0615 vs 39.5232; speed: 3.0557 vs 1.9697.

L4 system-deficit MAE: Rainstorm demand 0.9544, efficiency 0.6840; Typhoon demand 1.2379, efficiency 0.9591; PEMS04 demand 0.4964, efficiency 0.8321. Only Typhoon efficiency had a positive smoke Spearman correlation (0.1828); the remaining correlations were negative. These numbers are pipeline diagnostics only because training was intentionally capped at two epochs and 512 windows.

Artifacts: `D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\smoke`, `e_l4_1_smoke_metrics.csv`, `e_l4_1_smoke_report.md`, and `e_l4_1_smoke_decision.json`. Checkpoints and NPZ predictions remain outside Git.

Verification: compilation passed; 135 unittest tests passed (16 profile + 22 flow-speed + 41 L3 + 15 L4 + 36 pipeline + 5 prediction evaluation). `pytest` is unavailable (`No module named pytest`). E-L4-0R re-audit passed after the boundary extension.
## E-L4-1 formal N41 baseline (2026-07-27)
Command: `D:\soft\Python310\python.exe run_l4_prediction_baseline.py --run-tag formal_n41 --datasets bridge,rainstorm,typhoon,pems04,pems08 --max-nodes 41 --history 12 --horizon 12 --epochs 20 --batch-size 64 --hidden-dim 64 --num-blocks 2 --max-train-windows 0 --max-eval-windows 0 --seed 42 --device cuda`.

The run trained 9 models: Bridge flow; Rainstorm flow/speed; Typhoon flow/speed on the fixed 16-node matched subnet; PEMS04 flow/speed; PEMS08 flow/speed. Full strict windows used: Bridge 3,085/1,021/1,033 train/validation/test; Rainstorm 7,232/2,404/2,415; Typhoon 2,567/848/860; PEMS04 10,170/3,383/3,394; PEMS08 10,688/3,556/3,567 per variable.

Traffic MAE versus persistence:
- Bridge flow: 38.0659 vs 35.5208 (worse).
- Rainstorm flow/speed: 38.3059 vs 36.6612 (worse); 5.0076 vs 5.7574 (better).
- Typhoon flow/speed: 9.9867 vs 10.2149 (better); 3.5407 vs 3.7871 (better).
- PEMS04 flow/speed: 34.0178 vs 33.5419 (worse); 2.1794 vs 2.1948 (slightly better).
- PEMS08 flow/speed: 30.8721 vs 29.9787 (worse); 2.5295 vs 2.4798 (worse).

The nine continuous L4 deficit Spearman correlations were all positive: 0.5523, 0.6116, 0.7672, 0.9114, 0.7728, 0.6501, 0.8141, 0.6688 and 0.7896 in the output row order. Formal output: `D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\formal_n41`, plus `e_l4_1_formal_n41_metrics.csv`, `e_l4_1_formal_n41_report.md` and `e_l4_1_formal_n41_decision.json`.

Verification after the formal run: compilation passed and 135 unittest tests passed. pytest remains unavailable in the configured Python environment.
## E-L4-1E event-level L4 evaluation (2026-07-27)
Command: `D:\soft\Python310\python.exe evaluate_l4_event_prediction.py --datasets bridge,rainstorm,typhoon,pems04,pems08 --prediction-dir D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\formal_n41 --output-dir D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\event_evaluation --bootstrap-repetitions 1000 --block-length 12 --seed 42`.

The evaluator reused frozen train-only L4 profiles, train q90/q99 thresholds and q50/q75/q90 recovery sensitivity. It used the original Bridge explicit event, Rainstorm event signal and three Typhoon event segments. PEMS04/PEMS08 were normal-state diagnostics only. Outputs include event comparison, process, state, recovery, sensitivity and persistence CSV files plus UTF-8 report and decision JSON.

Key results: Rainstorm demand/efficiency event means were 2.1434/2.4939 versus non-event 0.8502/0.6680; both-high accuracy was 0.8630. Typhoon efficiency-only accuracy was 0.8542 and 0.8415 for observable windows 1 and 2. Bridge demand high-state precision/recall were both 0. Typhoon window 0 had no formal test targets and was marked unavailable. Rainstorm demand peak timing error was 16.25 hours; efficiency was 0.67 hours. Observable Typhoon peak errors were 0.17-1.00 hours.

Verification: event evaluator compilation passed; 5 event tests passed; previous 135 tests also passed; total unittest count is 140. No pytest run because pytest is not installed.
## E-L4-2A corrected alignment stop (2026-07-28)
Environment: Windows PowerShell; Python `D:\soft\Python310\python.exe`; branch `main`; starting baseline `9773f81`; prior audit commit `52ba2e6`.

The audit was rerun with explicit per-event test coverage. Existing checks passed for train-only frozen profiles/scalers, disjoint target timestamps, event/weather-free features, Bridge flow-only handling, Typhoon fixed 16-node matching, and finite DSTSGCN/DCRNN smoke inference. The new required coverage check failed: Bridge 0/1 and Typhoon 2/3 fixed events are fully in the default test target period; Rainstorm is 1/1.

A formal command using three seeds, 20 epochs and four neural models had begun before the coverage defect was discovered. It was stopped after 24/60 completed result files. No stderr failure occurred, but the protocol itself is invalid for the requested final event comparison. Partial outputs were preserved outside Git and are explicitly non-scientific.

Verification after correction: `py_compile` passed; `tests.test_l4_prediction_pipeline` passed 38 tests, including two new event-coverage tests. The complete existing suite passed 142 unittest tests. `pytest` remains unavailable (`No module named pytest`). Stage B was not completed, analyzed, committed or accepted.
## E-L4-2A-R final event-external boundary audit (2026-08-01)
Environment: Windows PowerShell, `D:\soft\Python310\python.exe`, branch `main`, start commit `c1d6da0`, seed not applicable, no training run.

Command: `python audit_final_l4_a3_alignment.py --split-protocol final_event_external --output-dir D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2a_r`.

Outputs: ten required repair-audit artifacts under `e_l4_2a_r`. All CSV/JSON/Markdown artifacts were reloaded successfully. Event coverage passed for Bridge 127/127, Rainstorm 237/237, and three separate Typhoon segments at 237/237 each. All event train/validation counts are zero. Scaler mean/std are unchanged when only val_end changes, inverse roundtrips pass, thresholds and frozen profile files were not changed, Typhoon remains fixed at 16 matched nodes, and Bridge remains flow-only.

Failure: requested train ends 3110/7257/2592 differ by +2 from frozen profile train ends 3108/7255/2590. The audit therefore returned `stage_passed=false`, `e_l4_2b_authorized=false`, and stopped. No DCRNN or DSTSGCN process was trained. The 24 partial `result.json` hashes, 120-file count and 492153111-byte formal directory total were unchanged.

Verification: four-file compilation passed; pipeline unittest 66/66 passed; full unittest 170/170 passed in 2.532 seconds. Pytest invocation failed because the module is not installed; this is not recorded as pytest success.
## E-L4-2A-R2 audit run (2026-08-01)
Command: `D:\soft\Python310\python.exe audit_final_l4_a3_alignment.py --split-protocol profile_compatible_event_external --output-dir D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\e_l4_2a_r2`.

No neural training was started. The profile-compatible split passed target disjointness, expected window counts, event-external coverage, exact profile train boundaries, train-only scaler checks, Typhoon fixed-16 matching, Bridge flow-only behavior, event/weather exclusion, inverse roundtrip, and partial-output integrity.

Canonical threshold comparison failed only for speed/efficiency. Rainstorm reloaded q90/q99 were 1.1925512791904394/1.8627887815816262 versus canonical 1.1506422622965709/1.780246632339144. Typhoon reloaded values were 1.5592506861505906/2.6820813337557605 versus canonical 1.5524600428879205/2.668365734884944. Loader diagnostics found 578 and 146 raw missing speed observations converted to zero, respectively.

Verification: compilation passed; pipeline unittest 86/86; full unittest 190/190; pytest unavailable. All nine CSV outputs, the decision JSON and Chinese Markdown report reloaded successfully. Formal partial outputs remained 24 result files, 120 files and 492153111 bytes with unchanged result hashes.
