# PROJECT MEMORY

## Research question
Unknown-disruption traffic resilience prediction under event-scarce training data. Estimate train-only conditional normal distributions, construct a unified latent traffic-performance index, and later study tail-risk-aware forecasting and uncertainty.

## Frozen architecture
DGCN-style dynamic graph + STSGCN-style synchronous convolution + existing quality fusion. Do not add graph gates, OOD scales, weather/event graphs, CVaR, or resilience heads until the statistical target is validated.

## Current statistical decision
The supported statistical definition is two-dimensional: demand/service-volume deficit from flow and operating-efficiency deficit from speed, reported separately with event quadrants and separate recovery processes. Universal flow-only, P3b, L3 one-factor, scalar L4 recombination, and automatic occupancy inclusion are rejected or restricted.

## Authoritative outputs
- D:\TrafficGNN\outputs\statistical_resilience_profile_n41
- D:\TrafficGNN\outputs\flow_speed_resilience_analysis
- D:\TrafficGNN\outputs\latent_traffic_performance_l3
- D:\TrafficGNN\outputs\two_factor_traffic_resilience_l4

## Important corrected facts
- PEMS test high-state rates: P3b PEMS04 6.50%, PEMS08 17.11%; speed-only PEMS04 8.77%, PEMS08 13.58%.
- Typhoon: 70 flow columns and 45 speed columns overall; only 16 of the first 41 flow nodes have matched speed.
- ECDF fallback must be audited by source level before L3. Current boolean fallback mainly indicates cell ECDF was unavailable; it does not prove parametric fallback.
- Bridge has one observed modality and cannot independently identify a full one-factor measurement model.

## Next step
The offline descriptive definition is complete. Stop before neural-network integration. A new preregistered study must choose whether the network should predict a vector state, use multi-task supervision, or optimize event-process objectives.
## Git workflow
1. Before editing code, inspect `git status` and preserve unrelated user changes.
2. Complete one coherent task, run the relevant compilation and tests, and inspect the diff before committing.
3. After verification, commit the task with a descriptive message and push it to the private `origin` repository.
4. Do not commit raw traffic data, model checkpoints, large generated arrays, experiment-output directories, credentials, tokens, or private keys.
5. Do not mark failing or unverified code as a stable version. Record any test limitation explicitly.
6. Never use destructive history operations or force-push unless the user explicitly requests them.
## L3 completed decision (2026-07-15)
Stage L3-0 and L3 are complete. The auditable CDF hierarchy is cell -> node-daytype -> node -> global -> robust parametric -> unavailable. In the N41 audit, Typhoon flow and speed use node-daytype empirical ECDFs rather than an untracked parametric fallback. The five negative Typhoon speed observations occur outside the 16 matched speed nodes in the first-41-flow analysis subnet.

The missing-data one-factor implementation, posterior variance, train-only latent profile, event process, initialization diagnostics, natural-day blocked stability, and 1,000-repetition moving-block bootstrap all passed engineering verification. However, L3 is rejected as a universal traffic-performance definition:
- Rainstorm L3 Cliff's delta is -0.8850 with a fully negative median-difference bootstrap interval; event high-state overlap is 0.
- Typhoon L3 remains positive but is materially weaker than speed-only: delta 0.2346 versus 0.5514, and overlap 9.56% versus 44.16%.
- PEMS04/PEMS08 L3 test high-state rates are 17.24% and 18.70%, so normal-state stability is not improved.
- Rainstorm initialization posterior agreement falls to about 0.3898, and the single factor learns stable negative flow loadings versus positive speed loadings.

Do not connect L3 to the neural network. Keep the architecture and training loss frozen. If research continues, preregister L4 as two separate factors (demand level and operating efficiency); do not implement L4 automatically.
## L4 two-dimension completed decision (2026-07-15)
The preregistered L4 study is complete. Traffic resilience is supported as a two-dimensional statistical description rather than one universal scalar:
- demand/service-volume state: flow conditional lower-tail deficit, interpreted as unusually low realized service volume or demand rather than operating efficiency;
- operating-efficiency state: speed conditional lower-tail deficit, with flow prohibited from entering this dimension.

Event quadrants resolve the previous contradiction. Rainstorm is predominantly both demand/service and efficiency loss (92.83% both-high during the main segment). Typhoon is predominantly efficiency-only loss: 40.08%, 40.51%, and 34.60% across its three segments, while demand-only/both-high rates are much smaller. Rainstorm efficiency delta is 0.9463. Typhoon efficiency deltas are positive in all three segments (0.5274, 0.5296, 0.4051).

The PEMS speed+reversed-occupancy efficiency extension is restricted/rejected for the core definition. Its bootstrap loadings are stable and positive, but test high-state rates deteriorate to 15.77% on PEMS04 and 17.13% on PEMS08, compared with speed-only 7.78% and 12.74%. Stable loadings do not override normal-state calibration failure.

Current recommended statistical definition:
1. report demand/service-volume deficit and operating-efficiency deficit separately;
2. report both-high, demand-only, efficiency-only, and neither event states;
3. compute peak, cumulative loss, duration, and recovery separately by dimension;
4. Bridge remains demand/service-volume only;
5. occupancy remains an auxiliary diagnostic, not a universal core efficiency input;
6. do not create a weighted scalar L4 score.

This completes the offline definition study at the descriptive statistical level, but does not authorize neural-network integration. A separate preregistered study is required to choose vector supervision, multi-task learning, or event-process objectives.
## E-L4-0 prediction-pipeline audit decision (2026-07-15)
E-L4-0 is complete and did not authorize training. The frozen L4 definition remains unchanged. The current DSTSGCN model is shape-compatible with independent flow and speed forecasting, but the surrounding pipeline fails the stage gate:
- chronological splits are made by window start, causing 11 shared target timestamps at each train/validation and validation/test boundary for horizon=12;
- the scaler raw-time cutoff includes 14 validation-input steps and 2 validation-target steps;
- the suffix-only loader cannot explicitly lock Typhoon to the fixed 16 matched flow-speed nodes with matching adjacency order;
- checkpoints contain only the state_dict and omit variable, node order, scaler, split and timestamp metadata;
- evaluate() does not export aligned multi-horizon predictions;
- current run_experiments dataset commands include event/weather features that are prohibited for E-L4-1.

No model architecture, training loss, resilience head, CVaR, uncertainty weighting or conformal method was changed or run. E-L4-1, smoke training, N41 training and multi-seed experiments remain prohibited. The next study must preregister a minimal E-L4-0R pipeline repair before any forecasting experiment.

## E-L4-0R prediction-pipeline repair decision (2026-07-15)
The minimal E-L4-0R repair is complete and passed the five-dataset N41 stage-gate audit. The frozen two-dimensional L4 definition, `model.py`, the DGCN-STSGCN architecture and the training-loss mathematics were not changed.

The repaired forecasting utilities now provide:
1. raw-time target-disjoint train/validation/test window splits;
2. one strictly train-only scaler per predicted variable;
3. explicit ordered traffic-column selection and fixed Typhoon 16-node flow-speed pairing;
4. checkpoint metadata containing dataset, variable, node order, history/horizon, split boundaries, scaler, timestamps, seed and feature contract;
5. aligned multi-horizon prediction NPZ export in scaled and physical spaces;
6. an event-free single-traffic-variable feature contract excluding weather, event signals, scalar L4 targets, resilience heads, CVaR and uncertainty weighting.

Verification: compilation passed; 129 unittest tests passed (16 profile + 22 flow-speed + 41 L3 + 15 L4 + 35 pipeline); pytest is not installed in `D:\soft\Python310\python.exe`. The repaired audit reports `stage_a_passed=true`, `e_l4_1_authorized=true`, no blockers and `training_run=false`.

Next authorized step: a separate event-free, single-seed, five-node E-L4-1 smoke training for Rainstorm, Typhoon and PEMS04. It must use independent flow/speed models, the repaired strict bundle, train-only static adjacency, frozen L4 profiles and aligned physical-space prediction export. Full N41 or multi-seed training is not yet authorized by this repair alone.
## E-L4-1 five-node smoke decision (2026-07-27)
The event-free E-L4-1 five-node smoke pipeline is now implemented and executed for Rainstorm, Typhoon and PEMS04, with independent flow and speed models (six models total), seed 42, two epochs, 512 training windows and 256 validation/test windows. The run used the existing DSTSGCN backbone and existing Huber/temporal/sparsity training formulation. `model.py`, the architecture, loss mathematics and frozen two-dimensional L4 definition were not modified.

A previously hidden split-boundary compatibility issue was found before training: the frozen statistical profiles use raw-time boundaries derived from usable forecasting windows (`int(num_windows * ratio) + history`), while the repaired strict bundle default used raw-row ratios. The boundaries differed by two steps on all three smoke datasets. The strict splitter now accepts explicit raw-time boundaries so it can preserve the frozen profile split while still omitting all target-crossing windows. No L4 profile was refitted. Verified train ends are Rainstorm 7255, Typhoon 2590 and PEMS04 10193. Typhoon flow nodes map to frozen demand-profile indices 25-29; speed uses indices 0-4 of the fixed 16-node subnet.

Engineering smoke result: passed. All six models trained, saved/reloaded auditable checkpoints, exported/reloaded aligned multi-horizon scaled/physical predictions, produced finite forecasts and completed frozen L4 demand/efficiency postprocessing. Compilation and 135 unittest tests passed; pytest is not installed. E-L4-0R re-audit also remains passed with no blockers.

Scientific boundary: the smoke configuration is deliberately undertrained and is not evidence of forecasting quality. Every neural traffic MAE was worse than persistence in this two-epoch run, and most L4 deficit correlations were weak or negative. Therefore E-L4-1 is engineering-operational but not scientifically accepted. The next permitted experiment is a preregistered single-seed full-training baseline; three-seed experiments, resilience auxiliary supervision and model/loss changes remain unauthorized.
## E-L4-1 formal N41 single-seed baseline (2026-07-27)
The complete single-seed N41 baseline was executed with seed 42, 20 epochs, hidden dimension 64, two DSTSGCN blocks, full strict train/validation/test windows and no event/weather/occupancy inputs. Bridge trained flow only; Rainstorm, Typhoon, PEMS04 and PEMS08 trained independent flow and speed models. The frozen two-dimensional L4 definition, profiles, model architecture and loss mathematics were not changed.

All nine model runs completed with finite physical predictions, aligned target timestamps, checkpoint reloads, prediction NPZ reloads and frozen L4 demand/efficiency postprocessing. Typhoon uses 16 matched nodes for both flow and speed in this implementation; Bridge has no efficiency result.

Traffic MAE beat persistence for Rainstorm speed, Typhoon flow, Typhoon speed and PEMS04 speed, but not Bridge flow, Rainstorm flow, PEMS04 flow or either PEMS08 variable. All nine system L4 deficit Spearman correlations were positive: Bridge 0.5523; Rainstorm flow/speed 0.6116/0.7672; Typhoon flow/speed 0.9114/0.7728; PEMS04 flow/speed 0.6501/0.8141; PEMS08 flow/speed 0.6688/0.7896.

Decision boundary: the complete baseline establishes that indirect L4 prediction is technically executable and that several dimensions have positive continuous-deficit association, but it does not establish universal forecasting superiority over persistence or complete event-level resilience prediction. No three-seed experiment, auxiliary resilience supervision, network change or loss change is authorized yet. Formal event-segment, peak/recovery and block-bootstrap evaluation remains required before accepting E-L4-1 scientifically.
## E-L4-1E event-level evaluation decision (2026-07-27)
Event-level evaluation was completed from the saved formal N41 predictions. No model was retrained, no event window was moved, no L4 profile or threshold was refit, and no three-seed or auxiliary-head experiment was run. Duplicate forecast-origin/horizon predictions for the same target timestamp were averaged before evaluation.

Frozen event windows recovered by the existing event logic: Bridge one explicit window (raw 4032-4158), Rainstorm one signal-defined window (10394-10630), and Typhoon three signal-defined windows (3194-3430, 3482-3718, 3770-4006). Typhoon window 0 is outside the formal test prediction timestamps and is explicitly marked unavailable rather than imputed.

Layered conclusion:
1. Definition layer: supported with Bridge restriction. Rainstorm demand and efficiency event means are well above non-event values, and both-high accuracy is about 0.863. Typhoon observed windows 1 and 2 have efficiency-only classification accuracy about 0.854 and 0.841. Bridge flow-only proxy did not identify a high-state event in this prediction slice, which does not invalidate the multi-variable L4 definition.
2. Variable prediction layer: mixed. Some formal N41 flow/speed models beat persistence, others do not.
3. Event-process layer: partially successful. Rainstorm efficiency peak timing error is about 0.67 hours, while demand peak timing error is about 16.25 hours. Observable Typhoon peak timing errors are at most 1 hour, with recovery correspondence about 0.43-0.90; some recoveries are censored by the available prediction window.

E-L4-1 is therefore restricted provisional evidence for indirect event-level L4 prediction, not a universally accepted neural resilience predictor. Keep L4 frozen. Do not run three seeds or add an auxiliary resilience head. The detailed event outputs are under `D:\TrafficGNN\outputs\e_l4_1_resilience_prediction_baseline\event_evaluation`.
## E-L4-2A corrected final-protocol audit (2026-07-28)
The original E-L4-2A decision was corrected after adding explicit event/test coverage to the strict split audit. Although target timestamps remain disjoint, scalers and frozen profiles remain train-only, inputs exclude event/weather variables, Typhoon node matching remains fixed at 16, and both model smoke shapes are `(2,12,5,1)`, the default 80% validation/test boundary does not satisfy the preregistered final event-evaluation contract.

Bridge uses `val_end=4147` while its fixed event is 4032-4158, so 0/1 events are fully in the test target period. Typhoon uses `val_end=3456` while its first fixed event is 3194-3430, so only 2/3 events are fully in test. Rainstorm passes with 1/1. The corrected stage decision is `stage_passed=false` and `e_l4_2b_authorized=false`.

A formal run had already started under the erroneous pass and was stopped immediately when the mismatch was found. It completed 24/60 neural tasks; these files remain under `D:\TrafficGNN\outputs\e_l4_2_final_aligned_a3\formal` only as invalid partial audit evidence and must not be used for scientific comparison or a paper claim. No A3-L4, L4-auxiliary, or CVaR conclusion is accepted.

The next permitted step is a separate E-L4-2A-R repair. Pre-register validation end boundaries Bridge=4032, Rainstorm=9676, Typhoon=3194; verify strict target non-overlap, train-only scaler/profile compatibility, and full event test coverage; then rerun Stage A. L4 remains frozen and no model/loss change is authorized before that gate passes.
## E-L4-2A-R event-external split repair audit (2026-08-01)
The preregistered `final_event_external` validation boundaries were implemented as an explicit opt-in protocol: Bridge train/val ends 3110/4032, Rainstorm 7257/9676, and Typhoon 2592/3194. The default legacy behavior remains unchanged. Strict target-based splitting produces 3087/911/1141 Bridge windows, 7234/2408/2409 Rainstorm windows, and 2569/591/1115 Typhoon windows, with no target timestamp overlap or boundary-crossing target window.

Event externality now passes: Bridge is 127/127 test targets, Rainstorm is 237/237, and all three Typhoon segments are separately 237/237. Train and validation contain zero fixed-event targets. Typhoon remains the fixed 16-node same-name flow/speed subnetwork; Bridge remains flow-only; event and weather features remain excluded. The frozen profiles, ECDFs, q75/q90/q99 thresholds, recovery thresholds, model and loss were not modified. The 24 partial formal `result.json` files and all 120 formal files remain byte-for-byte unchanged.

The stage still fails because the preregistered train ends do not equal the frozen profile metadata: Bridge 3110 versus 3108, Rainstorm 7257 versus 7255 for flow and speed, and Typhoon 2592 versus 2590 for flow and speed. This consistent +2 mismatch is a protocol-contract blocker. `stage_passed=false` and `e_l4_2b_authorized=false`; no training or final model selection is allowed. Compilation and 170 unittest tests passed. Pytest is unavailable in `D:\soft\Python310\python.exe`.
## E-L4-2A-R2 profile-compatible event-external audit (2026-08-01)
The explicit `profile_compatible_event_external` protocol now uses frozen-profile train ends and event-external validation ends: Bridge 3108/4032, Rainstorm 7255/9676, and Typhoon 2590/3194. The strict window counts are 3085/913/1141, 7232/2410/2409, and 2567/593/1115. All target timestamp splits are disjoint, every fixed event is fully in test, and train/validation contain zero fixed-event targets. All five dataset-variable profile boundaries match exactly.

The stage still fails because reloaded speed-profile q90/q99 values do not match the canonical L4 output. Rainstorm speed differs by +0.0419090168939 at q90 and +0.0825421492425 at q99; Typhoon speed differs by +0.00679064326267 and +0.0137155988708. Demand thresholds match exactly. Read-only tracing shows that the prediction loader applies `np.nan_to_num(nan=0.0)`: 578 Rainstorm speed missing values and 146 Typhoon speed missing values become zero before frozen-profile evaluation, while canonical statistical L4 preserves missing values. This is the likely threshold-path cause and requires a separately preregistered missing-preserving physical-space pipeline repair.

Decision: `stage_passed=false`, `e_l4_2b_authorized=false`. No training, profile refit, ECDF refit, threshold overwrite, model change, loss change, or final-model selection occurred. The 24 partial result files remain unchanged. Compilation, 86 pipeline unittests and 190 total unittests passed; pytest is unavailable.
## E-L4-2A-R3 dual-space missing-value audit (2026-08-02)
The missing-value contract was repaired and audited without training. The new physical-space loader preserves raw NaN values for frozen L4 postprocessing, while a separate model-input copy uses train-only per-node median imputation and produces finite values for scaling/model tensors. Rainstorm speed retains 578 missing observations and Typhoon speed retains 146; no missing physical value is converted to zero in the L4 space.

Using the missing-preserving physical values reproduces all five canonical L4 q90/q99 pairs exactly: Bridge demand, Rainstorm demand and efficiency, and Typhoon demand and efficiency. Profile train boundaries remain exact at 3108/7255/2590, event-external splits remain target-disjoint, all fixed events remain fully in test, and Typhoon remains a separate three-segment evaluation with fixed 16 matched nodes. Missing truth remains invalid for L4 metrics; imputed values are only for the future model-input path.

E-L4-2A-R3 passed with `training_run=false` and `e_l4_2b_authorized=true`, but authorization only permits a separately controlled next phase; no E-L4-2B training was started in this stage. Compilation and 203 unittest tests passed; pytest is unavailable. The 24 partial formal result files remain unchanged.
## E-L4-2B-0 runner-contract audit (2026-08-02)
The read-only formal runner audit was executed after R3. The dual-space data utilities themselves passed: physical speed NaN values remain preserved, model-input copies are finite and train-only median-imputed, canonical L4 thresholds reproduce, and forward-only DGCN/DCRNN shape checks are finite with no backward or optimizer step.

The stage failed because the existing formal runners are not wired to the R3 contract. `run_l4_prediction_baseline.py` still uses the legacy zero-filling loader and legacy derived split; the public DCRNN runner uses `load_wide_traffic_csv`, `SplitScaler` and its own split path; evaluation and checkpoint/NPZ schemas do not yet carry the required dual-space masks and imputation metadata; and M2/M3 formal runner contracts are not implemented. Therefore no B-S smoke training was run. `stage_passed=false` and `e_l4_2b_training_authorized=false`.

B-0 forward-only checks were finite for Bridge/Rainstorm/Typhoon flow/speed smoke tensors. Complete unittest discovery passed 208 tests; pytest is unavailable. No model or train file was modified and the 24 partial formal result files remain protected.

## E-L4-2B-0R runner-contract repair (2026-08-02)
The formal runner wiring blocker is repaired and B-0 now passes without training. The DGCN runner and public DCRNN runner both expose explicit opt-in R3 arguments for `profile_compatible_event_external` split and `dual_space_train_only_median` missing-space handling while preserving their legacy defaults. Event/weather inputs remain excluded in the dual-space contract.

The shared pipeline now exports imputation metadata and physical/L4 valid masks through checkpoint and prediction NPZ schemas. Evaluation can load NaN-preserving physical truth for dual-space archives. B-0 re-audit reports `stage_passed=true`, `e_l4_2b_training_authorized=true`, `training_run=false`, `optimizer_step_called=false`, `backward_called=false`, and `blockers=[]`.

No E-L4-2B-S smoke training, DCRNN training, M1/M2/M3 training, CVaR run, model selection, or thesis conclusion was performed in this repair. `model.py`, `train.py`, the frozen L4 definition, profiles, ECDFs, q90/q99 and recovery thresholds remain unchanged. The 24 partial formal result files remain byte-for-byte protected.

## E-L4-2B-S0 traffic-only runner smoke (2026-08-02)
The repaired dual-space runners were exercised on Rainstorm and Typhoon with five matched nodes, seed 42, two epochs, 512 training windows and 256 validation/test windows. M0 public DCRNN and M1 DSTSGCN traffic-only completed all eight dataset-variable runs, saved/reloaded auditable checkpoints and prediction NPZ archives, preserved physical truth masks, and produced finite physical traffic and L4 postprocessing results.

This is only an engineering smoke. All eight traffic MAEs were worse than persistence under the deliberately capped two-epoch configuration. DCRNN was better than DSTSGCN on Rainstorm flow/speed and Typhoon speed; DSTSGCN was better on Typhoon flow. L4 deficit correlations were positive for all eight runs, but high-state F1 remained unstable, including zero for DSTSGCN Typhoon flow.

The complete E-L4-2B-S stage is not accepted because final L4-aligned M2 auxiliary supervision and M3 tail-risk/CVaR runners are not implemented or run. Formal training, three seeds, bootstrap, L4 auxiliary-supervision claims, CVaR claims and final-model selection remain unauthorized. This is an implementation-stage blocker, not evidence against the dataset or frozen L4 definition.
