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