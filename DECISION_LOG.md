# DECISION LOG

## Rejected: universal flow deficit
Reason: Typhoon flow effect is negative and its bootstrap interval crosses zero.

## Rejected: P3b softmax as universal label
Reason: weaker than speed-only on Typhoon and raises non-event high-state on Rainstorm/PEMS08.

## Rejected: lag adjustment to rescue Typhoon flow
Reason: flow and joint lag distributions are unstable; speed response is synchronous at lag 0.

## Active candidate: L3 latent traffic performance
Use train-only conditional ECDF normal scores and a missing-modality one-factor model. Validate offline before modifying DGCN-STSGCN.
## Rejected: L3 one-factor latent traffic performance (2026-07-15)
Engineering correctness was accepted, but the statistical definition was rejected.
Reasons:
1. Rainstorm event semantics reverse: L3 Cliff's delta -0.8850, bootstrap interval fully below zero, and event high-state overlap 0.
2. Typhoon L3 is clearly weaker than speed-only despite remaining positive.
3. PEMS04 and PEMS08 normal-state high-state rates are not improved and PEMS04 deteriorates sharply.
4. The one-factor model consistently learns negative flow loadings and positive speed/occupancy loadings, showing that demand level and operating efficiency are not one common performance axis.
5. Rainstorm has poor multi-initialization latent-state agreement, even though blocked-fold loading signs are stable.
6. Occupancy communalities near 1 on PEMS indicate that the factor is dominated by occupancy rather than a balanced universal performance construct.
Consequence: do not add L3 labels, auxiliary heads, CVaR, uncertainty weighting, conformal prediction, graph gates, or other network complexity. Keep Bridge as a flow-only service proxy and Typhoon speed-only as the strongest current event-performance signal. A future L4 study may separate demand level from operating efficiency, but L4 was not implemented in this stage.
## Supported with restriction: L4 demand/service and operating-efficiency dimensions (2026-07-15)
Accepted:
1. Traffic resilience should be described with separate demand/service-volume and operating-efficiency processes.
2. Flow lower-tail deficit is a demand/service-volume signal, not a universal efficiency loss.
3. Speed lower-tail deficit is the current core operating-efficiency signal.
4. Event interpretation uses both-high, demand-only, efficiency-only, and neither states.
5. Peak, cumulative deficit, duration, and recovery are reported separately for each dimension.

Evidence:
- Rainstorm is mainly both-high, explaining why both flow and speed were informative.
- Typhoon is mainly efficiency-only, explaining why speed worked while flow-only failed.
- All three Typhoon efficiency effects are positive without moving event windows or lag.

Restricted/rejected:
- Do not automatically add occupancy to the core efficiency definition. Although speed/occupancy loadings are bootstrap-stable, PEMS test high-state calibration worsens.
- Do not combine demand and efficiency into a weighted scalar L4 score.
- Do not treat Bridge as evidence for an efficiency dimension.

Consequence: the descriptive statistical resilience definition is now two-dimensional. Neural integration remains unauthorized until a separate study chooses and validates vector supervision, multi-task targets, or event-process objectives.
## Rejected stage gate: E-L4-1 training after E-L4-0 audit (2026-07-15)
Decision: do not start indirect L4 neural forecasting yet.
Reasons:
1. Current window-start splits overlap targets across train/validation and validation/test.
2. Current scaler fitting reaches into validation targets.
3. Typhoon's fixed 16-node paired subnet cannot be explicitly selected by the suffix-only loader.
4. Current checkpoints are insufficient for auditable physical-space inverse transformation.
5. Current evaluation does not export timestamp-aligned predictions.
6. Existing experiment commands include event/weather features prohibited by the preregistered baseline.

Consequence: preserve the two-dimensional L4 statistical definition and keep model.py and the training loss frozen. A separate minimal pipeline-repair stage must fix splitting, explicit node selection, scaler/checkpoint metadata, event-free configuration and prediction export. Only after that repair passes the same audit may E-L4-1 training begin.

## Accepted stage gate: E-L4-0R prediction-pipeline repair (2026-07-15)
Decision: accept the minimal engineering repair and authorize the next E-L4-1 five-node smoke experiment.

Accepted evidence:
1. Strict raw-time split assignment yields disjoint target timestamps across train/validation/test.
2. Flow and speed use independent scalers fitted only on the strict training prefix.
3. Explicit ordered loading locks Typhoon to the fixed 16 matched flow-speed nodes.
4. Checkpoints now preserve scaler, node, timestamp, split, seed and feature-contract metadata.
5. Multi-horizon truth/prediction archives preserve aligned target timestamps and both scaled/physical values.
6. The enforced baseline contract excludes event and weather inputs, scalar L4 targets, resilience heads, CVaR and uncertainty weighting.
7. Compilation, 129 unittest tests and the five-dataset N41 repair audit passed with no blockers.

Boundary of this acceptance: no neural-network training has occurred, so this does not establish that DGCN-STSGCN predicts flow, speed or L4 resilience well. It authorizes only the preregistered event-free, single-seed, five-node E-L4-1 smoke training. The L4 statistical profiles remain frozen; `model.py`, architecture and loss remain frozen; full N41 and three-seed experiments require the smoke stage to pass first.
## Engineering-accepted, scientifically unresolved: E-L4-1 five-node smoke (2026-07-27)
Decision: accept the end-to-end engineering smoke, but do not accept neural L4 predictive validity from this run.

Accepted engineering evidence:
1. Six independent flow/speed DSTSGCN smoke models trained without event or weather input.
2. Frozen L4 split boundaries were reused exactly; validation/test data did not refit scalers or profiles.
3. Strict train/validation/test targets remain disjoint.
4. Typhoon uses five nodes drawn from the fixed 16-node matched subnet, with demand-profile indices 25-29 correctly preserved.
5. Checkpoint and prediction NPZ roundtrips passed, timestamps aligned and physical-space predictions were finite.
6. Frozen demand and efficiency postprocessing ran for truth and predictions using the same profile and train thresholds.
7. 135 unittest tests and the E-L4-0R re-audit passed.

Not accepted as predictive evidence:
1. All six two-epoch neural traffic forecasts were worse than persistence MAE.
2. Five of six L4 deficit Spearman correlations were negative; Typhoon efficiency was only weakly positive.
3. Some demand high-state predictions collapsed to no alarms in the smoke subset.
4. The capped smoke sample and two epochs are intentionally insufficient for model comparison.

Consequence: do not write that DGCN-STSGCN predicts L4 resilience. A single-seed, fully trained and preregistered E-L4-1 baseline may be run next to distinguish undertraining from a real forecasting failure. Three-seed experiments, auxiliary resilience supervision, network changes and loss changes remain prohibited until that formal baseline passes its own stage gate.
## E-L4-1 formal N41 baseline: technically valid, scientifically provisional (2026-07-27)
Decision: accept the complete single-seed N41 run as a valid engineering baseline, but do not accept it as proof that the network universally predicts L4 resilience better than persistence.

Evidence supporting technical validity:
1. Nine expected models completed without NaN/Inf prediction failure.
2. Checkpoints and multi-horizon NPZ archives reloaded successfully.
3. Target timestamps aligned with all forecast horizons.
4. Frozen train-only L4 profiles and thresholds were reused for truth and prediction.
5. Bridge produced no fabricated efficiency output.
6. All nine L4 continuous deficit Spearman correlations were positive.
7. 135 unittest tests and compilation passed.

Evidence limiting scientific acceptance:
1. Four of nine traffic-variable forecasts were better than persistence; five were worse.
2. PEMS08 flow and speed did not beat persistence.
3. This run has one seed and does not include event-segment bootstrap, peak timing, recovery correspondence or formal four-state event evaluation.
4. Positive continuous L4 correlation alone does not establish event-level resilience prediction or causal understanding.

Consequence: retain the frozen two-dimensional L4 definition and the frozen network/loss. Do not run three seeds or add an auxiliary resilience head yet. The next required analysis is event-process and baseline comparison evaluation using the saved full-N41 predictions, followed by a pre-registered decision on whether E-L4-1 is scientifically accepted.
## Restricted provisional result: E-L4-1E event evaluation (2026-07-27)
Decision: retain the frozen L4 definition and classify neural indirect prediction as restricted provisional evidence, not universal acceptance.

Definition layer: supported with Bridge flow-only restriction. Rainstorm preserves the expected two-dimensional event pattern; Typhoon observable segments preserve efficiency-dominant behavior. The Bridge prediction miss is treated as a limitation of the flow-only external proxy/prediction slice, not as evidence to collapse L4 into another definition.

Variable prediction layer: mixed. The formal N41 models do not uniformly beat persistence.

Event-process layer: partially successful. Rainstorm efficiency timing and Typhoon observable timing are useful, but Rainstorm demand timing is poor, Bridge high-state is missed, and some recovery statuses are censored. The first Typhoon segment is unavailable in the formal test horizon and is not filled or treated as a success.

Consequence: do not run three seeds, do not add a resilience auxiliary head, do not modify the network or loss, and do not reject L4. Keep the result out of the paper as a claim of complete neural resilience learning. Any later work must be separately preregistered and should first address event-process limitations or compare a vector/multi-task objective without changing the current conclusion retroactively.
## Rejected stage gate: corrected E-L4-2A final event coverage (2026-07-28)
Decision: reject entry to E-L4-2B under the current default 80% validation/test boundaries.

Reason: strict split leakage checks alone are insufficient for the preregistered event comparison. Bridge's fixed event begins before `val_end=4147`, and Typhoon event 1 ends before `val_end=3456`. Only Bridge 0/1 and Typhoon 2/3 events are fully test-external; the required complete coverage is absent. Validation traffic MAE can also see those disturbances, so they cannot be treated as untouched final test evidence.

Consequence: the 24/60 partial neural runs are invalid for final A3-L4/DCRNN inference. Do not report M2 auxiliary-supervision gains, M3 CVaR gains, A3-L4 superiority, three-seed stability or bootstrap conclusions from them. Keep L4 frozen. The next allowed task is E-L4-2A-R with one-time preregistered validation ends Bridge=4032, Rainstorm=9676 and Typhoon=3194. Only a fresh passing audit may authorize formal training.
## Rejected gate: E-L4-2A-R frozen-boundary compatibility (2026-08-01)
Decision: reject E-L4-2A-R and do not authorize E-L4-2B.

The one-time event-external validation boundaries solve the original event-coverage problem without target leakage: every fixed event is fully in test, validation is event-free, and the Typhoon segments remain separate. However, the prescribed train boundaries are uniformly two rows later than the frozen L4 profile boundaries. Because the protocol simultaneously forbids profile/ECDF refitting and requires exact train-boundary equality, the corrected split cannot be paired with the existing frozen L4 artifacts as a final aligned task.

This is a protocol compatibility failure, not evidence for or against the L4 statistical definition, L4 auxiliary supervision, CVaR, DCRNN, or any candidate final model. Do not train M0/M1/M2/M3, do not try alternative boundaries in this stage, and do not use the 24 partial runs. A new explicitly preregistered repair decision is required before any final comparison can proceed.
## Rejected gate: E-L4-2A-R2 canonical speed-threshold mismatch (2026-08-01)
Decision: reject E-L4-2A-R2 and keep E-L4-2B unauthorized.

The +2 train-boundary contradiction is resolved: model split train ends now exactly equal the frozen profile reports while all fixed events remain external to validation. The remaining blocker is variable-space consistency for speed. The current prediction loader converts missing speed observations to zero before frozen-profile evaluation, and the resulting efficiency q90/q99 values differ from the canonical L4 study output. Demand thresholds are exact because those loaded flow series contain no corresponding missing-value conversion.

Do not overwrite canonical thresholds, refit profiles, train M0/M1/M2/M3, or select a final model. The next allowed work must be a separate read-only/minimal pipeline audit that preserves missing physical speed values for L4 postprocessing while defining an explicit, training-safe missing-data policy for neural inputs.
## Accepted gate: E-L4-2A-R3 dual-space missing-value contract (2026-08-02)
Decision: accept the read-only R3 protocol repair and permit a separately controlled E-L4-2B preparation. This does not select a final model and does not claim any forecasting result.

The physical L4 space now preserves missing observations, and the model-input space uses a separate train-only node-median imputation copy. Canonical q90/q99 thresholds are exactly reproducible, including speed/efficiency. Missing truth remains excluded from L4 metrics rather than treated as zero or low speed. Split, event, profile, scaler, node and feature contracts all pass, and the legacy partial formal outputs remain untouched.

E-L4-2B is not automatically started. Before any formal training, the future DCRNN/M1/M2/M3 runner must explicitly consume the same dual-space contract, save imputation metadata and masks, use identical profile-compatible event-external boundaries, and pass a no-training smoke audit. No final model has been selected.
## Rejected gate: E-L4-2B-0 runner contract (2026-08-02)
Decision: reject B-0 and do not run E-L4-2B-S.

R3 established that the dual-space utilities can preserve physical missing values and create finite train-only model inputs. However, the actual DGCN and public DCRNN runners have not been migrated to that contract. They still use legacy zero-filling/split/scaler paths, and the required prediction-mask, imputation-metadata, checkpoint and NPZ schemas are incomplete. M2/M3 fairness contracts are also absent.

This is an engineering integration blocker, not a failure of the L4 definition or a model-quality result. The next phase must minimally wire the existing runners to the R3 contract and rerun B-0 without training. B-S, formal seeds, model selection and scientific conclusions remain unauthorized.

## Accepted gate: E-L4-2B-0R runner contract repair (2026-08-02)
Decision: accept the repaired no-training runner contract and authorize only a separately launched E-L4-2B-S smoke stage.

Accepted evidence:
1. DGCN and public DCRNN runners now expose explicit opt-in `profile_compatible_event_external` and `dual_space_train_only_median` protocols while preserving legacy defaults.
2. Physical truth space preserves NaN values for L4 and metric masks; model-input space uses train-only node-median imputation and finite train-only scaling.
3. Checkpoint metadata and prediction NPZ archives now carry imputation metadata, raw/model missing counts, physical truth masks and L4 valid masks.
4. Evaluation can explicitly use the NaN-preserving physical loader for dual-space archives.
5. B-0 forward-only DGCN/DCRNN probes are finite and no training action occurs.
6. `model.py`, `train.py`, frozen profiles, thresholds and partial formal outputs remain unchanged.
7. Compilation, B-0 tests and full unittest discovery pass.

Boundary of this acceptance: no DCRNN/M1/M2/M3 training has occurred, no CVaR or L4 auxiliary-supervision result exists, and no final thesis model is selected. B-S smoke is now allowed as the next separate stage; formal three-seed experiments and scientific claims remain unauthorized until the smoke and subsequent gates pass.

## Partial smoke acceptance: E-L4-2B-S0 M0/M1 (2026-08-02)
Decision: accept the M0 DCRNN and M1 DSTSGCN traffic-only smoke as an engineering runner check, but reject completion of the full E-L4-2B-S stage.

Accepted evidence:
1. Eight Rainstorm/Typhoon flow/speed runs completed under the same dual-space split, node, scaler and mask contract.
2. DCRNN and DSTSGCN checkpoints and prediction archives reloaded successfully.
3. DCRNN checkpoint selection now uses validation physical-space MAE in the formal dual-space path.
4. Physical missing truth remains masked rather than converted to zero for metrics and L4 postprocessing.
5. All L4 deficit Spearman correlations in the smoke are positive.

Reasons full B-S is not accepted:
1. M2 final frozen-L4 auxiliary supervision has not been implemented or run.
2. M3 tail-risk/CVaR training has not been implemented or run.
3. Every capped two-epoch traffic run is worse than persistence MAE.
4. High-state performance is unstable and one DSTSGCN Typhoon-flow run has F1=0.

Consequence: do not run formal seeds, bootstrap or final model comparison. The next stage must implement and audit the true frozen-L4 supervision target and M2/M3 smoke runner without changing `model.py`, the frozen L4 definition, event windows or test thresholds. No final thesis model is selected.

## E-L4-2B final aligned decision (2026-08-02)
Decision: reject the joint E-L4-2B candidate and do not select a final model.

Accepted component: frozen two-dimensional L4 auxiliary supervision has limited evidence. Under the required common postprocessing rule, M2 improves continuous L4 MAE over M1 in 9/15 dataset-variable-seed comparisons. This supports further study of L4 auxiliary supervision but does not establish universal event resilience prediction.

Rejected components: M3 improves q90 tail MAE over M2 in only 7/15 comparisons, so the preregistered CVaR tail-risk increment is not accepted. M3 improves ordinary traffic MAE over public DCRNN in only 3/15 comparisons, so A3-L4 superiority over DCRNN is not accepted. The joint stage gate therefore remains failed despite nondegenerate high-state predictions and some positive event-level bootstrap differences.

Evaluation rule: final L4 evidence must always be computed from physical flow/speed forecasts through the read-only frozen L4 profile. M2/M3 auxiliary-head archives are training diagnostics and cannot replace `g_flow(Q_hat)` or `g_speed(V_hat)` in the final fair comparison.

Consequence: keep the L4 statistical definition frozen, retain these results as a negative/partial ablation finding, do not tune CVaR on the test events, do not add a fourth seed, and do not describe any current candidate as the final thesis model. A separate preregistered decision is required before any new modeling stage.

## 2026-08-04 A4 decision

1. Reject capacity scaling alone as the next route; 128/160 dimensions did not improve validation consistently.
2. Retain the locked hybrid pipeline as an engineering candidate because ordinary traffic metrics improved 15/15 against raw DCRNN.
3. Do not call the hybrid result pure DSTSGCN superiority: the public DCRNN is an explicit component.
4. Do not mark all resilience metrics as exceeded: Bridge seed 3407 q90 L4 tail MAE remains worse, giving 14/15 tail wins.
5. Do not select a final thesis model from the previously inspected test archive. The next model stage must use a new outer holdout or external validation and focus on a pure temporal residual decoder rather than further ensemble-weight tuning.

## 2026-08-04 A7 decision

1. Reject A7 scheduled sampling as the next candidate: it improves direct-head M1 in 15/15 validation comparisons but improves free-running A6 in only 6/15.
2. Do not tune the teacher-forcing decay, start ratio, gradient clipping or epoch budget from this result; the locked A7 gate failed.
3. Do not evaluate A7 on the already-inspected event-test archive.
4. Preserve the useful negative result: the A6 gain comes from autoregressive decoding itself, while teacher forcing does not materially improve its validation behavior and particularly weakens Rainstorm flow.
5. Authorize design work for a materially new pure model combining traffic deep learning with explicit train-only statistical structure. DCRNN may be used only as a comparison baseline, never as an input, ensemble component or hidden-state source.
6. Final confirmation must use a newly locked temporal holdout or external dataset, with configuration frozen before comparison.

## 2026-08-04 A8 validation decision

1. Accept A8 as the first pure post-A3 candidate to pass both seed-42 and three-seed train/validation gates.
2. Freeze the three-expert architecture, robust gate inputs, equal gate initialization, auxiliary weight 0.10, hidden size 64, learning rate 0.001, 20 epochs and validation-MAE checkpoint selection.
3. Do not evaluate or tune A8 on the previously inspected Bridge/Rainstorm/Typhoon event-test archive.
4. Authorize external confirmation on PEMS04 and PEMS08 using fixed first-41-node volume and speed tasks, real timestamp sorting, train-only scaling/adjacency and a common chronological split.
5. Compare against the public official-code-adapted DCRNN under the same data and metric pipeline. DCRNN remains comparison-only.
6. A8 is not yet the final thesis model; success requires external test superiority across the locked ordinary metrics and statistical uncertainty analysis.

## 2026-08-05 A8 external decision

1. Reject A8 as the final model because the new PEMS external ordinary-metric gate failed.
2. Do not adjust A8 expert weights, gate features, epochs or losses using the observed PEMS test ratios.
3. Preserve the positive ablation: A8 is effective for PEMS08 speed, but its short-history DSTSGCN backbone remains inadequate for flow demand dynamics.
4. Authorize A9 as a materially new model based on spatial identity, temporal identity, train-only seasonal statistics, robust traffic-state features and train-only correlation context.
5. Use PEMS04/08 train and validation periods only for A9 selection. Their test periods are now historical evidence and cannot confirm A9.
6. A9 confirmation must use a separately locked external dataset or untouched rolling-origin blocks not used for architecture or hyperparameter selection.

## 2026-08-05 A9 validation decision

1. Accept A9 as the validation-locked candidate because it improves frozen DCRNN validation MAE in all 12 PEMS04/08 dataset-variable-seed comparisons.
2. Freeze all A9 architecture dimensions, seasonal shrinkage 7, 50-epoch budget, optimizer, loss and checkpoint selection before external evaluation.
3. Do not evaluate A9 on PEMS04/08 test segments; those segments were already used to reject A8.
4. Use PEMS03 as the new external confirmation dataset. The downloaded Zenodo artifact and checksum are frozen before training.
5. Construct PEMS03 timestamps from its complete 26208-sample September-November 2018 five-minute sequence, then lock strict 60/20/20 chronological target-disjoint splits.
6. Compare A9 against the same official-code-adapted DCRNN under common first-41-node data, train-only scaler/adjacency and physical metrics.

## 2026-08-05 A9 external and A10 feasibility decision

1. Reject A9 after its PEMS03 ordinary external gate failed; do not tune A9 from PEMS03 test results.
2. Accept the strategic pivot to A10: retain the published DCRNN temporal backbone and add a transparent hierarchical statistical calibration layer.
3. Freeze A10 components before new external testing: five-value persistence blend grid, node-by-horizon median residual, correction factor 0.25 and nonnegative clipping.
4. The blend weight is selected only on validation data by minimizing the maximum of MAE/RMSE/SMAPE/WAPE ratios, with mean ratio as tie-breaker.
5. A10 passed chronological half-validation evaluation in all 60 metric comparisons across five tasks and three seeds.
6. Use PEMS07 as the next and final new external dataset. Download and checksum must be frozen before DCRNN training or calibration.

## 2026-08-05 A10 PEMS07 complete confirmation

1. The PEMS07 artifact, node order, strict chronological split, train-only statistics and pre-test isolation audit passed before test predictions were generated.
2. A10 passed the locked ordinary gate: 3/3 wins for MAE, RMSE, SMAPE and WAPE, all four mean ratios below 1.0, 12/12 strict moving-block bootstrap improvements and 36/36 horizon-MAE improvements.
3. The train-only flow-deficit resilience gate passed under the preregistered A9-consistent rule: deficit MAE, deficit RMSE and q90-tail deficit MAE each improved in 3/3 seeds, all three mean ratios below 1.0, and mean high-state F1 did not decrease.
4. Mean ordinary ratios are 0.988701 MAE, 0.995057 RMSE, 0.952364 SMAPE and 0.988701 WAPE. Mean resilience ratios are 0.947488 deficit MAE, 0.928830 deficit RMSE and 0.894058 q90-tail MAE. Mean high-state F1 is 0.717946 for A10 versus 0.689878 for DCRNN.
5. Select A10 as the current confirmed model for the frozen PEMS07 protocol. Do not claim universal superiority, causal event detection or superiority of a pure new graph architecture. Preserve the stop rules for future datasets and event interpretations.
6. The Chinese delivery document is `A10成功方案过程说明.md`; authoritative machine-readable evidence is in `D:\TrafficGNN\outputs\a10_pems07_external_confirmation`.
