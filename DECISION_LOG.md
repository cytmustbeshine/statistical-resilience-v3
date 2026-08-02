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
