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