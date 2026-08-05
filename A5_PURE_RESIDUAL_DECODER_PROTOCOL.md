# A5 Pure Persistence-Residual Decoder Protocol

## Objective

Test whether the existing DSTSGCN can close the temporal-forecasting gap without using DCRNN predictions at inference. The graph learner, STSGCN blocks, frozen L4 profiles, data splits, node sets, history and horizon remain unchanged.

## Candidate

- Control: existing traffic-only DSTSGCN (`M1`), hidden dimension 64.
- Candidate: the same DSTSGCN wrapped as `prediction = last_observation + residual`.
- The final residual projection is initialized to exactly zero, so the initial prediction equals persistence instead of a random absolute forecast.
- Primary residual scale: 1.0. Sensitivity scale: 0.5.

## Prohibited Inputs and Losses

Do not use event labels, weather/event adjacency, L4 auxiliary targets, CVaR, OOD scaling, test thresholds or test-event metrics during candidate selection.

## A5-0 Validation Gate

Use Bridge flow, Rainstorm flow/speed and Typhoon flow/speed with the existing train/validation split and seed 42.

The candidate advances only if:

1. all five runs are finite and checkpoint correctly;
2. validation MAE improves over standard M1 in at least four of five tasks;
3. validation MAE improves over persistence in at least four of five tasks;
4. mean validation-MAE ratio versus M1 is at most 0.97;
5. no task deteriorates more than 3% versus the better of M1 and persistence.

## Later Gate

If A5-0 passes, use blocked inner validation to freeze the residual scale, learning rate and epoch budget. The locked model must then be evaluated on a new outer holdout or external dataset because the existing event-test archive has already been inspected during prior research.