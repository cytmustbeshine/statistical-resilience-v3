# A4 Locked Hybrid Candidate Protocol

## Scope

A4 evaluates a validation-locked hybrid forecast pipeline against the public adapted DCRNN (`M0`) on the existing three-seed test archives. It is an engineering/statistical candidate, not yet the final thesis model.

## Candidate

- Rainstorm flow: `0.90*M0 + 0.00*M1 + 0.10*persistence`.
- Rainstorm speed: `0.85*M0 + 0.10*M1 + 0.05*persistence`.
- Typhoon flow: `0.85*M0 + 0.15*M1 + 0.00*persistence`.
- Typhoon speed: `0.85*M0 + 0.05*M1 + 0.10*persistence`.
- Bridge flow: use M0 with a per-seed, per-horizon median residual correction multiplied by `0.75`. The correction is fitted on that seed's validation predictions only.

Here `M0` is the public DCRNN baseline, `M1` is the existing traffic-only DSTSGCN, and persistence repeats the last observed value.

## Selection

Weights were selected from pooled validation predictions for seeds 42, 2024 and 3407 using a grid step of `0.05`. The selection objective minimized the worst validation ratio across MAE, RMSE, SMAPE and WAPE, relative to M0, while requiring all validation ratios to be no greater than one. Bridge calibration structure and shrinkage were selected by the same validation-only rule.

No test metric was used to select weights or calibration.

## Test Evidence

On the existing three-seed test archives, the locked candidate improves on M0 for all four ordinary traffic metrics in all 15 seed-task comparisons:

- MAE: 15/15 lower; mean ratio 0.9648.
- RMSE: 15/15 lower; mean ratio 0.9680.
- SMAPE: 15/15 lower; mean ratio 0.9403.
- WAPE: 15/15 lower; mean ratio 0.9648.

A 1,000-repetition moving-block bootstrap with block length 12 gives a strictly negative candidate-minus-M0 95% interval in 58/60 seed-task-metric comparisons. The two intervals crossing zero are Bridge seed 3407 SMAPE and WAPE; neither reverses the point estimate.

## Scientific Limitation

This candidate contains the public DCRNN itself. Therefore it demonstrates that a validation-locked hybrid/post-processing pipeline can exceed the raw public baseline on ordinary prediction metrics, but it does not demonstrate that the pure DSTSGCN architecture has learned a superior representation. It also does not yet establish superiority on every frozen L4 resilience metric. A separate A4-L4 audit is required before any thesis-level claim.