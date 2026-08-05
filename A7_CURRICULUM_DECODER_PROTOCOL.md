# A7 DSTSGCN Curriculum Decoder Protocol

## Objective

Test whether the A6 validation-to-test failure is caused by training an autoregressive decoder only in free-running mode. Keep the DSTSGCN graph learner, static-dynamic fusion, DSTSGCN blocks, frozen L4 definition, event windows and test thresholds unchanged. Modify only the decoder training inputs.

## Model Boundary

1. The existing DSTSGCN backbone produces the final node state.
2. The node-shared GRUCell autoregressive decoder from A6 is retained.
3. The last observed traffic value is always the first decoder input.
4. During training only, the next decoder input is selected between the previous prediction and the corresponding observed target.
5. Validation and all later evaluations are strictly free-running and never receive future labels.
6. DCRNN predictions, weights and hidden states are prohibited.

## Locked Curriculum

The teacher-forcing probability is fixed before validation:

1. Start at 1.0 on the first optimizer step.
2. Decrease linearly to 0.0 over the first 80% of planned optimizer steps.
3. Remain at 0.0 for the final 20% of training.
4. Sampling is performed independently for each sample, node and output variable.
5. Missing or non-finite targets can never be selected as decoder inputs.

The schedule shape, decay fraction, loss, optimizer and gradient clipping must not be changed using event-test results.

## Training Contract

Use traffic history only, train-only correlation adjacency, train-only imputation and scaling, Huber traffic loss, Adam with weight decay 1e-4, and gradient clipping at norm 5.0. Checkpoint selection uses validation physical-space MAE. Event labels, weather/event adjacency, L4 auxiliary supervision, CVaR and test metrics are prohibited during A7 selection.

## Engineering Gate

Before any full validation run:

1. Python compilation passes.
2. Decoder unit tests cover teacher-forcing ratio 0 and 1, invalid targets, output shape and gradients.
3. A dry-run completes one optimizer step and one free-running validation pass.
4. Five-task smoke runs complete with finite losses and physical validation metrics.

## Validation Gate

Run Bridge flow, Rainstorm flow/speed and Typhoon flow/speed for seeds 42, 2024 and 3407. The A7 candidate advances only if all conditions hold:

1. All 15 runs are finite.
2. A7 improves A6 validation MAE in at least 9 of 15 comparisons.
3. Mean A7/A6 validation MAE ratio is at most 0.99.
4. A7 remains better than direct-head M1 in at least 12 of 15 comparisons.
5. Mean A7/M1 validation MAE ratio is at most 0.97.
6. No dataset-variable mean A7/A6 ratio exceeds 1.05.

Failure stops A7 without evaluating the existing event-test archive.

## Confirmation Boundary

Passing validation does not establish superiority over a public baseline. Because the existing event-test archive has already been inspected, final evidence must come from a newly locked temporal holdout or external PEMS04/PEMS08 evaluation. Hyperparameters and thresholds must be frozen before that confirmation.
