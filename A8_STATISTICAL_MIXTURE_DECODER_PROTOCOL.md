# A8 Statistical Traffic-State Mixture Decoder Protocol

## Research Question

Can a pure DSTSGCN model outperform its single-decoder variants by combining complementary temporal forecasts with an explicitly traffic-informed, statistically robust regime gate? DCRNN is used only as a public comparison baseline and is prohibited as a model input or ensemble component.

## Architecture

The A8 model retains the existing dynamic graph learner, static-dynamic fusion and DSTSGCN blocks. A single backbone state feeds three forecast experts:

1. Direct multi-horizon DSTSGCN projection.
2. Node-shared free-running GRU autoregressive decoder.
3. Parameter-free persistence forecast from the last observation.

The experts are fused per sample, horizon and node using non-negative softmax weights. The gate receives the final DSTSGCN node state and three robust traffic-state summaries calculated only from the observed history:

1. Robust level: last observation minus history median, divided by history MAD.
2. Trend: first-to-last observed change per history step.
3. Volatility: median absolute first difference.

These summaries introduce statistical traffic-state structure without event labels, test statistics or future observations.

## Locked Training

1. Train from scratch with seed-specific initialization.
2. Use train-only correlation adjacency, imputation and scaling.
3. Optimize mixture Huber loss plus auxiliary Huber losses for the direct and autoregressive experts.
4. Lock the expert auxiliary weight at 0.10.
5. Use Adam, learning rate 0.001, weight decay 1e-4 and gradient clipping at norm 5.0.
6. Use 20 epochs and validation physical-space MAE checkpoint selection.
7. Validation is free-running; future labels are never decoder inputs.
8. Event labels, weather/event adjacency, L4 auxiliary supervision, CVaR, DCRNN predictions and test metrics are prohibited during selection.

The gate bias is initialized to equal expert weights. No task-specific gate initialization or task-specific hyperparameter is allowed.

## Engineering Gate

Compilation, unit tests, a CUDA dry-run and a five-task smoke must pass before full validation. Tests must cover shape, finite weights, weights summing to one, gradients through both neural experts and robust behavior under constant history.

## A8-0 Validation Gate

Run seed 42 on Bridge flow, Rainstorm flow/speed and Typhoon flow/speed. Advance to three seeds only if:

1. All five runs are finite.
2. A8 improves A6 validation MAE in at least four of five tasks.
3. Mean A8/A6 validation MAE ratio is at most 0.98.
4. No task regresses more than 2% relative to A6.

## A8-1 Three-Seed Gate

Run seeds 42, 2024 and 3407 only after A8-0 passes. Advance to external confirmation only if:

1. All 15 runs are finite.
2. A8 improves A6 in at least 10 of 15 comparisons.
3. Mean A8/A6 validation MAE ratio is at most 0.98.
4. A8 improves direct M1 in all 15 comparisons.
5. No dataset-variable mean A8/A6 ratio exceeds 1.03.
6. Gate weights are non-degenerate: no single expert has mean weight above 0.95 on every task.

## Confirmation Boundary

The event-test archive has already been inspected in prior stages and cannot provide confirmatory evidence for A8. Passing A8-1 only authorizes a newly locked temporal holdout or external PEMS04/PEMS08 comparison against public DCRNN. All architecture and training choices must be frozen before that comparison.
