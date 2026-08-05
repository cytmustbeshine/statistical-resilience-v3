# A9 Statistical Identity Residual Traffic Model Protocol

## Motivation

A8 improved short-horizon speed forecasting but failed PEMS flow demand prediction. The failure is attributed to a short-history graph backbone that lacks explicit spatial identity, calendar identity and long-period demand structure. A9 is a new model rather than another DSTSGCN decoder modification.

## Model

A9 combines five sources of information for each node:

1. The previous 12 traffic observations.
2. A learned node identity embedding.
3. Prediction-origin time-of-day and day-of-week embeddings.
4. Train-only correlation-graph context obtained by propagating the history embedding once through the static adjacency.
5. Robust history summaries: median-centered level divided by MAD, first-to-last trend and median absolute first difference.

The fused representation is processed by three residual multilayer perceptron blocks. The network predicts a correction to a train-only seasonal baseline rather than the raw traffic level.

## Statistical Seasonal Baseline

For each node, fit median traffic by five-minute time-of-day slot and weekday/weekend type using training timestamps only. Shrink each stratum median toward the corresponding node-by-time-slot median with weight `n / (n + 7)`. Missing strata fall back to node-by-time-slot median and then node median. Validation and future baselines use only calendar identities and these frozen train estimates.

The baseline, scaler and correlation adjacency are all fitted before the validation boundary. No validation or test observation contributes to them.

## Locked Architecture

1. History embedding dimension: 32.
2. Node embedding dimension: 32.
3. Time-of-day embedding dimension: 32 with 288 slots.
4. Day-of-week embedding dimension: 32 with seven categories.
5. Robust-state embedding dimension: 32.
6. Correlation-context dimension: 32.
7. Three residual MLP blocks with hidden width 256 and dropout 0.10.
8. Horizon 12 and one output variable.
9. Final residual projection is zero-initialized so the initial forecast equals the statistical seasonal baseline.

## Locked Training

Use physical values sorted by parsed timestamps, strict train-only imputation/scaling, first 41 nodes, history/horizon 12, Adam learning rate 0.001, weight decay 1e-4, gradient clipping 5.0, batch size 64, masked MAE loss and 50 epochs. Select checkpoints by validation physical MAE.

DCRNN predictions, weights and hidden states are prohibited as A9 inputs. PEMS test metrics are prohibited during A9 selection.

## Validation Tasks

Use only the 60% train and following 20% validation portions of:

1. PEMS04 flow.
2. PEMS04 speed.
3. PEMS08 flow.
4. PEMS08 speed.

The already-viewed final 20% test portions remain inaccessible to A9 evaluation.

## A9-0 Gate

Run seed 42. Advance only if:

1. All four runs are finite.
2. A9 improves validation MAE over the previously frozen public DCRNN in at least three of four tasks.
3. Mean A9/DCRNN validation MAE ratio is at most 0.98.
4. No task ratio exceeds 1.03.

## A9-1 Gate

If A9-0 passes, run seeds 2024 and 3407. Advance only if:

1. A9 improves DCRNN in at least 9 of 12 comparisons.
2. Mean A9/DCRNN validation MAE ratio is at most 0.98.
3. Every dataset-variable task mean ratio is at most 1.03.
4. Node and temporal embeddings receive nonzero gradients.
5. The seasonal baseline is unchanged across seeds.

## Confirmation Boundary

PEMS04/08 test segments can no longer confirm A9. Passing A9-1 authorizes a separately preregistered confirmation on a new public dataset such as PEMS03/PEMS07, or on untouched rolling-origin blocks whose boundaries are fixed before A9 predictions are generated.

## Stop Rule

If A9 fails either validation gate, do not inspect PEMS test predictions and do not tune from the failed task ratios. Record the result and redesign using a new train/validation hypothesis.
