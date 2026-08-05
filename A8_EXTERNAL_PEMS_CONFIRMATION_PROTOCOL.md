# A8 External PEMS Confirmation Protocol

## Purpose

Provide a genuinely new confirmation test for the validation-locked A8 model. The Bridge, Rainstorm and Typhoon event-test archive has already been inspected and is excluded from confirmatory model selection. PEMS04 and PEMS08 have not been used to choose A8 architecture or training hyperparameters.

## Locked Tasks

Evaluate four tasks:

1. PEMS04 volume.
2. PEMS04 speed.
3. PEMS08 volume.
4. PEMS08 speed.

For computational parity with the event studies, use the first 41 node identifiers in CSV column order. Match variables by explicit suffix and never by positional pairing. Occupancy is reserved for resilience postprocessing and is not a traffic-prediction target in this confirmation.

## Data Contract

1. Parse `Time` with `pandas.to_datetime` and sort by the parsed timestamp.
2. Use history 12 and horizon 12.
3. Use strict chronological target-disjoint boundaries at 60% train, 20% validation and 20% test of timestamps.
4. Fit missing-value imputation, scaler and correlation adjacency on train timestamps only.
5. Preserve physical truth separately from finite model inputs.
6. Use the same selected nodes, windows, scaler, adjacency and physical masks for A8 and DCRNN.
7. Do not inspect test metrics until both model checkpoints are selected and all run metadata pass audit.

## Locked Models

### A8

Use the exact validation-accepted configuration: hidden size 64, two DSTSGCN blocks, direct/autoregressive/persistence experts, robust history-state gate, equal gate initialization, auxiliary expert weight 0.10, Adam learning rate 0.001, weight decay 1e-4, gradient clipping 5.0, 20 epochs and validation physical MAE checkpoint selection.

### Public Baseline

Use `OfficialAdaptedDCRNN` with 256 units, two recurrent layers, diffusion step 1, inverse-sigmoid curriculum decay 2000, Adam learning rate 0.01 with epsilon 1e-3, the published milestone scheduler, physical-space masked RMSE training loss, gradient clipping 5.0 and 20 epochs. Select its checkpoint by validation physical MAE so both models use the same selection metric.

The DCRNN architecture is deliberately larger and keeps its published curriculum and loss; it is not weakened to favor A8.

## Execution Gate

Before full confirmation:

1. Compile all new code.
2. Unit-test timestamp ordering, strict split disjointness, train-only scaler/adjacency and common model inputs.
3. Run one-batch dry-runs for both models.
4. Run all four tasks with capped windows and two epochs.
5. Verify prediction archive shapes, physical units, finite metrics and checkpoint reload.

## Confirmatory Runs

Run seeds 42, 2024 and 3407 for both models on all four tasks. This produces 24 trained runs and 12 paired test comparisons.

Primary metrics are MAE, RMSE, SMAPE and WAPE. A8 satisfies the external ordinary-metric gate only if:

1. A8 improves every primary metric in at least 10 of 12 paired comparisons.
2. A8 improves MAE in all four dataset-variable task means.
3. Mean A8/DCRNN ratio is below 1.0 for every primary metric.
4. No task-mean primary-metric ratio exceeds 1.03.
5. A 1,000-repetition paired moving-block bootstrap with block length 12 has an interval strictly below zero for at least 40 of 48 task-seed-metric comparisons.

## Statistical and Resilience Checks

After ordinary metrics are frozen, apply the existing train-only statistical profile to physical predictions. Report continuous deficit error, high-state classification, tail error, gate-weight stability and horizon-specific error. These analyses cannot change the selected checkpoint or ordinary-metric decision.

## Stop Rule

If A8 fails the external ordinary-metric gate, do not tune it on PEMS test results. Record the failure and design a separately preregistered successor using only train/validation evidence. If it passes, freeze A8 and proceed to the final statistical-resilience analysis and repository delivery.
