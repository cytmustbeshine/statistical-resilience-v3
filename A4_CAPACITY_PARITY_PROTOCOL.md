# A4 Capacity-Parity Validation Protocol

## Purpose

The completed E-L4-2B stage compared a 1.19M-parameter DSTSGCN configuration with a 3.55M-parameter adapted public DCRNN. A4 tests whether the ordinary-forecasting gap is primarily caused by model capacity and optimization budget before any new graph module, resilience target, or tail-risk loss is introduced.

## Frozen Components

- Keep `LaplaceMatrixLatentNetwork`, `DSTSGCNBlock`, graph fusion, frozen L4 profiles, data splits, node sets, history, horizon, scaler and missing-space protocol unchanged.
- Use traffic-only training. Do not use event labels, weather/event adjacency, L4 auxiliary supervision, CVaR, OOD scaling or test-event metrics during candidate selection.
- Do not change the DCRNN baseline or reuse E-L4-2B test results to choose A4 hyperparameters.

## Stage A4-0: Validation-Only Capacity Screen

- Candidate hidden dimensions: 64, 128 and 160.
- The 160-dimensional model is the capacity-parity candidate because it has about 3.40M parameters, close to the 3.55M DCRNN.
- Use the existing train-only correlation graph, Huber loss, Adam optimizer and validation physical MAE checkpoint selection.
- Evaluate Bridge flow, Rainstorm flow/speed and Typhoon flow/speed.
- The screening program must not construct a test loader, calculate test metrics or write test predictions.

## Selection Rule

For each dataset-variable task, divide the candidate's best validation MAE by the corresponding 64-dimensional control MAE. Select one global configuration using the lowest mean ratio across all five tasks.

A candidate advances only if:

1. all five validation tasks complete with finite predictions;
2. it improves at least four of five tasks over the 64-dimensional control;
3. its mean validation-MAE ratio is at most 0.97;
4. no task deteriorates by more than 5%; and
5. the choice is made without inspecting new test predictions.

## Later Stages

- A4-1 may tune learning rate, scheduler, patience and training epochs using train/validation data only after A4-0 freezes the capacity range.
- A4-2 will run one locked three-seed test comparison against DCRNN, persistence and M1.
- Failure of A4-0 stops capacity scaling. It does not authorize graph-module stacking or test-set tuning.