# A9 PEMS03 External Confirmation Protocol

## Frozen Data

Use `D:\TrafficGNN\data\public\PEMS03\PEMS03.npz`, downloaded from Zenodo record 7816008 before A9 external training.

1. File size: 15,800,415 bytes.
2. MD5: `651add9bb9eaf7f5eda2f2ee8778a182`.
3. Array key: `data`.
4. Shape: 26,208 timestamps x 358 nodes x 1 flow variable.
5. Physical range: 0 to 1,852.
6. Calendar: complete five-minute sequence from 2018-09-01 00:00 through 2018-11-30 23:55.

Use the first 41 nodes in NPZ order. PEMS03 was not used to select A9 architecture or hyperparameters.

## Split and Statistics

1. History and horizon are both 12.
2. Strict chronological target-disjoint boundaries are 60% train, 20% validation and 20% test of timestamps.
3. Fit missing-value imputation, scaler, correlation adjacency and seasonal median baseline on train timestamps only.
4. Preserve physical truth separately from model inputs.
5. Use identical windows, nodes, scaler, adjacency and masks for A9 and DCRNN.

## Locked Models

### A9

Use the exact validation-accepted configuration from `A9_STATISTICAL_IDENTITY_RESIDUAL_PROTOCOL.md`: embedding dimension 32, hidden width 256, three residual blocks, dropout 0.10, seasonal shrinkage 7, Adam learning rate 0.001, weight decay 1e-4, masked MAE, gradient clipping 5.0, batch size 64 and 50 epochs.

### DCRNN

Use the official-code-adapted DCRNN configuration from the A8 external comparison: 256 units, two layers, diffusion step 1, curriculum decay 2000, Adam learning rate 0.01 with epsilon 1e-3, published milestone scheduler, physical masked RMSE, gradient clipping 5.0, batch size 16 and 20 epochs.

Both checkpoints are selected by validation physical MAE.

## Isolation

Training and evaluation are separate stages. The training stage must not construct a test loader or produce test predictions. Evaluation is authorized only after all six model-seed checkpoints exist and pass a combination/checksum audit.

## Confirmatory Runs

Train seeds 42, 2024 and 3407 for A9 and DCRNN, producing six checkpoints and three paired test comparisons.

Primary metrics are MAE, RMSE, SMAPE and WAPE. The ordinary confirmation gate passes only if:

1. A9 improves every primary metric in all three seed comparisons.
2. Mean A9/DCRNN ratio is below 1.0 for every primary metric.
3. A 1,000-repetition paired moving-block bootstrap with block length 12 has an interval strictly below zero for at least 10 of 12 seed-metric comparisons.
4. Prediction archives have identical physical truth, test indices and node order.

## Statistical Resilience Gate

If and only if the ordinary gate passes, fit the existing train-only flow statistical profile on PEMS03 training timestamps and apply the same frozen transformation to physical A9 and DCRNN predictions. Report continuous deficit MAE/RMSE, q90 tail MAE, high-state F1 and block bootstrap. The final thesis candidate requires A9 to improve continuous deficit MAE/RMSE and q90 tail MAE in at least two of three seeds without lowering mean high-state F1.

## Stop Rule

If the ordinary or resilience gate fails, do not tune A9 using PEMS03 test results. Record the failure and continue only under a separately preregistered new-data protocol.
