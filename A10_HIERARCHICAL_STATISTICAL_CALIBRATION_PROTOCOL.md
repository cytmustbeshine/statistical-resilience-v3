# A10 Hierarchical Statistical Calibration Protocol

## Model Definition

A10 is an enhanced traffic forecasting model built from:

1. The official-code-adapted DCRNN deep sequence predictor.
2. A traffic persistence forecast from the last observed value.
3. A robust node-by-horizon median residual calibration layer.
4. Fixed partial pooling of the calibration residual and a nonnegative physical constraint.

DCRNN remains the deep forecasting backbone, while the new contribution is a transparent train/validation-only statistical correction designed for temporally correlated traffic residuals.

## Frozen Calibration

For each trained DCRNN seed:

1. Generate physical DCRNN predictions on the validation period only.
2. Construct persistence predictions from the final observed traffic value.
3. For each weight in `{0.85, 0.90, 0.95, 1.00, 1.05}`, form `P_w = w * P_DCRNN + (1-w) * P_persistence`.
4. Estimate the node-by-horizon median residual `C = median(Y - P_w)` over validation windows.
5. Apply fixed shrinkage `0.25 * C`.
6. Clip the calibrated physical forecast at zero.
7. Select the weight by lexicographically minimizing the maximum validation ratio across MAE, RMSE, SMAPE and WAPE relative to raw DCRNN; use the mean ratio as tie-breaker.

The correction factor 0.25, weight grid, residual hierarchy, objective and clipping rule are frozen before PEMS07 test evaluation.

## Validation Evidence

Chronological half-validation evaluation was completed on PEMS04 flow/speed, PEMS08 flow/speed and PEMS03 flow, three seeds each. The first half fitted A10 and the second half evaluated it. A10 improved all four metrics in all 15 task-seed comparisons, giving 60/60 improvements. Mean ratios were 0.981220 MAE, 0.985296 RMSE, 0.963417 SMAPE and 0.981220 WAPE.

## Final External Dataset

Use `D:\TrafficGNN\data\public\PEMS07\PEMS07.npz` from Zenodo record 7816008.

1. Size: 43,705,518 bytes.
2. MD5: `978d3d9b85fe640a446983a34271a48d`.
3. Shape: 28,224 timestamps x 883 nodes x 1 flow variable.
4. Use the first 41 nodes in NPZ order.
5. Use strict chronological 60/20/20 target-disjoint splits, history 12 and horizon 12.
6. Fit scaler and correlation adjacency on training rows only.

The dataset contains five-minute flow samples from the published PEMS07 period. A10 calibration does not require reconstructing omitted calendar dates; it uses chronological window order, horizon and node identity only.

## DCRNN Training

Use 256 units, two recurrent layers, diffusion step 1, curriculum decay 2000, Adam learning rate 0.01 with epsilon 1e-3, published milestone scheduler, physical masked RMSE, gradient clipping 5.0, batch size 16 and 20 epochs. Select the checkpoint by validation physical MAE.

## Isolation

The training/calibration stage cannot construct a test loader. Test evaluation is authorized only after all three DCRNN checkpoints and all three A10 calibration archives exist, checksums match and zero test prediction files are present.

## Final Ordinary Gate

Run seeds 42, 2024 and 3407. A10 must:

1. Improve MAE, RMSE, SMAPE and WAPE in all three paired comparisons.
2. Have a mean A10/DCRNN ratio below 1.0 for every metric.
3. Obtain strictly negative 95% moving-block bootstrap intervals in at least 10 of 12 seed-metric comparisons using 1,000 repetitions and block length 12.
4. Preserve identical physical truth, test indices and node order.

## Statistical Gate

If the ordinary gate passes, quantify calibration residual dependence, block-bootstrap uncertainty, horizon-specific gains and train-only flow-deficit performance. The final model requires no material degradation of frozen statistical resilience metrics.

Use the same confirmatory rule preregistered for A9: continuous flow-deficit MAE, continuous flow-deficit RMSE and q90-tail flow-deficit MAE must each improve in at least two of three seeds, their three-seed mean ratios must remain below 1.0, and mean q90 high-state F1 must not decrease. Fit the hierarchical conditional ECDF, deficit clipping value and q90 threshold on PEMS07 training rows only. Because PEMS07 does not provide a trustworthy complete calendar, use the observed five-minute sequence index modulo 288 for intraday grouping and disable weekday/weekend grouping; do not fabricate calendar dates. Use 1,000 moving-block bootstrap repetitions with block length 12 as uncertainty evidence without changing this gate after observing the results.

## Stop Rule

If A10 fails PEMS07, do not alter its grid, shrinkage or residual hierarchy from PEMS07 test results. Record the failure and do not claim the project goal is complete.
