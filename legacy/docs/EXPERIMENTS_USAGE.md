# Experiment Runner Usage

Code folder:

```text
D:\TrafficGNN\dstsgcn_code
```

Prepared English-path data files:

```text
D:\TrafficGNN\data\rainstorm_traffic_state.csv
D:\TrafficGNN\data\bridge_collapse_flow.csv
```

## 1. Open in VSCode

Open this folder:

```text
D:\TrafficGNN\dstsgcn_code
```

Then open a new PowerShell terminal in VSCode.

## 2. First check commands without training

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_quick --dry-run
```

This only prints the planned commands. It does not train.

## 3. Run the basic rainstorm comparison

This runs `static` and `quality` once, both with seed 42.

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_quick
```

Summary output:

```text
D:\TrafficGNN\outputs\experiments\rainstorm_quick\experiment_summary.csv
```

## 4. Run three-seed repeat experiments

This runs:

```text
static  seed=42,2024,3407
quality seed=42,2024,3407
```

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_seeds
```

Summary output:

```text
D:\TrafficGNN\outputs\experiments\rainstorm_seeds\experiment_summary.csv
```

## 5. Run bridge dataset comparison

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite bridge_quick --epochs 10
```

Summary output:

```text
D:\TrafficGNN\outputs\experiments\bridge_quick\experiment_summary.csv
```

## 6. Run both datasets quickly

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite all_quick --epochs 10
```

## 7. Recommended next experiment

Start with:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py --suite rainstorm_seeds
```

Send the generated `experiment_summary.csv` back for analysis.

## 8. Current mainline probability/OOD models

For the simplified paper track, prefer these model names:

```text
quality_v2              point-prediction baseline
quantile or quantile_plain
                        equal-weight quantile prediction with combined loss
quantile_cqr            quantile_plain plus CQR interval calibration
causal_quality          quality_v2 with Granger static graph prior
causal_quantile_cqr     quantile_cqr with Granger static graph prior
```

The mainline quantile models use:

```text
--quantile-loss-mode combined
--quantile-mae-alpha 0.5
--quantile-weights 1.0 1.0 1.0
```

CQR is explicit, not automatic. Only `*_cqr` models pass:

```text
--calibration-mode cqr
--target-picp 0.80
```

The older weight-search models remain available for reproducibility, but are
treated as legacy ablations rather than the recommended research path:

```text
quantile_search_mae
quantile_search_multi
quantile_search_optuna
quantile_search_optuna_constrained
adaptive_quantile_quality_v2
```

Recommended short check:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py ^
  --suite all_quick ^
  --models quality_v2,quantile_plain,quantile_cqr,causal_quality,causal_quantile_cqr ^
  --epochs 8 ^
  --max-nodes 41 ^
  --calib-ratio 0.1 ^
  --output-root D:\TrafficGNN\outputs\experiments_mainline_ood_n41_e8
```

## 9. Statistical Residual Graph Fusion Track

The current model-development track returns to the core DGCN-STSGCN fusion
problem. The recommended first-stage comparison is:

```text
quality_v2              quality-gated dynamic/static graph fusion baseline
stat_residual_fusion    statistical-prior anchored residual dynamic graph fusion
```

`stat_residual_fusion` uses the correlation graph as `A_stat` and learns a
small residual correction:

```text
A_fused(t) = normalize(A_stat + gamma * (A_dynamic(t) - A_stat))
```

This is intended to test whether a stable statistical graph prior can anchor
the dynamic graph learner and reduce noisy graph corrections.

Recommended dry-run:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py ^
  --suite rainstorm_quick ^
  --models quality_v2,stat_residual_fusion ^
  --epochs 2 ^
  --max-nodes 5 ^
  --dry-run ^
  --output-root D:\TrafficGNN\outputs\dryrun_stat_residual_fusion
```

Recommended short experiment:

```powershell
D:\soft\Python310\python.exe .\run_experiments.py ^
  --suite all_quick ^
  --models quality_v2,stat_residual_fusion ^
  --epochs 8 ^
  --max-nodes 41 ^
  --output-root D:\TrafficGNN\outputs\experiments_stat_residual_fusion_n41_e8
```

Keep the older quantile weight-search, Optuna, adaptive-weight, CQR, and
Granger variants as legacy/auxiliary experiments. They are not part of the
first-stage statistical residual fusion validation.
