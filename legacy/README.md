# 历史代码归档

本目录保存历史探索脚本和旧版说明文档。这些内容仅用于追溯研究过程，不再属于当前交通韧性论文主线。

## 目录结构

- `analysis/`：旧版消融分析、天气分布偏移分析、论文结果分析、预测脚本和 shift 诊断脚本。
- `run_scripts/`：旧版 PowerShell 实验入口。
- `docs/`：旧版使用说明和早期 README 内容。
- `internal_baselines/`：`dgcn`、`quality_v2`、`quality_ood_scaled_v1` 等内部基线历史副本。

## 当前主线

当前工作请使用上一级目录中的主线文件：

- `run_experiments.py`
- `train.py`
- `model.py`
- `resilience_metrics.py`
- `analyze_resilience_results.py`
- `baselines/dcrnn_resilience/`

如果后续确实需要重新启用某个历史脚本，应先确认其与当前论文主线的关系，再有意识地移回或复制，并同步检查导入路径、数据路径和输出路径。
