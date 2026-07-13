# D-STSGCN 交通韧性预测主线代码

本目录保留当前硕士论文主线代码。与论文主线无关的历史探索脚本没有删除，已统一归档到 `legacy/`，便于追溯但不再作为主实验入口。

## 当前研究范围

当前论文方向为：

```text
面向突发扰动事件的交通韧性预测：
以 D-STSGCN 作为交通状态预测引擎，并在预测结果之上计算统计韧性指标。
```

当前事件数据集：

- `bridge`：桥梁坍塌，结构性突发扰动；
- `rainstorm`：暴雨，气象扰动；
- `typhoon`：台风，极端天气扰动。

正常交通对照数据集：

- `pems04`
- `pems08`

## 主线文件

- `data.py`：CSV 数据读取、时间排序、标准化、滑动窗口数据集、邻接矩阵构建。
- `model.py`：当前 D-STSGCN 主线模型。
- `train.py`：D-STSGCN 训练入口。
- `run_experiments.py`：实验批量运行入口。
- `resilience_metrics.py`：交通韧性指标计算层。
- `analyze_resilience_results.py`：基于预测结果的韧性指标分析与报告生成。
- `statistical_tests.py`：配对检验、多重比较校正等统计分析工具。
- `audit_dataset_splits.py`：数据集时间顺序、划分与事件分布审计。
- `analyze_public_baselines.py`：STSGCN-style 与 DGCN-style 基线汇总。
- `baselines/dcrnn_resilience_official_adapted/`：轻量化 DCRNN-Resilience 公开基线实现。

说明：`dgcn`、`quality_v2` 和 `quality_ood_scaled_v1` 已降级为历史内部基线，不再作为当前交通韧性论文主模型。相关旧版入口已归档到 `legacy/internal_baselines/`；如需复现实验，必须在 `run_experiments.py` 中显式使用 `--allow-legacy-internal-models`。

## 推荐编译检查

```powershell
D:\soft\Python310\python.exe -m py_compile `
  D:\TrafficGNN\dstsgcn_code\data.py `
  D:\TrafficGNN\dstsgcn_code\model.py `
  D:\TrafficGNN\dstsgcn_code\train.py `
  D:\TrafficGNN\dstsgcn_code\run_experiments.py `
  D:\TrafficGNN\dstsgcn_code\resilience_metrics.py `
  D:\TrafficGNN\dstsgcn_code\analyze_resilience_results.py `
  D:\TrafficGNN\dstsgcn_code\baselines\dcrnn_resilience_official_adapted\model.py `
  D:\TrafficGNN\dstsgcn_code\baselines\dcrnn_resilience_official_adapted\train.py
```

## 历史代码归档

旧的天气分布偏移、panel 特征、消融分析和早期论文实验脚本没有物理删除，而是移动到了 `legacy/`。这样根目录只展示当前交通韧性论文主线，同时保留历史实验的可追溯性。
