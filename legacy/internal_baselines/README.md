# 内部基线历史归档

本目录保存转向“交通韧性预测”主线之前的内部模型入口副本。

归档内容：

- `model_pre_resilience_aux.py`：包含 STSGCN-style、DGCN-style、quality_v2、quality_ood_scaled_v1 的旧版模型组件。
- `train_pre_resilience_aux.py`：旧版 D-STSGCN 训练入口。
- `run_experiments_pre_resilience_aux.py`：旧版主实验批量入口。

这些模型不再作为当前论文主模型：

- `dgcn`：动态图结构消融基线，回答的是交通预测问题，不是交通韧性主问题。
- `quality_v2`：静态图-动态图融合强基线，但统计韧性含义不足。
- `quality_ood_scaled_v1`：OOD 韧性感知扩展，对 performance loss 有一定帮助，但没有形成稳定全面提升。

当前论文主线应转向“统计韧性辅助交通预测模型”，并与公开交通韧性相关基线对比。若确实需要复现实验，可在主入口中显式加入 `--allow-legacy-internal-models`。
