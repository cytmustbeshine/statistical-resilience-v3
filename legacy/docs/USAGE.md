# D-STSGCN 代码使用说明

代码位置：

```text
D:\TrafficGNN\dstsgcn_code
```

输出位置：

```text
D:\TrafficGNN\outputs
```

## 1. 环境准备

先打开 PowerShell，进入代码目录：

```powershell
cd D:\TrafficGNN\dstsgcn_code
```

建议创建虚拟环境：

```powershell
python -m venv D:\TrafficGNN\.venv
D:\TrafficGNN\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install numpy pandas scikit-learn
pip install torch torchvision torchaudio
```

如果你有 NVIDIA 显卡，后面可以再安装 GPU 版 PyTorch；初期先用 CPU 版把流程跑通。

检查 PyTorch 是否安装成功：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. 先跑一个小规模训练

建议先用暴雨数据集，因为它是宽表格式，最容易接入。

示例命令：

```powershell
python train.py `
  --csv "C:\Users\Administrator\Desktop\交通\_inspect\2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据集\2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据.csv" `
  --time-col "Time" `
  --value-suffix "_volume" `
  --history 12 `
  --horizon 12 `
  --max-nodes 30 `
  --batch-size 16 `
  --epochs 5 `
  --output-dir "D:\TrafficGNN\outputs\rainstorm_volume_test"
```

这条命令的含义：

- 用过去 12 个时间片预测未来 12 个时间片。
- 数据粒度是 5 分钟，所以相当于用过去 1 小时预测未来 1 小时。
- `--max-nodes 30` 表示先只取 30 个检测器，方便快速验证。
- 模型会保存到 `D:\TrafficGNN\outputs\rainstorm_volume_test\best_dstsgcn.pt`。

## 3. 预测速度

如果要预测速度，把后缀改成 `_speed`：

```powershell
python train.py `
  --csv "C:\Users\Administrator\Desktop\交通\_inspect\2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据集\2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据.csv" `
  --time-col "Time" `
  --value-suffix "_speed" `
  --history 12 `
  --horizon 12 `
  --max-nodes 30 `
  --batch-size 16 `
  --epochs 5 `
  --output-dir "D:\TrafficGNN\outputs\rainstorm_speed_test"
```

## 4. 正式实验建议

小规模跑通后，再逐步扩大：

```text
max-nodes: 30 -> 80 -> 全部节点
epochs: 5 -> 50 -> 100
batch-size: 根据内存/GPU 显存调整
```

正式实验至少要跑：

- STSGCN baseline
- DGCN baseline
- D-STSGCN static-only
- D-STSGCN dynamic-only
- D-STSGCN fusion

当前代码已经实现 D-STSGCN fusion 原型，后续可以继续补 baseline 和消融开关。

## 5. 常见问题

如果提示 `No module named torch`：

```powershell
pip install torch torchvision torchaudio
```

如果训练很慢：

- 先减少 `--max-nodes`
- 减少 `--batch-size`
- 减少 `--epochs`
- 后续配置 GPU 版 PyTorch

如果路径包含中文导致问题：

- 可以把数据复制到 `D:\TrafficGNN\data`
- 然后用英文路径运行
