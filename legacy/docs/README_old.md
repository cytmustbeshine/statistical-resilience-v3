# D-STSGCN 研究原型

这个目录给出一个可扩展的 PyTorch 原型，用来实现：

**DGCN 动态图学习 + STSGCN 局部时空同步图卷积**

建议模型名称：

- `D-STSGCN`
- `DSTSGCN`
- `Dynamic Spatial-Temporal Synchronous Graph Convolutional Network`

## 模型思路

原始 STSGCN 使用固定路网邻接矩阵构造局部时空图。这里加入 DGCN 的动态图思想：

1. 根据历史交通序列学习动态邻接矩阵 `A_dynamic`。
2. 将真实道路拓扑/相关性图 `A_static` 与动态图融合：

```text
A_fused = alpha * A_static + (1 - alpha) * A_dynamic
```

3. 用 `A_fused` 构造 STSGCN 的 3 时间片局部时空图。
4. 进行同步图卷积，预测未来交通流量/速度/TPI。

## 文件说明

- `model.py`：D-STSGCN 模型主体。
- `data.py`：滑动窗口数据集、标准化、宽表 CSV 读取、相关性邻接矩阵构建。
- `train.py`：最小训练脚本。

## 输入格式

当前训练脚本支持宽表 CSV，例如：

```text
Time,1111570_volume,1116139_volume,1108486_volume,...
2023/7/15 0:00,212,135,86,...
2023/7/15 0:05,164,125,65,...
```

这和“暴雨打击下高速路网交通状态演化数据集”的格式比较接近。

## 运行示例

需要先安装 PyTorch。当前机器环境里尚未检测到 `torch`。

```bash
python train.py \
  --csv "../_inspect/2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据集/2021YFB1600102-002 暴雨打击下高速路网交通状态演化数据.csv" \
  --time-col "Time" \
  --value-suffix "_volume" \
  --history 12 \
  --horizon 12 \
  --epochs 50
```

如果预测速度，可以改成：

```bash
python train.py \
  --csv "你的数据.csv" \
  --time-col "Time" \
  --value-suffix "_speed"
```

## 后续可做的实验

建议做以下消融实验：

1. `STSGCN`：只用固定图，不用动态图。
2. `DGCN-like`：只用动态图，不做 STSGCN 局部同步图。
3. `D-STSGCN-static`：`alpha=1`，只用静态图。
4. `D-STSGCN-dynamic`：`alpha=0`，只用动态图。
5. `D-STSGCN-fusion`：学习 `alpha`，融合静态图和动态图。

论文中可以重点说明：动态图学习模块提高了扰动场景下模型对隐含交通依赖变化的适应能力。
