# baselines/DiffusionDet/diffusiondet/util/

DiffusionDet 训练/推理通用工具与辅助模块。

## 功能与作用

- bbox 几何计算与格式转换。
- 分布式/日志辅助、`NestedTensor` 等结构体。
- EMA（Exponential Moving Average）模型维护。
- 可视化/配色辅助。

## 结构与关键文件

- `box_ops.py`：bbox 格式转换、IoU/GIoU、面积等几何操作。
- `misc.py`：分布式训练 helper、张量打包、随机性/日志辅助。
- `model_ema.py`：EMA 模型更新与保存/加载逻辑。
- `plot_utils.py`：训练曲线/指标绘制工具。
- `colormap.py`：可视化配色表。
- `__init__.py`：导出。

## 如何验证功能是否实现

- 随 DiffusionDet smoke test/完整训练一起验证：
  - 无 bbox/分布式/EMA 相关报错。
  - 若启用 EMA（config 中 `MODEL_EMA`），checkpoint 中应包含 EMA 权重。

