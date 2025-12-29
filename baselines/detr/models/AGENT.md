# baselines/detr/models/

该目录包含 DETR 基线的模型实现与构建逻辑，是训练/推理的核心代码区。

## 功能与作用

- 定义 DETR 的 **backbone + transformer + set-based head** 结构。
- 提供 set prediction 的 Hungarian matching 与损失计算。
- 支持可选的实例/全景分割头（上游能力）。

## 结构与关键文件

- `backbone.py`：构建特征提取 backbone（默认 ResNet50/101），包含 `FrozenBatchNorm2d` 与 `BackboneBase`，输出多尺度特征供 Transformer 使用。
- `transformer.py`：DETR Transformer 编码器/解码器实现（在 PyTorch Transformer 上做了位置编码与中间层输出修改）。
- `position_encoding.py`：正弦/学习式位置编码构建。
- `matcher.py`：HungarianMatcher，用于 set-based bipartite matching（预测与 GT 一一对应）。
- `detr.py`：DETR 主模型、criterion 与 `build(args)` 工厂函数；本项目通过 `args.num_classes` 支持自定义类别数。
- `segmentation.py`：可选分割分支（`--masks` 时启用）。
- `__init__.py`：模块导出。

## 如何验证功能是否实现

1. **构建链路可用**
   - 在 `baselines/detr/` 下运行 smoke test（见 `baselines/detr/AGENT.md`），若能正常创建模型并开始训练，说明 `build()`、backbone、transformer 等均可用。
2. **类别数传入生效**
   - 训练命令传入 `--num_classes 4` 后不应出现类别维度不匹配报错；如需快速确认，可观察初始化日志中分类头的输出维度。
3. **（可选）分割分支**
   - 传入 `--masks` 训练或 eval，若能正常前向并产出 mask loss/指标，则 `segmentation.py` 链路完整。

