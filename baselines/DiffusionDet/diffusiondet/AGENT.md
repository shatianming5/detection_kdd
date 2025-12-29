# baselines/DiffusionDet/diffusiondet/

DiffusionDet 的核心实现区，按 Detectron2 的 Meta-Arch 组织，包含模型、动态 head、扩散过程、损失、数据 mapper 等。

## 功能与作用

- 实现 DiffusionDet 主体模型（扩散提案 + 动态检测头）。
- 定义训练/推理所需的扩散调度、matching 与 loss。
- 提供 Detectron2 训练管线可调用的数据预处理与 TTA。

## 结构与关键文件

- `detector.py`：`DiffusionDet` Meta-Arch（`@META_ARCH_REGISTRY.register()`），包含扩散采样、动态 K matching、后处理等。
- `head.py`：`DynamicHead` 与 Transformer/ROI Pooler 组件。
- `loss.py`：`SetCriterionDynamicK`、`HungarianMatcherDynamicK` 等损失/匹配实现。
- `dataset_mapper.py`：`DiffusionDetDatasetMapper`，Detectron2 dataset dict → DiffusionDet 输入；包含 Resize/Flip/Crop 等增强。
- `config.py`：`add_diffusiondet_config(cfg)`，补充 DiffusionDet/Swin/solver/TTA 配置项。
- `predictor.py`：可视化 demo 的 `VisualizationDemo`（Detectron2 默认 predictor 的封装）。
- `swintransformer.py`：Swin backbone 支持。
- `test_time_augmentation.py`：`DiffusionDetWithTTA`，推理阶段测试时增强。
- `util/`：bbox/misc/EMA 等通用工具（见下级 `AGENT.md`）。

## 如何验证功能是否实现

1. **Meta-Arch 与配置可用**
   - `cd baselines/DiffusionDet`
   - `python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1`
   - 若能构建 `DiffusionDet` 并开始迭代，说明 `detector.py`、`config.py`、`dataset_mapper.py` 正常。
2. **Loss/Matching 链路**
   - 训练前几十 iter 不应出现维度或 assignment 错误；loss 应持续下降或稳定输出。
3. **TTA/可视化（可选）**
   - 使用 `demo.py` 或 `DiffusionDetWithTTA` 推理，若能输出实例与可视化结果，则 predictor/TTA 链路完整。

