# baselines/detr/d2/detr/

Detectron2 wrapper 的核心实现区，负责把 DETR 以 Detectron2 的模型/数据接口形式提供。

## 功能与作用

- 注册 `Detr` Meta-Arch，供 `train_net.py` 调用。
- 定义与 DETR 对齐的数据预处理/增强策略。
- 扩展 Detectron2 配置项以支持 DETR 参数。

## 结构与关键文件

- `config.py`：`add_detr_config(cfg)`，向 Detectron2 `CfgNode` 注入 DETR 所需超参。
- `dataset_mapper.py`：`DetrDatasetMapper`，将 Detectron2 的 dataset dict 转为 DETR 训练输入，并应用匹配的增强。
- `detr.py`：Detectron2 Meta-Arch 实现，封装 DETR 前向与损失。
- `__init__.py`：导出与注册。

## 如何验证功能是否实现

- 通过 `baselines/detr/d2/train_net.py` 训练或评估：
  - 能正确构建 `Detr` 模型并加载数据。
  - 训练日志出现分类/回归 loss。
  - `--eval-only` 输出 COCO 指标。

