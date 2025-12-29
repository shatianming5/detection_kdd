# baselines/detr/d2/

DETR 的 Detectron2 封装（上游提供），用于更方便地接入 Detectron2 生态的数据集、backbone 与训练器。本项目主基线默认使用 `main.py`，此封装为可选路径。

## 功能与作用

- 以 Detectron2 的 `META_ARCH_REGISTRY` 方式注册 DETR。
- 复用 Detectron2 的数据与训练/评估框架。
- 提供权重转换脚本，把原版 DETR checkpoint 转成 Detectron2 wrapper 可读格式。

## 结构与关键文件

- `train_net.py`：Detectron2 风格训练/评估入口。
- `configs/`：Detectron2 YAML 配置（检测/分割两套）。
- `converter.py`：权重格式转换。
- `README.md`：上游使用说明。
- `detr/`：
  - `config.py`：`add_detr_config(cfg)`。
  - `dataset_mapper.py`：Detectron2 数据 mapper（匹配 DETR 增强）。
  - `detr.py`：Detectron2 Meta-Arch 实现。

## 如何验证功能是否实现

1. **权重转换 + 评估**
   ```bash
   python converter.py --source_model <url_or_path.pth> --output_model /tmp/converted_model.pth
   python train_net.py --eval-only --config configs/detr_256_6_6_torchvision.yaml MODEL.WEIGHTS /tmp/converted_model.pth
   ```
   若能输出 COCO AP，则 wrapper 与 evaluator 正常。
2. **快速训练 Smoke Test**
   ```bash
   python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 1 SOLVER.MAX_ITER 100
   ```
   若能启动并迭代，说明 Meta-Arch 与 mapper 链路可用。

