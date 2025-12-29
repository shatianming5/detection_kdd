# baselines/DiffusionDet/configs/

DiffusionDet 的 Detectron2 配置文件集合。

## 功能与作用

- 定义模型结构、训练超参、数据集名称与推理参数。
- 通过 `_BASE_` 继承方式复用上游默认配置。

## 结构与关键文件

- `Base-DiffusionDet.yaml`：上游基础配置（模型/solver/TTA 默认值）。
- `diffdet.coco.*.yaml`、`diffdet.lvis.*.yaml`：上游 COCO/LVIS 训练配置。
- `diffdet.repro_10k.yaml`：本项目 Repro-10k 配置：
  - `DATASETS.TRAIN/TEST` = `repro_10k_train` / `repro_10k_val`
  - `MODEL.DiffusionDet.NUM_CLASSES = 3`
  - 当前为完整 baseline 日程（`MAX_ITER=10000`，`STEPS=(8000,9500)`）；需要短跑验证时用命令行覆盖 `SOLVER.MAX_ITER`。

## 如何验证功能是否实现

1. **配置可解析**
   ```bash
   cd baselines/DiffusionDet
   python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 SOLVER.MAX_ITER 50
   ```
   若能进入训练循环，则 YAML 与 `add_diffusiondet_config` 匹配。
2. **数据集名称正确**
   - 启动时若能打印 “Registered datasets … repro_10k_*”，且无找不到 dataset 的报错，说明 `TRAIN/TEST` 指向有效注册名。
