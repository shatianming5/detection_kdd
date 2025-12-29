# baselines/DiffusionDet/

DiffusionDet 基线代码，基于 Detectron2 训练框架的扩散模型目标检测器。本项目在上游代码基础上加入了 `repro_10k` 数据注册与配置。

## 功能与作用

- 提供 DiffusionDet 在 Repro 子集上的训练/评估。
- 作为与 DETR 对比的扩散模型检测基线。

## 关键结构

- `train_net.py`：Detectron2 风格训练入口。
  - 顶部增加了 `repro_10k_train`、`repro_10k_val` 的 COCO 数据集注册，默认读取 `../data/repro_10k/`。
- `configs/`
  - `diffdet.repro_10k.yaml`：本项目使用的 Repro-10k 配置（NUM_CLASSES=3，短跑迭代数用于验证）。
  - 其他文件为上游 COCO/LVIS 配置。
- `diffusiondet/`：模型/mapper/EMA 等核心实现。
- `demo.py`：推理可视化 demo（上游保持）。

## 项目内的适配点

- **数据集注册**：`train_net.py` 自动注册：
  - `repro_10k_train` → `baselines/data/repro_10k/annotations/instances_train2017.json`
  - `repro_10k_val` → `baselines/data/repro_10k/annotations/instances_val2017.json`
- **Repro 配置**：`configs/diffdet.repro_10k.yaml`
  - `DATASETS.TRAIN/TEST` 指向上述注册名。
  - 当前为完整 baseline 配置（`MAX_ITER=10000`，`STEPS=(8000,9500)`）；smoke test 时可通过命令行覆盖 `SOLVER.MAX_ITER`。

## 如何验证功能是否实现

1. **确认数据路径**
   - `../data/repro_10k/annotations/instances_train2017.json`
   - `../data/repro_10k/train2017/`、`val2017/`
2. **快速训练 Smoke Test**
   - 在本目录执行：
     ```bash
     python train_net.py \
       --config-file configs/diffdet.repro_10k.yaml \
       --num-gpus 1 \
       SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100 OUTPUT_DIR /tmp/diffdet_smoke
     ```
   - 预期：
     - 启动时打印 “Registered datasets …/baselines/data”。
     - 训练迭代到 500 step 后结束。
     - `/tmp/diffdet_smoke/` 下生成日志与 checkpoint。
3. **评估（可选）**
   - 指定权重：
     ```bash
     python train_net.py \
       --config-file configs/diffdet.repro_10k.yaml \
       --eval-only MODEL.WEIGHTS <path/to/model.pth> \
       --num-gpus 1
     ```
