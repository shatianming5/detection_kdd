# baselines/detr/

DETR（Detection Transformer）基线代码。该目录基本为上游 DETR 仓库拷贝，面向 COCO 风格数据训练/评估；本项目做了轻量适配以支持小类别 Repro 数据集。

## 功能与作用

- 提供 DETR 在 `repro_10k` COCO 子集上的训练与评估能力。
- 作为与 DiffusionDet 对比的经典 Transformer 检测基线。

## 关键结构

- `main.py`：训练/评估入口脚本，支持 `--dataset_file coco` 与 `--coco_path <coco_root>`。
- `models/`：DETR 模型定义与构建逻辑。
  - `models/detr.py` 中支持从命令行传入 `--num_classes`。
- `datasets/`：COCO 数据加载与预处理。
- `engine.py`、`util/`：训练循环、分布式与工具函数。
- `requirements.txt`：最小依赖（PyTorch、torchvision 等）。

## 项目内的适配点

- **自定义类别数**：
  - `main.py` 增加参数 `--num_classes`。
  - `models/detr.py:build()` 在 `args.num_classes` 非空时使用该值。
- **COCO-pretrained 细调（推荐 baseline 路径）**：
  - `run_baselines.sh` 会先运行 `utils/prepare_detr_coco_ids.py` 生成一个“DETR 视角”的 COCO 数据根目录（类别 id 映射到 COCO：person=1, car=3, motorcycle=4）。
  - 然后以 `--num_classes 91` 加载官方 COCO 预训练 DETR 权重并进行 fine-tune。

## 如何验证功能是否实现

1. **数据可读性检查**
   - 确保 `../data/repro_10k/` 存在 COCO 结构：
     - `annotations/instances_train2017.json`
     - `train2017/`、`val2017/`
   - 生成 DETR 使用的 COCO-id 视图（一次即可）：
     ```bash
     python ../../utils/prepare_detr_coco_ids.py --src ../data/repro_10k --dst ../data/repro_10k_detr_coco
     ```
2. **快速训练 Smoke Test**
   - 在本目录执行：
     ```bash
     python main.py \
       --dataset_file coco \
       --coco_path ../data/repro_10k_detr_coco \
       --num_classes 91 \
       --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
       --output_dir /tmp/detr_smoke \
       --epochs 1 \
       --batch_size 2 \
       --num_workers 2
     ```
   - 预期：
     - 控制台打印 loss/学习率等训练日志。
     - `/tmp/detr_smoke/` 下生成 `checkpoint.pth` 等文件。
3. **（可选）运行上游单元测试**
   - `python test_all.py`，用于验证模型与数据管线的基本一致性。
