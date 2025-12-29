# baselines/

该目录包含本项目使用的两套目标检测基线及其数据/输出约定，用于在 `repro_10k` 等子集上训练与评估。

## 功能与作用

- **DETR 基线**：`detr/`，基于 Transformer 的端到端检测器。
- **DiffusionDet 基线**：`DiffusionDet/`，基于 Detectron2 的扩散模型检测器。
- **训练数据**：`data/`，COCO 风格数据（当前主要提供 `repro_10k`）。
- **实验输出**：`output/`，每次运行生成一个时间戳目录，内部包含两条基线的结果。

## 结构

- `detr/`
  - 上游 DETR 代码拷贝，新增 `--num_classes` 以适配小类别数据。
  - 训练入口：`main.py`
- `DiffusionDet/`
  - 上游 DiffusionDet 代码拷贝，加入 `repro_10k` 数据集注册和配置。
  - 训练入口：`train_net.py`
  - 配置：`configs/diffdet.repro_10k.yaml`
- `data/repro_10k/`
  - `annotations/instances_{train2017,val2017}.json`
  - `train2017/`、`val2017/`（多为符号链接到 VOC 原图）
- `output/<timestamp>/`
  - `repro_10k_diffdet/`
  - `repro_10k_detr/`

## 验证方式

1. **数据验证**
   - 确认 `data/repro_10k/` 的 COCO 结构完整（见 `baselines/data/AGENT.md`）。
2. **单模型快速训练**
   - DiffusionDet：`cd DiffusionDet && python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1`
   - DETR（COCO-pretrained fine-tune 路径）：
     - `python ../utils/prepare_detr_coco_ids.py --src data/repro_10k --dst data/repro_10k_detr_coco`
     - `cd detr && python main.py --dataset_file coco --coco_path ../data/repro_10k_detr_coco --num_classes 91 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_dir /tmp/detr_smoke --epochs 1`
3. **一键跑完整基线**
   - 回到仓库根目录执行：`bash run_baselines.sh`
   - 预期：`output/<timestamp>/` 下出现两套模型的训练输出。
