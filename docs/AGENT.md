# 项目总览（tiasha/archives）

该仓库用于“Repro”系列数据集上的目标检测基线实验存档与复现。核心流程是：

1. **准备/解包原始数据**：将 `repro_10k*.tar.gz`、`repro_50k*.tar.gz`、`repro_200k*.tar.gz` 放在某个目录中（推荐 `datasets/` 或仓库根目录），运行 `unpack_datasets.py` 解包，生成 `repro_10k/`、`repro_50k/`、`repro_200k/`（VOC2012 风格）。
2. **转换为 COCO 格式（供训练）**：用 `utils/prepare_coco_data.py` 把 VOC 子集转换为 COCO 格式数据，输出到 `baselines/data/repro_10k/`（当前仅内置了 10k 的 COCO 转换数据）。
3. **运行基线**：执行 `run_baselines.sh`，依次训练 DiffusionDet 与 DETR，并把结果写入 `baselines/output/<timestamp>/`。

## check.md 工程栈对齐（MMDet3 + Diffusers）

如果你的目标是按 `check.md:6.1` 的工程形态（MMDetection 3.x + Diffusers Pipeline + 固定 N=500 图数据 collate）跑通训练/推理，本仓库已提供独立实现：

- `mmdet_diffusers/mmdet3/`：MMDet3 插件（`GraphDiffusionDetector` + `coco_graph_collate` + `MMDet3HybridDiffusionPipeline`）
- `mmdet_diffusers/mmdet_configs/`：toy smoke + COCO/LVIS/CrowdHuman 配置模板

最小 smoke（要求单独 conda env：Python3.11 + mmdet3）：

```bash
conda run -n mmdet3 python mmdet_diffusers/tools/make_toy_coco.py --out datasets/toy_coco
conda run -n mmdet3 mim train mmdet mmdet_diffusers/mmdet_configs/toy_graph_diffusion_r50_fpn.py --work-dir /tmp/mmdet_toy_graphdiff --gpus 1 --launcher none -y
```

## 目录结构

- `baselines/`：两套检测基线（DETR、DiffusionDet）及训练所需的 COCO 格式数据与输出。
- `datasets/`：原始 tar.gz 数据包放置目录（当前为空）。
- `repro_10k/`、`repro_50k/`、`repro_200k/`：解包后的 VOC2012 风格子集。
- `utils/`：数据转换与准备脚本。
- `run_baselines.sh`：一键跑 10k 基线脚本。
- `unpack_datasets.py`：批量解 tar.gz 数据包脚本。
- `install_d2.log`、`baselines_execution.log`：安装/运行日志存档。

## 快速验证（Smoke Test）

在依赖已安装（PyTorch、Detectron2 等）前提下：

1. **确认 COCO 数据存在**：
   - `baselines/data/repro_10k/annotations/instances_train2017.json`
   - `baselines/data/repro_10k/train2017/`、`val2017/`
2. **跑最小基线**（可选，耗时取决于机器）：
   - DiffusionDet：`cd baselines/DiffusionDet && python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100 OUTPUT_DIR /tmp/diffdet_smoke`
     - `diffdet.repro_10k.yaml` 为完整 baseline 配置；smoke test 通过命令行覆盖迭代数。
   - DETR（推荐 baseline 路径：COCO-pretrained fine-tune）：
     - `python utils/prepare_detr_coco_ids.py --src baselines/data/repro_10k --dst baselines/data/repro_10k_detr_coco`
     - `cd baselines/detr && python main.py --dataset_file coco --coco_path ../data/repro_10k_detr_coco --num_classes 91 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_dir /tmp/detr_smoke --epochs 1 --batch_size 2`
3. **检查输出**：
   - `baselines/output/<timestamp>/repro_10k_diffdet/` 下有日志与 checkpoint。
   - `/tmp/detr_smoke/` 下有 `checkpoint.pth` 与训练日志。
