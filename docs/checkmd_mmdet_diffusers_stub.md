# checkmd_mmdet_diffusers_stub.md — `docs/check.md` 的 MMDet+Diffusers 口径在本仓库内的对齐说明（已落地）

`docs/check.md` 的 Phase1 明确要求 **PyTorch + MMDetection + Diffusers** 的工程栈（`HybridDiffusionPipeline` + 自定义 collate）。

本仓库主线仍是 **Detectron2/DiffusionDet**；为对齐 `docs/check.md` 的工程栈（MMDet3 + Diffusers + 固定 N 图数据），本仓库新增并补齐：

- `mmdet_diffusers/`：提供 `HybridDiffusionPipeline`（Diffusers 风格接口）、box scheduler（DDIM）与 label scheduler（D3PM），以及 `collate_coco_graph`；并提供 Detectron2/DiffusionDet 适配入口（推理侧）。
- `mmdet_diffusers/mmdet3/`：提供 **可直接训练/推理** 的 MMDet3 插件：
  - `coco_graph_collate`：固定 N 节点图 collate（GT + 噪声 + unk）
  - `GraphDiffusionDetector`：固定 N proposals 的扩散检测器（含 `GraphDenoisingNetwork`、可选质量头/能量引导、可选拓扑损失）
  - `MMDet3HybridDiffusionPipeline`：Diffusers 风格 pipeline（支持 `init_state=(boxes,label_state)` 作为 tuple 输入）

注意：当前仓库默认运行环境是 Python 3.13，`mmdet/mmengine/mmcv` 往往不可用/不稳定，因此 **MMDet3 训练仍建议在独立 conda 环境里运行**（下述 `mmdet3`）。

## 1) 严格同栈：独立环境（推荐）

目标：按 `docs/check.md:6.1` 的栈与形态运行（MMDet3 + Diffusers + 固定 N 图 collate）。

建议做法（最小骨架）：
1. 新建独立 repo 或独立 conda 环境（Python 3.10/3.11 更现实）
2. 固定版本矩阵：PyTorch / MMEngine / MMDetection / Diffusers / MMCV
3. 先跑通 toy smoke（本仓库已提供 config 与脚本）
4. 准备 COCO/LVIS/CrowdHuman 数据后，再跑真实训练并产出 ckpt+tsv 验收证据

本仓库已提供一份“可直接复现的 MMDet3 环境 + 训练 smoke（toy COCO）”作为落地入口：

```bash
# 1) 创建独立环境（避免与本仓库 Python3.13 主环境冲突）
conda create -y -n mmdet3 python=3.11

# 2) 安装 PyTorch 2.1 + CUDA 12.1 wheels（docs/check.md:6.1 口径）
conda run -n mmdet3 python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3) 安装 openmim + 兼容版本（torch2.1 需要 numpy<2；并降 opencv 以避免 numpy>=2 的约束）
conda run -n mmdet3 python -m pip install -U openmim \"numpy<2\" \"opencv-python<4.12\"
conda run -n mmdet3 mim install \"mmcv==2.1.0\"
conda run -n mmdet3 mim install \"mmdet==3.3.0\"

# 4) 生成 toy COCO 数据并跑 1-epoch 训练 smoke
conda run -n mmdet3 python mmdet_diffusers/tools/make_toy_coco.py --out datasets/toy_coco
conda run -n mmdet3 mim train mmdet mmdet_diffusers/mmdet_configs/toy_faster_rcnn_r50_fpn.py --work-dir /tmp/mmdet_toy --gpus 1 --launcher none -y

# 5) 扩散检测训练 smoke（固定 N proposals 图 + box diffusion + label state + 图自注意力）
conda run -n mmdet3 mim train mmdet mmdet_diffusers/mmdet_configs/toy_graph_diffusion_r50_fpn.py --work-dir /tmp/mmdet_toy_graphdiff --gpus 1 --launcher none -y
```

## 2) 工程形态对齐（本仓库内已补齐）

如果你的真实目标是 `docs/check.md` 的功能点与实验口径，而非严格 MMDet 同栈，则当前仓库已覆盖：
- Diffusers 风格 Pipeline/Scheduler + 固定 N 节点 collate：`mmdet_diffusers/`
- 固定 N proposals（`NUM_PROPOSALS=500`）
- 离散 label diffusion（D3PM）
- 图交互（self-attn + 可学习几何 bias）
- 各向异性噪声
- 能量引导（quality head + Langevin）
- 多 seed 复验落盘（ckpt+tsv）

对照与证据入口：`docs/checkmd_parity.md`、`results_manifest.tsv`
