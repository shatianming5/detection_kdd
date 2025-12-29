# mmdet_diffusers/

本目录用于把 `check.md` 里“MMDetection + Diffusers Pipeline/Scheduler + 固定 N 节点图数据 collate”的工程形态，在**不破坏本仓库 Detectron2/DiffusionDet 主线**的前提下补齐为可复用模块。

要点：

- `HybridDiffusionPipeline`：基于 `diffusers.DiffusionPipeline` 的推理封装（Detectron2/DiffusionDet 适配），内部使用 `diffusers.DDIMScheduler` 管理 box diffusion；label diffusion 用自定义 D3PM(mask/uniform) 调度器。
- `collate_coco_graph`：把检测数据打包为固定 `N` 节点的图（GT 节点 + 噪声补齐 + `unk` 类），对应 `check.md:6.1` “自定义 collate”。
- 适配器：`mmdet_diffusers/adapters/diffusiondet.py` 提供把本仓库的 Detectron2 `DiffusionDet` 模型包装成 Pipeline 的最小入口（推理用）。

限制与说明：

- 本仓库运行环境是 Python 3.13；`mmdet`/`mmengine` 在该版本下通常不可用，因此 **MMDet3 训练需要独立 conda 环境（Python 3.11）**。
- 本仓库同时提供：
  - Diffusers 形态的 Pipeline/Scheduler/Collate（推理侧与口径对齐用）：`mmdet_diffusers/`
  - **MMDet3 可直接训练的插件**（`check.md` 工程栈对齐）：`mmdet_diffusers/mmdet3/`
    - `GraphDiffusionDetector`：固定 N proposals 的扩散检测器（含 `GraphDenoisingNetwork`、可选质量头/能量引导、可选拓扑损失）
    - `coco_graph_collate`：MMEngine `collate_fn`（固定 N 节点图）
    - `MMDet3HybridDiffusionPipeline`：Diffusers 风格 pipeline（支持 `init_state=(boxes,label_state)` 作为 tuple 输入）

## 快速用法（推理：DiffusionDet → HybridDiffusionPipeline）

如果你直接在仓库根目录运行 Python，需要让 `diffusiondet/` 包可 import：

```bash
export PYTHONPATH="$PWD/baselines/DiffusionDet:$PYTHONPATH"
```

```python
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from diffusiondet import add_diffusiondet_config
from mmdet_diffusers import HybridDiffusionPipeline

cfg = get_cfg()
add_diffusiondet_config(cfg)
cfg.merge_from_file("baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml")
cfg.MODEL.WEIGHTS = "baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable.pth"
cfg.freeze()

model = build_model(cfg)
model.eval()
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

pipe = HybridDiffusionPipeline.from_diffusiondet(model)

# batched_inputs 仍使用 detectron2 的输入格式：[{ "image": CHW tensor, "height":..., "width":... }, ...]
out = pipe(batched_inputs=[{"image": img_tensor, "height": H, "width": W}], num_inference_steps=1)
instances = out["instances"]
```

## Collate：固定 N 节点图数据

`collate_coco_graph` 适用于“标准 PyTorch DataLoader + 自定义 Dataset”的 batch 组装：

```python
from torch.utils.data import DataLoader
from mmdet_diffusers import collate_coco_graph

loader = DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_coco_graph(b, num_nodes=500, unk_id=80))
batch = next(iter(loader))
batch["boxes"]   # (B, N, 4) xyxy (abs)
batch["labels"]  # (B, N) long, padding=unk_id
batch["mask"]    # (B, N) bool, True=GT节点
```

## MMDet3 环境与 smoke

严格 MMDet 3.x 同栈训练建议使用独立环境（Python 3.11）：

```bash
conda create -y -n mmdet3 python=3.11
conda run -n mmdet3 python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
conda run -n mmdet3 python -m pip install -U openmim \"numpy<2\" \"opencv-python<4.12\"
conda run -n mmdet3 mim install \"mmcv==2.1.0\"
conda run -n mmdet3 mim install \"mmdet==3.3.0\"
conda run -n mmdet3 python -m pip install diffusers==0.34.0 transformers==4.46.3 accelerate==1.1.1

# 生成 toy COCO 并跑 smoke
conda run -n mmdet3 python mmdet_diffusers/tools/make_toy_coco.py --out datasets/toy_coco
conda run -n mmdet3 mim train mmdet mmdet_diffusers/mmdet_configs/toy_faster_rcnn_r50_fpn.py --work-dir /tmp/mmdet_toy --gpus 1 --launcher none -y

# 扩散检测训练 smoke（固定 N 节点图 collate + box diffusion + label state + 图自注意力）
conda run -n mmdet3 mim train mmdet mmdet_diffusers/mmdet_configs/toy_graph_diffusion_r50_fpn.py --work-dir /tmp/mmdet_toy_graphdiff --gpus 1 --launcher none -y
```

### COCO/LVIS/CrowdHuman 配置模板（仅提供工程对齐，不包含数据）

- COCO（Phase1 baseline：Swin-B + “DiffusionDet 形态”）：`mmdet_diffusers/mmdet_configs/coco_diffusiondet_swinb_baseline.py`
  - 数据根目录：`COCO_ROOT=/path/to/coco`
  - Swin 预训练：`SWIN_BASE_CKPT=...`（默认指向 Swin 官方 release）
- LVIS：`mmdet_diffusers/mmdet_configs/lvis_graph_diffusion_r50_fpn.py`
  - 数据根目录：`LVIS_ROOT=/path/to/lvis_v1`
- CrowdHuman：`mmdet_diffusers/mmdet_configs/crowdhuman_graph_diffusion_r50_fpn.py`
  - 数据根目录：`CROWDHUMAN_ROOT=/path/to/CrowdHuman`

## Diffusers Pipeline import 说明

若你遇到 `diffusers.DiffusionPipeline` import 失败且报错链路指向 `flash_attn`，通常是 `flash_attn` 与当前 `torch` ABI 不匹配。解决方式是卸载该包：

```bash
python -m pip uninstall -y flash-attn flash_attn
```
