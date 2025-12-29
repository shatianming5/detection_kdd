# checkmd_parity.md — 对照 `check.md` 的“差距清单”与补齐路线图

更新：2025-12-21  
仓库：`/home/ubuntu/tiasha/archives`

> 说明  
> - `check.md` 是 proposal 文档抽取文本（更像“研究方案/路线图”，不是精确的工程验收表）。  
> - 本仓库当前主线是 **Detectron2/DiffusionDet**（Repro-10k 数据集）上的 C2O-GND MVP 改造与实验归档；与 `check.md` 里写的 **MMDetection + Diffusers pipeline** 存在“工程栈不一致”。  
> - 本文件做两件事：  
>   1) 把 `check.md` 中可操作的“工程/模块/实验”逐条落到本仓库的代码与证据（ckpt+tsv）。  
>   2) 对缺口给出“在当前仓库内可补齐”的执行路径；对需要外部数据/算力/新工程栈的条目明确标注。

---

## 0. 口径（本仓库的“可复验证据”）

对齐原则：**任何结论必须能仅靠 `ckpt + tsv` 复验**（不依赖 `/dev/shm` 临时目录）。

- ckpt：`baselines/checkpoints/*.pth`（同名 `.sha256`）
- tsv：仓库根目录 `*.tsv`（优先由 `scripts/eval_multiseed.py` 生成，含 ckpt sha256）
- 总索引：`results_manifest.tsv`
- 冻结交付（step1, 5-seed）：`deliverables/step1_5seed_geofeat_mix/manifest.tsv`

---

## 1. check.md:5.* 训练目标与损失（核心缺口集中在这里）

### 1.1 5.1 IoU-Aware Varifocal Regression（`check.md:202`）

- check.md 要求：回归损失引入类似 VFL 的“质量/IoU 加权”（配合 GIoU）。
- 本仓库现状：
  - ✅ 盒回归：L1 + GIoU（`baselines/DiffusionDet/diffusiondet/loss.py`）
  - ✅ “IoU-aware/VFL 风格加权”已实现（默认关闭）：`MODEL.DiffusionDet.BOX_LOSS_IOU_WEIGHT_POWER`（`baselines/DiffusionDet/diffusiondet/config.py` / `baselines/DiffusionDet/diffusiondet/loss.py`）
- 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth` + `checkmd_mvp_seed0_iter200_step1_results.tsv`

### 1.2 5.2 Hybrid Quality Focal Loss（QFL）（`check.md:212`）

- check.md 要求：分类分数反映定位质量，引入 QFL（软标签 y∈[0,1]）。
- 本仓库现状：
  - ✅ 分类：sigmoid focal（默认；`CLS_LOSS_TYPE=focal`）
  - ✅ QFL 已实现（默认关闭）：`MODEL.DiffusionDet.CLS_LOSS_TYPE=qfl`、`MODEL.DiffusionDet.QFL_BETA`（`baselines/DiffusionDet/diffusiondet/config.py` / `baselines/DiffusionDet/diffusiondet/loss.py`）
- 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth` + `checkmd_mvp_seed0_iter200_step1_results.tsv`

### 1.3 5.3 Graph Consistency Loss（`check.md:224`）

- check.md 要求：监督 Graph Transformer 注意力邻接与 GT 邻接一致。
- 本仓库现状（✅ 已实现，但缺“验收证据”）：
  - ✅ 可捕获最后一层 proposal self-attn（`baselines/DiffusionDet/diffusiondet/head.py:901`）
  - ✅ `loss_graph` 实现（`baselines/DiffusionDet/diffusiondet/loss.py:256`）
  - ✅ 配置开关：`MODEL.DiffusionDet.GRAPH_TOPO_LOSS_WEIGHT`（`baselines/DiffusionDet/diffusiondet/config.py:196`）
- 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth` + `checkmd_mvp_seed0_iter200_step1_results.tsv`（配置 `GRAPH_TOPO_LOSS_WEIGHT=1.0`）

---

## 2. check.md:6.* 工程阶段（对齐到“同等能力”而非同栈）

### 2.1 6.1 Skeleton & Baseline（`check.md:242`）

- check.md 要求（MMDet+Diffusers）：`HybridDiffusionPipeline`、自定义 collate、COCO val mAP≈46、Swin-B baseline。
- 本仓库现状：
  - ✅ Baseline 复现链路：`run_baselines.sh` + `baselines/DiffusionDet/train_net.py`（见 `plan2.md` A 验收）
  - ✅ Swin-B baseline（可选）：`plan.md` 已列（本仓库存在 Swin 配置）
  - ✅ 固定 proposals（N=500）：`MODEL.DiffusionDet.NUM_PROPOSALS=500`（Repro-10k 配方）
  - ✅ torch.compile 开关（实验性）：`SOLVER.TORCH_COMPILE*`（`baselines/DiffusionDet/diffusiondet/config.py` + `baselines/DiffusionDet/train_net.py`）
  - ✅ Detectron2 builtin 数据集注册（COCO/LVIS/VOC）：`baselines/DiffusionDet/train_net.py` 现在会注册 `coco_2017_*` / `lvis_v1_*` / `voc_*`（依赖 `$DETECTRON2_DATASETS`）
  - ✅ Diffusers Pipeline/Scheduler 形态 + 固定 N 节点 collate：已补齐在 `mmdet_diffusers/`（并提供 Detectron2/DiffusionDet 适配入口）
- 可补齐/不可补齐：
  - 可补齐（在本仓库内）：完善“复现/运行手册”和 smoke tests（已基本具备）。
  - 🟡 MMDet 3.x 同栈训练：仍建议单独新工程/环境（Python 3.13 下 mmdet/mmengine 依赖通常不可用；见 `checkmd_mmdet_diffusers_stub.md`）

### 2.2 6.2 Hybrid & Graph Modules（`check.md:253`）

#### 6.2.1 D3PM Scheduler（`check.md:257`）
- ✅ 已实现并用于主线：`LABEL_D3PM`（`baselines/DiffusionDet/diffusiondet/detector.py:290`）
- ✅ 有证据：`d3pm_qhead_step1_results.tsv` / `d3pm_qhead_warmstart_step1_results.tsv` 等（见 `plan2.md`、`results_manifest.tsv`）

#### 6.2.2 Graph Transformer + Geometric Bias（`check.md:258`）
- check.md 要求：带“几何偏置网络 g_phi”的 attention（可 FlashAttention-2 加速）。
- 本仓库现状：
  - ✅ proposal self-attn（图交互骨架）：`baselines/DiffusionDet/diffusiondet/head.py:894`
  - ✅ 几何信息注入（learnable）：`GEO_FEAT`（主线已验证并用于交付）
  - ✅ 几何 attention bias（非学习）：`GEO_BIAS`（distance/iou；含可学习 scale，但没有 learnable g_phi）
  - ✅ learnable “几何偏置网络 g_phi”（MVP 版）：`GEO_BIAS_TYPE=mlp`（`baselines/DiffusionDet/diffusiondet/head.py`）
  - ✅ FlashAttention-2（via PyTorch SDPA）已显式集成：`SELF_ATTN_IMPL=sdpa`（`baselines/DiffusionDet/diffusiondet/util/sdpa_attention.py`）
    - 模板 config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sdpa.yaml`
    - Benchmark：`scripts/benchmark_attention.py`
    - 注意1：当 `GRAPH_TOPO_LOSS_WEIGHT>0` 需要返回 attention weights 时，会自动回退到 `nn.MultiheadAttention`
    - 注意2：PyTorch 的 FlashAttention kernel 不支持通用的 additive `attn_mask`；开启 `GEO_BIAS`（会生成 float bias mask）时通常会回退到 math kernel（仍可正确但不等价于“FlashAttention 加速”）
- 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth` + `checkmd_mvp_seed0_iter200_step1_results.tsv`（配置 `GEO_BIAS=True, GEO_BIAS_TYPE=mlp`）

#### 6.2.3 Anisotropic Noise（`check.md:259`）
- ✅ 代码已实现（可开关）：`ANISO_NOISE*`（`baselines/DiffusionDet/diffusiondet/detector.py:169` + `q_sample/ddim_sample`）
- ✅ 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth` + `checkmd_mvp_seed0_iter200_step1_results.tsv`（配置 `ANISO_NOISE=True`）

#### 6.2 验证（VOC 小数据集收敛检查）
- check.md 建议：在小规模数据集（如 VOC）上验证混合损失是否可收敛。
- 本仓库现状：
  - ✅ VOC 数据集名注册：`baselines/DiffusionDet/train_net.py`（detectron2 builtin）
  - ✅ VOC D3PM 配方模板 config：`baselines/DiffusionDet/configs/diffdet.voc2007.res50_d3pm_mask_dist_qhead.yaml`
  - ❌ 缺“跑过/收敛曲线/结论”的 ckpt+tsv 证据（需要 VOC 数据与训练跑一次）

### 2.3 6.3 Guidance & Distillation（`check.md:264`）

- ✅ Quality Head：已实现+有证据（`guidance_sweep_results_qhead_seed42*.tsv`，见 `plan2.md`）
- ✅ Energy Sampling（Langevin guidance）：已实现（`baselines/DiffusionDet/diffusiondet/detector.py:511`）且有 sweep 表
- 🟡 Consistency Distillation（teacher->student 同噪声/同 t 的 distill）：
  - ✅ 入口与 loss 已实现（`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_consistency_distill.yaml`；`baselines/DiffusionDet/diffusiondet/detector.py:782`）
  - ✅ 证据（ckpt+tsv，可复验）：`baselines/checkpoints/checkmd_consistency_smoke_seed0_iter50.pth` + `checkmd_consistency_smoke_seed0_iter50_step1_results.tsv`（仅 smoke 验收链路）
- ✅ Sampler distill（teacher step20→student step1 真加速）：已实现+已做成 5-seed 冻结交付（见 `deliverables/...`）
- 🟡 check.md 里“逐步把采样步数 1000→4→2”的 *学生多步采样蒸馏*：
  - ✅ 已补齐“多步学生”的工程能力：`SAMPLER_DISTILL_STUDENT_SAMPLE_STEP>1` 会在训练时实际运行学生 sampler（`baselines/DiffusionDet/diffusiondet/detector.py`）
  - ✅ 已提供 progressive distill 模板 config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_{20to4,4to2,2to1}.yaml`
  - ✅ 已提供一键模板脚本（会产出 3 个 ckpt + 3 个 tsv，注意磁盘）：`scripts/run_progressive_sampler_distill.sh`
  - ❌ 仍未补齐的部分：check.md 口径里的 teacher=1000steps/COCO 全量/大算力设置（数据与资源缺口，不建议在本仓库强行复刻）

### 2.4 6.4 Scale-up（`check.md:280`）

- check.md 要求：8×A100、AMP/BF16、EMA、Objects365→COCO/LVIS。
- 本仓库现状：
  - ✅ EMA 基础设施已接入 Detectron2 流程（`baselines/DiffusionDet/train_net.py:357`）
  - 🟡 AMP（Detectron2 支持，但本仓库没有按该口径跑过）
  - ❌ 外部大规模数据/多卡训练与证据：缺失（资源/数据缺口）
- 已补齐的“模板”（不代表已跑过/已验收）：
  - AMP+EMA 开关示例 config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_amp_ema.yaml`
  - COCO/LVIS/CrowdHuman/Objects365 AMP+EMA 模板：
    - `baselines/DiffusionDet/configs/diffdet.coco.res50_d3pm_mask_dist_qhead_amp_ema.yaml`
    - `baselines/DiffusionDet/configs/diffdet.lvis.res50_d3pm_mask_dist_qhead_amp_ema.yaml`
    - `baselines/DiffusionDet/configs/diffdet.crowdhuman.res50_d3pm_mask_dist_qhead_amp_ema.yaml`
    - `baselines/DiffusionDet/configs/diffdet.objects365.res50_d3pm_mask_dist_qhead_amp_ema.yaml`
  - 训练启动脚本模板：`scripts/run_train_detectron2.sh`
  - Objects365 预训练模板 config（需自行注册数据集名）：`baselines/DiffusionDet/configs/diffdet.objects365.res50_d3pm_mask_dist_qhead.yaml`
  - 额外 COCO-style 数据集注册入口：`baselines/DiffusionDet/train_net.py`（环境变量 `EXTRA_COCO_DATASETS=...`）
- 结论：这部分无法在“当前数据与单卡实验归档仓库”内完全补齐；只能补“开关与脚本模板”，最终结论仍需要外部数据与算力验收。

---

## 3. check.md:7.* 实验设计（本仓库当前覆盖与缺口）

### 3.1 7.1 指标（`check.md:294`）
- ✅ 标准检测指标（AP/AP50/AP75/...）：已覆盖（COCOeval）
- ✅ 稳定性（多 eval_seed 复验 mean/std）：已作为主口径（`scripts/eval_multiseed.py` + `final_step1_5seed_geofeat_mix.tsv`）
- ✅ Inference FPS（实现“可落盘的计时口径”）：`scripts/eval_multiseed.py` 现在会在 TSV 里写 `inference_s_per_img` / `inference_fps`（由 detectron2 eval 日志里的 `s / iter per device` + `SOLVER.IMS_PER_BATCH` 推导；硬件相关）
  - 证据（TSV 示例）：`ablation_graph_topo_full_seed0_stable_step1_results.tsv`

### 3.2 7.3 消融（`check.md:330`）

1) 图结构影响（独立节点 / Full / Sparse kNN）  
- ✅ Full Graph（默认 self-attn 全连接）：存在  
- ✅ Sparse kNN（k-NN 动态图）：`GEO_BIAS_TOPK>0` 会把 attention mask 稀疏化；已补齐示例 config 与 TSV
  - config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_graph_topo_sparse_knn_topk50.yaml`
  - TSV（eval-only 示例）：`ablation_graph_topo_sparse_knn_topk50_seed0_stable_step1_results.tsv`
- ✅ 独立节点（无交互）：已补齐开关 `DISABLE_SELF_ATTN`（`baselines/DiffusionDet/diffusiondet/config.py` / `baselines/DiffusionDet/diffusiondet/head.py`）
  - config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_graph_topo_none.yaml`
  - TSV（eval-only 示例）：`ablation_graph_topo_none_seed0_stable_step1_results.tsv`

2) Hybrid vs Gaussian（D3PM vs continuous label）  
- ✅ 部分覆盖：已有 `LABEL_D3PM` vs baseline 对照（Repro-10k）  
- 🟡 LVIS/COCO 的 D3PM 配方模板：已补齐配置文件，但缺数据与验收证据  
  - COCO 模板：`baselines/DiffusionDet/configs/diffdet.coco.res50_d3pm_mask_dist_qhead.yaml`  
  - LVIS 模板：`baselines/DiffusionDet/configs/diffdet.lvis.res50_d3pm_mask_dist_qhead.yaml`
  - CrowdHuman 模板（需 `EXTRA_COCO_DATASETS` 注册）：`baselines/DiffusionDet/configs/diffdet.crowdhuman.res50_d3pm_mask_dist_qhead.yaml`
  - Objects365 模板（需 `EXTRA_COCO_DATASETS` 注册）：`baselines/DiffusionDet/configs/diffdet.objects365.res50_d3pm_mask_dist_qhead.yaml`

3) Energy guidance 强度 sweep  
- ✅ 已覆盖（见 `guidance_sweep_results_qhead_seed42*.tsv`）

4) Anisotropic noise  
- ✅ 代码在  
- ✅ 证据链（见 2.2.3）

---

## 4. “补齐差缺”执行顺序（在本仓库内可落地）

优先级按：**代码缺口（影响面大）→ 功能验收证据（已有代码但没跑）→ 大规模/外部资源项（仅留模板）**。

1) 补齐 5.1/5.2：新增 QFL 与 IoU 加权回归（默认关闭）  
2) 补齐 6.2.2：新增 learnable 几何偏置网络（`GEO_BIAS_TYPE=mlp`，默认关闭）  
2.5) 补齐 6.1：Diffusers Pipeline/Scheduler + 固定 N 节点 collate（✅ `mmdet_diffusers/`）  
3) 补齐 5.3：跑一次 `GRAPH_TOPO_LOSS_WEIGHT>0` smoke 并落盘（✅ 已用 checkmd_mvp smoke 覆盖）  
4) 补齐 6.2.3 / 7.3-4：跑一次 `ANISO_NOISE=True` smoke 并落盘（✅ 已用 checkmd_mvp smoke 覆盖）  
5) 补齐 6.3 consistency distill：跑一次 smoke 并落盘（✅ 已落盘）  
6) 7.1 FPS：扩展 `scripts/eval_multiseed.py` 抽取推理耗时并写入 TSV（✅ 已补齐）  
7) 7.3-1：图结构“无交互/稀疏/全连接”消融（✅ 已补齐开关 + 示例 config + TSV）  
7.5) FlashAttention-2 显式集成 + benchmark（✅ `SELF_ATTN_IMPL=sdpa` + `scripts/benchmark_attention.py`）  
7.6) check.md:7.2 主结果表生成器（✅ `scripts/build_checkmd_main_results.py` + `deliverables/checkmd_main_results_spec.example.json`）  
8) 6.4/7.* 外部数据项：写模板与 TODO（不在本仓库直接完成）

---

## 5. 仍未补齐（需要外部数据/算力/新工程栈）

以下条目“在本仓库内只能补模板/入口”，无法补出 **ckpt+tsv** 的验收证据：

1) **MMDetection + Diffusers 同栈训练**（严格 MMDet 3.x pipeline）  
   - 状态：🟡（已补齐：1) Diffusers 形态的 Pipeline/Scheduler/Collate：`mmdet_diffusers/`；2) 可运行的 MMDet3 环境与训练 smoke：`mmdet_diffusers/mmdet_configs/toy_faster_rcnn_r50_fpn.py`；但仍缺“在 MMDet 内实现完整 C2O-GND / detection diffusion 的训练工程”）  
   - 说明/命令：见 `checkmd_mmdet_diffusers_stub.md`

2) **COCO/LVIS/CrowdHuman/Objects365 的真实训练与结论**  
   - 状态：🟡（已补齐 config + 数据集注册入口，但缺数据与训练验收）  
   - 需要：准备数据目录并设置 `$DETECTRON2_DATASETS` / `EXTRA_COCO_DATASETS`，然后跑训练并用 `scripts/eval_multiseed.py` 落盘

3) **Scale-up（8×A100 / BF16 / 大规模预训→微调）的最终验收**  
   - 状态：🟡（已补 AMP+EMA 开关模板，但缺多卡与数据）  
   - 需要：多卡资源 + 数据 + 长跑训练

5) **check.md:7.2 主结果表（COCO 上的 AP/AP75/FPS 对比表）**  
   - 状态：🟡（已补齐“表格生成器”，但缺 COCO/LVIS 等真实训练与统一硬件口径）  
   - 需要：按上面数据/算力补齐后再产出对比表
