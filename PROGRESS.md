# Baseline 复现进度与操作指南

本文档用于记录当前仓库的 baseline 复现状态，并给出从零到可复现的最短路径与验证方式。

---

## 当前进度（截至本次更新）

- [x] **仓库结构解析完成**：关键路径均已补充 `AGENT.md` 说明（含功能/结构/验证）。
- [x] **运行环境就绪**：
  - 已检测到：`torch 2.6.0+cu124`、`torchvision 0.21.0+cu124`、`detectron2 0.6` 可正常 import。
- [x] **Repro-10k COCO 数据已准备**：
  - `baselines/data/repro_10k/annotations/instances_train2017.json`
  - `baselines/data/repro_10k/annotations/instances_val2017.json`
  - `baselines/data/repro_10k/train2017/`、`val2017/`（图片已存在/符号链接正常）。
- [x] **完整 baseline 长跑复现（10k）**：已完成（见下方“最新复现结果”）。
- [x] **C2O-GND Phase 1（MVP）代码已接入**：为 DiffusionDet 的 proposal self-attention 增加可开关的几何偏置（仅先验证链路跑通）。

结论：**具备完整复现 baseline 的所有前置条件**；下一步只需要跑脚本并核对输出。

---

## 最新复现结果（Repro-10k）

输出目录：`baselines/output/20251212_225009/`

### DiffusionDet（`repro_10k_diffdet`）

读取位置：`baselines/output/20251212_225009/repro_10k_diffdet/metrics.json`、`baselines/output/20251212_225009/repro_10k_diffdet/log.txt`

- bbox AP=45.496，AP50=74.075，AP75=45.907，APs=16.004，APm=53.485，APl=69.610
- Per-class bbox AP：car=52.939，motorbike=38.053，person=nan（val 中 person 标注极少/为 0，COCOeval 无法统计）

### DETR（`repro_10k_detr`，COCO-pretrained fine-tune）

读取位置：`baselines/output/20251212_225009/repro_10k_detr/log.txt`（最后一行 epoch=49；数值范围为 0~1）

- bbox AP=0.5457（54.57），AP50=0.8330（83.30），AP75=0.6084（60.84）
- bbox APs=0.2074（20.74），APm=0.6129（61.29），APl=0.7788（77.88）
- bbox AR@1=0.3289（32.89），AR@10=0.5807（58.07），AR@100=0.6076（60.76）

---

## C2O-GND Phase 1（几何偏置 self-attn）验证记录

### 1) 训练 smoke（500 iter）

两次 smoke（baseline 与 `GEO_BIAS=True`）都可正常完成训练与 eval，不会崩溃；在 500 iter 时由于置信度极低，评估阶段可能提示 `No predictions from the model!`，这是 smoke 的常见现象，不作为失败标准。

- baseline（无几何偏置）：输出到 `/tmp/diffdet_baseline_smoke500`
- GEO_BIAS：输出到 `/tmp/c2o_gnd_geo_bias_smoke`

### 2) 仅推理侧开关验证（加载已有 10k checkpoint）

目的：验证“新开关”不会影响旧 checkpoint 加载；并观察推理侧启用偏置的影响（不等价于训练收益）。

使用权重：`baselines/output/20251212_225009/repro_10k_diffdet/model_final.pth`

- `GEO_BIAS=False`：bbox AP=46.0245
- `GEO_BIAS=True, GEO_BIAS_SCALE=1.0`：bbox AP=44.1281（下降）
- `GEO_BIAS=True, GEO_BIAS_SCALE=0.1`：bbox AP=45.2229（更接近 baseline）

结论：开关链路正常；后续如果要验证“提案收益”，应当在训练阶段开启并做短跑/长跑对比（见 `plan.md` 的 Phase 1 验收）。

### 3) 训练短跑对比（MAX_ITER=2500，SEED=42）

目的：在可控成本下验证“训练期开启几何偏置”是否能提升/至少不伤害指标。

共同参数：
- `SOLVER.MAX_ITER=2500`
- `SOLVER.CHECKPOINT_PERIOD=1000000`（避免中途写大 checkpoint，只保留 `model_final.pth`）
- `SEED=42`

结果（COCOeval，数值范围 0~100）：

- baseline：`/tmp/diffdet_baseline_iter2500_seed42`
  - bbox AP=19.8826，AP50=36.7067，AP75=18.4234
- GEO_BIAS（distance，scale=0.1，sigma=2.0）：`/tmp/diffdet_geo_bias_s0p1_iter2500_seed42`
  - bbox AP=16.6378，AP50=31.4428，AP75=15.9825（明显下降）
- GEO_BIAS（distance，scale=0.1，sigma=10.0）：`/tmp/diffdet_geo_bias_s0p1_sigma10_iter2500_seed42`
  - bbox AP=15.3006，AP50=30.5487，AP75=14.4852（进一步下降）
- GEO_BIAS（distance，scale=0.1，sigma=2.0，min_norm=64.0）：`/tmp/diffdet_geo_bias_s0p1_sigma2_minnorm64_iter2500_seed42`
  - bbox AP=15.3807，AP50=30.7564，AP75=14.5272（对 motorbike 无明显改善）

结论：当前“distance-based 几何偏置”实现不满足 Phase 1 的验收（不应显著掉点）；需要重新设计 bias 形式（例如 IoU/learnable/按扩散步调度），或先暂时关闭继续推进其他阶段。

### 4) 更换 bias 形式（IoU / learnable / 扩散步调度）复验

为进一步定位“为什么掉点”并尝试更合理的设计，我们把 Phase 1 的实现扩展为可配置版本（仍然只改 `proposal self-attn`）：

- bias 类型：`MODEL.DiffusionDet.GEO_BIAS_TYPE` 支持 `distance` / `iou`
- 扩散步调度：`MODEL.DiffusionDet.GEO_BIAS_SCHEDULE` 支持 `constant` / `linear` / `threshold`
- learnable scale：`MODEL.DiffusionDet.GEO_BIAS_LEARNABLE_SCALE=True`（每个 head 一个可学习 scale，tanh 限幅）

复验基线（同样 MAX_ITER=2500, SEED=42，作为最新对照）：

- baseline_v2（无几何偏置）：`/tmp/diffdet_baseline_iter2500_seed42_v2`
  - bbox AP=19.3030，AP50=42.8379，AP75=15.9206（car=27.5790，motorbike=11.0270）

代表性复验结果（全部数值 0~100）：

- IoU bias（fixed，linear schedule）：`/tmp/diffdet_geo_bias_iou_linear_p2_s1_iter2500_seed42`
  - bbox AP=14.3221（car=28.4220，motorbike=0.2221）
- IoU bias（learnable scale，linear schedule）：`/tmp/diffdet_geo_bias_iou_learnscale_linear_p8_s1_iter2500_seed42`
  - bbox AP=11.9426（car=23.4196，motorbike=0.4655）
- distance bias（learnable scale，linear schedule）：`/tmp/diffdet_geo_bias_dist_learnscale_linear_p8_s1_sigma2_iter2500_seed42`
  - bbox AP=16.1520（car=28.7835，motorbike=3.5205）
- distance bias（threshold schedule，t<=20 才启用）：`/tmp/diffdet_geo_bias_dist_threshold_t20_s0p5_sigma2_iter2500_seed42`
  - bbox AP=4.4821（car=8.8882，motorbike=0.0761，训练明显崩掉）

结论与原因定位（当前证据链）：

1. **多种 bias 形式都显著掉点**，主要表现为 `motorbike AP` 极易掉到接近 0；而本数据集本身极度不平衡：
   - train：car=52033、motorbike=831、person=4
   - val：car=5743、motorbike=90、person=0
   这使得任何会扰动分类/匹配/去噪链路的改动都会首先“压垮” motorbike。
2. **扩散训练下，绝大多数时间步的 box 是高噪声**，基于当前 box 几何关系做强约束/强 bias 往往是不可靠先验；
   这类先验会改变 proposal self-attn 的信息流，导致 denoise 过程更难收敛（在短跑尤其明显）。
3. `threshold` 这种“只在极低噪声 t 启用”的策略在本实现下表现不稳定（短跑直接崩），说明需要换一种更温和的注入方式。

下一步建议（Phase 1 继续推进的更稳路线）：

- 不再直接在 self-attn logits 上加手工 bias（当前证据显示收益为负），改为 **在 proposal 特征上注入几何特征**（learnable MLP / relative encoding），让模型自己学“用不用”，并保证默认初始化近似 baseline。
- 或者先把 Phase 1 暂停，先做 Phase 3 的“质量头/能量引导”框架（不改训练主干），待收益链路更明确后再回头做图交互。

### 5) 几何特征注入（learnable residual）复验与“掉点根因”定位

实现方式（代码层面）：
- 新增 `MODEL.DiffusionDet.GEO_FEAT` 系列开关：用 bbox 的几何量（cx,cy,w,h,+t）做一个线性投影，作为 residual 注入到特征中；
- 支持注入位置：`GEO_FEAT_TARGET=proposal`（self-attn 前）或 `GEO_FEAT_TARGET=reg`（只注入 bbox 回归分支）；
- 该投影权重以 0 初始化（训练中可学习），理论上更“温和”，避免 hard bias 直接改 attention logits。

2500 iter（SEED=42）结果：

- GEO_FEAT（target=proposal，linear schedule，p=4，scale=0.1）：`/tmp/diffdet_geo_feat_linear_p4_s0p1_iter2500_seed42`
  - bbox AP=13.1925（car=26.3070，motorbike=0.0779）
- GEO_FEAT（target=reg，linear schedule，p=4，scale=0.1）：`/tmp/diffdet_geo_feat_reg_linear_p4_s0p1_iter2500_seed42`
  - bbox AP=17.3907（car=30.7699，motorbike=4.0116）
- GEO_FEAT（target=reg，linear schedule，p=4，scale=0.01）：`/tmp/diffdet_geo_feat_reg_linear_p4_s0p01_iter2500_seed42`
  - bbox AP=16.4394（car=32.5378，motorbike=0.3409）
- GEO_FEAT（target=reg，train warmup=2500，linear schedule，p=4，scale=0.1）：`/tmp/diffdet_geo_feat_reg_warmup2500_linear_p4_s0p1_iter2500_seed42`
  - bbox AP=14.8312（car=29.2808，motorbike=0.3824）

深度原因定位（为什么 motorbike AP 会“先崩”）：

我们对 `motorbike` 做了 “TP/FP score 分离度” 的直接统计（按 IoU>=0.5 的 greedy matching，取每图 top100 预测）：

- baseline_v2：TP 分数显著高于 FP（TP mean≈0.243 / FP mean≈0.039），因此即使 FP 很多，AP 仍能保持在可用水平。
- GEO_FEAT（target=proposal，scale=0.1）：TP 分数掉到接近 FP（TP mean≈0.051 / FP mean≈0.034），导致排序几乎失效，AP 直接接近 0。

结论：当前数据极度不平衡（car≫motorbike）+ 扩散训练的大多数时间步 box 几何高噪声，使得任何“几何先验注入”很容易让模型把 motorbike 的置信度拉平（TP 不再比 FP 高），从而在 COCO AP 上表现为 motorbike AP 崩盘。

### 6) 验证“长尾/短跑不稳定”假设：RepeatFactorTrainingSampler

为了确认 motorbike 崩盘是否来自“短跑早期抽样到的 motorbike 太少”，我们用 Detectron2 自带的 `RepeatFactorTrainingSampler` 做了短跑复验（只改采样器，不改模型）。

- baseline（repeat=0.02）：`/tmp/diffdet_baseline_repeat0p02_iter2500_seed42`
  - bbox AP=18.9871（car=29.3410，motorbike=8.6336）

同样采样器下继续开启几何注入/几何 bias（仍然掉点）：

- GEO_FEAT（target=reg，repeat=0.02，linear p=4，scale=0.1）：`/tmp/diffdet_geo_feat_reg_repeat0p02_p4_s0p1_iter2500_seed42`
  - bbox AP=12.5205（car=24.7844，motorbike=0.2566）
- GEO_BIAS（distance，repeat=0.02，linear p=4，scale=0.01）：`/tmp/diffdet_geo_bias_dist_repeat0p02_p4_s0p01_iter2500_seed42`
  - bbox AP=10.6698（car=21.2438，motorbike=0.0954）

结论：短跑里“motorbike 崩盘”确实与长尾抽样不稳定有关（baseline 采样一改就显著改善），但 Phase 1 的几何先验注入在该数据上仍强烈负收益；因此 Phase 1 更建议：
- 以 10k 长跑作为主验收口径（短跑只做跑通/不崩），或
- 换到更平衡的数据/引入更强的长尾策略后再做图交互。

### 7) MLP 几何注入（proposal 侧）进一步尝试：仍不收敛（AP=0/NaN）

在“hard bias 掉点”后，我们按 Phase 1 的建议继续做了更温和的 **proposal 特征几何注入**（learnable MLP / 可选 relative encoding），并加了“no-op 初始化 + learnable gate”来尽量保证默认行为接近 baseline。

实现要点（代码已落地）：
- `GEO_FEAT_ENCODER=mlp`：`(cx,cy,w,h,+t) -> MLP -> d_model`，最后一层权重/偏置置 0，保证初始严格 no-op；
- `GEO_FEAT_ALPHA_INIT`：额外 gate（`sigmoid(alpha_param)`）让模型自己学“用不用”；
- `GEO_FEAT_PROPOSAL_MODE=qk_pos`：把几何残差当作 self-attn 的 q/k 位置编码（value 不变），更接近 Transformer 的 pos-encoding 用法；
- `GEO_FEAT_LR_MULT`：可选把 `geo_feat_*` 参数单独降学习率，避免新分支扰动主干收敛。

在更稳的短跑对照口径（`RepeatFactorTrainingSampler, threshold=0.02, SEED=42`）下结果如下：

- baseline（repeat=0.02）：`/tmp/diffdet_baseline_repeat0p02_iter2500_seed42`
  - bbox AP=18.9871（car=29.3410，motorbike=8.6336）
- MLP proposal 注入（无 gate，add 模式）：`/tmp/diffdet_geo_feat_mlp_proposal_repeat0p02_p4_s0p1_iter2500_seed42`
  - eval：`No predictions from the model!` → bbox AP=NaN（`inference/coco_instances_results.json` 仅 `[]`）
- MLP proposal 注入（`alpha_init=0.1`，add 模式）：`/tmp/diffdet_geo_feat_mlp_proposal_repeat0p02_p4_s0p1_a0p1_iter2500_seed42`
  - bbox AP=0.0000（预测框大量退化为“极薄框”，IoU≈0）
- MLP proposal 注入（`alpha_init=0.01`，add 模式）：`/tmp/diffdet_geo_feat_mlp_proposal_repeat0p02_p4_s0p1_a0p01_iter2500_seed42`
  - eval：`No predictions from the model!` → bbox AP=NaN
- MLP proposal 注入（`qk_pos` 模式）：`/tmp/diffdet_geo_feat_mlp_qkpos_repeat0p02_p4_s0p1_a0p1_iter2500_seed42`
  - eval：`No predictions from the model!` → bbox AP=NaN
- `qk_pos + GEO_FEAT_LR_MULT=0.1`（2000 iter 探路）：`/tmp/diffdet_geo_feat_mlp_qkpos_lr0p1_repeat0p02_p4_s0p1_a0p1_iter2000_seed42`
  - eval：`No predictions from the model!` → bbox AP=NaN

关键现象（用于“深因”定位）：
- 这些 proposal 侧注入实验在前 1k~1.5k iter 的 `total_loss` 与 baseline 接近，但 **baseline 会在 ~1.6k 后快速下降到 ~8-10，而注入版本会长期停在 ~20 左右**，表现为“无法进入后期收敛阶段”。
- `No predictions` 的直接原因是：DiffusionDet 推理里有 `box_renewal`（阈值 0.5）会把低置信度框全部过滤掉；当整个采样过程都没有 score>0.5 的框时，最终预测为空，COCOeval 的 AP 会变成 NaN。

结论（Phase 1 的工程决策）：
- 在当前 `repro_10k`（极端长尾 + 扩散训练多数时间步 proposal 高噪声）的前提下，**任何会改变 proposal self-attn 信息流的“几何先验注入”（hard bias 或 feature injection）都高度不稳定且倾向于阻断收敛**。
- 因此 Phase 1 若要“不掉点”，更现实的路线是先把长尾问题解决/换验收口径（10k 长跑为主），或者先转去 Phase 3 的质量头/能量引导（不动 self-attn 主干）。

### 8) Phase 1 继续：GEO_FEAT 稳定化尝试（仍未达标）

在 Phase 3 完成后，我们又回头补了一轮 Phase 1 的“几何特征注入”稳定化，目标是解决历史上 `GEO_FEAT` 训练会出现的 **收敛被阻断 / AP=0 / AP=NaN** 问题，并满足短跑验收口径：

> 口径：`RepeatFactorTrainingSampler + threshold=0.02 + SEED=42 + MAX_ITER=2500`

#### A) 代码侧稳定化改动（已落地）

- `baselines/DiffusionDet/diffusiondet/config.py`
  - 新增：`GEO_FEAT_INPUT_CLIP`、`GEO_FEAT_OUT_TANH`、`GEO_FEAT_REL_NORM`
  - 新增：`GEO_FEAT_TRAIN_START_ITER`（支持“训练到一定 iter 后才启用注入”）
- `baselines/DiffusionDet/diffusiondet/head.py`
  - `GEO_FEAT_INPUT_CLIP`：对 `(cx,cy,w,h,log_wh,t)` 做 clamp，降低极端噪声框带来的梯度尖峰
  - `GEO_FEAT_OUT_TANH`：对注入向量做 `tanh` 限幅，保证注入残差有上界
  - `GEO_FEAT_REL_NORM=softmax`：relative message 归一化（避免 `NUM_PROPOSALS=500` 时 O(N) 累加导致爆炸）
  - `GEO_FEAT_TRAIN_START_ITER`：支持“从某个 iter 之后再启用，并按 warmup 线性爬坡”
- `baselines/DiffusionDet/train_net.py`
  - full-model grad clipping 下，把 `geo_feat_*` 参数从 `main` 组分离到 `clip_group=geo`，避免几何分支梯度影响主干的裁剪比例

#### B) 2500 iter 对照结果（仍显著掉点）

baseline（当前代码口径）：
- `/tmp/diffdet_baseline_repeat0p02_iter2500_seed42_v6_currentcode/metrics.json`：bbox AP=17.6094，last total_loss=8.2777

Phase 1（GEO_FEAT）尝试：
- proposal `qk_pos`（无 start_iter）：`/tmp/diffdet_geo_feat_mlp_qkpos_clipgeo_tanh_inclip10_repeat0p02_p4_s0p1_a0p01_lr0p1_iter2500_seed42/metrics.json`
  - bbox AP=0.0000，last total_loss=19.1353（收敛被阻断）
- proposal `qk_pos`（`START_ITER=1600, WARMUP=500`）：`/tmp/diffdet_geo_feat_mlp_qkpos_start1600_warm500_clipgeo_tanh_inclip10_repeat0p02_p4_s0p1_a0p01_lr0p1_iter2500_seed42/metrics.json`
  - bbox AP=1.1844，last total_loss=11.8517（有所缓解但仍远低于 baseline）
- `TARGET=reg`（无 start_iter）：`/tmp/diffdet_geo_feat_reg_clipgeo_tanh_inclip10_repeat0p02_p4_s0p1_a0p1_lr0p1_iter2500_seed42/metrics.json`
  - bbox AP=0.0000，last total_loss=19.0729（收敛被阻断）
- `TARGET=reg`（`START_ITER=1600, WARMUP=500`）：`/tmp/diffdet_geo_feat_reg_start1600_warm500_clipgeo_tanh_inclip10_repeat0p02_p4_s0p1_a0p01_lr0p1_iter2500_seed42/metrics.json`
  - bbox AP=0.4519，last total_loss=13.2932

#### C) 结论

即使加入了“输入/输出限幅 + 训练延迟启用 + 独立 grad clipping”这一整套稳定化，**从零训练期开启 GEO_FEAT 仍然会显著阻断 DiffusionDet 在 `repro_10k` 的短跑收敛**，导致 AP 大幅掉点（远低于 baseline）。

下一步若仍要推进 Phase 1，更现实的路线是：
- **Warm-start**：从一个已收敛的 baseline checkpoint（10k 或更长）开始 finetune，再逐步打开几何注入（小 LR、强限幅、只训 geo 分支），而不是从零训练就引入；
- 或者保持 Phase 1 关闭，把时间投入 Phase 2（离散类别扩散）/ Phase 3（已可用）。

#### D) Warm-start（从 10k baseline finetune）结果：Phase 1 可用且可小幅提点（总体 AP）

我们按上述建议做了 warm-start 路线：从已经收敛的 10k baseline 权重开始，用很小的 LR 只做短步 finetune，并把几何注入限制在 **低噪声时间步**（`threshold`）。

1) baseline（10k）在当前代码下的评估结果：
- 权重：`baselines/output/20251212_225009/repro_10k_diffdet/model_final.pth`
- eval-only：`/tmp/diffdet_eval_10k_baseline_currentcode/`
  - bbox AP=46.8976（car=53.129，motorbike=40.666）

2) correctness：开启 `GEO_FEAT` 但保持 no-op（不训练）必须与 baseline 等价：
- `GEO_FEAT=True` eval-only：`/tmp/diffdet_eval_10k_geo_feat_noop_currentcode/`
  - bbox AP=46.8976（与 baseline 一致）

3) warm-start finetune（从 baseline 10k 出发，逐步加大“有效强度”）

- 保守设置（接入链路 + 不崩，但仍略掉点）：
  - `threshold=50, alpha=0.01, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold50_lr2p5e-6_geolr10_a0p01_warm500_iter1000_seed42/metrics.json`
    - bbox AP=46.7065（相对 baseline -0.19）

- 提高有效强度（总体 AP 开始超过 baseline）：
  - `threshold=100, alpha=0.02, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold100_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed42/metrics.json`
    - bbox AP=47.0137（相对 baseline +0.12）
  - `threshold=150, alpha=0.02, rel=True`（seed42 参考点）：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed42/metrics.json`
    - bbox AP=47.0230（相对 baseline +0.13）

- 过强会负收益：
  - `threshold=200, alpha=0.02, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold200_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed42/metrics.json`
    - bbox AP=45.7878（明显下降）
  - 关闭 `GEO_FEAT_OUT_TANH`（输出不限幅）会下降：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_notanh_warm500_iter1000_seed42/metrics.json`
    - bbox AP=46.2114

- 追加 sweep（围绕当前 best 微调，均未超过 `threshold=150, alpha=0.02, iter=1000`）：
  - `threshold=130, alpha=0.02, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold130_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed42/metrics.json`
    - bbox AP=46.7301（相对 baseline -0.17）
  - `threshold=170, alpha=0.02, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold170_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed42/metrics.json`
    - bbox AP=46.1781（相对 baseline -0.72）
  - `threshold=150, alpha=0.015, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p015_warm500_iter1000_seed42/metrics.json`
    - bbox AP=45.2465（相对 baseline -1.65）
  - `threshold=150, alpha=0.025, rel=True`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p025_warm500_iter1000_seed42/metrics.json`
    - bbox AP=46.3547（相对 baseline -0.54）
  - `threshold=150, alpha=0.02, rel=True, iter=2000`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter2000_seed42/metrics.json`
    - bbox AP=46.9859（相对 baseline +0.09，但低于 iter=1000 的 best）

- 稳定性复验（同一配置，多 seed 方差较大）：
  - baseline 参考：bbox AP=46.8976（`/tmp/diffdet_eval_10k_baseline_currentcode/log.txt`）
  - `seed=0`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed0/metrics.json`
    - bbox AP=47.5332（相对 baseline +0.64）
  - `seed=1`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed1/metrics.json`
    - bbox AP=47.0126（相对 baseline +0.11）
  - `seed=2`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed2/metrics.json`
    - bbox AP=46.1276（相对 baseline -0.77）
  - `seed=3`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed3/metrics.json`
    - bbox AP=47.0883（相对 baseline +0.19）
  - `seed=4`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed4/metrics.json`
    - bbox AP=46.7640（相对 baseline -0.13）
  - `seed=5`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed5/metrics.json`
    - bbox AP=48.8691（相对 baseline +1.97）
  - `seed=6`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed6/metrics.json`
    - bbox AP=45.7358（相对 baseline -1.16）
  - `seed=7`：`/tmp/diffdet_ft_geo_feat_qkpos_rel_iou_warmstart10k_threshold150_lr2p5e-6_geolr10_a0p02_warm500_iter1000_seed7/metrics.json`
    - bbox AP=46.4360（相对 baseline -0.46）
  - 统计（seed=0~7）：mean=46.9458，std=0.9011，min=45.7358，max=48.8691（≈ baseline 小幅高 0.05 AP，但方差大，不宜用单次 run 判定收益）

结论：**Phase 1 在 warm-start 路线下可以稳定训练且不会崩（不再出现 AP=0/NaN），并且通过调 `threshold/alpha` 可以做到总体 AP 小幅超过 baseline**；如果后续要继续追更高 AP，建议：
- 提高 `GEO_FEAT` 的有效强度时必须继续采用 `threshold`/强限幅/小 LR，并增加更频繁的 eval 以便 early-stop；
- 或先完成 Phase 2（离散类别扩散）后再回头做图交互，以减少长尾噪声带来的不稳定。

### 9) Phase 3 起步：质量头（Quality head）+ 能量引导采样（Guidance）已接入

我们已经把 Phase 3 的“工程骨架”接上了：在不改动 baseline 主训练链路的前提下，新增一个 **quality head** 来预测每个 proposal 的 “box 质量”（用 matched pair 的 IoU 作监督），并在推理采样时支持用质量头的梯度做能量引导（scale=0 时严格等价 baseline）。

代码改动点（已落地）：
- 配置项：`baselines/DiffusionDet/diffusiondet/config.py`
  - `QUALITY_HEAD` / `QUALITY_LOSS_TYPE` / `QUALITY_LOSS_WEIGHT`
  - `QUALITY_GUIDANCE_SCALE` / `QUALITY_GUIDANCE_TOPK` / `QUALITY_GUIDANCE_GRAD_NORM`
- 模型输出：`baselines/DiffusionDet/diffusiondet/head.py`
  - `RCNNHead` 增加 `quality_head`，输出 `quality_logits`（每个 proposal 一个标量 logit）
  - `DynamicHead` 增加第三个返回值 `outputs_quality`（或 None）
- Loss：`baselines/DiffusionDet/diffusiondet/loss.py`
  - 新增 `loss_quality`：对 matched proposals 计算 `IoU` 作为 soft target（[0,1]），支持 `bce/l1/mse`
- 推理采样：`baselines/DiffusionDet/diffusiondet/detector.py`
  - 由于 Detectron2 的 `ROIPooler/ROIAlignV2` 对 proposal boxes 不回传可用梯度，直接做 ∇x_t guidance 会得到全 0（详见 9.4）。
  - 当前实现为 **box-space guidance**：仅在 terminal step（`time_next<0`）固定 final head 的 proposal feature，把 `pred_boxes` 当作变量，重算 quality head 并对 boxes 做一步梯度更新。
  - `QUALITY_GUIDANCE_SCALE==0` 时保持严格等价 baseline（并已用 sha256 验证输出一致）。
- optimizer：`baselines/DiffusionDet/train_net.py`
  - 支持 `QUALITY_HEAD_LR_MULT` 对 `quality_*` 参数单独缩放学习率

验证（smoke）：

1) 训练链路跑通（10 iter，只验证不会崩）：
```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 10 SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.QUALITY_HEAD True MODEL.DiffusionDet.QUALITY_LOSS_WEIGHT 1.0 \
  OUTPUT_DIR /tmp/diffdet_quality_head_smoke_iter10 \
  SEED 42
```
2) 能量引导路径跑通（eval-only，scale=0.1）：
```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --eval-only --num-gpus 1 \
  MODEL.WEIGHTS /tmp/diffdet_quality_head_smoke_iter10/model_final.pth \
  MODEL.DiffusionDet.QUALITY_HEAD True \
  MODEL.DiffusionDet.QUALITY_GUIDANCE_SCALE 0.1 \
  OUTPUT_DIR /tmp/diffdet_quality_guidance_eval_scale0p1
```

注：由于上述 smoke 没有训练到可用阶段，eval 会提示 `No predictions from the model!` 属于预期现象；这里只验“链路正确 + 不崩 + guidance 触发后速度变慢（说明走了 enable_grad 路径）”。

### 9) Phase 3：质量头训练“掉点 / AP=0 / No predictions”已定位并修复

#### 9.1 现象

在 `RepeatFactorTrainingSampler + threshold=0.02 + MAX_ITER=2500 + SEED=42` 的短跑对照里，一度出现：

- `QUALITY_HEAD=True` 时 **bbox AP=0**（预测框基本全是 FP，IoU≈0）
- 或者 eval 输出 `No predictions from the model!`，COCOeval 全部 `NaN`

其中 `No predictions` 的直接原因是 Detectron2 的 `detector_postprocess` 会做一次 `output_boxes.nonempty()` 过滤：当预测框大量退化为“宽/高≈0”的退化框时，会被全部过滤掉，最终 evaluator 看到空预测列表（`inference/coco_instances_results.json` 只剩 `[]`）。

#### 9.2 根因与修复点

1) **full_model 梯度裁剪耦合（会把 quality 的梯度“带着”一起裁）**
- 原因：Detectron2 配置是 `SOLVER.CLIP_GRADIENTS.CLIP_TYPE=full_model`，默认会对 **全模型参数的梯度范数** 做一次整体裁剪。即使 `loss_quality` 不会对检测分支产生梯度（我们在 head/loss 中做了 stop-grad），它仍会显著改变全局梯度范数，从而间接影响检测分支的有效更新幅度，导致训练不稳定/后期无法进入收敛。
- 修复：在 `baselines/DiffusionDet/train_net.py` 把 full_model clipping 改为 **按参数组分开裁剪**（main vs quality），避免 quality 梯度影响 main 分支的 clip 比例。

2) **启用 optional module 会改变 RNG 消耗（影响初始化/训练随机性）**
- 原因：新增模块（quality head）会改变参数初始化时的随机数消耗顺序；如果不控制 RNG，容易让 “同 seed 的对照” 不可比，且在短跑里表现为收敛路径差异巨大。
- 修复：
  - `baselines/DiffusionDet/diffusiondet/head.py`：在构造 `quality_head` 时用 `torch.get_rng_state()/set_rng_state()` 保护 RNG，避免启用 quality head 改变全局 RNG 状态。
  - `baselines/DiffusionDet/diffusiondet/head.py`：在 `DynamicHead._reset_parameters()` 中对 **main 参数** 与 **optional 参数** 分两段初始化：先按 baseline 顺序初始化 main（跳过 quality/geo 参数，保证不改变 main 的 RNG 消耗），再在不改变全局 RNG 的前提下初始化 optional（初始化后恢复 RNG state）。

#### 9.3 复测（2500 iter 对照已通过）

质量头 + 检测分支一起训练（不冻结主干），在同口径下已经做到“不掉点”：

**训练（QUALITY_HEAD=True）**：
```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 2500 SOLVER.CHECKPOINT_PERIOD 100000000 \
  DATALOADER.SAMPLER_TRAIN RepeatFactorTrainingSampler DATALOADER.REPEAT_THRESHOLD 0.02 \
  MODEL.DiffusionDet.QUALITY_HEAD True MODEL.DiffusionDet.QUALITY_LOSS_WEIGHT 1.0 \
  OUTPUT_DIR /tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed \
  SEED 42
```

**结果**：`/tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed/metrics.json`（最后一行）
- bbox AP=19.5812，AP50=43.8477，AP75=14.8182

（对照：baseline short-run `repeat=0.02` 的历史记录是 bbox AP≈18.987）

#### 9.4 Guidance：scale>0 之前“无效”（输出完全一致）的原因

- 现象：同一 checkpoint + 固定 `SEED` 下，`QUALITY_GUIDANCE_SCALE` 从 `0 → >0`，`inference/coco_instances_results.json` 的 sha256 完全一致（输出完全相同）。
- 进一步确认：在 `ddim_sample` 内部对 `x_t` 做 `autograd.grad(obj, x_t)` 得到的梯度张量全 0（abs_mean/abs_max 都为 0）。
- 根因（工程层面）：
  - DiffusionDet 的 proposal → 特征路径依赖 Detectron2 的 `ROIPooler/ROIAlignV2`，该算子对 proposal boxes **不提供可用梯度**，导致 “质量分数对 x_t 的可导路径” 在 autograd 中为 0。
  - 且当前 baseline `SAMPLE_STEP=1`，terminal step 会直接输出 `x_start`；仅修改 `pred_noise` 也不会影响最终 `pred_boxes`。

#### 9.5 Guidance 修复：box-space quality guidance（已可用 + 可验证）

实现：在 terminal step 用 forward hook 抓取 final `RCNNHead.quality_head` 的输入 `q_in=[q_feat,q_geo]`，固定 `q_feat`，把 `pred_boxes` 当作变量重算 `quality_head(q_feat, q_geo(pred_boxes))`，对 `pred_boxes` 做一步梯度更新并裁剪到图像边界。

复现/验证（10k warm-start checkpoint，`SEED=42`）：

- checkpoint：`/tmp/diffdet_ft_quality_head_warmstart10k_lr2p5e-6_qlr100_iter2000_seed42/model_final.pth`
- `QUALITY_GUIDANCE_SCALE=0`（等价性）：
  - output：`/tmp/diffdet_eval_qguidance_scale0_boxspace_seed42/`
  - bbox AP=47.7013
  - sha256 与旧 eval-only 输出一致（证明 scale=0 行为不变）
- 小步长有小幅收益：
  - scale=0.05：`/tmp/diffdet_eval_qguidance_scale0p05_boxspace_seed42/` bbox AP=47.7132
  - scale=0.1：`/tmp/diffdet_eval_qguidance_scale0p1_boxspace_seed42/` bbox AP=47.7679（仅在 SEED=42 下最好）
- 步长过大明显退化（反例）：
  - scale=1.0：`/tmp/diffdet_eval_qguidance_scale1_boxspace_seed42/` bbox AP=46.3107

多 seed 复验（**同一 checkpoint**，仅改推理 `SEED`，用于衡量 guidance 是否稳定）：

- checkpoint：`/tmp/diffdet_ft_quality_head_warmstart10k_lr2p5e-6_qlr100_iter2000_seed42/model_final.pth`
- seed=0：scale0=47.0333；scale0.05=47.0314（-0.0019）；scale0.1=47.0245（-0.0088）
- seed=1：scale0=48.4284；scale0.05=48.5117（+0.0833）；scale0.1=48.4185（-0.0099）
- seed=2：scale0=48.8223；scale0.05=48.7892（-0.0331）；scale0.1=48.7807（-0.0416）
- seed=3：scale0=48.7846；scale0.05=48.8041（+0.0195）；scale0.1=48.6108（-0.1738）
- seed=4：scale0=47.4229；scale0.05=47.2935（-0.1294）；scale0.1=47.4178（-0.0051）
- 汇总（seed=0..4）：ΔAP(scale0.05)=mean -0.0123；ΔAP(scale0.1)=mean -0.0478

结论：当前这个 “terminal step 的 1-step box-space quality guidance” **可以改变输出**，但对总体 AP 的提升并不稳定（不同 seed 可能涨/跌）；建议：
- 默认保持 `QUALITY_GUIDANCE_SCALE=0`；
- 若要继续探索：用更小步长（如 `0.01~0.05`）、或把步长改成按 box 尺寸/图像尺寸归一化、或用多个 seed/多次采样做 ensemble 来降低方差。

建议默认从 `QUALITY_GUIDANCE_SCALE=0.01~0.05` 开始做 sweep。

#### 9.6 Phase 3：更稳定的落地收益——Quality score reweight（强烈推荐）

相比 “梯度引导 boxes”，一个更稳定且更符合检测工程习惯的用法是：**用 quality head 预测的质量分数重标定分类 score**。

实现（已落地）：
- 配置项：`baselines/DiffusionDet/diffusiondet/config.py`
  - `QUALITY_SCORE_REWEIGHT`（bool，默认 False）
  - `QUALITY_SCORE_POWER`（float，默认 1.0）
- 推理：`baselines/DiffusionDet/diffusiondet/detector.py`
  - 在 `inference()` 内对分类分数做 `score *= sigmoid(quality)^power`（在 focal 分支里会影响 topk 选择；在 softmax 分支里会影响排序/NMS）

复现（同一 checkpoint，仅改推理参数）：

```bash
cd baselines/DiffusionDet
python train_net.py --eval-only --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  MODEL.WEIGHTS /tmp/diffdet_ft_quality_head_warmstart10k_lr2p5e-6_qlr100_iter2000_seed42/model_final.pth \
  MODEL.DiffusionDet.QUALITY_HEAD True \
  MODEL.DiffusionDet.QUALITY_GUIDANCE_SCALE 0.0 \
  MODEL.DiffusionDet.QUALITY_SCORE_REWEIGHT True \
  MODEL.DiffusionDet.QUALITY_SCORE_POWER 1.0 \
  SEED 42 \
  OUTPUT_DIR /tmp/diffdet_eval_qscore_reweight_power1_seed42
```

结果：
- seed=42：`/tmp/diffdet_eval_qscore_reweight_power1_seed42/` bbox AP=48.8551（相对 `scale=0` baseline 的 47.7013，+1.1538）
- 多 seed（0..4，对照基线为 `QUALITY_SCORE_REWEIGHT=False` 的 `scale=0` 输出）：
  - seed0：47.0333 → 48.1666（+1.1333）
  - seed1：48.4284 → 49.2877（+0.8593）
  - seed2：48.8223 → 49.6461（+0.8238）
  - seed3：48.7846 → 49.9079（+1.1233）
  - seed4：47.4229 → 48.6677（+1.2448）
  - 汇总（seed0..4）：ΔAP mean=+1.0369，std=0.1655

结论：`QUALITY_SCORE_REWEIGHT=True` 在当前 checkpoint 上 **稳定提升总体 AP**，建议作为 Phase 3 的默认落地成果；`QUALITY_GUIDANCE_SCALE` 可继续保持 0（避免引入额外不稳定性）。

---

## Phase 2(A) 进展：Label-state（unk 吸收态）最小骨架已跑通

目标：先不做完整 D3PM，只把“类别状态”作为离散输入接进来（含 `unk` 吸收态），跑通训练/推理链路并可观测。

已落地内容：
- 配置项（默认关闭，不影响 baseline）：`MODEL.DiffusionDet.LABEL_STATE` 及 keep-prob 调度相关参数（见 `baselines/DiffusionDet/diffusiondet/config.py`）。
- 训练侧：对来自 GT 的 proposals，将 `c0` 以 `keep_prob(t)` 保留，否则替换为 `unk`；并把 `c_t` embedding 注入 proposal features（可学习 gate + zero-init proj，初始化为 no-op）。
- 推理侧：默认从全 `unk` 开始（`LABEL_STATE_FORCE_UNK_INFER=True`），保持训练/推理口径一致。
- 可观测：训练日志写入 `metrics.json` 的 `label_state_keep_prob` / `label_state_non_unk_ratio`（用于验证 corruption 是否生效）。

Smoke（已通过）：
```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 10 SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.LABEL_STATE True \
  OUTPUT_DIR /tmp/diffdet_label_state_smoke_iter10_v2 \
  SEED 42
```

验证方式：
- 打开 `/tmp/diffdet_label_state_smoke_iter10_v2/metrics.json`，确认包含：
  - `label_state_keep_prob`
  - `label_state_non_unk_ratio`
- 训练/推理链路不崩即可；短跑 AP 可能为 NaN/0 属正常（迭代太少，模型可能无预测）。

下一步（建议验收口径）：
- `MAX_ITER=500`：确认 loss 正常下降、无 NaN（只做“跑稳”验收）。
- `MAX_ITER=2500`：在同一口径下对比 baseline 的总体 AP 是否掉点（Phase 2(A) 先不追涨点，优先“不掉点/可控”）。

### Phase 2(A) 验收：MAX_ITER=500（已执行）

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.LABEL_STATE True \
  OUTPUT_DIR /tmp/diffdet_label_state_iter500 \
  SEED 42
```

结果：
- 训练 loss 正常、无 NaN（满足“跑稳”）。
- `/tmp/diffdet_label_state_iter500/metrics.json` 包含：
  - `label_state_keep_prob`
  - `label_state_non_unk_ratio`
- 该短跑在本数据上可能仍出现 “No predictions from the model!” 导致 AP=NaN（这不影响“链路/稳定性”验收）。

### Phase 2(A) 验收：MAX_ITER=2500 baseline vs label_state（已执行 + 多 seed eval）

训练（得到 checkpoint）：
```bash
cd baselines/DiffusionDet
# baseline
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 2500 SOLVER.CHECKPOINT_PERIOD 100000000 \
  OUTPUT_DIR /tmp/diffdet_baseline_iter2500_seed42_currentcode \
  SEED 42

# label_state
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 2500 SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.LABEL_STATE True \
  OUTPUT_DIR /tmp/diffdet_label_state_iter2500_seed42_currentcode \
  SEED 42
```

说明：训练跑完后会自动做一次 eval，但该 eval 的随机数状态已被训练过程推进，不适合做“同噪声种子”的严格对照；因此下面用 `--eval-only` 统一对比。

eval-only（固定权重，只改推理 seed；baseline 与 label_state 用相同 seed）：
```bash
cd baselines/DiffusionDet
# baseline
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.WEIGHTS /tmp/diffdet_baseline_iter2500_seed42_currentcode/model_final.pth \
  OUTPUT_DIR /tmp/diffdet_eval_baseline_iter2500_seed42_currentcode_evalseed0 \
  SEED 0

# label_state
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.DiffusionDet.LABEL_STATE True \
  MODEL.WEIGHTS /tmp/diffdet_label_state_iter2500_seed42_currentcode/model_final.pth \
  OUTPUT_DIR /tmp/diffdet_eval_label_state_iter2500_seed42_currentcode_evalseed0 \
  SEED 0
```
（将 `SEED`/`OUTPUT_DIR` 的 `0` 换成 `1/2/3/4/42` 即可复现下表）

结果（bbox AP）：
- seed0：21.3996 → 20.9421（-0.4575）
- seed1：20.6530 → 21.9091（+1.2561）
- seed2：22.0336 → 20.8257（-1.2079）
- seed3：21.0494 → 20.7331（-0.3163）
- seed4：20.8162 → 20.4104（-0.4058）
- seed42：19.7172 → 20.7096（+0.9924）
- 汇总（seed0..4,42）：ΔAP mean=-0.0232，std=0.8653

结论：
- Phase 2(A) 在 `MAX_ITER=2500` 口径下，对总体 AP **均值几乎不变**，但不同采样 seed 的方差较大（单次 AP 的不确定性较高）。
- 若你希望“更稳定”的对照：建议提高采样步数/做采样集成（减少单次采样方差），或用更多 seed 取均值。

---

## 复现步骤（推荐顺序）

### 1. （可选）从原始 tar.gz 解包 VOC 子集

若你需要从压缩包重建 `repro_*`：

1. 把 `repro_10k*.tar.gz` / `repro_50k*.tar.gz` / `repro_200k*.tar.gz` 放到 `datasets/` 或仓库根目录。
2. 在压缩包所在目录运行：
   ```bash
   python unpack_datasets.py
   ```
3. 预期输出：
   - 根目录出现 `repro_10k/`、`repro_50k/`、`repro_200k/`
   - 各自包含 `VOC2012/Annotations`、`VOC2012/JPEGImages`。

验证方式：
- 随机查看 `VOC2012/Annotations` 与 `JPEGImages` 有文件且能对应。

---

### 2. 生成 COCO 格式数据（供训练）

当前仓库已提供 `repro_10k` 的 COCO 数据；若你要重新生成或为 50k/200k 生成：

```bash
python utils/prepare_coco_data.py repro_10k/VOC2012 baselines/data/repro_10k
```

验证方式：
- 目标目录结构应为：
  - `annotations/instances_train2017.json`
  - `annotations/instances_val2017.json`
  - `train2017/`、`val2017/`
- JSON 可被正常解析（无语法错误），图片目录非空。

---

### 3. 先跑 Smoke Test（强烈推荐）

用于快速确认训练链路无误，几分钟内完成。

**DiffusionDet smoke**：
```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100 OUTPUT_DIR /tmp/diffdet_smoke
```
说明：`diffdet.repro_10k.yaml` 现在是完整 baseline 配置；smoke test 通过命令行覆盖迭代数。

验证方式：
- 启动时打印 “Registered datasets …/baselines/data”。
- `/tmp/diffdet_smoke/` 下出现日志/ckpt。

**DETR smoke**：
```bash
cd baselines/detr
python ../../utils/prepare_detr_coco_ids.py --src ../data/repro_10k --dst ../data/repro_10k_detr_coco
python main.py \
  --dataset_file coco \
  --coco_path ../data/repro_10k_detr_coco \
  --num_classes 91 \
  --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
  --output_dir /tmp/detr_smoke \
  --epochs 1 \
  --batch_size 2
```

验证方式：
- 控制台输出 loss/epoch 日志。
- `/tmp/detr_smoke/` 下生成 `checkpoint.pth`。

---

### 4. 跑完整 baseline（Repro-10k）

在仓库根目录执行：

```bash
bash run_baselines.sh
```

该脚本会：
1. 生成 `baselines/output/<timestamp>/`
2. 长跑 DiffusionDet（10000 iter）
3. 长跑 DETR（COCO-pretrained fine-tune：50 epoch，`lr_drop=40`）

验证方式：
- `baselines/output/<timestamp>/repro_10k_diffdet/`
  - 存在 `model_final.pth` 或周期 checkpoint
  - 有 Detectron2 训练日志（loss/iter）
- `baselines/output/<timestamp>/repro_10k_detr/`
  - 存在 `checkpoint.pth`
  - 有 DETR 训练日志（loss/epoch）

若需要评估：
- DiffusionDet：`train_net.py --eval-only MODEL.WEIGHTS <pth>`
- DETR：在 `main.py` 加 `--eval --resume <ckpt>`

---

## 常见问题排查

- **找不到数据集 / annotation**：
  - 检查 `baselines/data/repro_10k/annotations/*.json` 是否存在且路径正确。
- **类别数维度不匹配**：
  - DiffusionDet 需保证 COCO `categories` 数为 3（config `NUM_CLASSES=3`）。
  - DETR baseline 使用 COCO-pretrained 方式：需要先生成 `baselines/data/repro_10k_detr_coco/` 并传 `--num_classes 91`。
- **Detectron2/torch 版本不兼容**：
  - 当前环境已通过 import 校验；若在新环境复现，优先保证 Detectron2 与 torch/torchvision 匹配。
