# plan.md — 当前进度 + 下一步逐一执行（更新：2025-12-15）

本文件只回答三件事：
1) 现在完成了什么（用“可复验的数据”说话）
2) 下一步做什么（按 Phase 1/2/3 拆分，逐一执行）
3) 每一步怎么验证（具体命令 + 通过标准 + 产物在哪）

> 注意：根盘 `/` 目前接近满（`df -h /` 显示 ~99%），新训练建议优先写到 `/dev/shm`（tmpfs，约 63G 空间），避免 `/tmp` 再堆大 checkpoint。

---

## 0. 快速结论（你现在做到了什么）

### 0.1 数据集与评估注意点（避免误判 “nan/0”）

- 数据集：`baselines/data/repro_10k/`（COCO 格式，3 类：car / motorbike / person）。
- **val 集里 person=0**，所以 COCOeval 的 `AP-person` 常见为 `nan`，这不是训练崩。
- DiffusionDet 打印的 AP 表是 **0~100** 标尺（例如 `21.3996` 表示 21.3996）。

### 0.2 Phase 1（2500 iter 快速口径）现有对照结果

口径：`--eval-only` + `MODEL.DiffusionDet.SAMPLE_STEP=1` + seeds=`0,1,2,3,4,42`。

| Variant | mean AP | std | 结果来源（log.txt） |
|---|---:|---:|---|
| baseline | 20.9448 | 0.7085 | `/tmp/diffdet_eval_baseline_iter2500_seed42_currentcode_evalseed{seed}/log.txt` |
| label_state（Phase 2(A)） | 20.9217 | 0.4702 | `/tmp/diffdet_eval_label_state_iter2500_seed42_currentcode_evalseed{seed}/log.txt` |
| D3PM(mask, distribution) 旧实现 | 17.0203 | 0.3985 | `/tmp/diffdet_eval_label_d3pm_mask_iter2500_seed42_evalseed{seed}/log.txt` |
| D3PM(mask, distribution) **新实现（train 采样）** | **27.3704** | **0.4129** | `/dev/shm/diffdet_eval_label_d3pm_mask_iter2500_seed42_sampletrain_evalseed{seed}/log.txt` |

结论：
- baseline 与 label_state：**不掉点**（均值几乎相同）。
- 旧 D3PM(mask, distribution)：**显著掉点**（均值 -3.9 AP）。
- 新 D3PM(mask, distribution)（训练时采样离散 label state）：**显著提升**（均值 +6.4 AP）。

### 0.3 Phase 1（完整 10k）对照组已跑通（同口径 seeds=0/1/2）

| Variant | mean AP | std | 备注 |
|---|---:|---:|---|
| vanilla DiffusionDet baseline | **45.8896** | 1.3481 | 见 3.8 |
| D3PM(mask, dist) + QUALITY_HEAD | 44.7976 | 2.2864 | 见 3.7（seed42 单跑 46.6839，见 3.6） |

### 0.4 公平口径（10k checkpoint eval-only，eval seeds=0..4）

同一 checkpoint 下用多个 eval seed 复验（避免“只看一次 eval”误判）：

| Variant | mean over train_seeds | flatten std（15 runs） | 备注 |
|---|---:|---:|---|
| vanilla baseline | 46.4293 | 0.7817 | 见 3.10 |
| **baseline + GEO_FEAT finetune（+1k iter）** | **46.7774** | 0.6955 | **+0.3481 AP**，见 3.11 |
| D3PM+QHead + GEO_FEAT finetune（+1k iter） | 45.2418 | — | 见 2.4.4 |
| **D3PM+QHead warmstart（从 baseline 10k 起步，+2.5k iter）** | **46.9068** | 0.6829 | **修复 seed0/2 掉点**，见 3.12 |

### 0.5 与论文 6.1–6.3 对照（已做/缺失/怎么验证）

| 论文项（图中 6.x） | 仓库对应 | 当前状态 | 怎么验证（最小命令） |
|---|---|---|---|
| 6.1 Baseline 复现 | `run_baselines.sh` + `baselines/DiffusionDet/configs/diffdet.repro_10k.yaml` | ✅ 已跑通（R50-FPN） | `bash run_baselines.sh` 或 `cd baselines/DiffusionDet && python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 SOLVER.MAX_ITER 500 OUTPUT_DIR /dev/shm/diffdet_smoke` |
| 6.1 Swin-B baseline | `baselines/DiffusionDet/configs/diffdet.repro_10k.swinbase.yaml` | ✅ 已补齐（可选跑） | `RUN_DIFFDET_SWIN=1 DIFFDET_MODELS_DIR=/dev/shm/diffdet_models OUTPUT_BASE=/dev/shm bash run_baselines.sh` |
| 6.2 D3PM Scheduler / label diffusion | `baselines/DiffusionDet/diffusiondet/detector.py:_label_d3pm_forward` | ✅ 已实现（mask/uniform） | 见 `docs/plan.md` 3.3/3.6/3.12（D3PM 训练与 eval-only 复验） |
| 6.2 各向异性 box 噪声 | `baselines/DiffusionDet/diffusiondet/detector.py:q_sample/ddim_sample` | ✅ 已实现（可开关） | `--eval-only` 对比：加 `MODEL.DiffusionDet.ANISO_NOISE True MODEL.DiffusionDet.ANISO_NOISE_SIGMA_W 2.0 MODEL.DiffusionDet.ANISO_NOISE_SIGMA_H 2.0` 看 AP/是否 NaN |
| 6.2 Graph / 拓扑约束 | `baselines/DiffusionDet/diffusiondet/loss.py:loss_graph`（用 attention 做邻接） | ✅ 已实现（可开关） | 训练时加 `MODEL.DiffusionDet.GRAPH_TOPO_LOSS_WEIGHT 1.0`，观察 log 里 `loss_graph` 非 0 且训练不崩 |
| 6.3 Quality Head | `baselines/DiffusionDet/diffusiondet/head.py` | ✅ 已实现 | 用 `configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml` 跑 500 iter，log 中出现 `loss_quality` |
| 6.3 Energy/Langevin guidance | `baselines/DiffusionDet/diffusiondet/detector.py:ddim_sample` | ✅ 已补齐（跨 timestep + 多步） | `--eval-only` 固定同一权重，改 `QUALITY_GUIDANCE_*`（见 4.2 的 sweep 模板） |
| 6.3 Consistency Distillation | `baselines/DiffusionDet/train_net.py` + `diffusiondet/detector.py` | ✅ 已给出入口与配置 | 用 `configs/diffdet.repro_10k_d3pm_mask_dist_qhead_consistency_distill.yaml` 并在命令行传 `MODEL.DiffusionDet.CONSISTENCY_TEACHER_WEIGHTS <teacher.pth>`；训练 loss 多出 `loss_consistency_*` |
| 图里提到的 MMDetection/Diffusers pipeline | — | ❌ 本仓库未采用（当前是 Detectron2 线） | 如需对齐，需要新增独立工程分支（不建议和当前 baseline 混跑） |

---

## 1. 结果在哪里（不要扫大目录）

### 1.1 一键 baseline 输出（大跑）

脚本：`run_baselines.sh`（仓库根目录）。

- 输出目录：`baselines/output/<timestamp>/`
- 找最新 run：

```bash
ls -1dt baselines/output/20* | head -n 5
```

`run_baselines.sh` 当前会跑：
- DiffusionDet baseline：`repro_10k_diffdet`
- （可选）DiffusionDet Swin-B baseline：`repro_10k_diffdet_swinb`（`RUN_DIFFDET_SWIN=1`）
- DiffusionDet D3PM(mask)+QUALITY_HEAD baseline：`repro_10k_diffdet_d3pm_qhead`
- DETR baseline：`repro_10k_detr`

可选环境变量：
- `SEED=42`（默认 42）
- `OUTPUT_BASE=/path/to/output`（默认 `baselines/output`；磁盘紧张时可用 `/dev/shm` 临时跑）
- `RUN_DIFFDET_SWIN=1`（额外跑 Swin-B backbone）
- `DIFFDET_MODELS_DIR=/path/to/models`（Swin 权重下载目录；磁盘紧张建议 `/dev/shm/diffdet_models`）

### 1.2 2500 iter 快速实验输出（短回路）

你当前用到的关键 checkpoint（都很大，~1.3G）：

- baseline：`/tmp/diffdet_baseline_iter2500_seed42_currentcode/model_final.pth`
- label_state：`/tmp/diffdet_label_state_iter2500_seed42_currentcode/model_final.pth`
- 旧 D3PM(mask, distribution)（掉点）：`/tmp/diffdet_label_d3pm_mask_iter2500_seed42/model_final.pth`
- **新 D3PM(mask, distribution)（train 采样）**：`/dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth`
  - `/dev/shm` 会随重启清空；要长期保存请复制到磁盘（见 3.4）。

### 1.3 从 log.txt 抽取 AP（eval-only 统一口径）

eval-only 目录下的 `log.txt` 最后会有：

```
copypaste: 20.7096,44.4036,16.3641,4.6759,24.5689,35.6250
```

第一个数就是 `AP`。

### 1.4 一键统计多个 seed 的 mean/std（不扫大目录）

```bash
python - <<'PY'
import re, statistics
from pathlib import Path

def ap_from_log(p: Path) -> float:
    txt = p.read_text(errors="ignore")
    for line in reversed(txt.splitlines()):
        if "copypaste:" in line:
            m = re.search(r"copypaste:\\s*([0-9]+\\.[0-9]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"AP not found: {p}")

def summarize(name, tmpl, seeds):
    aps = {}
    for s in seeds:
        p = Path(tmpl.format(seed=s))
        aps[s] = ap_from_log(p)
    vals = list(aps.values())
    print(f"{name}: n={len(vals)} mean={statistics.mean(vals):.4f} std={statistics.pstdev(vals):.4f}")
    print("  " + " ".join([f"{s}:{aps[s]:.4f}" for s in sorted(aps)]))

seeds = [0,1,2,3,4,42]
summarize(
    "baseline_samp1",
    "/tmp/diffdet_eval_baseline_iter2500_seed42_currentcode_evalseed{seed}/log.txt",
    seeds,
)
summarize(
    "label_state_samp1",
    "/tmp/diffdet_eval_label_state_iter2500_seed42_currentcode_evalseed{seed}/log.txt",
    seeds,
)
summarize(
    "d3pm_mask_dist_old_samp1",
    "/tmp/diffdet_eval_label_d3pm_mask_iter2500_seed42_evalseed{seed}/log.txt",
    seeds,
)
summarize(
    "d3pm_mask_dist_sampletrain_samp1",
    "/dev/shm/diffdet_eval_label_d3pm_mask_iter2500_seed42_sampletrain_evalseed{seed}/log.txt",
    seeds,
)
PY
```

---

## 2. Phase 1（Skeleton & Baseline）— 怎么逐一验收

目标：把“能跑通”变成“可验收可重复”。

### 2.1 Baseline（2500 iter）eval-only 复验模板

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  MODEL.WEIGHTS /tmp/diffdet_baseline_iter2500_seed42_currentcode/model_final.pth \
  OUTPUT_DIR /tmp/diffdet_eval_baseline_iter2500_seed42_currentcode_evalseed42 \
  SEED 42
```

**通过标准**：`log.txt` 里出现 `copypaste: AP,...` 且 AP > 0（3 类数据集通常 >10）。

### 2.2 Label-state（Phase 2(A)）eval-only 复验模板

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  MODEL.DiffusionDet.LABEL_STATE True \
  MODEL.WEIGHTS /tmp/diffdet_label_state_iter2500_seed42_currentcode/model_final.pth \
  OUTPUT_DIR /tmp/diffdet_eval_label_state_iter2500_seed42_currentcode_evalseed42 \
  SEED 42
```

**通过标准**：均值与 baseline 差距在 0.5 AP 以内（当前已满足）。

### 2.3 Phase 1 的“短回路”推荐口径

- 训练：`MAX_ITER=2500`（只用于快速对比）
- 评估：`--eval-only` + seeds=`0,1,2,3,4,42`
- 统计：用 1.4 的脚本出 mean/std

### 2.4 （可选）GEO_FEAT（proposal 几何特征注入）— 现状与可用做法

结论先说：
- **GEO_FEAT 从头训练在 2500 iter 口径下会“回归退化”（预测框高度趋近 0），导致 AP=0。**
- 但 **在强 checkpoint 上做小 LR finetune**（主要让 GEO_FEAT 学），是可用的：能保持总体 AP、并显著降低采样方差。

#### 2.4.1 现象复现（不推荐继续用作“短回路对比”）

下面这些都是从头训练（`MAX_ITER=2500`）的产物（都在 `/dev/shm`），最终 `metrics.json` 为 AP=0 或 nan：
- `/dev/shm/diffdet_baseline_geofeat_iter2500_seed42/`
- `/dev/shm/diffdet_baseline_geofeat_const_iter2500_seed42/`
- `/dev/shm/diffdet_baseline_geofeat_qkpos_iter2500_seed42/`

其中一个直观证据：`qk_pos` 这次虽然输出了很多预测，但回归退化（COCO `bbox` 的 `h` 极小），IoU≈0 → AP=0。

#### 2.4.2 可用做法：从“强 baseline 10k checkpoint”上 finetune GEO_FEAT（推荐）

目的：让 GEO_FEAT 学“用不用/怎么用”，但不破坏已收敛的 detector。

1) 选一个强 baseline checkpoint（示例用 seed0）：
- `/dev/shm/diffdet_baseline_iter10000_seed0/model_final.pth`

2) finetune（`BASE_LR` 很小，`GEO_FEAT_LR_MULT` 很大，只跑 1000 iter；建议 seeds=0/1/2 都做一遍）：

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 1000 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  SOLVER.BASE_LR 2.5e-6 \
  MODEL.WEIGHTS /dev/shm/diffdet_baseline_iter10000_seed0/model_final.pth \
  MODEL.DiffusionDet.GEO_FEAT True \
  MODEL.DiffusionDet.GEO_FEAT_LR_MULT 100.0 \
  MODEL.DiffusionDet.GEO_FEAT_SCHEDULE constant \
  OUTPUT_DIR /dev/shm/diffdet_baseline_seed0_geofeat_ft_iter1000 \
  SEED 42
```

3) 用同口径 eval-only 对比（固定 `SAMPLE_STEP=1`，seeds=`0/1/2`）：

```bash
cd baselines/DiffusionDet
BASE=/dev/shm/diffdet_baseline_iter10000_seed0/model_final.pth
FT=/dev/shm/diffdet_baseline_seed0_geofeat_ft_iter1000/model_final.pth

for s in 0 1 2; do
  python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
    MODEL.DiffusionDet.SAMPLE_STEP 1 \
    MODEL.WEIGHTS ${BASE} \
    OUTPUT_DIR /dev/shm/diffdet_eval_baseline_seed0_evalseed${s} \
    SEED ${s}
done

for s in 0 1 2; do
  python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
    MODEL.DiffusionDet.SAMPLE_STEP 1 \
    MODEL.DiffusionDet.GEO_FEAT True \
    MODEL.WEIGHTS ${FT} \
    OUTPUT_DIR /dev/shm/diffdet_eval_baseline_seed0_geofeat_ft_evalseed${s} \
    SEED ${s}
done
```

### 2.5 （可选）Swin-B backbone smoke（repro_10k）— 已跑到哪一步

说明：Swin-B 配置与权重下载逻辑已补齐（见 `run_baselines.sh` 的 `RUN_DIFFDET_SWIN=1` 分支）。

已完成的 smoke（都能稳定训练，无 NaN）：
- 50 iter：`/dev/shm/diffdet_swin_smoke/`（早期预测框退化为点，`detector_postprocess` 会过滤掉全部预测 → AP=nan，属正常现象）
- 500 iter：`/dev/shm/diffdet_swin_iter500_seed42/`（同上，AP=nan）
- 2500 iter：`/dev/shm/diffdet_swin_iter2500_seed42/`（已能正常出预测并计算 AP）
  - `copypaste: 21.2076,45.2352,16.4910,4.7961,24.5826,37.3041`

如何验证：

```bash
tail -n 5 /dev/shm/diffdet_swin_iter2500_seed42/log.txt
ls -la /dev/shm/diffdet_models/swin_base_patch4_window7_224_22k.pkl
```

下一步（对齐图片里“Baseline=Swim-B + DiffusionDet”的口径）：
- 跑满 `MAX_ITER=10000`，并记录最终 `copypaste: AP,...`（建议输出到 `/dev/shm` 防止根盘爆满）：

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.swinbase.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 10000 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.WEIGHTS /dev/shm/diffdet_models/swin_base_patch4_window7_224_22k.pkl \
  OUTPUT_DIR /dev/shm/diffdet_swin_iter10000_seed42 \
  SEED 42
```

本次实测（seed=42，`SAMPLE_STEP=1`，train 到 10k 后自动 eval）：
- 输出目录：`/dev/shm/diffdet_swin_iter10000_seed42/`
- checkpoint：`/dev/shm/diffdet_swin_iter10000_seed42/model_final.pth`
- `copypaste: 46.7461,74.8295,50.1701,17.2384,54.3206,68.0537`

本次实测（seeds=0/1/2，`SAMPLE_STEP=1`）：
- baseline mean AP = 46.9571，std = 0.5566
- GEO_FEAT finetune mean AP = 47.2693，std = 0.0911

补充：baseline 的 GEO_FEAT finetune 已扩展到训练 seeds=0/1/2（产物）：
- `/dev/shm/diffdet_baseline_seed0_geofeat_ft_iter1000/model_final.pth`
- `/dev/shm/diffdet_baseline_seed1_geofeat_ft_iter1000/model_final.pth`
- `/dev/shm/diffdet_baseline_seed2_geofeat_ft_iter1000/model_final.pth`

#### 2.4.3 同思路迁移到 D3PM+QHead（已验证可跑，均值有提升）

finetune（基于 D3PM+QHead 的 10k checkpoint，示例用 seed1）：
- baseline checkpoint：`/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed1/model_final.pth`
- finetune 输出：`/dev/shm/diffdet_d3pm_qhead_seed1_geofeat_ft_iter1000/model_final.pth`

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 1000 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  SOLVER.BASE_LR 2.5e-6 \
  MODEL.WEIGHTS /dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed1/model_final.pth \
  MODEL.DiffusionDet.GEO_FEAT True \
  MODEL.DiffusionDet.GEO_FEAT_LR_MULT 100.0 \
  MODEL.DiffusionDet.GEO_FEAT_SCHEDULE constant \
  OUTPUT_DIR /dev/shm/diffdet_d3pm_qhead_seed1_geofeat_ft_iter1000 \
  SEED 42
```

eval-only 对比（seeds=0/1/2，`SAMPLE_STEP=1`）：
- seeds=0..4，`SAMPLE_STEP=1`
- baseline mean AP = 47.5358，std = 0.1912
- GEO_FEAT finetune mean AP = 48.0113，std = 0.4971（mean +0.4755）

结论：均值有提升（约 +0.48 AP），但 std 变大；若要把它变成“默认配方”，建议再扩展到不同训练 seed/不同 checkpoint 做稳健性确认。

#### 2.4.4 扩展到 D3PM+QHead 训练 seeds=0/1/2（已执行，eval seeds=0..4）

finetune（每个训练 seed 都从各自 10k checkpoint 起步）：

```bash
cd baselines/DiffusionDet
for train_seed in 0 1 2; do
  python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 1000 \
    SOLVER.CHECKPOINT_PERIOD 100000000 \
    SOLVER.BASE_LR 2.5e-6 \
    MODEL.WEIGHTS /dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed${train_seed}/model_final.pth \
    MODEL.DiffusionDet.GEO_FEAT True \
    MODEL.DiffusionDet.GEO_FEAT_LR_MULT 100.0 \
    MODEL.DiffusionDet.GEO_FEAT_SCHEDULE constant \
    OUTPUT_DIR /dev/shm/diffdet_d3pm_qhead_seed${train_seed}_geofeat_ft_iter1000 \
    SEED 42
done
```

eval-only（每个训练 seed，对 eval seeds=0..4 做均值/方差；`SAMPLE_STEP=1`）：

```bash
cd baselines/DiffusionDet
for train_seed in 0 1 2; do
  BASE=/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed${train_seed}/model_final.pth
  FT=/dev/shm/diffdet_d3pm_qhead_seed${train_seed}_geofeat_ft_iter1000/model_final.pth
  for s in 0 1 2 3 4; do
    python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 --eval-only \
      MODEL.DiffusionDet.SAMPLE_STEP 1 \
      MODEL.WEIGHTS ${BASE} \
      OUTPUT_DIR /dev/shm/diffdet_eval_d3pm_qhead_seed${train_seed}_evalseed${s} \
      SEED ${s}
  done
  for s in 0 1 2 3 4; do
    python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 --eval-only \
      MODEL.DiffusionDet.SAMPLE_STEP 1 \
      MODEL.DiffusionDet.GEO_FEAT True \
      MODEL.WEIGHTS ${FT} \
      OUTPUT_DIR /dev/shm/diffdet_eval_d3pm_qhead_seed${train_seed}_geofeat_ft_evalseed${s} \
      SEED ${s}
  done
done
```

结果（每行的 mean/std 都是 **对 eval seeds=0..4** 统计；不同于 10k 训练结束 `metrics.json` 的“单次 eval”口径）：

| train_seed | base mean AP | base std | GEO_FEAT-FT mean AP | GEO_FEAT-FT std | delta(mean) |
|---:|---:|---:|---:|---:|---:|
| 0 | 44.2543 | 0.3293 | 44.4941 | 0.8080 | +0.2398 |
| 1 | 47.5358 | 0.1912 | 48.0113 | 0.4971 | +0.4755 |
| 2 | 42.6303 | 0.2021 | 43.2201 | 0.6297 | +0.5898 |

汇总（train seeds=0/1/2 的 mean-of-means；每个 mean 先对 eval seeds=0..4 平均）：
- base mean AP = 44.8068
- GEO_FEAT-FT mean AP = 45.2418（+0.4350）

---

## 3. Phase 2（Hybrid 离散×连续）— D3PM(mask) 的落地与修复

### 3.1 旧问题（为什么 D3PM(mask, distribution) 会掉点）

旧实现里，当 `label_state` 是分布 `(N,nr,K+1)` 时，直接用 `E[emb]=probs@W` 注入 proposal feature。

这会把离散扩散的离散状态变成“软插值”，训练侧缺少离散采样带来的随机性，容易出现：
- 有效噪声不足（forward 不像真实的离散马尔可夫跳变）
- 类别稀有类（motorbike）更容易被压垮（我们观测到 motorbike AP 掉得最厉害）

### 3.2 已应用修复（训练时采样离散 state）

修改点：
- 文件：`baselines/DiffusionDet/diffusiondet/head.py`
- 逻辑：当 `label_state` 为分布时
  - **train**：`torch.multinomial` 采样离散 state → embedding（更贴近离散扩散）
  - **eval**：仍使用 `probs @ embedding_weight`（类似 dropout：train 随机，eval 用期望）

### 3.3 逐一执行：重训 D3PM(mask, distribution) 2500 iter（推荐写 /dev/shm）

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 2500 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.LABEL_STATE True \
  MODEL.DiffusionDet.LABEL_D3PM True \
  MODEL.DiffusionDet.LABEL_D3PM_KERNEL mask \
  MODEL.DiffusionDet.LABEL_D3PM_USE_DISTRIBUTION True \
  OUTPUT_DIR /dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain \
  SEED 42
```

产物：
- `model_final.pth`：`/dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth`
- `metrics.json`：最后一行有 `bbox/AP`

### 3.4 逐一执行：eval-only 多 seed 复验（同口径对照 baseline）

```bash
cd baselines/DiffusionDet
for s in 0 1 2 3 4 42; do
  python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
    MODEL.DiffusionDet.SAMPLE_STEP 1 \
    MODEL.DiffusionDet.LABEL_STATE True \
    MODEL.DiffusionDet.LABEL_D3PM True \
    MODEL.DiffusionDet.LABEL_D3PM_KERNEL mask \
    MODEL.DiffusionDet.LABEL_D3PM_USE_DISTRIBUTION True \
    MODEL.WEIGHTS /dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth \
    OUTPUT_DIR /dev/shm/diffdet_eval_label_d3pm_mask_iter2500_seed42_sampletrain_evalseed${s} \
    SEED ${s}
done
```

**通过标准**（Phase 2(A)→Phase 2 强形式的“门槛”）：
- mean AP ≥ baseline mean AP（当前：27.3704 vs 20.9448，已大幅超过）
- std 不要暴涨（当前 std≈0.41，OK）

### 3.5 如果要长期保存 /dev/shm 的 best checkpoint

根盘空间紧张，建议只保留“best 的 1~2 个”：

```bash
mkdir -p baselines/checkpoints
cp -v /dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth baselines/checkpoints/d3pm_mask_dist_sampletrain_iter2500_seed42.pth
```

（复制前建议先 `df -h /` 看空间；如空间不足，先清理旧的 `/tmp/diffdet*/model_final.pth`。）

### 3.6 把当前配方跑到完整 10k（已执行）

训练（从头训练 10k；关闭中间 checkpoint，输出写 `/dev/shm`）：

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  MODEL.DiffusionDet.LABEL_STATE True \
  MODEL.DiffusionDet.LABEL_D3PM True \
  MODEL.DiffusionDet.LABEL_D3PM_KERNEL mask \
  MODEL.DiffusionDet.LABEL_D3PM_USE_DISTRIBUTION True \
  MODEL.DiffusionDet.QUALITY_HEAD True \
  OUTPUT_DIR /dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed42 \
  SEED 42
```

产物：
- checkpoint：`/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed42/model_final.pth`
- 指标：`/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed42/metrics.json`（最后一行 `iteration=10000`）

最终指标（seed=42，`SAMPLE_STEP=1`）：
- `bbox/AP=46.6839`, `AP50=74.0916`, `AP75=49.4306`
- `APs=18.9108`, `APm=53.8278`, `APl=72.7984`
- per-class：`AP-car=54.1862`, `AP-motorbike=39.1816`（`AP-person=nan` 属于 val 无 person）

### 3.7 10k 多 seed 统计（seeds=0/1/2，已跑完）

脚本：
- `scripts/run_d3pm_qhead_10k_seeds012.sh`（顺序跑 seeds=`0 1 2`，输出到 `/dev/shm`）

后台启动日志：
- `/dev/shm/run_d3pm_qhead_10k_seeds012.nohup.log`

监控进度：

```bash
tail -n 50 /dev/shm/run_d3pm_qhead_10k_seeds012.nohup.log
```

每个 seed 的输出目录：
- `/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed0/`
- `/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed1/`
- `/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed2/`

结果（读取各自 `metrics.json` 最后一行，`iteration=10000`）：

| seed | bbox/AP | AP50 | AP75 | AP-car | AP-motorbike |
|---:|---:|---:|---:|---:|---:|
| 0 | 43.5812 | 68.6614 | 43.9209 | 52.1697 | 34.9927 |
| 1 | 48.0004 | 74.6821 | 52.9782 | 54.0398 | 41.9609 |
| 2 | 42.8114 | 67.8255 | 41.5076 | 51.5733 | 34.0495 |

汇总（0/1/2）：
- mean AP = 44.7976
- std AP = 2.2864

对照（seed=42 的完整 10k 单跑）：`bbox/AP=46.6839`（见 3.6）。

### 3.8 10k baseline 多 seed 统计（seeds=0/1/2，已跑完）

脚本：
- `scripts/run_diffdet_baseline_10k_seeds012.sh`（顺序跑 seeds=`0 1 2`，输出到 `/dev/shm`）

后台启动日志：
- `/dev/shm/run_diffdet_baseline_10k_seeds012.nohup.log`

每个 seed 的输出目录：
- `/dev/shm/diffdet_baseline_iter10000_seed0/`
- `/dev/shm/diffdet_baseline_iter10000_seed1/`
- `/dev/shm/diffdet_baseline_iter10000_seed2/`

结果（读取各自 `metrics.json` 最后一行，`iteration=10000`）：

| seed | bbox/AP | AP50 | AP75 | AP-car | AP-motorbike |
|---:|---:|---:|---:|---:|---:|
| 0 | 47.4855 | 74.3472 | 49.4843 | 54.7424 | 40.2286 |
| 1 | 44.1883 | 70.0184 | 42.7293 | 53.4001 | 34.9765 |
| 2 | 45.9950 | 72.8857 | 47.5120 | 53.3039 | 38.6860 |

汇总（0/1/2）：
- mean AP = 45.8896
- std AP = 1.3481

### 3.9 10k 对比结论（先追总体 AP）

同口径（10k、seeds=0/1/2）下：
- baseline mean AP **45.8896**
- D3PM+QHead mean AP **44.7976**

结论：
- **当前“总体 AP 最优”是 vanilla baseline（均值高 +1.0920 AP，且 std 更小）。**
- D3PM+QHead 在 seed1 上很强（48.0004），但整体方差更大；如果下一步继续追总体 AP，建议先做“稳定性/方差”优化（见 4 节与后续可选实验）。

### 3.10 公平口径：10k checkpoint 做 eval-only（seeds=0..4）

说明：这里把 **train_seed（checkpoint）** 和 **eval seed（采样/随机性）** 分开统计，避免只看单次 eval 造成误判。

命令模板（vanilla baseline）：

```bash
cd baselines/DiffusionDet
BASE=/dev/shm/diffdet_baseline_iter10000_seed{TRAIN_SEED}/model_final.pth
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  MODEL.WEIGHTS ${BASE} \
  OUTPUT_DIR /dev/shm/diffdet_eval_baseline_seed{TRAIN_SEED}_evalseed{EVAL_SEED} \
  SEED {EVAL_SEED}
```

结果（每个 train_seed 对 eval seeds=0..4 统计 mean/std）：

| train_seed | mean AP | std | per eval-seed AP（0..4） |
|---:|---:|---:|---|
| 0 | 47.0893 | 0.5136 | 47.3515 / 46.1699 / 47.3498 / 47.6471 / 46.9283 |
| 1 | 45.8847 | 0.8397 | 46.9127 / 45.8030 / 46.7428 / 45.2041 / 44.7610 |
| 2 | 46.3138 | 0.3445 | 45.8750 / 46.4856 / 46.6263 / 46.6601 / 45.9218 |

汇总：
- mean over train_seeds（对每个 train_seed 的 mean 再平均）= **46.4293**（std across train_seed means = 0.4985）
- flatten all 15 runs（train_seed∈{0,1,2} × eval_seed∈{0..4}）mean=**46.4293**，std=0.7817

对比（同样是 eval seeds=0..4 的口径）：
- D3PM+QHead base：mean over train_seeds = 44.8068（见 2.4.4）
- D3PM+QHead + GEO_FEAT finetune：mean over train_seeds = 45.2418（见 2.4.4）

结论：在这个更公平的统计口径下，**vanilla baseline 仍然领先**（46.4293 vs 45.2418）。

### 3.11 公平口径：baseline + GEO_FEAT finetune（train seeds=0/1/2，eval seeds=0..4）

说明：这里的 checkpoint 是 “10k baseline” 上再 finetune 1000 iter 得到（见 2.4.2）。

命令模板（baseline + GEO_FEAT finetune）：

```bash
cd baselines/DiffusionDet
FT=/dev/shm/diffdet_baseline_seed{TRAIN_SEED}_geofeat_ft_iter1000/model_final.pth
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  MODEL.DiffusionDet.GEO_FEAT True \
  MODEL.WEIGHTS ${FT} \
  OUTPUT_DIR /dev/shm/diffdet_eval_baseline_seed{TRAIN_SEED}_geofeat_ft_evalseed{EVAL_SEED} \
  SEED {EVAL_SEED}
```

结果（每个 train_seed 对 eval seeds=0..4 统计 mean/std）：

| train_seed | mean AP | std | per eval-seed AP（0..4） |
|---:|---:|---:|---|
| 0 | 47.1365 | 0.2482 | 47.3186 / 47.1416 / 47.3478 / 46.6628 / 47.2118 |
| 1 | 46.4122 | 0.8886 | 47.5261 / 46.5913 / 47.1693 / 45.4011 / 45.3732 |
| 2 | 46.7834 | 0.5812 | 46.0899 / 46.1012 / 47.5150 / 47.1078 / 47.1030 |

汇总：
- mean over train_seeds = **46.7774**（std across train_seed means = 0.2957）
- flatten all 15 runs mean=**46.7774**，std=0.6955

对比：vanilla baseline（3.10）mean over train_seeds=46.4293 → **+0.3481 AP**。

### 3.12 稳定性方案：D3PM+QHead warmstart（从 baseline 10k checkpoint 起步）

目标：专门解决 D3PM 的 **train_seed=0/2 掉点**（见 2.4.4），把 mean 拉回到 baseline 水平以上。

做法：不再从 ImageNet 训练 D3PM，而是从各自的 **baseline 10k checkpoint** 起步，启用 `LABEL_D3PM+QUALITY_HEAD` 后再 finetune。

#### 3.12.1 训练（+2500 iter，冻结 backbone 保稳）

```bash
cd baselines/DiffusionDet
for train_seed in 0 1 2; do
  python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 \
    SOLVER.MAX_ITER 2500 \
    SOLVER.CHECKPOINT_PERIOD 100000000 \
    SOLVER.BASE_LR 2.5e-6 \
    SOLVER.BACKBONE_MULTIPLIER 0.0 \
    MODEL.WEIGHTS /dev/shm/diffdet_baseline_iter10000_seed${train_seed}/model_final.pth \
    OUTPUT_DIR /dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed${train_seed}_iter2500 \
    SEED ${train_seed}
done
```

产物：
- `/dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed{0,1,2}_iter2500/model_final.pth`

#### 3.12.2 公平评测（eval seeds=0..4，SAMPLE_STEP=1）

```bash
cd baselines/DiffusionDet
for train_seed in 0 1 2; do
  CKPT=/dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed${train_seed}_iter2500/model_final.pth
  for s in 0 1 2 3 4; do
    python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 --eval-only \
      MODEL.DiffusionDet.SAMPLE_STEP 1 \
      MODEL.WEIGHTS ${CKPT} \
      OUTPUT_DIR /dev/shm/diffdet_eval_d3pm_qhead_warmstart_baseline_seed${train_seed}_evalseed${s} \
      SEED ${s}
  done
done
```

结果（每行 mean/std 都是 **对 eval seeds=0..4** 统计）：

| train_seed | mean AP | std | per eval-seed AP（0..4） |
|---:|---:|---:|---|
| 0 | 47.4814 | 0.3443 | 48.0077 / 46.9841 / 47.3221 / 47.4153 / 47.6778 |
| 1 | 46.1575 | 0.5623 | 46.2323 / 46.0714 / 47.1625 / 45.4819 / 45.8392 |
| 2 | 47.0817 | 0.2049 | 47.0137 / 47.0243 / 47.4278 / 47.1414 / 46.8012 |

汇总：
- mean over train_seeds = **46.9068**（std across train_seed means = 0.5545）
- flatten all 15 runs mean=**46.9068**，std=0.6829

解读：
- **train_seed=0/2 的掉点被“抬回去了”**（从 44/42 → 47+），整体均值已超过 vanilla baseline（46.4293）与 baseline+GEO_FEAT（46.7774）。
- train_seed=1 这条 warmstart 路线不如“从头训 D3PM seed1”的峰值强，但作为“保稳配方”整体更均衡。

---

## 4. Phase 3（Guidance & Distillation）— 下一步怎么做

你之前要做的是：固定同一权重，只改推理参数 `QUALITY_GUIDANCE_SCALE=0.05~0.1` 做 3–5 seed 复验。

### 4.1 前置条件（必须满足）

Quality guidance 只有在 **训练时启用了 quality head** 的 checkpoint 上才有效：
- `MODEL.DiffusionDet.QUALITY_HEAD=True`（checkpoint 里会带 `quality_head` 参数）

当前 best 的 D3PM(sampletrain) checkpoint **没有 quality head**；要做 Phase 3 有两条路：

**路 A（最快）**：用你现有的 quality-head checkpoint 做推理参数 sweep：
- 例如：`/tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed/model_final.pth`

**路 B（更一致）**：在 D3PM(sampletrain) 的基础上，加上 QUALITY_HEAD 再训练/finetune（建议等 Phase 2 稳后再做）。

### 4.2 路 A：对现有 quality-head checkpoint 做 guidance scale sweep

1) 先确认 checkpoint 真的有 quality_head（只读）：

```bash
python - <<'PY'
import torch
sd = torch.load('/tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed/model_final.pth', map_location='cpu')['model']
print(any('quality_head' in k for k in sd))
PY
```

2) eval-only（seed 固定，同一权重，只改 `QUALITY_GUIDANCE_SCALE`）：

```bash
cd baselines/DiffusionDet
CKPT=/tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed/model_final.pth
for scale in 0.0 0.05 0.1; do
  for s in 0 1 2 3 4; do
    python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 --eval-only \
      MODEL.DiffusionDet.SAMPLE_STEP 1 \
      MODEL.DiffusionDet.QUALITY_HEAD True \
      MODEL.DiffusionDet.QUALITY_GUIDANCE_SCALE ${scale} \
      MODEL.WEIGHTS ${CKPT} \
      OUTPUT_DIR /tmp/diffdet_eval_qguidance_scale${scale}_seed${s} \
      SEED ${s}
  done
done
```

**通过标准**：
- `scale=0.05` 或 `0.1` 的 mean AP ≥ `scale=0.0`
- 如果 mean 提升但 std 明显变大，再考虑调 `QUALITY_GUIDANCE_GRAD_NORM`/`QUALITY_GUIDANCE_TOPK`（见配置项）。

### 4.3 已跑结果（本次 sweep，固定同一权重）

固定 checkpoint：
- `/tmp/diffdet_quality_head_repeat0p02_iter2500_seed42_v8_fixed/model_final.pth`

eval 输出目录（每个目录都有 `log.txt` 和 `inference/coco_instances_results.json`）：
- `/dev/shm/diffdet_eval_qguidance_sweep_iter2500_seed42/scale{SCALE}_seed{SEED}/`

口径：`SAMPLE_STEP=1`，seeds=`0,1,2,3,4`。

| QUALITY_GUIDANCE_SCALE | mean AP | std | 备注 |
|---:|---:|---:|---|
| 0.0 | 19.7760 | 0.7536 | baseline（该 checkpoint） |
| 0.05 | 19.7658 | 0.7554 | 几乎无差异 |
| 0.1 | 19.7720 | 0.7496 | 几乎无差异 |

结论：在 `0.05~0.1` 这个范围内，**quality guidance 对总体 AP 没有带来可见提升**（变化 < 0.02 AP）。

### 4.4 路 B：把 QUALITY_HEAD 接到“最强 D3PM(sampletrain)”再复验 guidance（已执行）

> 目的：让 quality head 训练在更强的 detector 上，再看 guidance 是否能带来增益。

warmstart checkpoint（无 quality head）：
- `/dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth`

finetune（启用 QUALITY_HEAD，尽量只让 quality 头学习：`BASE_LR` 很小、`QUALITY_HEAD_LR_MULT` 很大）：

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k.yaml --num-gpus 1 \
  SOLVER.MAX_ITER 1000 \
  SOLVER.CHECKPOINT_PERIOD 100000000 \
  SOLVER.BASE_LR 2.5e-6 \
  MODEL.WEIGHTS /dev/shm/diffdet_label_d3pm_mask_iter2500_seed42_sampletrain/model_final.pth \
  MODEL.DiffusionDet.LABEL_STATE True \
  MODEL.DiffusionDet.LABEL_D3PM True \
  MODEL.DiffusionDet.LABEL_D3PM_KERNEL mask \
  MODEL.DiffusionDet.LABEL_D3PM_USE_DISTRIBUTION True \
  MODEL.DiffusionDet.QUALITY_HEAD True \
  MODEL.DiffusionDet.QUALITY_HEAD_LR_MULT 100.0 \
  OUTPUT_DIR /dev/shm/diffdet_label_d3pm_mask_sampletrain_qhead_ft_iter1000_seed42 \
  SEED 42
```

finetune 产物：
- checkpoint：`/dev/shm/diffdet_label_d3pm_mask_sampletrain_qhead_ft_iter1000_seed42/model_final.pth`

guidance sweep（固定同一权重，只改推理参数）：
- 输出目录：`/dev/shm/diffdet_eval_d3pm_qhead_guidance_sweep_iter1000_seed42/scale{SCALE}_seed{SEED}/`
- 口径：`SAMPLE_STEP=1`，seeds=`0,1,2,3,4`

| QUALITY_GUIDANCE_SCALE | mean AP | std | 相对 scale=0 |
|---:|---:|---:|---:|
| 0.0 | 30.0186 | 0.4548 | +0.0000 |
| 0.05 | 30.0672 | 0.4290 | +0.0486 |
| 0.1 | 30.1000 | 0.4402 | +0.0814 |

结论：
- 在更强的 D3PM 上，`QUALITY_GUIDANCE_SCALE=0.1` 带来 **小但稳定** 的总体 AP 增益（约 +0.08 AP，5 seeds 一致为正）。
- 目前可默认把推理参数定为：`QUALITY_GUIDANCE_SCALE=0.1`（如果后续做更大规模训练/更多类，再重新 sweep 一次）。

### 4.5 在完整 10k checkpoint 上再做一次 sweep（已执行）

固定 checkpoint：
- `/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed42/model_final.pth`

eval 输出目录：
- `/dev/shm/diffdet_eval_d3pm_qhead_iter10000_guidance_sweep_seed42/scale{SCALE}_seed{SEED}/`

口径：`SAMPLE_STEP=1`，seeds=`0,1,2,3,4`。

| QUALITY_GUIDANCE_SCALE | mean AP | std | 相对 scale=0 |
|---:|---:|---:|---:|
| 0.0 | 46.1882 | 0.4964 | +0.0000 |
| 0.1 | 46.2052 | 0.4476 | +0.0170 |

结论：10k 训练后的模型上，`scale=0.1` 的提升 **非常小且不稳定**（更像采样噪声范围内波动）；默认可继续用 `scale=0.0`（或保留 `0.1` 但不作为关键增益点）。

### 4.6 （补充）把 `SAMPLE_STEP` 提到 5，复验 “跨 time steps guidance”（已执行）

> 背景：`SAMPLE_STEP=1` 时 diffusion 只有 1 个采样步，`QUALITY_GUIDANCE_MODE=final/all` 实际等价；要验证“跨 time steps”必须让 `SAMPLE_STEP>1`。

固定 checkpoint：
- `/dev/shm/diffdet_d3pm_mask_dist_qhead_iter10000_seed42/model_final.pth`

口径：
- `SAMPLE_STEP=5`
- `QUALITY_GUIDANCE_MODE` ∈ {`final`,`all`}
- `QUALITY_GUIDANCE_LANGEVIN_STEPS` ∈ {1,2}
- `QUALITY_GUIDANCE_SCALE` ∈ {0.0,0.05,0.1}
- eval seeds=`0..4`

产物：
- 逐条结果：`guidance_sweep_results_qhead_seed42_step5.tsv`
- eval 输出目录：`/dev/shm/diffdet_eval_guidance_sweep_qhead_seed42_step5/step5_mode{mode}_k{k}_scale{scale}_seed{seed}/`

核心结论（只看总体 AP）：
- **仅把 `SAMPLE_STEP` 从 1 提到 5，本身就把 mean AP 从 ~46.19 提升到 ~48.54（+2.35 AP）**；这是目前最显著的“推理参数增益”。
- 在 `SAMPLE_STEP=5` 下，guidance 的增益非常小：最佳组合 `mode=all,k=2,scale=0.1` 仅比 `scale=0.0` 高 **+0.025 AP**（5 seeds）。
- 注意一个实现细节：当 `SAMPLE_STEP>1` 时，当前推理会走 **ensemble 分支**，且最后一个采样步（`time_next<0`）不会被加入 ensemble，因此：
  - `QUALITY_GUIDANCE_MODE=final`（只在最后一步施加）对最终输出 **基本无效**；
  - 要让 guidance 影响输出，优先用 `mode=all`（每个采样步都施加），或后续考虑改造 ensemble 逻辑。

数值摘要（eval seeds=0..4）：

| SAMPLE_STEP | mode | k | scale | mean AP | std |
|---:|---:|---:|---:|---:|---:|
| 5 | final | 1 | 0.0 | 48.5373 | 0.2462 |
| 5 | all | 2 | 0.1 | 48.5622 | 0.2813 |

### 4.7 （补充）`SAMPLE_STEP` 对总体 AP 的影响（已执行：baseline vs warmstart）

> 目的：既然 `SAMPLE_STEP` 对 AP 提升显著，就把它纳入“主口径”里，避免后续口径混淆。

#### 4.7.1 vanilla baseline（train seeds=0/1/2）在 `SAMPLE_STEP=5` 的总体 AP

结果表：
- `baseline_step5_results.tsv`（共 15 条：train_seed×eval_seed）

汇总（train_seeds=0/1/2 × eval_seeds=0..4）：
- mean AP = **48.2897**，std = **0.6804**

#### 4.7.2 D3PM+QHead warmstart（从 baseline 起步 +2500 iter）在 `SAMPLE_STEP=5`

结果表：
- `d3pm_qhead_warmstart_step5_results.tsv`（共 15 条）

汇总（train_seeds=0/1/2 × eval_seeds=0..4）：
- mean AP = **48.5016**，std = **0.6158**

结论：在 `SAMPLE_STEP=5` 下，warmstart 版本相对 vanilla baseline **+0.21 AP**，且方差略小。

#### 4.7.3 进一步把 `SAMPLE_STEP` 提到 10（只测最强 ckpt，追 AP 上限）

ckpt：
- `/dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed0_iter2500/model_final.pth`

结果表：
- `warmstart_seed0_step10_results.tsv`（eval seeds=0..4）

汇总（eval seeds=0..4）：
- mean AP = **49.2828**，std = **0.3333**

结论：`SAMPLE_STEP=10` 相对 `SAMPLE_STEP=5`（同 ckpt）再提升约 **+0.22 AP**，但推理成本也更高（速度约线性变慢）。

#### 4.7.4 继续提高到 `SAMPLE_STEP=20`（已执行）

结果表：
- `warmstart_seed0_step20_results.tsv`（eval seeds=0..4）

汇总（eval seeds=0..4）：
- mean AP = **49.6059**，std = **0.4056**

结论：`SAMPLE_STEP=20` 相对 `SAMPLE_STEP=10` 再提升约 **+0.32 AP**；是否值得取决于你的推理时延预算。

#### 4.7.5 速度/精度折中（同一最强 ckpt：warmstart seed0）

ckpt：
- `/dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed0_iter2500/model_final.pth`

| SAMPLE_STEP | mean AP（eval seeds=0..4） | 推理秒/图（mean） | 相对 step1 速度 | 结果表 |
|---:|---:|---:|---:|---|
| 1 | 47.4822 | 0.0359 | 1.0× | `warmstart_seed0_step1_results.tsv` |
| 5 | 49.0658 | 0.1042 | 2.9× | `d3pm_qhead_warmstart_step5_results.tsv`（train_seed=0 子集） |
| 10 | 49.2828 | 0.1903 | 5.3× | `warmstart_seed0_step10_results.tsv` |
| 20 | 49.6059 | 0.3614 | 10.1× | `warmstart_seed0_step20_results.tsv` |
| 50 | 49.2875 | 0.8891 | 24.8× | `warmstart_seed0_step50_results.tsv` |

---

## 6. Phase 3：step20 → step1 的“真正加速蒸馏”（已补齐入口）

目标：用 teacher 的多步采样（例如 `SAMPLE_STEP=20`）作为监督，把 student 的推理压到 `SAMPLE_STEP=1`。

新增配置：
- `baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_20to1.yaml`

关键点：
- teacher 会在构建时把 `SAMPLE_STEP` 设置为 `SAMPLER_DISTILL_TEACHER_SAMPLE_STEP`（默认 20）；
- 默认会把 teacher 的 `use_ensemble/box_renewal` 关掉，以保持 proposal identity 稳定，便于做 per-proposal distill；
- distill loss 出现在训练日志里：`loss_sampler_distill_box` / `loss_sampler_distill_cls`。

Smoke 命令（只验收链路，50 iter）：

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_20to1.yaml --num-gpus 1 \
  MODEL.WEIGHTS /dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed0_iter2500/model_final.pth \
  MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS /dev/shm/diffdet_d3pm_qhead_warmstart_baseline_seed0_iter2500/model_final.pth \
  SOLVER.MAX_ITER 50 SOLVER.CHECKPOINT_PERIOD 100000000 TEST.EVAL_PERIOD 100000000 \
  OUTPUT_DIR /dev/shm/diffdet_sampler_distill_smoke_iter50_seed0 SEED 0
```

本次实测（train_seed=0）：
- smoke（50 iter）：`/dev/shm/diffdet_sampler_distill_smoke_iter50_seed0/`（目录可能已清理/重启丢失，可按本节命令重跑）
  - `log.txt` 中出现 `loss_sampler_distill_box`（链路 OK）
  - eval-only（自动跑）`copypaste: 46.2275,...`
- 注意：默认 LR（2.5e-5）+ 全 proposal 蒸馏会明显掉点（`/dev/shm/diffdet_sampler_distill_20to1_iter500_seed0/`，目录可能已清理；eval_seed=0 时 `AP≈41.38`）。
- 稳定版（teacher `eta=0` + `topk=100` + `BASE_LR=2.5e-6`）：
  - checkpoint：`/dev/shm/diffdet_sampler_distill_20to1_iter500_seed0_stable/model_final.pth`
  - eval-only（`SAMPLE_STEP=1`，eval seeds=0..4）：mean AP = **47.5271**，std = **0.6677**
  - 结果表：`sampler_distill_20to1_seed0_iter500_stable_step1_results.tsv`
- 续跑（`--resume`）到 2500 iter（同一目录覆盖 `model_final.pth`）：
  - checkpoint：`/dev/shm/diffdet_sampler_distill_20to1_iter500_seed0_stable/model_final.pth`
  - eval-only（`SAMPLE_STEP=1`，eval seeds=0..4）：mean AP = **49.4829**，std = **0.4277**
  - 结果表：`sampler_distill_20to1_seed0_iter2500_stable_step1_results.tsv`
- 同配方扩展到 train_seed=1/2（各自从对应 warmstart ckpt 起步，训练 2500 iter）：
  - train_seed=1：mean AP = **46.7410**，std = **0.6178**（结果表：`sampler_distill_20to1_seed1_iter2500_stable_step1_results.tsv`）
  - train_seed=2：mean AP = **48.0701**，std = **0.9205**（结果表：`sampler_distill_20to1_seed2_iter2500_stable_step1_results.tsv`）
  - 汇总（train_seeds=0/1/2 × eval_seeds=0..4）：mean AP = **48.0980**，std = **1.3130**

- （推荐）在 stable student 上追加 `GEO_FEAT` finetune（+1k iter，用于把 step1 表现“拉齐并降方差”）：
  - 配方：`GEO_FEAT=True`、`GEO_FEAT_SCHEDULE=constant`、`BASE_LR=2.5e-6`、`BACKBONE_MULTIPLIER=0.0`、`GEO_FEAT_LR_MULT=150`、`MAX_ITER=1000`
  - train_seed=1：mean AP = **47.4178**，std = **0.3882**（结果表：`sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`；ckpt：`baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`）
  - train_seed=2：mean AP = **48.7277**，std = **0.7317**（结果表：`sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`；ckpt：`baselines/checkpoints/sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`）
  - train_seed=0：mean AP = **49.0169**，std = **0.4120**（结果表：`sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`；略降，所以 seed0 建议保持 stable 不动）
  - 推荐汇总（seed0 用 stable；seed1/2 用 geofeat_ft）：mean AP = **48.5428**，std = **1.0087**
  - train_seed=1 的 `GEO_FEAT_LR_MULT` 小 sweep（同样 `MAX_ITER=1000`；eval seeds=0..4）：
    - `LR_MULT=100`：mean AP = **47.4932**，std = **0.5409**（结果表：`sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult100_step1_results.tsv`）
    - `LR_MULT=150`：mean AP = **47.4178**，std = **0.3882**（结果表：`sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`）
    - `LR_MULT=200`：mean AP = **46.7669**，std = **0.4305**（结果表：`sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult200_step1_results.tsv`）
    - 汇总表：`sampler_distill_20to1_seed1_stable_geofeat_lrmult_sweep_iter1000_step1_summary.tsv`（结论：更偏“降方差”用 150；只看均值用 100）
  - （未采用）重跑 train_seed=1 的 sampler distill（teacher 改为 warmstart seed0）：mean AP = **46.2054**，std = **0.8526**（结果表：`sampler_distill_20to1_seed1_teacher_seed0warmstart_iter2500_step1_results.tsv`；比 stable seed1 更差）

## 5. 空间与“卡住”规避清单（必看）

### 5.1 避免卡住（大目录）

- 不要对 `baselines/data/.../images` 这类目录做 `find`/`ls -R`/`du -a`。
- 只读 JSON：`baselines/data/repro_10k/annotations/*.json`

### 5.2 根盘快满：建议的工作流

- 新训练：`OUTPUT_DIR=/dev/shm/...`（不占根盘）
- 只把 best checkpoint 复制到 `baselines/checkpoints/`（最多留 1~2 个）

查看 `/tmp` 里有多少大 checkpoint：

```bash
find /tmp -maxdepth 2 -path '/tmp/diffdet*' -name 'model_final.pth' -printf '%s %p\n' 2>/dev/null | sort -n | tail
```

---
