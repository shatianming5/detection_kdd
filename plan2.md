# plan2.md — 实验台账（Single Source of Truth）+ 下一步执行清单

更新：2025-12-21

目标：把当前仓库里“已经跑过/已经验证过/已经证伪”的所有关键实验，用**同一口径**汇总成一份可执行、可复验、可追溯的台账与下一步计划，避免结果口径混乱。

> 说明：`plan.md` 仍保留作为“长文说明/命令模板/历史笔记”，且你之前引用过 `plan.md:43 / plan.md:903` 的行号；为避免行号漂移，本文件 `plan2.md` 作为新的主索引与总计划，不再依赖 `plan.md` 的行号引用。

---

## 0. 现在的处境（3 句话）

1) **A 复现链路已确认可跑通**（`run_baselines.sh` 口径一致），所以现在的问题不是“环境/数据/代码跑不起来”，而是“哪条改动/哪份权重在什么口径下更好”。  
2) 当前主战场是 **Phase 3：step20 → step1 的真正加速**（sampler distill 20→1），并用 **Phase 2 的 GEO_FEAT finetune** 在 step1 上做“拉齐均值、降方差”。  
3) 目前已经有两条“可交付”的 step1 口径结论：**stable distill 已扩到 train_seed=0..4**；且 **seed0 的 GEO_FEAT ft 通过 `LR_MULT=25` 可不掉点甚至小幅增益**（seed1/2 继续用 GEO_FEAT ft 拉齐）。

对外冻结交付已生成：
- `deliverables/step1_5seed_geofeat_mix/manifest.tsv`
- `results/final_step1_5seed_geofeat_mix.tsv`

---

## 1. 统一口径（所有表与结论都按这里算）

### 1.1 数据与指标

- 数据集：Repro-10k COCO（`baselines/data/repro_10k`，3 类：car / motorbike / person）。
- `person` 在 val 中极少/为 0，COCOeval 的 per-class 可能出现 `nan`，不作为失败标准。
- **DiffusionDet 的 bbox AP 是 0~100 标尺**（如 46.7 就是 46.7）。  
- DETR 在日志里常见 0~1（如 0.54 对应 54 AP）。

### 1.2 严格 train_seed × eval_seed 分离（核心）

为了避免“训练过程推进随机数状态导致的误判”，所有对比均按：

- `train_seed`：训练随机种子（决定训练轨迹）
- `eval_seed`：推理采样随机种子（决定采样方差）
- 复验：固定 checkpoint，遍历 `eval_seed=0..4`，只在推理端改 `SEED`，得到 5 个 AP，计算 mean/std。

### 1.3 step1 统一口径

- 推理：`MODEL.DiffusionDet.SAMPLE_STEP=1`
- 评估：`--eval-only`
- 结果抽取：看 `log.txt` 里的 `copypaste:` 第一项（AP）

---

## 2. 证据链（哪些文件是“权威证据”）

### 2.1 现存 checkpoint（`baselines/checkpoints/`）

> 只列“当前仍有价值/仍在使用”的权重；其它临时权重一般在 `/dev/shm`，已按策略清理。

- 10k baseline（DiffusionDet）：`baseline_iter10000_seed{0,1,2}.pth`
- 10k D3PM(mask, dist)+QHead warmstart 起点：`d3pm_qhead_warmstart_baseline_seed{0,1,2,3,4}_iter2500.pth`
- warmstart + GEO_FEAT finetune（D3PM+QHead warmstart，Phase2）：  
  - `d3pm_qhead_warmstart_geofeat_ft_seed0_iter1000*.pth`  
  - `d3pm_qhead_warmstart_geofeat_ft_seed1_iter1000*.pth`  
  - `d3pm_qhead_warmstart_geofeat_ft_seed2_iter1000*.pth`
- sampler distill（Phase3，加速 step20→step1）：  
  - stable student：`sampler_distill_20to1_seed{0,1,2,3,4}_iter2500_stable.pth`
  - stable + GEO_FEAT(1k)（当前推荐；按 seed 选 LR_MULT）：  
    - `sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth`  
    - `sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`  
    - `sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`
    - `sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth`  
    - `sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`

### 2.2 结果 TSV（`results/*.tsv`）

这些 TSV 是当前最“硬”的可复验证据（每行都是一次 eval-only 的结果；可用脚本重新汇总）。

#### Baseline / D3PM / warmstart（step1 & step5）

- `results/baseline_step1_results.tsv`：10k baseline，train_seed=0/1/2 × eval_seed=0..4（step1）  
  - flatten mean/std(15 runs) = **46.4293 / 0.7817**
- `results/baseline_step5_results.tsv`：同上（step5）  
  - flatten mean/std(15 runs) = **48.2897 / 0.6804**
- `results/d3pm_qhead_step1_results.tsv`：D3PM+QHead（非 warmstart）  
  - flatten mean/std = **44.8068 / 2.0555**（方差大且均值低）
- `results/d3pm_qhead_step5_results.tsv`：同上（step5）  
  - flatten mean/std = **46.8046 / 1.8905**
- `results/d3pm_qhead_warmstart_step1_results.tsv`：D3PM+QHead warmstart baseline（seed0/1/2）  
  - flatten mean/std = **46.9069 / 0.6829**
- `results/d3pm_qhead_warmstart_step5_results.tsv`：同上（step5）  
  - flatten mean/std = **48.5016 / 0.6158**
- `results/d3pm_qhead_warmstart_seed3_step1_results.tsv`：warmstart baseline（train_seed=3，eval_seed=0..4）  
  - mean/std = **47.5083 / 0.4325**
- `results/d3pm_qhead_warmstart_seed4_step1_results.tsv`：warmstart baseline（train_seed=4，eval_seed=0..4）  
  - mean/std = **47.3521 / 0.2592**

#### warmstart seed0 的 step 数/速度折中（单 train_seed=0，eval_seed=0..4）

- `results/warmstart_seed0_step1_results.tsv`：mean/std = **47.4822 / 0.3432**
- `results/warmstart_seed0_step5_results.tsv`：mean/std = **49.0658 / 0.4201**
- `results/warmstart_seed0_step10_results.tsv`：mean/std = **49.2828 / 0.3333**
- `results/warmstart_seed0_step20_results.tsv`：mean/std = **49.6059 / 0.4056**
- `results/warmstart_seed0_step50_results.tsv`：mean/std = **49.2875 / 0.3312**

#### Phase2：GEO_FEAT warmstart sweep（train_seed=1）

- `results/phase2_geofeat_sweep_seed1_iter1000_step1_results.tsv`：best = `lrmult150` mean/std **47.2633 / 0.3848**
- `results/phase2_geofeat_sweep_seed1_iter1500_step1_results.tsv`：best = `lrmult100` mean/std **47.2753 / 0.7134**
- `results/phase2_geofeat_sweep_seed1_iter2000_step1_results.tsv`：整体偏弱（46.64~46.88）

#### Phase3：Quality guidance / sweep（历史记录）

- `results/guidance_sweep_results_qhead_seed42.tsv`：step1 下的 guidance sweep（含多个 eval_seed）
- `results/guidance_sweep_results_qhead_seed42_step5.tsv`：step5 下的 guidance sweep（含多个 eval_seed）

#### Phase3：sampler distill（step20→step1）

stable student（eval_seed=0..4）：

- `results/sampler_distill_20to1_seed0_iter2500_stable_step1_results.tsv`：mean/std **49.4829 / 0.4277**
- `results/sampler_distill_20to1_seed1_iter2500_stable_step1_results.tsv`：mean/std **46.7410 / 0.6178**
- `results/sampler_distill_20to1_seed2_iter2500_stable_step1_results.tsv`：mean/std **48.0701 / 0.9205**
- `results/sampler_distill_20to1_seed3_iter2500_stable_step1_results.tsv`：mean/std **48.5907 / 0.3326**
- `results/sampler_distill_20to1_seed4_iter2500_stable_step1_results.tsv`：mean/std **47.6969 / 0.5274**
- 汇总（seed0/1/2 × eval0..4）：mean/std **48.0980 / 1.3130**
- 汇总（seed0..4 × eval0..4）：mean/std **48.1163 / 1.0920**

stable + GEO_FEAT finetune（`MAX_ITER=1000, BASE_LR=2.5e-6, BACKBONE_MULTIPLIER=0.0`）：

- `results/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv`：mean/std **49.5638 / 0.2476**（✅ 推荐；已固化 ckpt）
- `results/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult50_step1_results.tsv`：mean/std **49.4513 / 0.1524**（均值略降但方差更低）
- `results/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult100_step1_results.tsv`：mean/std **49.3992 / 0.2449**（均值略降）
- `results/sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **47.4178 / 0.3882**（已固化 ckpt）
- `results/sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **48.7277 / 0.7317**（已固化 ckpt）
- `results/sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv`：mean/std **48.8636 / 0.3135**（✅ 推荐；均值更高）
- `results/sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **48.6745 / 0.1742**（备选；方差更低）
- `results/sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **48.1256 / 0.2314**（✅ 推荐）
- `results/sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv`：mean/std **47.7380 / 0.5768**（不推荐）
- `results/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **49.0169 / 0.4120**（历史记录；掉点，不推荐）
- seed1 `GEO_FEAT_LR_MULT` sweep 汇总：`results/sampler_distill_20to1_seed1_stable_geofeat_lrmult_sweep_iter1000_step1_summary.tsv`
  - `lrmult100`：mean/std **47.4932 / 0.5409**
  - `lrmult150`：mean/std **47.4178 / 0.3882**
  - `lrmult200`：mean/std **46.7669 / 0.4305**
- 推荐汇总（seed0 g25 / seed1 g150 / seed2 g150 / seed3 g25 / seed4 g150；train_seed=0..4 × eval_seed=0..4）：mean/std **48.5397 / 0.8388**

失败/证伪的 distill 尝试：

- `results/sampler_distill_20to1_seed1_teacher_geofeat150_iter2500_step1_results.tsv`：mean/std **46.4670 / 0.9310**（teacher=geo_feat，掉点）
- `results/sampler_distill_20to1_seed1_teacher_seed0warmstart_iter2500_step1_results.tsv`：mean/std **46.2054 / 0.8526**（teacher 换 seed0 warmstart 更差）
- distill loss sweep（`SAMPLER_DISTILL_CLS_WEIGHT`）：
  - `results/sampler_distill_20to1_seed1_iter2500_cls0p1_step1_results.tsv`：mean/std **46.0795 / 0.9304**
  - `results/sampler_distill_20to1_seed1_iter2500_cls0p1_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **47.2258 / 0.7211**
  - `results/sampler_distill_20to1_seed1_iter2500_cls0p2_step1_results.tsv`：mean/std **46.0946 / 0.9961**
  - `results/sampler_distill_20to1_seed1_iter2500_cls0p2_geofeat_ft_iter1000_lrmult150_step1_results.tsv`：mean/std **47.4020 / 0.5271**
- distill knob（`SAMPLER_DISTILL_TOPK`）：
  - `results/sampler_distill_20to1_seed1_iter2500_topk50_step1_results.tsv`：mean/std **46.1499 / 0.6418**（比 stable seed1 更差；停止该方向）
  - `results/sampler_distill_20to1_seed1_iter2500_topk150_step1_results.tsv`：mean/std **45.9879 / 1.0463**（更差；停止该方向）
  - `results/sampler_distill_20to1_seed1_iter2500_topk200_step1_results.tsv`：mean/std **46.0025 / 0.9646**（更差；停止该方向）
- distill knob（`SAMPLER_DISTILL_BOX_WEIGHT`）：
  - `results/sampler_distill_20to1_seed1_iter2500_boxw0p5_step1_results.tsv`：mean/std **45.8648 / 1.0076**（更差；停止该方向）
  - `results/sampler_distill_20to1_seed1_iter2500_boxw2p0_step1_results.tsv`：mean/std **46.3139 / 0.7068**（仍低于 stable seed1=46.7410；停止该方向）

---

## 3. 实验台账（按阶段归档；每条给出“结论/证据/状态”）

### 3.1 A：baseline 复现验收（整条链路）

1) 历史 baseline（可追溯输出）  
- 证据：`PROGRESS.md:25`（`baselines/output/20251212_225009/`）  
- 结论：DiffusionDet bbox AP=45.496；DETR bbox AP=0.5457（54.57）

2) 最近一次 A（跑在 `/dev/shm`，已清理输出，仅保留本台账记录）  
- 命令：`OUTPUT_BASE=/dev/shm bash run_baselines.sh`  
- 结论（同口径）：DiffusionDet baseline AP≈45.625；D3PM+QHead AP≈45.057；DETR AP≈0.540  
- 状态：✅ 证明“当前环境+代码”仍可复现

### 3.2 Phase 1：几何注入（GEO_BIAS / GEO_FEAT）——结论：从零训练不稳，warmstart 可用但方差大

证据来源：`PROGRESS.md:44` 起的整段实验记录。

- 从零训练期开启 `GEO_FEAT`：短跑 2500 iter 多次出现 AP≈0/极低（收敛被阻断）  
  - 代表结果：`PROGRESS.md:223`（baseline AP=17.61；GEO_FEAT 多配置 AP≈0~1）
  - 结论：❌ 不建议继续“从零训练期开启 GEO_FEAT”的路线
- warmstart（从已收敛 10k baseline finetune）：可小幅提点，但多 seed 方差大  
  - 代表结果：`PROGRESS.md:246`（best 单点可到 47.02；seed0..7 mean=46.95 std=0.90）  
  - 结论：✅ 作为“可用路线”；但要按多 seed/均值口径验收

### 3.3 Phase 2(A)：label_state（unk 吸收态）——结论：不掉点（均值几乎不变）

证据来源：`PROGRESS.md:479`。

- 2500 iter（seed42 训练）+ eval-only 多 seed 对照：ΔAP mean=-0.0232，std=0.8653  
  - 结论：✅ “不掉点/链路可控”验收通过；单次 eval 波动大，必须多 seed

### 3.4 Phase 2（离散 label diffusion / D3PM）与 warmstart

证据来源：根目录 TSV（见 2.2）+ `plan.md` 的命令模板。

- 非 warmstart 的 D3PM+QHead：均值低且方差大（step1：44.81±2.06）  
  - 结论：❌ 不作为主线
- warmstart 的 D3PM+QHead：step1 接近 baseline，step5 明显更强（48.50±0.62）  
  - 结论：✅ 作为“高精度/可提步数”的 teacher 候选
- warmstart(seed0) step 数折中：step20 均值约 49.61（单 seed0）  
  - 结论：✅ 作为 distill teacher（step20）很合理（比 step1 高）

### 3.5 Phase 3：Quality head / guidance

证据来源：`PROGRESS.md:313` 起 + `results/guidance_sweep_results_qhead_seed42*.tsv`。

- 质量头工程接入已完成，并定位修复过“掉点/AP=0/无预测”等问题（详见 PROGRESS）  
- 更稳定的落地收益：`QUALITY_SCORE_REWEIGHT=True`（见 `PROGRESS.md:470`）在多 seed 下稳定增益  
- guidance sweep 已留表，但当前主线优先级低于 distill（因为 distill 直接解决推理成本）

### 3.6 Phase 3：sampler distill（step20→step1 真加速）——当前主线

证据来源：`plan.md` Phase 3 章节 + distill TSV + 已固化的 checkpoints。

1) stable 配方（teacher eta=0, topk=100, base_lr=2.5e-6）已跑通，并在 seed0 上非常强：49.48±0.43  
2) stable distill 已扩展到 train_seed=0..4（eval_seed=0..4）：flatten mean/std **48.1163 / 1.0920**（见 2.2）。  
3) seed1/seed2 的 stable student 波动更大/均值偏低，因此引入 GEO_FEAT finetune 拉齐：  
   - seed1：46.74±0.62 → 47.42±0.39（lrmult150, +1k）  
   - seed2：48.07±0.92 → 48.73±0.73（lrmult150, +1k）  
4) seed0 的 GEO_FEAT 之前用 `lrmult150` 会掉点，但 `lrmult25` 可以不掉点并小幅增益：49.56±0.25（已固化 ckpt）。  
5) “改 teacher”与 “CLS_WEIGHT sweep”都更差，已证伪（见 2.2）。  

**当前推荐交付组合（step1）：**  
- seed0：`baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth`  
- seed1：`baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`（若只看均值，可考虑 lrmult100，但方差更大）  
- seed2：`baselines/checkpoints/sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`  
- seed3：`baselines/checkpoints/sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth`（均值更高；追方差更低可用 lrmult150）  
- seed4：`baselines/checkpoints/sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth`

冻结交付（对外口径）：
- 清单与 sha256：`deliverables/step1_5seed_geofeat_mix/manifest.tsv`
- 合并 25-run 表：`results/final_step1_5seed_geofeat_mix.tsv`

---

## 4. 为什么你会觉得“结果不明确”（我们需要消除的歧义）

1) **不同实验的口径不同**：2500 iter 短跑 / 10k 长跑 / warmstart finetune / distill student —— 这些不能直接横比。  
2) **采样方差显著**：step1 采样下单次 eval 波动大，必须按 `eval_seed=0..4` 取均值，必要时再扩展更多 eval_seed。  
3) **证据强弱不一致**：有些结果写在文档里但输出目录已清理；以后需要做到“每个结论都有 TSV + ckpt 指针”。  

因此本文件把“可复验 TSV”作为主证据，并把“仅文档描述/已清理输出”的条目标注为历史结论。

---

## 5. 下一步应该怎么做（按优先级；先澄清不确定性，再追收益）

### 5.1 统一“结果落盘”规范（强烈推荐）

- 目标：以后任何新实验必须同时产出：  
  1) `baselines/checkpoints/<name>.pth`（只保留 best/最终）  
  2) `results/*.tsv`（包含 eval_seed、AP、out_dir、以及 ckpt 的名字/哈希）  
- 通过标准：任何结论都可以仅靠 `ckpt + tsv` 被复验，不依赖 `/dev/shm` 临时目录。

已落地工具（现在开始所有新实验都用它们来落盘）：
- 固化 checkpoint + 生成 sha256：`scripts/finalize_checkpoint.py`
- 严格 eval_seed=0..4 的 eval-only 并写 TSV（含 ckpt sha256）：`scripts/eval_multiseed.py`
- 汇总现有 TSV → 生成总索引：`scripts/build_results_manifest.py`（产物：`results_manifest.tsv`）

已完成冻结交付（对外口径）：
- `deliverables/step1_5seed_geofeat_mix/manifest.tsv`
- `results/final_step1_5seed_geofeat_mix.tsv`

### 5.2 distill 的下一轮优化（只动一个旋钮）

已执行（train_seed=1；eval_seed=0..4；其它保持 stable 配方不动）：

- `CLS_WEIGHT=0.1`：mean/std **46.0795 / 0.9304**（ckpt：`baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p1.pth`）  
  - + GEO_FEAT(+1k,lrmult150)：mean/std **47.2258 / 0.7211**
- `CLS_WEIGHT=0.2`：mean/std **46.0946 / 0.9961**（ckpt：`baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p2.pth`）  
  - + GEO_FEAT(+1k,lrmult150)：mean/std **47.4020 / 0.5271**

结论：均未超过当前 best（seed1 stable+GEO_FEAT150：**47.4178 / 0.3882**），且方差更差 → ✅ **停止该方向**。

已执行（train_seed=1；eval_seed=0..4；其它保持 stable 配方不动）：

- `SAMPLER_DISTILL_BOX_WEIGHT=0.5`：mean/std **45.8648 / 1.0076**
- `SAMPLER_DISTILL_BOX_WEIGHT=2.0`：mean/std **46.3139 / 0.7068**

结论：均未超过 stable seed1（**46.7410 / 0.6178**），且不满足“distill 本体有提升再接 GEO_FEAT ft”的门槛 → ✅ **停止该方向**。

已执行（train_seed=1；eval_seed=0..4；其它保持 stable 配方不动）：

- `SAMPLER_DISTILL_TOPK=50`：mean/std **46.1499 / 0.6418**
- `SAMPLER_DISTILL_TOPK=150`：mean/std **45.9879 / 1.0463**
- `SAMPLER_DISTILL_TOPK=200`：mean/std **46.0025 / 0.9646**

结论：均未超过 stable seed1（TOPK=100；**46.7410 / 0.6178**）→ ✅ **停止 TOPK 方向**（保持 `TOPK=100`）。

### 5.3 GEO_FEAT 在 seed0 上“掉点”的归因/修复（可选）

如果你希望最终三 seed 都用同一“GEO_FEAT ft”策略（更一致），需要找 seed0 不适配的原因：

已执行（train_seed=0；从 `sampler_distill_20to1_seed0_iter2500_stable.pth` 起步；+1k iter；eval_seed=0..4）：

- `LR_MULT=25`：mean/std **49.5638 / 0.2476**（✅ 通过；ckpt：`baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth`）  
- `LR_MULT=50`：mean/std **49.4513 / 0.1524**（均值略降但方差更低）  
- `LR_MULT=100`：mean/std **49.3992 / 0.2449**（均值略降）

结论：seed0 GEO_FEAT 不适配并非“必然掉点”，主要是 `LR_MULT` 过大；当前最佳是 **lrmult25**。

### 5.4 扩展 train_seed（长期）

已执行（stable distill 20→1；新增 train_seed=3/4；eval_seed=0..4）：

- train_seed=3：`results/sampler_distill_20to1_seed3_iter2500_stable_step1_results.tsv` mean/std **48.5907 / 0.3326**  
- train_seed=4：`results/sampler_distill_20to1_seed4_iter2500_stable_step1_results.tsv` mean/std **47.6969 / 0.5274**

汇总（train_seed=0..4 × eval_seed=0..4）：flatten mean/std **48.1163 / 1.0920**（稳定性更可信，可用于对外口径）。

已执行“GEO_FEAT 拉齐”补齐（train_seed=3/4；固定配方 `MAX_ITER=1000, BASE_LR=2.5e-6, BACKBONE_MULTIPLIER=0.0`；eval_seed=0..4）：

- train_seed=3：`LR_MULT=25` mean/std **48.8636 / 0.3135**；`LR_MULT=150` mean/std **48.6745 / 0.1742**
- train_seed=4：`LR_MULT=25` mean/std **47.7380 / 0.5768**；`LR_MULT=150` mean/std **48.1256 / 0.2314**

结论：✅ 这两条 seed 都建议纳入 “GEO_FEAT ft 拉齐” 的 5-seed 最终交付口径（seed3 用 25；seed4 用 150）。

---

### 5.5 对齐 `check.md` 的缺口补齐（MVP + 证据落盘）

已补齐（代码项，默认关闭，不影响现有结论）：

- QFL（soft target）分类：`CLS_LOSS_TYPE=qfl`、`QFL_BETA`（见 `baselines/DiffusionDet/diffusiondet/loss.py`）
- IoU-aware 回归加权：`BOX_LOSS_IOU_WEIGHT_POWER`（见 `baselines/DiffusionDet/diffusiondet/loss.py`）
- learnable 几何偏置网络（g_phi）：`GEO_BIAS_TYPE=mlp`（见 `baselines/DiffusionDet/diffusiondet/head.py`）

已补齐（证据链：ckpt + tsv，可复验）：

- check.md Phase2/5.* “混合损失 + 图模块 + 各向异性噪声”组合 smoke（train_seed=0，200 iter）：
  - config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_checkmd_mvp.yaml`
  - ckpt：`baselines/checkpoints/checkmd_mvp_seed0_iter200.pth`
  - TSV：`results/checkmd_mvp_seed0_iter200_step1_results.tsv`（eval_seed=0..4 mean/std **47.5333 / 0.3146**）
- check.md Phase3 “Consistency Distillation” smoke（train_seed=0，50 iter；仅验收 loss 链路 + 落盘）：
  - config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_consistency_distill.yaml`
  - ckpt：`baselines/checkpoints/checkmd_consistency_smoke_seed0_iter50.pth`
  - TSV：`results/checkmd_consistency_smoke_seed0_iter50_step1_results.tsv`（eval_seed=0..4 mean/std **47.4060 / 0.5396**）

### 5.6 对齐 `check.md` 的指标/消融口径（FPS + 图拓扑）

- FPS 落盘：`scripts/eval_multiseed.py` 已新增 `inference_s_per_img` / `inference_fps` 两列（从 detectron2 eval 日志解析；硬件相关）
- 图拓扑消融开关：
  - 无交互（independent nodes）：`MODEL.DiffusionDet.DISABLE_SELF_ATTN=True`（示例 config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_graph_topo_none.yaml`）
  - 稀疏 kNN：`MODEL.DiffusionDet.GEO_BIAS_TOPK>0`（示例 config：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_graph_topo_sparse_knn_topk50.yaml`）
- TSV 示例（eval-only，train_seed=0 stable student；eval_seed=0..4）：  
  - Full：`results/ablation_graph_topo_full_seed0_stable_step1_results.tsv`  
  - Sparse：`results/ablation_graph_topo_sparse_knn_topk50_seed0_stable_step1_results.tsv`  
  - None：`results/ablation_graph_topo_none_seed0_stable_step1_results.tsv`

### 5.7 Progressive sampler distill（多步学生：20→4→2→1 模板）

为对齐 `check.md` 中“逐步把采样步数从 1000 减到 4，再到 2”的工程能力，本仓库已补齐：

- 多步学生训练支持：当 `SAMPLER_DISTILL_STUDENT_SAMPLE_STEP>1` 时，distill loss 会在训练时真实运行学生 sampler（会更慢、更占显存）
- 模板 config：  
  - teacher20→student4：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_20to4.yaml`  
  - teacher4→student2：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_4to2.yaml`  
  - teacher2→student1：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_2to1.yaml`
- 一键模板脚本（会生成多份 ckpt/tsv，注意磁盘）：`scripts/run_progressive_sampler_distill.sh`

### 5.8 Scale-up 开关模板（AMP/EMA）

- 仅提供开关示例（不代表已在大规模数据/多卡上验收）：`baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead_amp_ema.yaml`

### 5.9 check.md Phase1/Phase2 的“工程补齐”（torch.compile + COCO/LVIS/VOC 模板）

- torch.compile（实验性）：`SOLVER.TORCH_COMPILE*`（代码入口：`baselines/DiffusionDet/train_net.py`）
- Detectron2 builtin 数据集注册：`baselines/DiffusionDet/train_net.py` 会注册 COCO/LVIS/VOC（依赖 `$DETECTRON2_DATASETS`）
- D3PM 配方模板：
  - COCO：`baselines/DiffusionDet/configs/diffdet.coco.res50_d3pm_mask_dist_qhead.yaml`
  - LVIS：`baselines/DiffusionDet/configs/diffdet.lvis.res50_d3pm_mask_dist_qhead.yaml`
  - VOC：`baselines/DiffusionDet/configs/diffdet.voc2007.res50_d3pm_mask_dist_qhead.yaml`
- 额外 COCO-style 数据集注册（CrowdHuman / Objects365 / 自定义）：`baselines/DiffusionDet/train_net.py`（环境变量 `EXTRA_COCO_DATASETS=...`）
  - CrowdHuman 模板：`baselines/DiffusionDet/configs/diffdet.crowdhuman.res50_d3pm_mask_dist_qhead.yaml`
  - Objects365 模板：`baselines/DiffusionDet/configs/diffdet.objects365.res50_d3pm_mask_dist_qhead.yaml`
- MMDet+Diffusers 同栈迁移说明：`checkmd_mmdet_diffusers_stub.md`

## Appendix A：常用命令模板（复制就能跑）

### A.1 eval-only（单 ckpt，多 eval_seed）

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 --eval-only \
  MODEL.WEIGHTS /path/to/model_final.pth \
  MODEL.DiffusionDet.SAMPLE_STEP 1 \
  OUTPUT_DIR /dev/shm/eval_tmp/evalseed0 \
  SEED 0
```

### A.2 sampler distill 20→1（训练入口）

```bash
cd baselines/DiffusionDet
python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_20to1.yaml --num-gpus 1 \
  MODEL.WEIGHTS <student_init.pth> \
  MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS <teacher.pth> \
  SOLVER.MAX_ITER 2500 SOLVER.CHECKPOINT_PERIOD 100000000 TEST.EVAL_PERIOD 100000000 \
  OUTPUT_DIR /dev/shm/distill_run \
  SEED <train_seed>
```
