每一次更新完都要push结果上去

# detection_kdd

本仓库用于目标检测（Detectron2/DiffusionDet 主线）实验归档与复现，并提供对齐 `check.md` 的 **MMDet3 + Diffusers 形态**实现（`mmdet_diffusers/`，定位为对照/口径对齐，不替代主训练栈）。

## 结果总览（表格）

说明：
- AP 为百分制（×100），std 为总体标准差（pop std）。
- 大数据/ckpt/训练输出不随 git 跟踪（见 `.gitignore`）；结果表以仓库内 TSV/summary 为准。

### Detectron2/DiffusionDet（Repro-10k）

来源：`results_manifest.tsv`

| Stack | Dataset | Setting | n_runs | mean_AP | std_AP | Source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| DiffusionDet baseline | repro_10k | step1 | 15 | 46.43 | 0.78 | `results/baseline_step1_results.tsv` |
| DiffusionDet baseline | repro_10k | step5 | 15 | 48.29 | 0.68 | `results/baseline_step5_results.tsv` |
| D3PM+QHead warmstart | repro_10k | step1 | 15 | 46.91 | 0.68 | `results/d3pm_qhead_warmstart_step1_results.tsv` |
| Sampler distill + GEO_FEAT (deliverable) | repro_10k | step1 | 25 | 48.54 | 0.84 | `results/final_step1_5seed_geofeat_mix.tsv` |

完整索引：`results_manifest.tsv`（包含更多消融/扫参条目）。

### MMDet3（LVIS / CrowdHuman）

来源：`results/rerun_mmdet3_lvis_crowdhuman_summary.tsv`

| Dataset | baseline_name | step | mean_AP | std_AP | mean_fps | ckpt |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| lvis | lvis_iter_10000_lvis_graph_diffusion_r50_fpn_teacher_focal_iter10k_step10 | 10 | 0.00 | 0.00 | 3.74 | iter_10000.pth |
| lvis | lvis_iter_10000_lvis_graph_diffusion_r50_fpn_teacher_focal_iter10k_step1 | 1 | 0.00 | 0.00 | 13.82 | iter_10000.pth |
| crowdhuman | crowdhuman_iter_10000_crowdhuman_graph_diffusion_r50_fpn_teacher_focal_iter10k_step10 | 10 | 45.99 | 0.06 | 5.05 | iter_10000.pth |
| crowdhuman | crowdhuman_iter_10000_crowdhuman_graph_diffusion_r50_fpn_teacher_focal_iter10k_step1 | 1 | 59.33 | 0.15 | 24.44 | iter_10000.pth |
| lvis | lvis_iter_10000_lvis_graph_diffusion_r50_fpn_teacher_focal_iter150k_bs2_lr5e5_step10 | 10 | 0.00 | 0.00 | 1.71 | iter_10000.pth |
| lvis | lvis_iter_10000_lvis_graph_diffusion_r50_fpn_teacher_focal_iter150k_bs2_lr5e5_step1 | 1 | 0.00 | 0.00 | 13.85 | iter_10000.pth |

## 目录结构（省略数据与大产物）

- `baselines/`：Detectron2 baselines（DiffusionDet、DETR 等）。`baselines/data/`、`baselines/checkpoints/`、`baselines/output/`、`baselines/evals/` 为本地数据/产物目录（默认不入库）。
- `mmdet_diffusers/`：MMDet3 插件 + Diffusers 风格 pipeline/scheduler/collate（对齐 `check.md` 的工程形态）。
- `results/`：评测 TSV、rerun manifest/summary 等轻量结果文件。
- `scripts/`：评测多 seed、生成 manifest/summary 的脚本。
- `deliverables/`：小体量“冻结交付”与规范示例（不包含大 ckpt）。
- `check.md`：需求/路线文档；对齐差距与路线图见 `checkmd_parity.md`、`checkmd_mmdet_diffusers_stub.md`。

## 复现与更新口径

- Detectron2 结果索引更新：`python scripts/build_results_manifest.py`
- MMDet3 summary 更新：`python scripts/build_mmdet3_manifest.py --help`
- 评测入口：
  - Detectron2 多 eval_seed：`scripts/eval_multiseed.py`
  - MMDet3 多 eval_seed：`scripts/eval_mmdet3_multiseed.py`

## Weights & Biases（W&B）

本仓库的 `scripts/eval_multiseed.py` / `scripts/eval_mmdet3_multiseed.py` 支持自动把评测 TSV 与汇总指标在线记录到 W&B：

- 配置方式（本地，不入库）：
  - `cp .env.example .env`，填入 `WANDB_API_KEY`、`WANDB_PROJECT`（以及可选 `WANDB_ENTITY`）
  - `.env` 已在 `.gitignore` 里，禁止提交
- 开关：
  - 默认：当检测到 `WANDB_PROJECT` 或 `WANDB_API_KEY` 时自动开启
  - 可显式关闭：在命令行加 `--no-wandb`
