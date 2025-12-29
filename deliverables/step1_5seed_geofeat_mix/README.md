# Step1 最终交付（5 train_seed，GEO_FEAT mix）— Frozen

冻结时间：2025-12-21

本目录用于“对外口径”的最终交付冻结：**只认 5 个 checkpoint + 对应 TSV**，任何对外结论必须可由 `ckpt + tsv` 复验，不依赖 `/dev/shm` 临时目录。

## 统一口径

- 数据：Repro-10k COCO（`baselines/data/repro_10k`）
- 推理：`MODEL.DiffusionDet.SAMPLE_STEP=1`
- 严格 `train_seed × eval_seed` 分离：固定 ckpt，遍历 `eval_seed=0..4`，取 mean/std

## 交付清单

- 逐 seed 明细：`deliverables/step1_5seed_geofeat_mix/manifest.tsv`
- 最终 25-run 合并表（25 行数据）：`final_step1_5seed_geofeat_mix.tsv`

