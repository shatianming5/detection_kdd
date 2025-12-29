#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values")
    return statistics.mean(values), statistics.pstdev(values)


def load_aps(tsv_path: Path) -> list[float]:
    with tsv_path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = list(r)
    if not rows:
        raise ValueError(f"no rows: {tsv_path}")
    if "AP" not in rows[0]:
        raise ValueError(f"missing AP column: {tsv_path}")
    return [float(row["AP"]) for row in rows]


@dataclass(frozen=True)
class Entry:
    exp_id: str
    tsv_path: str
    ckpt_path_or_pattern: str
    notes: str = ""


ENTRIES: list[Entry] = [
    Entry(
        exp_id="baseline_step1_flat_train012",
        tsv_path="baseline_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/baseline_iter10000_seed{train_seed}.pth",
        notes="DiffusionDet baseline 10k; train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="baseline_step5_flat_train012",
        tsv_path="baseline_step5_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/baseline_iter10000_seed{train_seed}.pth",
        notes="DiffusionDet baseline 10k; train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=5",
    ),
    Entry(
        exp_id="d3pm_qhead_step1_flat_train012",
        tsv_path="d3pm_qhead_step1_results.tsv",
        ckpt_path_or_pattern="(see out_dir in TSV; multiple ckpts)",
        notes="D3PM(mask, dist)+QHead (non-warmstart); train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="d3pm_qhead_step5_flat_train012",
        tsv_path="d3pm_qhead_step5_results.tsv",
        ckpt_path_or_pattern="(see out_dir in TSV; multiple ckpts)",
        notes="D3PM(mask, dist)+QHead (non-warmstart); train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=5",
    ),
    Entry(
        exp_id="d3pm_qhead_warmstart_step1_flat_train012",
        tsv_path="d3pm_qhead_warmstart_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed{train_seed}_iter2500.pth",
        notes="D3PM+QHead warmstart baseline; train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="d3pm_qhead_warmstart_step5_flat_train012",
        tsv_path="d3pm_qhead_warmstart_step5_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed{train_seed}_iter2500.pth",
        notes="D3PM+QHead warmstart baseline; train_seed=0/1/2 × eval_seed=0..4; SAMPLE_STEP=5",
    ),
    Entry(
        exp_id="warmstart_seed0_step1",
        tsv_path="warmstart_seed0_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth",
        notes="Warmstart seed0 speed/quality curve; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="warmstart_seed0_step5",
        tsv_path="warmstart_seed0_step5_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth",
        notes="Warmstart seed0 speed/quality curve; eval_seed=0..4; SAMPLE_STEP=5",
    ),
    Entry(
        exp_id="warmstart_seed0_step10",
        tsv_path="warmstart_seed0_step10_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth",
        notes="Warmstart seed0 speed/quality curve; eval_seed=0..4; SAMPLE_STEP=10",
    ),
    Entry(
        exp_id="warmstart_seed0_step20",
        tsv_path="warmstart_seed0_step20_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth",
        notes="Warmstart seed0 speed/quality curve; eval_seed=0..4; SAMPLE_STEP=20",
    ),
    Entry(
        exp_id="warmstart_seed0_step50",
        tsv_path="warmstart_seed0_step50_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth",
        notes="Warmstart seed0 speed/quality curve; eval_seed=0..4; SAMPLE_STEP=50",
    ),
    Entry(
        exp_id="phase2_geofeat_sweep_seed1_iter1000",
        tsv_path="phase2_geofeat_sweep_seed1_iter1000_step1_results.tsv",
        ckpt_path_or_pattern="(multiple ckpts in /dev/shm; see out_dir in TSV)",
        notes="Phase2 GEO_FEAT warmstart sweep (train_seed=1); eval_seed=0..4; MAX_ITER=1000",
    ),
    Entry(
        exp_id="phase2_geofeat_sweep_seed1_iter1500",
        tsv_path="phase2_geofeat_sweep_seed1_iter1500_step1_results.tsv",
        ckpt_path_or_pattern="(multiple ckpts in /dev/shm; see out_dir in TSV)",
        notes="Phase2 GEO_FEAT warmstart sweep (train_seed=1); eval_seed=0..4; MAX_ITER=1500",
    ),
    Entry(
        exp_id="phase2_geofeat_sweep_seed1_iter2000",
        tsv_path="phase2_geofeat_sweep_seed1_iter2000_step1_results.tsv",
        ckpt_path_or_pattern="(multiple ckpts in /dev/shm; see out_dir in TSV)",
        notes="Phase2 GEO_FEAT warmstart sweep (train_seed=1); eval_seed=0..4; MAX_ITER=2000",
    ),
    Entry(
        exp_id="distill_seed0_iter2500_stable",
        tsv_path="sampler_distill_20to1_seed0_iter2500_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable.pth",
        notes="Sampler distill 20→1 stable; train_seed=0; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_iter2500_stable",
        tsv_path="sampler_distill_20to1_seed1_iter2500_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_stable.pth",
        notes="Sampler distill 20→1 stable; train_seed=1; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed2_iter2500_stable",
        tsv_path="sampler_distill_20to1_seed2_iter2500_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed2_iter2500_stable.pth",
        notes="Sampler distill 20→1 stable; train_seed=2; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=1; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed2_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed2_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=2; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed0_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="(not promoted; see plan2)",
        notes="Stable distill + GEO_FEAT finetune; train_seed=0; eval_seed=0..4; SAMPLE_STEP=1 (slightly worse than stable)",
    ),
    Entry(
        exp_id="distill_seed1_teacher_geofeat150_failed",
        tsv_path="sampler_distill_20to1_seed1_teacher_geofeat150_iter2500_step1_results.tsv",
        ckpt_path_or_pattern="(student ckpt not promoted)",
        notes="Sampler distill from GEO_FEAT teacher (failed; worse).",
    ),
    Entry(
        exp_id="distill_seed1_teacher_seed0warmstart_failed",
        tsv_path="sampler_distill_20to1_seed1_teacher_seed0warmstart_iter2500_step1_results.tsv",
        ckpt_path_or_pattern="(student ckpt not promoted)",
        notes="Sampler distill with teacher swapped to warmstart seed0 (failed; worse).",
    ),
    Entry(
        exp_id="d3pm_qhead_warmstart_seed3_step1",
        tsv_path="d3pm_qhead_warmstart_seed3_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed3_iter2500.pth",
        notes="D3PM+QHead warmstart baseline; train_seed=3; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="d3pm_qhead_warmstart_seed4_step1",
        tsv_path="d3pm_qhead_warmstart_seed4_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed4_iter2500.pth",
        notes="D3PM+QHead warmstart baseline; train_seed=4; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed3_iter2500_stable",
        tsv_path="sampler_distill_20to1_seed3_iter2500_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed3_iter2500_stable.pth",
        notes="Sampler distill 20→1 stable; train_seed=3; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed4_iter2500_stable",
        tsv_path="sampler_distill_20to1_seed4_iter2500_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed4_iter2500_stable.pth",
        notes="Sampler distill 20→1 stable; train_seed=4; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed0_geofeat_ft_iter1000_lrmult25",
        tsv_path="sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=0; GEO_FEAT_LR_MULT=25; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed0_geofeat_ft_iter1000_lrmult50",
        tsv_path="sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult50_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult50.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=0; GEO_FEAT_LR_MULT=50; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed0_geofeat_ft_iter1000_lrmult100",
        tsv_path="sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult100_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable_geofeat_ft_iter1000_lrmult100.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=0; GEO_FEAT_LR_MULT=100; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_cls_weight_0p1",
        tsv_path="sampler_distill_20to1_seed1_iter2500_cls0p1_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p1.pth",
        notes="Sampler distill 20→1; train_seed=1; SAMPLER_DISTILL_CLS_WEIGHT=0.1; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_cls_weight_0p1_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed1_iter2500_cls0p1_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p1_geofeat_ft_iter1000_lrmult150.pth",
        notes="Distill (CLS_WEIGHT=0.1) + GEO_FEAT finetune; train_seed=1; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_cls_weight_0p2",
        tsv_path="sampler_distill_20to1_seed1_iter2500_cls0p2_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p2.pth",
        notes="Sampler distill 20→1; train_seed=1; SAMPLER_DISTILL_CLS_WEIGHT=0.2; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_cls_weight_0p2_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed1_iter2500_cls0p2_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_cls0p2_geofeat_ft_iter1000_lrmult150.pth",
        notes="Distill (CLS_WEIGHT=0.2) + GEO_FEAT finetune; train_seed=1; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed3_geofeat_ft_iter1000_lrmult25",
        tsv_path="sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=3; GEO_FEAT_LR_MULT=25; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed3_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed3_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=3; GEO_FEAT_LR_MULT=150; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed4_geofeat_ft_iter1000_lrmult25",
        tsv_path="sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult25_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult25.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=4; GEO_FEAT_LR_MULT=25; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed4_geofeat_ft_iter1000_lrmult150",
        tsv_path="sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed4_iter2500_stable_geofeat_ft_iter1000_lrmult150.pth",
        notes="Stable distill + GEO_FEAT finetune; train_seed=4; GEO_FEAT_LR_MULT=150; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="final_step1_5seed_geofeat_mix_flat_train0to4",
        tsv_path="final_step1_5seed_geofeat_mix.tsv",
        ckpt_path_or_pattern="(see deliverables/step1_5seed_geofeat_mix/manifest.tsv)",
        notes="FROZEN deliverable: seed0 g25 / seed1 g150 / seed2 g150 / seed3 g25 / seed4 g150; train_seed=0..4 × eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_topk50",
        tsv_path="sampler_distill_20to1_seed1_iter2500_topk50_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_topk50.pth",
        notes="R&D: sampler distill with SAMPLER_DISTILL_TOPK=50 (train_seed=1); eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_topk150",
        tsv_path="sampler_distill_20to1_seed1_iter2500_topk150_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_topk150.pth",
        notes="R&D: sampler distill with SAMPLER_DISTILL_TOPK=150 (train_seed=1); eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_topk200",
        tsv_path="sampler_distill_20to1_seed1_iter2500_topk200_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_topk200.pth",
        notes="R&D: sampler distill with SAMPLER_DISTILL_TOPK=200 (train_seed=1); eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_boxw0p5",
        tsv_path="sampler_distill_20to1_seed1_iter2500_boxw0p5_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_boxw0p5.pth",
        notes="R&D: sampler distill with SAMPLER_DISTILL_BOX_WEIGHT=0.5 (train_seed=1); eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="distill_seed1_boxw2p0",
        tsv_path="sampler_distill_20to1_seed1_iter2500_boxw2p0_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed1_iter2500_boxw2p0.pth",
        notes="R&D: sampler distill with SAMPLER_DISTILL_BOX_WEIGHT=2.0 (train_seed=1); eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="checkmd_mvp_seed0_iter200_step1",
        tsv_path="checkmd_mvp_seed0_iter200_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/checkmd_mvp_seed0_iter200.pth",
        notes="check.md MVP smoke: GEO_BIAS_TYPE=mlp + ANISO_NOISE + GRAPH_TOPO_LOSS + QFL + IoU-weighted reg; train_seed=0; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="checkmd_consistency_smoke_seed0_iter50_step1",
        tsv_path="checkmd_consistency_smoke_seed0_iter50_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/checkmd_consistency_smoke_seed0_iter50.pth",
        notes="check.md Phase3 smoke: CONSISTENCY_DISTILL entry; train_seed=0; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="ablation_graph_topo_full_seed0_stable_step1",
        tsv_path="ablation_graph_topo_full_seed0_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable.pth",
        notes="check.md:7.3-1 ablation (eval-only toggle): full graph (default self-attn); train_seed=0 stable student; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="ablation_graph_topo_sparse_knn_topk50_seed0_stable_step1",
        tsv_path="ablation_graph_topo_sparse_knn_topk50_seed0_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable.pth",
        notes="check.md:7.3-1 ablation (eval-only toggle): sparse kNN via GEO_BIAS_TOPK=50 (distance bias mask); train_seed=0 stable student; eval_seed=0..4; SAMPLE_STEP=1",
    ),
    Entry(
        exp_id="ablation_graph_topo_none_seed0_stable_step1",
        tsv_path="ablation_graph_topo_none_seed0_stable_step1_results.tsv",
        ckpt_path_or_pattern="baselines/checkpoints/sampler_distill_20to1_seed0_iter2500_stable.pth",
        notes="check.md:7.3-1 ablation (eval-only toggle): no interaction via DISABLE_SELF_ATTN=True; train_seed=0 stable student; eval_seed=0..4; SAMPLE_STEP=1",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_manifest.tsv", help="Output TSV path.")
    args = ap.parse_args()

    out_path = (REPO_ROOT / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["generated_at_utc", now])
        w.writerow(
            [
                "exp_id",
                "tsv_path",
                "ckpt_path_or_pattern",
                "ckpt_sha256",
                "n_runs",
                "mean_AP",
                "std_AP_pop",
                "notes",
            ]
        )

        for e in ENTRIES:
            tsv = (RESULTS_DIR / e.tsv_path).resolve()
            aps = load_aps(tsv)
            mean, std = mean_std(aps)

            sha = ""
            ckpt = e.ckpt_path_or_pattern
            # Only compute sha256 for a concrete file path.
            if "{" not in ckpt and ckpt.endswith(".pth"):
                ckpt_path = (REPO_ROOT / ckpt).resolve()
                if ckpt_path.is_file():
                    sha = sha256_file(ckpt_path)

            w.writerow(
                [
                    e.exp_id,
                    str(Path("results") / e.tsv_path),
                    e.ckpt_path_or_pattern,
                    sha,
                    len(aps),
                    f"{mean:.5f}",
                    f"{std:.5f}",
                    e.notes,
                ]
            )

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
