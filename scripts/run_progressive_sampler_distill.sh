#!/usr/bin/env bash
set -euo pipefail

#
# Progressive sampler distillation template (check.md: Phase3 idea):
#   teacher(step20) -> student(step4) -> student(step2) -> student(step1)
#
# This script intentionally does NOT run by default in this repo's main plan,
# because it can be slow and will create multiple ~1.3GB checkpoints.
#
# Usage example (train_seed=0):
#   bash scripts/run_progressive_sampler_distill.sh \
#     --train-seed 0 \
#     --student-init baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth \
#     --teacher20 baselines/checkpoints/d3pm_qhead_warmstart_baseline_seed0_iter2500.pth \
#     --out-prefix progdist_seed0
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFFDET_DIR="$REPO_ROOT/baselines/DiffusionDet"

TRAIN_SEED=""
STUDENT_INIT=""
TEACHER20=""
OUT_PREFIX=""
OUT_BASE="${OUTPUT_BASE:-/dev/shm}"
MAX_ITER="${MAX_ITER:-2500}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-seed) TRAIN_SEED="$2"; shift 2 ;;
    --student-init) STUDENT_INIT="$2"; shift 2 ;;
    --teacher20) TEACHER20="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --out-base) OUT_BASE="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TRAIN_SEED" || -z "$STUDENT_INIT" || -z "$TEACHER20" || -z "$OUT_PREFIX" ]]; then
  echo "Missing required args." >&2
  echo "Required: --train-seed --student-init --teacher20 --out-prefix" >&2
  exit 2
fi

run_stage () {
  local stage_name="$1"
  local cfg_rel="$2"
  local teacher_ckpt="$3"
  local student_init="$4"
  local student_out_ckpt="$5"
  local tsv_out="$6"

  local out_dir="$OUT_BASE/${stage_name}"
  mkdir -p "$out_dir"

  echo "[train] $stage_name"
  (
    cd "$DIFFDET_DIR"
    python train_net.py --config-file "configs/${cfg_rel}" --num-gpus 1 \
      MODEL.WEIGHTS "$student_init" \
      MODEL.DiffusionDet.SAMPLER_DISTILL_TEACHER_WEIGHTS "$teacher_ckpt" \
      SOLVER.MAX_ITER "$MAX_ITER" \
      SOLVER.CHECKPOINT_PERIOD 100000000 \
      TEST.EVAL_PERIOD 100000000 \
      OUTPUT_DIR "$out_dir" \
      SEED "$TRAIN_SEED"
  )

  echo "[finalize] $student_out_ckpt"
  python "$REPO_ROOT/scripts/finalize_checkpoint.py" --src "$out_dir/model_final.pth" --dst "$student_out_ckpt"

  echo "[eval] $tsv_out"
  python "$REPO_ROOT/scripts/eval_multiseed.py" \
    --exp-name "$(basename "$student_out_ckpt" .pth)" \
    --config-file "$REPO_ROOT/baselines/DiffusionDet/configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml" \
    --weights "$student_out_ckpt" \
    --sample-step 1 \
    --eval-seeds 0 1 2 3 4 \
    --tsv-out "$tsv_out"
}

CKPT20="$TEACHER20"

CKPT4="$REPO_ROOT/baselines/checkpoints/${OUT_PREFIX}_20to4_iter${MAX_ITER}.pth"
TSV4="$REPO_ROOT/${OUT_PREFIX}_20to4_iter${MAX_ITER}_step1_results.tsv"
run_stage "${OUT_PREFIX}_20to4" "diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_20to4.yaml" "$CKPT20" "$STUDENT_INIT" "$CKPT4" "$TSV4"

CKPT2="$REPO_ROOT/baselines/checkpoints/${OUT_PREFIX}_4to2_iter${MAX_ITER}.pth"
TSV2="$REPO_ROOT/${OUT_PREFIX}_4to2_iter${MAX_ITER}_step1_results.tsv"
run_stage "${OUT_PREFIX}_4to2" "diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_4to2.yaml" "$CKPT4" "$CKPT4" "$CKPT2" "$TSV2"

CKPT1="$REPO_ROOT/baselines/checkpoints/${OUT_PREFIX}_2to1_iter${MAX_ITER}.pth"
TSV1="$REPO_ROOT/${OUT_PREFIX}_2to1_iter${MAX_ITER}_step1_results.tsv"
run_stage "${OUT_PREFIX}_2to1" "diffdet.repro_10k_d3pm_mask_dist_qhead_sampler_distill_2to1.yaml" "$CKPT2" "$CKPT2" "$CKPT1" "$TSV1"

echo "Done."
