#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mmdet3}"
WORK_DIR="${2:-/dev/shm/mmdet_toy_graphdiff}"
GPUS="${3:-1}"

conda run -n "${ENV_NAME}" python mmdet_diffusers/tools/make_toy_coco.py --out datasets/toy_coco
conda run -n "${ENV_NAME}" mim train mmdet mmdet_diffusers/mmdet_configs/toy_graph_diffusion_r50_fpn.py \
  --work-dir "${WORK_DIR}" \
  --gpus "${GPUS}" \
  --launcher none -y

