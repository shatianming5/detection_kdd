#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mmdet3}"
WORK_DIR="${2:-/tmp/mmdet_coco_diffusiondet_swinb}"
GPUS="${3:-1}"

if [[ -z "${COCO_ROOT:-}" ]]; then
  echo "COCO_ROOT is required (e.g. export COCO_ROOT=/path/to/coco)" >&2
  exit 2
fi

conda run -n "${ENV_NAME}" mim train mmdet mmdet_diffusers/mmdet_configs/coco_diffusiondet_swinb_baseline.py \
  --work-dir "${WORK_DIR}" \
  --gpus "${GPUS}" \
  --launcher none -y

