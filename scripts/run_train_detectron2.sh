#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_file> <output_dir> [num_gpus] [opts...]" >&2
  echo "Example: $0 baselines/DiffusionDet/configs/diffdet.coco.res50.yaml /dev/shm/diffdet_coco 8 SOLVER.MAX_ITER 1000" >&2
  exit 2
fi

CONFIG_FILE="$1"
OUTPUT_DIR="$2"
if [[ $# -ge 3 ]]; then
  NUM_GPUS="$3"
else
  NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
    NUM_GPUS=1
  fi
fi

shift 2
if [[ $# -ge 1 ]]; then
  shift 1  # consume num_gpus
fi

python baselines/DiffusionDet/train_net.py --config-file "${CONFIG_FILE}" --num-gpus "${NUM_GPUS}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  "$@"
