#!/bin/bash
set -e

# Configuration
# Get current timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATA_ROOT=$(pwd)/baselines/data/repro_10k
DETR_DATA_ROOT=$(pwd)/baselines/data/repro_10k_detr_coco
OUTPUT_BASE=${OUTPUT_BASE:-$(pwd)/baselines/output}
OUTPUT_ROOT=${OUTPUT_BASE}/${TIMESTAMP}
SEED=${SEED:-42}
RUN_DIFFDET_SWIN=${RUN_DIFFDET_SWIN:-0}  # set to 1 to also run Swin-B backbone

download_if_missing() {
  local url="$1"
  local dst="$2"
  if [ -f "$dst" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  echo "Downloading: $url"
  echo "        to: $dst"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$dst" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$dst" "$url"
  else
    echo "ERROR: need wget or curl to download weights" >&2
    exit 1
  fi
}

echo "=========================================================="
echo "   Experiment Output Directory: ${OUTPUT_ROOT}"
echo "=========================================================="
mkdir -p ${OUTPUT_ROOT}

echo "=========================================================="
echo "   Running DiffusionDet Baseline on repro_10k"
echo "=========================================================="
cd baselines/DiffusionDet
# Running for 10000 iters (~4 epochs) for a good baseline.
python train_net.py \
    --config-file configs/diffdet.repro_10k.yaml \
    --num-gpus 1 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.CHECKPOINT_PERIOD 100000000 \
    OUTPUT_DIR ${OUTPUT_ROOT}/repro_10k_diffdet \
    SEED ${SEED}

if [ "${RUN_DIFFDET_SWIN}" = "1" ]; then
  echo ""
  echo "=========================================================="
  echo "   Running DiffusionDet Swin-B Baseline on repro_10k"
  echo "=========================================================="

  # Prefer an explicit override dir to avoid writing large weights to a nearly-full root disk.
  # Example:
  #   DIFFDET_MODELS_DIR=/dev/shm/diffdet_models RUN_DIFFDET_SWIN=1 bash run_baselines.sh
  DIFFDET_MODELS_DIR=${DIFFDET_MODELS_DIR:-$(pwd)/models}
  SWIN_B_WEIGHTS=${SWIN_B_WEIGHTS:-${DIFFDET_MODELS_DIR}/swin_base_patch4_window7_224_22k.pkl}
  SWIN_B_URL="https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/swin_base_patch4_window7_224_22k.pkl"
  download_if_missing "${SWIN_B_URL}" "${SWIN_B_WEIGHTS}"

  python train_net.py \
      --config-file configs/diffdet.repro_10k.swinbase.yaml \
      --num-gpus 1 \
      SOLVER.MAX_ITER 10000 \
      SOLVER.CHECKPOINT_PERIOD 100000000 \
      MODEL.WEIGHTS ${SWIN_B_WEIGHTS} \
      OUTPUT_DIR ${OUTPUT_ROOT}/repro_10k_diffdet_swinb \
      SEED ${SEED}
fi

echo ""
echo "=========================================================="
echo "   Running DiffusionDet D3PM(mask)+QualityHead Baseline on repro_10k"
echo "=========================================================="
python train_net.py \
    --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml \
    --num-gpus 1 \
    SOLVER.MAX_ITER 10000 \
    SOLVER.CHECKPOINT_PERIOD 100000000 \
    OUTPUT_DIR ${OUTPUT_ROOT}/repro_10k_diffdet_d3pm_qhead \
    SEED ${SEED}

echo ""
echo "=========================================================="
echo "   Running DETR Baseline on repro_10k"
echo "=========================================================="
cd ../detr
# DETR is hard to train from scratch; fine-tune from the official COCO-pretrained weights.
# We remap repro_10k category ids to COCO ids (person=1, car=3, motorcycle=4) to align with pretrained DETR.
python ../../utils/prepare_detr_coco_ids.py --src ${DATA_ROOT} --dst ${DETR_DATA_ROOT}

DETR_URL="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
TORCH_HOME_DIR="${TORCH_HOME:-${HOME}/.cache/torch}"
DETR_CKPT_LOCAL="${TORCH_HOME_DIR}/hub/checkpoints/detr-r50-e632da11.pth"
if [ -f "${DETR_CKPT_LOCAL}" ]; then
  DETR_RESUME="${DETR_CKPT_LOCAL}"
else
  DETR_RESUME="${DETR_URL}"
fi

python main.py \
    --dataset_file coco \
    --coco_path ${DETR_DATA_ROOT} \
    --num_classes 91 \
    --resume ${DETR_RESUME} \
    --output_dir ${OUTPUT_ROOT}/repro_10k_detr \
    --epochs 50 \
    --lr_drop 40 \
    --batch_size 4 \
    --num_workers 4

echo "=========================================================="
echo "   All Baselines Finished"
echo "=========================================================="
