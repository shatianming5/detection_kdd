#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/baselines/DiffusionDet"

OUT_BASE="${OUT_BASE:-/dev/shm}"
MAX_ITER="${MAX_ITER:-10000}"
SEEDS="${SEEDS:-0 1 2}"
CKPT_PERIOD="${CKPT_PERIOD:-100000000}"

echo "OUT_BASE=${OUT_BASE}"
echo "MAX_ITER=${MAX_ITER}"
echo "SEEDS=${SEEDS}"

for seed in ${SEEDS}; do
  out="${OUT_BASE}/diffdet_d3pm_mask_dist_qhead_iter${MAX_ITER}_seed${seed}"
  echo "=========================================================="
  echo "Seed=${seed} OUTPUT_DIR=${out}"
  echo "=========================================================="
  mkdir -p "${out}"
  python train_net.py --config-file configs/diffdet.repro_10k_d3pm_mask_dist_qhead.yaml --num-gpus 1 \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    SOLVER.CHECKPOINT_PERIOD "${CKPT_PERIOD}" \
    OUTPUT_DIR "${out}" \
    SEED "${seed}"
done

python - <<'PY'
import json
import os
import statistics

out_base = os.environ.get("OUT_BASE", "/dev/shm")
max_iter = int(os.environ.get("MAX_ITER", "10000"))
seeds = [int(s) for s in os.environ.get("SEEDS", "0 1 2").split()]

aps = []
paths = []
for s in seeds:
    metrics = f"{out_base}/diffdet_d3pm_mask_dist_qhead_iter{max_iter}_seed{s}/metrics.json"
    with open(metrics, "r", encoding="utf-8") as f:
        last = f.read().splitlines()[-1]
    m = json.loads(last)
    ap = float(m["bbox/AP"])
    aps.append(ap)
    paths.append(metrics)

print("metrics_paths:")
for p in paths:
    print(" ", p)

per_seed = " ".join([f"{s}:{ap:.4f}" for s, ap in zip(seeds, aps)])
print(f"AP mean={statistics.mean(aps):.4f} std={statistics.pstdev(aps):.4f} per_seed={per_seed}")
PY
