#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DOTENV_PATH="${1:-$REPO_ROOT/.env}"

if [[ -f "$DOTENV_PATH" ]]; then
  echo "ERROR: $DOTENV_PATH already exists" >&2
  exit 2
fi

read -r -p "WANDB_PROJECT [detection_kdd]: " WANDB_PROJECT
WANDB_PROJECT="${WANDB_PROJECT:-detection_kdd}"

read -r -p "WANDB_ENTITY (optional): " WANDB_ENTITY

read -r -s -p "WANDB_API_KEY: " WANDB_API_KEY
echo

umask 077
cat >"$DOTENV_PATH" <<EOF
# Local-only secrets. DO NOT COMMIT THIS FILE.
WANDB_API_KEY=${WANDB_API_KEY}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_MODE=online
WANDB_DIR=wandb
EOF

echo "wrote: $DOTENV_PATH"
echo "next: run eval scripts; they will auto-log to W&B when .env is present."

