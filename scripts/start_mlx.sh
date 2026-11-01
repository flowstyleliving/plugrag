#!/usr/bin/env bash
set -euo pipefail

MLX_HOST="${MLX_HOST:-127.0.0.1}"
MLX_PORT="${MLX_PORT:-5368}"
MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3-8B-4bit}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

exec mlx_lm.server \
  --host "$MLX_HOST" \
  --port "$MLX_PORT" \
  --model "$MLX_MODEL"
