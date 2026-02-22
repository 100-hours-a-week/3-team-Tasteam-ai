#!/usr/bin/env bash
# RunPod Pod 학습용: /workspace에 모델 있으면 바로 train_qlora / 없으면 이미지→/workspace 복사 후 train_qlora.

set -e

MODEL_PATH="${MODEL_NAME:-/workspace/llm-models/Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_IMAGE_PATH="${MODEL_IMAGE_PATH:-/opt/llm-models/Qwen/Qwen2.5-0.5B-Instruct}"

if [[ -f "${MODEL_PATH}/config.json" ]]; then
  echo "[entrypoint] Model present at ${MODEL_PATH}, starting Qwen2.5-0.5B-Instruct."
elif [[ -f "${MODEL_IMAGE_PATH}/config.json" ]]; then
  echo "[entrypoint] Copying model from image ${MODEL_IMAGE_PATH} to ${MODEL_PATH}..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  cp -a "${MODEL_IMAGE_PATH}" "$(dirname "$MODEL_PATH")/"
  echo "[entrypoint] Copy done, starting Qwen2.5-0.5B-Instruct."
else
  echo "[entrypoint] ERROR: Model not found at ${MODEL_PATH} nor in image at ${MODEL_IMAGE_PATH}." >&2
  exit 1
fi

exec python /app/scripts/train_qlora.py --student-model "${MODEL_PATH}" "$@"