#!/usr/bin/env bash
# RunPod Pod vLLM: /workspace에 모델 있으면 바로 vLLM / 없으면 이미지→/workspace 복사 후 vLLM.

set -e

MODEL_PATH="${MODEL_NAME:-/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct}"
MODEL_IMAGE_PATH="${MODEL_IMAGE_PATH:-/opt/llm-models/Qwen/Qwen2.5-7B-Instruct}"

if [[ -f "${MODEL_PATH}/config.json" ]]; then
  echo "[entrypoint] Model present at ${MODEL_PATH}, starting vLLM."
elif [[ -f "${MODEL_IMAGE_PATH}/config.json" ]]; then
  echo "[entrypoint] Copying model from image ${MODEL_IMAGE_PATH} to ${MODEL_PATH}..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  cp -a "${MODEL_IMAGE_PATH}" "$(dirname "$MODEL_PATH")/"
  echo "[entrypoint] Copy done, starting vLLM."
else
  echo "[entrypoint] ERROR: Model not found at ${MODEL_PATH} nor in image at ${MODEL_IMAGE_PATH}." >&2
  exit 1
fi

exec python3 -m vllm.entrypoints.openai.api_server "$@"
