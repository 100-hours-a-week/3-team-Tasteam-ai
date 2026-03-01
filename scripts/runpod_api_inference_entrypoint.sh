#!/usr/bin/env bash
# API 추론용 RunPod Pod: 볼륨의 MODEL_PATH에 merged 모델이 있어야만 vLLM 기동.
# 모델은 S3로 추론용 네트워크 볼륨에 업로드해 둔 상태를 가정 (이미지 내 모델 없음).

set -e

MODEL_PATH="${MODEL_PATH:-/workspace/merged}"

if [[ -f "${MODEL_PATH}/config.json" ]]; then
  echo "[entrypoint] Model present at ${MODEL_PATH}, starting vLLM."
else
  echo "[entrypoint] ERROR: Model not found at ${MODEL_PATH}. Upload merged model to volume (e.g. S3 prefix merged/)." >&2
  exit 1
fi

exec python3 -m vllm.entrypoints.openai.api_server --model "${MODEL_PATH}" "$@"
