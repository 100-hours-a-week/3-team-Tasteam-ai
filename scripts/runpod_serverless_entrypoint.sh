#!/usr/bin/env bash
# RunPod Serverless vLLM 워커: 최초 1회 모델 다운로드 후 vLLM 기동
# MODEL_NAME이 로컬 경로(/runpod-volume/...)이고 해당 경로에 모델이 없으면
# HF_MODEL_ID로 snapshot_download 실행 후 handler 기동.

set -e

# MODEL_NAME이 절대 경로(네트워크 볼륨)이고, config.json이 없으면 다운로드
if [[ "$MODEL_NAME" == /* ]]; then
  if [[ ! -f "${MODEL_NAME}/config.json" ]]; then
    echo "[entrypoint] Model not found at ${MODEL_NAME}, downloading from HuggingFace (HF_MODEL_ID=${HF_MODEL_ID:-})..."
    python3 -c "
import os
from huggingface_hub import snapshot_download

model_name = os.environ.get('MODEL_NAME')
hf_id = os.environ.get('HF_MODEL_ID')
if not hf_id:
    # 경로에서 추론: /runpod-volume/llm-models/Qwen/Qwen2.5-7B-Instruct -> Qwen/Qwen2.5-7B-Instruct
    prefix = '/runpod-volume/llm-models/'
    if model_name.startswith(prefix):
        hf_id = model_name[len(prefix):].strip('/')
    else:
        hf_id = 'Qwen/Qwen2.5-7B-Instruct'
snapshot_download(hf_id, local_dir=model_name, local_dir_use_symlinks=False)
print('[entrypoint] Download done.')
"
    echo "[entrypoint] Model ready at ${MODEL_NAME}"
  else
    echo "[entrypoint] Model already present at ${MODEL_NAME}, skipping download."
  fi
fi

exec "$@"
