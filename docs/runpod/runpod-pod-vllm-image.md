[2 tools called]


이미지에 모델을 포함하고, 엔트리포인트에서 /workspace에 없으면 이미지 경로에서 복사하도록 구현합니다.


[2 tools called]


RUN에서 `MODEL_IMAGE_PATH`가 Python 문자열 안에서 치환되지 않으므로, Dockerfile에 경로를 하드코딩합니다.


[4 tools called]


구현 반영했습니다.

### 1. `Dockerfile.runpod-pod-vllm`

- **이미지 안에 모델 포함**: `RUN`에서 `huggingface_hub.snapshot_download`로 `Qwen/Qwen2.5-7B-Instruct`를 `/opt/llm-models/Qwen/Qwen2.5-7B-Instruct`에 받습니다.
- `ENV MODEL_IMAGE_PATH=/opt/llm-models/Qwen/Qwen2.5-7B-Instruct` 로 엔트리포인트에서 사용할 경로를 지정했습니다.
- 빌드 시 한 번만 다운로드하므로 이미지 크기는 커지고, 빌드/푸시 시간은 길어집니다.

### 2. `scripts/runpod_pod_entrypoint.sh`

동작 순서:

1. **`/workspace/.../config.json` 있음** → 그대로 vLLM 실행.
2. **없고, `/opt/llm-models/.../config.json`(이미지 내 모델) 있음** → 이미지 경로를 `/workspace/llm-models/Qwen/` 아래로 `cp -a`로 복사한 뒤 vLLM 실행.
3. **이미지에도 없음** (이상 케이스) → 기존처럼 HuggingFace에서 다운로드(재시도) 후 vLLM 실행.

실제 사용 시에는 보통 1번(볼륨에 이미 있음) 또는 2번(첫 기동 시 이미지→볼륨 복사)만 타고, 3번 다운로드는 거의 쓰이지 않습니다.