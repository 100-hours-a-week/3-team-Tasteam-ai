**구현**: `Dockerfile.api-llm`, `scripts/runpod_api_inference_entrypoint.sh`. 볼륨·경로 규약은 `api_llm_net_vol.md` 참고.

---

프로젝트에 이미 있는 Pod용 vLLM 이미지/엔트리포인트를 기준으로, **추론 전용(모델은 S3→볼륨)** 이미지에 들어가면 될 것들을 정리했습니다.

---

## 1. 들어가야 할 것

- **vLLM 베이스 이미지**  
  - 예: `vllm/vllm-openai:v0.11.0` (기존 `Dockerfile.labeling-llm`과 동일 베이스)
- **모델 경로 규약**  
  - 볼륨은 RunPod처럼 `/workspace` 마운트라고 가정.  
  - merged 모델을 S3로 올릴 때 **고정 경로 하나**로 맞추면 됨.  
  - 예: `/workspace/merged` (또는 `/workspace/api-llm-model`)
- **실행 방식**  
  - `python3 -m vllm.entrypoints.openai.api_server` 를 **`--model <위 경로>`** 로 기동하면 됨.

이미지 안에 모델을 넣을 필요는 없고, **베이스 + (선택) 엔트리포인트 + CMD**만 있으면 됩니다.

---

## 2. Dockerfile에 넣을 것

- **FROM**  
  - `vllm/vllm-openai:v0.11.0` (원하면 버전만 올려서 사용)
- **모델 경로 환경변수**  
  - 예: `ENV MODEL_PATH=/workspace/merged`  
  - S3에서 올릴 때 이 경로에 맞추면 됨.
- **엔트리포인트 (선택)**  
  - “볼륨에 모델이 있어야만 서버 기동” 하려면:
    - `$MODEL_PATH/config.json` 존재 여부만 확인
    - 있으면: `exec python3 -m vllm.entrypoints.openai.api_server "$@"`
    - 없으면: 에러 메시지 출력 후 `exit 1`
  - 이렇게 하면 모델 안 올린 상태로 Pod 띄우면 바로 실패해서 원인 파악이 쉬움.
- **RUN**  
  - 모델 다운로드/복사는 **하지 않음** (S3→볼륨만 사용).
- **EXPOSE 8000**  
  - OpenAI 호환 API 포트.
- **CMD**  
  - vLLM 서버 인자만 넘기면 됨.  
  - 예:  
    `["--model", "/workspace/merged", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "4096", "--tensor-parallel-size", "1"]`  
  - `MODEL_PATH`를 쓰려면 엔트리포인트에서 `$MODEL_PATH`로 넘기거나, CMD를 `["--model", "${MODEL_PATH}", ...]` 같이 쓰고 싶다면 엔트리포인트에서 `envsubst`로 치환 후 실행하는 식으로 할 수 있음 (보통은 고정 경로로 CMD만 두는 게 단순함).

정리하면, **Dockerfile 내용**은 대략:

- `FROM vllm/vllm-openai:...`
- `ENV MODEL_PATH=/workspace/merged`
- (선택) `COPY scripts/runpod_api_inference_entrypoint.sh /entrypoint.sh` + `ENTRYPOINT ["/entrypoint.sh"]`
- `EXPOSE 8000`
- `CMD ["--model", "/workspace/merged", "--host", "0.0.0.0", "--port", "8000", "--max-model-len", "4096", "--tensor-parallel-size", "1"]`

---

## 3. Docker CMD

- **역할**: vLLM OpenAI API 서버를 어떤 모델로 띄울지, 포트/옵션을 정하는 것.
- **형식**:  
  `python3 -m vllm.entrypoints.openai.api_server` 에 넘길 인자들.
- **최소한 필요한 것**  
  - `--model <경로>`: 위에서 정한 고정 경로 (예: `/workspace/merged`).  
    → **추론용 네트워크 볼륨**에 S3로 올릴 때 이 경로에 merged 모델이 오도록 하면 됨.
  - `--host 0.0.0.0`, `--port 8000` (필요 시)
  - `--max-model-len`, `--tensor-parallel-size` 등은 GPU/모델에 맞게.

즉, **“추론용 이미지 = vLLM 베이스 + (선택) 엔트리포인트 + CMD”**이고, **“모델은 이미지가 아니라 S3→추론용 네트워크 볼륨의 고정 경로”**라고 보면 됩니다.  
`api_llm_net_vol.md`에는 **볼륨 ID**와 함께 **S3 업로드 시 사용할 경로(예: `merged/` → 컨테이너에선 `/workspace/merged`)** 를 적어두면, 나중에 자동화할 때도 맞추기 쉽습니다.