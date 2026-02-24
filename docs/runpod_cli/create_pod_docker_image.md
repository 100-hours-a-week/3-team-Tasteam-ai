# Pod 생성 시 Docker Hub 이미지 (용도별)

- **labeling**: `jinsoo1218/runpod-pod-vllm:latest` → 환경변수 `RUNPOD_POD_IMAGE_NAME_LABELING`
- **train**: `jinsoo1218/train-llm:latest` → 환경변수 `RUNPOD_POD_IMAGE_NAME_TRAIN`

Prefect에서 Pod 생성 시 flow가 labeling이면 labeling용, train이면 train용 이미지를 사용.  
자세한 사용법: [docker_imgae_in_distill_prefect.md](../prefect/docker_imgae_in_distill_prefect.md)
