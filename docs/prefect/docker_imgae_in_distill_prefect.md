# Prefect Pod 생성 시 Docker 이미지 (환경변수)

Prefect flow에서 RunPod Pod를 생성할 때 사용하는 Docker 이미지는 **용도별 환경변수**로 지정한다.

| 용도     | 환경변수                           | 기본값                              |
|----------|------------------------------------|-------------------------------------|
| labeling | `RUNPOD_POD_IMAGE_NAME_LABELING`  | `jinsoo1218/runpod-pod-vllm:latest` |
| train    | `RUNPOD_POD_IMAGE_NAME_TRAIN`      | `jinsoo1218/train-llm:latest`       |

구현: `scripts/runpod_cli/pod_create_delete_cli.py`의 `get_default_pod_payload(use="labeling"|"train")`에서 해당 env 사용.  
labeling_with_pod_task는 `use="labeling"`, train Pod 생성 시에는 `use="train"` 전달.

참고: [create_pod_docker_image.md](../runpod_cli/create_pod_docker_image.md)
