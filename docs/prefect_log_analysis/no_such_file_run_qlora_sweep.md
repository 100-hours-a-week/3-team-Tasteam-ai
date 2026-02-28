
원인은 **RunPod에서 쓰는 학습용 이미지 안에 `run_qlora_sweep.py`가 없다는 것**입니다.

---

## 원인

1. **코드 쪽**
   - `run_sweep_on_pod_task`에서 Pod를 띄울 때  
     `dockerEntrypoint = ["python", "/app/scripts/run_qlora_sweep.py"]` 로 실행합니다.  
     (`distill_flows.py` 641행 근처)
   - 학습용 Pod 이미지는 `RUNPOD_POD_IMAGE_NAME_TRAIN`(기본값 `jinsoo1218/train-llm:latest`)입니다.  
     (`pod_create_delete_cli.py` 148행)

2. **이미지 쪽**
   - `Dockerfile.train-llm`에는 이미 다음이 들어가 있어서, **이 Dockerfile로 최신 코드 기준으로 빌드한 이미지**라면 `/app/scripts/run_qlora_sweep.py`가 있어야 합니다.

```46:48:Dockerfile.train-llm
# train_qlora.py, run_qlora_sweep.py (Pod에서 sweep 시 사용)
COPY scripts/train_qlora.py /app/scripts/
COPY scripts/run_qlora_sweep.py /app/scripts/
```

그런데 Pod에서는 이 파일이 없다고 하므로, **현재 RunPod에 배포된 이미지(`jinsoo1218/train-llm:latest`)는 예전에 빌드된 것**이거나, **`run_qlora_sweep.py`가 포함되기 전 Dockerfile / 빌드 컨텍스트로 만든 이미지**일 가능성이 큽니다.

---

## 해결 방법

**학습용 이미지를 `Dockerfile.train-llm`으로 다시 빌드하고, 같은 이름으로 푸시**하면 됩니다.  
(이미 사용 중인 `jinsoo1218/train-llm:latest`를 갱신하는 방식)

1. **로컬에서 빌드 & 푸시 (Docker Hub 사용 시)**

   ```bash
   docker build -f Dockerfile.train-llm -t jinsoo1218/train-llm:latest .
   docker push jinsoo1218/train-llm:latest
   ```

2. **다른 레지스트리/태그를 쓰는 경우**
   - `RUNPOD_POD_IMAGE_NAME_TRAIN`에 지정한 이미지 이름/태그를 사용해 위처럼 빌드·푸시하면 됩니다.

빌드 시 프로젝트 루트에서 실행해 `scripts/run_qlora_sweep.py`가 빌드 컨텍스트에 포함되는지 확인하세요.  
이미지 재빌드·푸시 후 Pod를 다시 띄우면 `/app/scripts/run_qlora_sweep.py`가 있어서 같은 에러는 사라져야 합니다.