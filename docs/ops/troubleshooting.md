# Troubleshooting (Docker/Distill)

## Quick Checks

1. `docker: command not found`  
   -> Docker Desktop/Engine 설치 및 실행 확인
2. API 키 오류 (`RUNPOD_API_KEY`, `WANDB_API_KEY`)  
   -> `.env` 존재/값 유효성 확인
3. `adapter path is not a directory`  
   -> `DISTILL_ADAPTER_PATH`가 실제 adapter 디렉터리인지 확인
4. Batch 미처리  
   -> `make up-batch` 후 `redis`, `spark-service`, `batch-worker` 상태 확인
5. Metrics 비노출  
   -> `/metrics` 엔드포인트 및 Prometheus target 상태 확인

## Compose Validation

```bash
docker compose -f compose.base.yml -f compose.app.yml config
docker compose -f compose.base.yml -f compose.batch.yml config
docker compose -f compose.base.yml -f compose.stack.yml config
docker compose -f compose.base.yml -f compose.stack.yml ps
```

## Related Docs

- RunPod Pod troubleshooting: `docs/troubleshooting/runpod/runpod_pod/trouble_shooting/TROUBLESHOOTING_SUMMARY.md`
- RunPod Serverless troubleshooting: `docs/troubleshooting/runpod/runpod_serverless/trouble_shooting/TROUBLESHOOTING_SUMMARY.md`
