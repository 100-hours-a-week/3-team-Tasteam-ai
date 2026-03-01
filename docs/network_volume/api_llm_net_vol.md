# API 추론용 네트워크 볼륨 (API_LLM)

| 항목 | 값 |
|------|-----|
| Name | API_LLM |
| Size | 10 GB |
| Data center | EU-SE-1 |
| Monthly cost | $0.70 |
| Volume ID | n5e655a7w5 |
| Bucket name | n5e655a7w5 |
| Endpoint URL | https://s3api-eu-se-1.runpod.io (리전에 맞게) |

## 모델 경로 규약

- **컨테이너 내 경로**: `/workspace/merged`
- **S3 업로드 prefix**: `merged/` — merged 모델 디렉터리 전체를 이 prefix 아래 업로드.
- 이미지: `Dockerfile.api-llm` / 엔트리포인트: `scripts/runpod_api_inference_entrypoint.sh`
- Pod 생성 시 이 볼륨을 `/workspace`에 마운트하면, vLLM이 `/workspace/merged`에서 모델 로드.

## merged 모델 업로드

**스크립트 (권장)** — `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_ACCESS_KEY` 설정 후:

```bash
# 최신 merged 디렉터리 자동 선택 후 업로드
python scripts/sync_merged_to_api_llm_volume.py --latest

# 특정 디렉터리 지정
python scripts/sync_merged_to_api_llm_volume.py --source-dir ./distill_pipeline_output/merged_for_serving/YYYYMMDD_HHMMSS

# 업로드 없이 대상만 확인
python scripts/sync_merged_to_api_llm_volume.py --latest --dry-run
```

**수동 aws s3 sync 예시:**

```bash
# merged 모델 로컬 경로: ./distill_pipeline_output/merged_for_serving/YYYYMMDD_HHMMSS/
aws s3 sync ./distill_pipeline_output/merged_for_serving/YYYYMMDD_HHMMSS/ s3://n5e655a7w5/merged/ \
  --endpoint-url https://s3api-eu-se-1.runpod.io --region eu-se-1
```

