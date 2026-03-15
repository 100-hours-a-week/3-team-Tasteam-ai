# DeepFM 배치 추론 Prefect Flow

배치 추론(S3 폴링 → 추론 → S3 업로드)을 **하나의 Prefect flow**로 묶어, Prefect 서버/UI에서 주기 실행·모니터링할 수 있도록 한 설정 가이드.

## Flow 개요

**Flow 이름**: `DeepFM Batch Inference Pipeline`  
**진입점**: `inference_flow.py` → `deepfm_inference_flow`

이 flow는 **워크플로우 한 종류만** 수행한다: S3 Raw 다운로드 → 후보 CSV 변환 → DeepFM 추론 → S3 업로드.

목록 조회만 / 다운로드만이 필요하면 Prefect가 아니라 **CLI 스크립트**를 사용한다.

- **목록만**: `python scripts/s3_raw_poll_download.py --env dev --list-only`
- **다운로드만**: `python scripts/s3_raw_poll_download.py --env dev --out-dir ./data/raw_download`

### Task 구성

- **deepfm-s3-download-raw**: S3 Raw 다운로드 (_SUCCESS 있는 파티션만)
- **deepfm-raw-to-candidates-csv**: Raw → 후보 CSV 변환
- **deepfm-inference-and-upload**: 추론 + S3 업로드 (part-00001.csv/json.gz + _SUCCESS)

---

## 로컬 실행 (CLI)

```bash
# deepfm_pipeline 디렉터리에서
cd ml/deepfm_pipeline

# Prefect flow: 폴링 → 추론 → 업로드 (한 번에)
python inference_flow.py --env dev --out-dir ./data/raw_download --run-dir ./output/deepfm-1.0.xxxx
```

**옵션**: `--profile jayvi`, `--dt 2025-03-15`, `--data-types events,restaurants,menus`, `--output-format json.gz` 등.

목록만/다운로드만은 flow가 아니라 스크립트로:

```bash
python scripts/s3_raw_poll_download.py --env dev --list-only
python scripts/s3_raw_poll_download.py --env dev --out-dir ./data/raw_download
```

---

## Prefect 서버/UI 연동

### 1. Prefect 서버 기동

```bash
# Prefect 2.x
prefect server start
# UI: http://127.0.0.1:4200
```

### 2. Flow 등록 (배포)

Flow를 Prefect에 등록해 스케줄·트리거로 실행하려면 **deployment**를 만든다.

```bash
cd ml/deepfm_pipeline

# 로컬 스크립트 기반 deployment (추천)
prefect deployment build inference_flow.py:deepfm_inference_flow \
  -n "batch-inference-daily" \
  -q default \
  --cron "0 2 * * *" \
  --param env=dev \
  --param out_dir=./data/raw_download \
  --param run_dir=./output/active_run \
  -a
```

- `--cron "0 2 * * *"`: 매일 02:00 (로컬 타임존)
- `-a`: 적용(apply)하여 서버에 등록
- `--param run_dir=...`: 실제 모델 run 경로로 변경

### 3. 수동 실행 (UI)

1. Prefect UI → **Deployments** → 해당 deployment 선택
2. **Run** → Parameters에서 `env`, `out_dir`, `run_dir` 등 필요 시 수정 후 실행

### 4. 워커 실행

Deployment가 큐에서 flow run을 가져가려면 **worker**를 띄운다.

```bash
cd ml/deepfm_pipeline
prefect worker start -q default
```

(실행 환경에 맞게 `prefect worker start` 또는 `prefect agent start` 사용.)

---

## 주기 배치 권장 설정

- **Prefect flow**: 추천 결과 일배치용. **cron 1회/일** (예: `0 2 * * *`) + `env`, `out_dir`, `run_dir` 고정.
- **목록/다운로드만**: Prefect가 아닌 `scripts/s3_raw_poll_download.py`를 수동 또는 별도 스크립트로 실행.

---

## 환경 변수

- **AWS_PROFILE**: S3 접근용 프로필 (예: `jayvi`). Flow 파라미터 `profile_name`으로도 전달 가능.
- **AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY**: 프로필 대신 사용 시.

---

## 참고

- 학습 파이프라인: `training_flow.py` → `deepfm_training_flow` (Prefect Training Pipeline).
- S3 계약: `docs/service_extraction/service_constract.md`
- Docker 추론 이미지: `README.docker.md` (엔트리포인트로도 동일 흐름 실행 가능).
