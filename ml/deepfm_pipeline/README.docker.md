# DeepFM Docker 이미지

## 빌드

컨텍스트는 `ml/deepfm_pipeline` 디렉터리.

```bash
cd ml/deepfm_pipeline

# 학습용
docker build -f Dockerfile.training -t deepfm-training .

# 추론용
docker build -f Dockerfile.inference -t deepfm-inference .
```

---

## 학습용 이미지 (deepfm-training)

**기본 흐름**: S3 폴링(_SUCCESS 있는 파티션만) → Raw 다운로드 → 파이프라인 CSV 변환 → `training_flow.py` (split → preprocess → train).

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `S3_ENV` | - | dev / stg / prod → `tasteam-{env}-analytics` 버킷 (S3_BUCKET 대신 사용 가능) |
| `S3_BUCKET` | - | S3 버킷 직접 지정 |
| `RAW_DOWNLOAD_DIR` | `/data/raw_download` | Raw 다운로드 기준 디렉터리 |
| `DATA_DIR` | `/data` | 변환 CSV·전처리 결과 저장 (raw, train.txt 등) |
| `OUTPUT_DIR` | `/output` | 모델·run 산출물 |
| `SKIP_S3_POLL` | (비설정) | 1 이면 S3 다운로드/변환 생략, 기존 `DATA_DIR/raw/training_dataset.csv` 사용 |

### 실행 예 (S3 → 학습)

```bash
docker run --rm \
  -e S3_ENV=dev \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  deepfm-training

# 학습 옵션 전달
docker run --rm -e S3_ENV=dev -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -v "$(pwd)/data:/data" -v "$(pwd)/output:/output" \
  deepfm-training --epochs 10 --exp-ablation "user_category_match,user_region_match,price_diff" --no-wandb
```

### 실행 예 (S3 없이, 기존 데이터로)

```bash
# data/raw/training_dataset.csv 가 이미 있을 때
docker run --rm \
  -e SKIP_S3_POLL=1 \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/output:/output" \
  deepfm-training
```

---

## 추론용 이미지 (deepfm-inference)

**기본 흐름**: S3 폴링 → Raw 다운로드 → 파이프라인 CSV 변환 → `score_batch --raw-candidates` → 추천 결과 CSV.

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `S3_ENV` / `S3_BUCKET` | - | S3 버킷 (S3 폴링 시) |
| `RAW_DOWNLOAD_DIR` | `/data/raw_download` | Raw 다운로드 위치 |
| `RUN_DIR` | `/model` | 모델 run 디렉터리 (model.pt, categorical_dicts.json 등) |
| `OUT_PATH` | `/data/recommendations.csv` | 추천 결과 CSV 출력 경로 |
| `CANDIDATES_CSV` | `/data/raw_candidates.csv` | 변환된 후보 CSV (S3 폴링 시 생성, SKIP_S3_POLL 시 직접 마운트) |
| `SKIP_S3_POLL` | (비설정) | 1 이면 다운로드/변환 생략, `CANDIDATES_CSV` 또는 인자로 후보 전달 |

### 실행 예 (S3 → 추론)

```bash
docker run --rm \
  -e S3_ENV=dev \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  -v /path/to/run_dir:/model \
  -v /path/to/data:/data \
  deepfm-inference
```

### 실행 예 (로컬 후보만, S3 미사용)

```bash
docker run --rm \
  -e SKIP_S3_POLL=1 \
  -e CANDIDATES_CSV=/data/raw_candidates.csv \
  -v /path/to/run_dir:/model \
  -v /path/to/data:/data \
  deepfm-inference
# RUN_DIR, OUT_PATH 등은 기본값 또는 인자로 덮어쓰기
```

### S3 업로드 (score_batch_to_s3)

```bash
docker run --rm \
  -v /path/to/run_dir:/model \
  -v /path/to/candidates.csv:/data/candidates.csv \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION \
  --entrypoint python deepfm-inference scripts/score_batch_to_s3.py \
  --pipeline-version deepfm-1.0.xxxx --run-dir /model \
  --candidates-path /data/candidates.csv --env dev
```

---

## 공통: S3 Raw 계약

- **폴링 기준**: 각 파티션(`dt=YYYY-MM-DD`)에 **`_SUCCESS`** 가 있을 때만 해당 파티션 다운로드.  
  구현: `scripts/s3_raw_poll_download.py`
- **Raw → 파이프라인 CSV**: events + restaurants + menus 조인·매핑.  
  구현: `utils/raw_to_pipeline.py`, `scripts/raw_to_pipeline_csv.py`  
  계약: `docs/service_extraction/service_constract.md` §4
