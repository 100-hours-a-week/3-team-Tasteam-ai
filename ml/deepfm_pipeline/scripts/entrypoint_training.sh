#!/usr/bin/env bash
# 학습 이미지 엔트리포인트: S3 폴링(_SUCCESS 기준) → Raw 다운로드 → 파이프라인 CSV 변환 → training_flow
# 환경변수: S3_ENV(dev|stg|prod) 또는 S3_BUCKET, RAW_DOWNLOAD_DIR(/data/raw_download), DATA_DIR(/data), OUTPUT_DIR(/output)
#          SKIP_S3_POLL=1 이면 다운로드/변환 생략 후 바로 training_flow

set -e
cd /app

RAW_DOWNLOAD_DIR="${RAW_DOWNLOAD_DIR:-/data/raw_download}"
DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$RAW_DOWNLOAD_DIR" "$DATA_DIR" "$OUTPUT_DIR"
mkdir -p "$DATA_DIR/raw"

if [ -z "${SKIP_S3_POLL}" ]; then
  POLL_ARGS=(--profile "${AWS_PROFILE:-jayvi}")
  if [ -n "${S3_BUCKET}" ]; then
    python scripts/s3_raw_poll_download.py --bucket "$S3_BUCKET" --out-dir "$RAW_DOWNLOAD_DIR" "${POLL_ARGS[@]}"
  elif [ -n "${S3_ENV}" ]; then
    python scripts/s3_raw_poll_download.py --env "$S3_ENV" --out-dir "$RAW_DOWNLOAD_DIR" "${POLL_ARGS[@]}"
  else
    echo "S3_BUCKET or S3_ENV not set, skipping S3 download. Set SKIP_S3_POLL=1 to use existing data."
    exit 1
  fi
  python scripts/raw_to_pipeline_csv.py --raw-dir "$RAW_DOWNLOAD_DIR" --out "$DATA_DIR/raw/training_dataset.csv"
  if [ ! -s "$DATA_DIR/raw/training_dataset.csv" ]; then
    echo "No rows after transform. Check S3 raw data (events/restaurants/menus with _SUCCESS)."
    exit 1
  fi
fi

# training_flow: source가 있으면 split → preprocess → train
exec python training_flow.py \
  --source "$DATA_DIR/raw/training_dataset.csv" \
  --raw-dir "$DATA_DIR/raw" \
  --processed-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
