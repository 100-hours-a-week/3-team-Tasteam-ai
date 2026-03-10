#!/usr/bin/env bash
# 추론 이미지 엔트리포인트: S3 폴링 → Raw 다운로드 → 파이프라인 CSV 변환 → score_batch (--raw-candidates)
# 환경변수: S3_ENV 또는 S3_BUCKET, RAW_DOWNLOAD_DIR(/data/raw_download), RUN_DIR(모델 run 디렉터리), OUT_PATH(/data/recommendations.csv)
#          SKIP_S3_POLL=1 이면 다운로드/변환 생략 (기존 후보 CSV 또는 인자로 전달)

set -e
cd /app

RAW_DOWNLOAD_DIR="${RAW_DOWNLOAD_DIR:-/data/raw_download}"
RUN_DIR="${RUN_DIR:-/model}"
OUT_PATH="${OUT_PATH:-/data/recommendations.csv}"
CANDIDATES_CSV="${CANDIDATES_CSV:-/data/raw_candidates.csv}"
mkdir -p "$(dirname "$OUT_PATH")" "$(dirname "$CANDIDATES_CSV")"

if [ -z "${SKIP_S3_POLL}" ]; then
  if [ -n "${S3_BUCKET}" ]; then
    python scripts/s3_raw_poll_download.py --bucket "$S3_BUCKET" --out-dir "$RAW_DOWNLOAD_DIR"
  elif [ -n "${S3_ENV}" ]; then
    python scripts/s3_raw_poll_download.py --env "$S3_ENV" --out-dir "$RAW_DOWNLOAD_DIR"
  else
    echo "S3_BUCKET or S3_ENV not set. Set SKIP_S3_POLL=1 and pass --raw-candidates /path/to/candidates.csv"
    exit 1
  fi
  python scripts/raw_to_pipeline_csv.py --raw-dir "$RAW_DOWNLOAD_DIR" --out "$CANDIDATES_CSV"
fi

exec python -m utils.score_batch \
  --run-dir "$RUN_DIR" \
  --raw-candidates "$CANDIDATES_CSV" \
  --out "$OUT_PATH" \
  "$@"
