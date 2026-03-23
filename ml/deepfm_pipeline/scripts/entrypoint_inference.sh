#!/usr/bin/env bash
# 추론 이미지 엔트리포인트: S3 폴링 → Raw 다운로드 → 파이프라인 CSV 변환 → 추론 → (선택) S3 업로드
# 환경변수:
#   S3_ENV 또는 S3_BUCKET  - S3 폴링 시 버킷 (S3_ENV=dev|stg|prod)
#   RAW_BASE_PREFIX        - S3 raw base prefix (기본: raw)
#   RAW_DOWNLOAD_DIR       - /data/raw_download
#   RUN_DIR                - /model (모델 run 디렉터리, 볼륨 마운트 또는 아티팩트 다운로드 대상)
#   PIPELINE_VERSION       - RUN_DIR에 모델이 없을 때 사용. wandb artifact에서 해당 버전 다운로드 (WANDB_API_KEY 필요)
#   OUT_PATH               - /data/recommendations.csv (UPLOAD_TO_S3 미설정 시 로컬 출력)
#   CANDIDATES_CSV         - /data/raw_candidates.csv
#   SKIP_S3_POLL=1         - 다운로드/변환 생략
#   UPLOAD_TO_S3=1         - 추론 후 결과를 S3에 업로드 (recommendations/... + _SUCCESS). S3_ENV 필요.
#   RECOMMENDATION_DT      - 선택. YYYY-MM-DD (기본: UTC 오늘)
#   RECOMMENDATION_OUTPUT_FORMAT - csv | json.gz (기본: csv). S3 업로드 파일 형식.
#   WANDB_API_KEY          - PIPELINE_VERSION 사용 시 아티팩트 다운로드에 필요.

set -e
cd /app

RAW_DOWNLOAD_DIR="${RAW_DOWNLOAD_DIR:-/data/raw_download}"
RAW_BASE_PREFIX="${RAW_BASE_PREFIX:-raw}"
RUN_DIR="${RUN_DIR:-/model}"
ARTIFACT_CACHE_DIR="${ARTIFACT_CACHE_DIR:-/model}"
OUT_PATH="${OUT_PATH:-/data/recommendations.csv}"
CANDIDATES_CSV="${CANDIDATES_CSV:-/data/raw_candidates.csv}"
mkdir -p "$(dirname "$OUT_PATH")" "$(dirname "$CANDIDATES_CSV")" "$ARTIFACT_CACHE_DIR"

# 모델 run_dir 또는 pipeline_version 인자 결정: RUN_DIR에 model.pt가 있으면 --run-dir, 없으면 --pipeline-version + 아티팩트 캐시
RUN_ARGS=()
if [ -f "${RUN_DIR}/model.pt" ] && [ -f "${RUN_DIR}/feature_sizes.txt" ]; then
  RUN_ARGS=(--run-dir "$RUN_DIR")
elif [ -n "${PIPELINE_VERSION}" ]; then
  RUN_ARGS=(--pipeline-version "$PIPELINE_VERSION" --artifact-cache-dir "$ARTIFACT_CACHE_DIR")
  [ -n "${WANDB_PROJECT}" ] && RUN_ARGS+=(--wandb-project "$WANDB_PROJECT")
  [ -n "${WANDB_ENTITY}" ] && RUN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  if [ -z "${WANDB_API_KEY}" ]; then
    echo "PIPELINE_VERSION is set but WANDB_API_KEY is not. Set WANDB_API_KEY to download model from artifact."
    exit 1
  fi
else
  echo "Model not found at RUN_DIR=${RUN_DIR} (need model.pt and feature_sizes.txt). Set RUN_DIR or PIPELINE_VERSION (and WANDB_API_KEY)."
  exit 1
fi

if [ -z "${SKIP_S3_POLL}" ]; then
  POLL_ARGS=()
  if [ -n "${AWS_PROFILE}" ]; then
    POLL_ARGS=(--profile "${AWS_PROFILE}")
  fi
  if [ -n "${S3_BUCKET}" ]; then
    python scripts/s3_raw_poll_download.py --bucket "$S3_BUCKET" --out-dir "$RAW_DOWNLOAD_DIR" --base-prefix "$RAW_BASE_PREFIX" "${POLL_ARGS[@]}"
  elif [ -n "${S3_ENV}" ]; then
    python scripts/s3_raw_poll_download.py --env "$S3_ENV" --out-dir "$RAW_DOWNLOAD_DIR" --base-prefix "$RAW_BASE_PREFIX" "${POLL_ARGS[@]}"
  else
    echo "S3_BUCKET or S3_ENV not set. Set SKIP_S3_POLL=1 and pass --raw-candidates /path/to/candidates.csv"
    exit 1
  fi
  python scripts/raw_to_pipeline_csv.py --raw-dir "$RAW_DOWNLOAD_DIR" --out "$CANDIDATES_CSV" --base-prefix "$RAW_BASE_PREFIX"
  if [ ! -s "$CANDIDATES_CSV" ]; then
    echo "No candidate rows after raw transform. Check RAW_BASE_PREFIX=${RAW_BASE_PREFIX} and source raw schema."
    exit 1
  fi
fi

if [ -n "${UPLOAD_TO_S3}" ] && [ "${UPLOAD_TO_S3}" = "1" ]; then
  if [ -z "${S3_ENV}" ]; then
    echo "UPLOAD_TO_S3=1 requires S3_ENV (dev|stg|prod)"
    exit 1
  fi
  S3_ARGS=("${RUN_ARGS[@]}" --raw-candidates "$CANDIDATES_CSV" --env "$S3_ENV")
  if [ -n "${AWS_PROFILE}" ]; then
    S3_ARGS+=(--profile "${AWS_PROFILE}")
  fi
  [ -n "${RECOMMENDATION_DT}" ] && S3_ARGS+=(--dt "$RECOMMENDATION_DT")
  [ -n "${RECOMMENDATION_OUTPUT_FORMAT}" ] && S3_ARGS+=(--output-format "$RECOMMENDATION_OUTPUT_FORMAT")
  exec python scripts/score_batch_to_s3.py "${S3_ARGS[@]}"
fi

exec python -m utils.score_batch \
  "${RUN_ARGS[@]}" \
  --raw-candidates "$CANDIDATES_CSV" \
  --out "$OUT_PATH" \
  "$@"
