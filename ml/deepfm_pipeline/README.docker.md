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

## 학습용 이미지 (deepfm-training)

- **역할**: 전처리 + 학습. `training_flow.py` 실행.
- **입력**: `data/training_dataset.csv`(기본) 또는 `data/raw/train.csv`, `data/raw/test.csv`
- **출력**: `data/`(전처리 결과), `output/<run_id>/`(모델, pipeline_version, 메타)

```bash
# 데이터·출력 디렉터리 마운트 후 실행 (기본 인자)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  deepfm-training

# 옵션 전달 (예: epoch, exp-ablation)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  deepfm-training --epochs 10 --exp-ablation "user_category_match,user_region_match,price_diff" --no-wandb
```

## 추론용 이미지 (deepfm-inference)

- **역할**: run 디렉터리 + 후보 CSV → 추천 결과 CSV (또는 S3 업로드).
- **기본 진입점**: `python -m utils.score_batch`

### 로컬 CSV → CSV

```bash
docker run --rm \
  -v /path/to/run_dir:/model \
  -v /path/to/candidates_dir:/data \
  deepfm-inference \
  --run-dir /model --candidates /data/candidates.csv --out /data/recommendations.csv
```

### Raw CSV로 추론 (run_dir에 categorical_dicts.json 등 필요)

```bash
docker run --rm \
  -v /path/to/run_dir:/model \
  -v /path/to/raw.csv:/data/raw_candidates.csv \
  deepfm-inference \
  --run-dir /model --raw-candidates /data/raw_candidates.csv --out /data/out.csv
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
