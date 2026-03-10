# Admin DeepFM API DTO 명세

`api_design.md` 기준 엔드포인트별 **Request / Response DTO** 정의.  
구현: `api/schemas.py`, `api/routers/deepfm.py`.

---

## 1) POST /admin/deepfm/train

학습 트리거. Prefect 학습 플로우 실행 → 모델 artifact + pipeline_version 산출.

### Request body (JSON, 전부 선택)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `raw_data_dir` | string \| null | (파이프라인 기본) | raw 데이터 디렉터리 S3 URL (train.csv 등, 예: s3://bucket/data/raw) |
| `processed_data_dir` | string \| null | (파이프라인 기본) | 전처리 결과 저장 S3 URL |
| `num_train_sample` | integer \| null | null | 학습 샘플 수 상한 |
| `num_test_sample` | integer \| null | null | 테스트 샘플 수 상한 |
| `num_val` | integer \| null | 1000 | 검증 샘플 수 |
| `epochs` | integer \| null | 5 | 에폭 수 |
| `batch_size` | integer \| null | 100 | 배치 크기 |
| `lr` | number \| null | 1e-4 | 학습률 |
| `output_dir` | string \| null | (파이프라인 기본) | 모델/run 산출물 S3 URL |
| `use_cuda` | boolean | false | GPU 사용 여부 |
| `skip_preprocess` | boolean | false | 전처리 생략 여부 |
| `use_sample_weight` | boolean | true | sample_weight 사용 여부 |
| `time_column` | string \| null | null | 시간 기준 split 컬럼명 |
| `train_end` | string \| null | null | train 구간 끝 (시간) |
| `valid_end` | string \| null | null | valid 구간 끝 |
| `test_end` | string \| null | null | test 구간 끝 |
| `group_column` | string \| null | null | recommendation 단위 그룹 컬럼 |
| `use_wandb` | boolean | true | W&B 로깅 여부 |

### Response (200)

| 필드 | 타입 | 설명 |
|------|------|------|
| `pipeline_version` | string | 발급된 pipeline_version (예: deepfm-1.0.20260227120000) |
| `model_path` | string | 저장된 model.pt S3 URL |
| `run_manifest_path` | string | run_manifest.json S3 URL |
| `metrics` | object \| null | 오프라인 지표 (NDCG@5, NDCG@10, AUC 등). test 없으면 null |

### 입출력 예시

**Request (최소):**
```json
{}
```

**Request (일부 옵션 지정):**
```json
{
  "raw_data_dir": "s3://my-bucket/deepfm/data/raw",
  "epochs": 10,
  "batch_size": 128,
  "use_cuda": true,
  "num_train_sample": 50000,
  "num_val": 2000
}
```

**Response (200):**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000",
  "model_path": "s3://my-bucket/deepfm/output/deepfm-1.0.20260227120000/model.pt",
  "run_manifest_path": "s3://my-bucket/deepfm/output/deepfm-1.0.20260227120000/run_manifest.json",
  "metrics": {
    "ndcg_at_5": 0.412,
    "ndcg_at_10": 0.388,
    "auc": 0.721
  }
}
```

**Response (test 없음 시 metrics null):**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000",
  "model_path": "s3://my-bucket/deepfm/output/deepfm-1.0.20260227120000/model.pt",
  "run_manifest_path": "s3://my-bucket/deepfm/output/deepfm-1.0.20260227120000/run_manifest.json",
  "metrics": null
}
```

### Error

- **404**: raw_data_dir 없음 등 FileNotFoundError
- **500**: 학습 중 예외

---

## 2) POST /admin/deepfm/score-batch

배치 스코어링/추천 생성 트리거. recommendation 형식 CSV 출력.  
**recommendation 테이블 INSERT는 호출 측(ETL/DB)에서 수행.**

### Request body (JSON)

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `pipeline_version` | string | O | - | 사용할 모델 버전 |
| `run_dir` | string \| null | - | null | run 디렉터리 S3 URL. 없으면 pipeline_version으로 output 하위에서 탐색 |
| `candidates_path` | string | O | - | 후보 CSV S3 URL (전처리된 feature 열) |
| `output_path` | string | O | - | recommendation CSV 출력 S3 URL |
| `meta_path` | string \| null | - | null | user_id, anonymous_id, restaurant_id, context_snapshot 메타 CSV S3 URL |
| `ttl_hours` | number | - | 24 | expires_at TTL(시간) |
| `batch_size` | integer | - | 256 | 추론 배치 크기 |

### Response (200)

| 필드 | 타입 | 설명 |
|------|------|------|
| `pipeline_version` | string | 사용된 pipeline_version |
| `output_path` | string | 출력 CSV S3 URL |
| `rows_written` | integer | 출력된 recommendation 행 수 |

### 입출력 예시

**Request:**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000",
  "candidates_path": "s3://my-bucket/deepfm/candidates.csv",
  "output_path": "s3://my-bucket/deepfm/recommendations.csv",
  "meta_path": "s3://my-bucket/deepfm/candidates_meta.csv",
  "ttl_hours": 24,
  "batch_size": 256
}
```

**Request (meta_path 생략):**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000",
  "candidates_path": "s3://my-bucket/deepfm/candidates.csv",
  "output_path": "s3://my-bucket/deepfm/recommendations.csv"
}
```

**Response (200):**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000",
  "output_path": "s3://my-bucket/deepfm/recommendations.csv",
  "rows_written": 15230
}
```

### Error

- **404**: run_dir/pipeline_version에 해당하는 run 없음, 또는 candidates_path 없음
- **500**: 스코어링 중 예외

---

## 3) GET /admin/deepfm/models

모델/버전 목록 조회 및 현재 활성(서빙) pipeline_version.

### Request

- Query 파라미터 없음.

### Response (200)

| 필드 | 타입 | 설명 |
|------|------|------|
| `models` | array | 모델(버전) 목록 |
| `models[].pipeline_version` | string | pipeline_version |
| `models[].run_dir` | string | run 디렉터리 S3 URL |
| `models[].created_at` | string \| null | run_manifest 기준 생성 시각 (ISO 8601) |
| `models[].metrics` | object \| null | run_manifest의 metrics (NDCG@K, AUC 등) |
| `active_version` | string \| null | 현재 활성(서빙) pipeline_version. 없으면 null |

### 입출력 예시

**Response (200):**
```json
{
  "models": [
    {
      "pipeline_version": "deepfm-1.0.20260227120000",
      "run_dir": "s3://my-bucket/deepfm/output/deepfm-1.0.20260227120000",
      "created_at": "2026-02-27T12:00:00+00:00",
      "metrics": { "ndcg_at_5": 0.412, "ndcg_at_10": 0.388, "auc": 0.721 }
    },
    {
      "pipeline_version": "deepfm-1.0.20260226100000",
      "run_dir": "s3://my-bucket/deepfm/output/deepfm-1.0.20260226100000",
      "created_at": "2026-02-26T10:00:00+00:00",
      "metrics": null
    }
  ],
  "active_version": "deepfm-1.0.20260227120000"
}
```

---

## 4) POST /admin/deepfm/activate

서빙용 pipeline_version 활성화.  
활성 버전은 `output/active_pipeline_version.txt`에 저장되며, 추천 API 등에서 참조.

### Request body (JSON)

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `pipeline_version` | string | O | 활성화할 pipeline_version |

### Response (200)

| 필드 | 타입 | 설명 |
|------|------|------|
| `active_version` | string | 현재 활성 pipeline_version |

### 입출력 예시

**Request:**
```json
{
  "pipeline_version": "deepfm-1.0.20260227120000"
}
```

**Response (200):**
```json
{
  "active_version": "deepfm-1.0.20260227120000"
}
```

### Error

- **404**: 해당 pipeline_version의 run이 output S3 prefix 하위에 없음

---

## CSV 예시

### 학습용 raw 데이터 (train.csv)

학습 트리거 시 `raw_data_dir`(S3 prefix) 내 `train.csv` (및 선택 시 `test.csv`) 형식.  
전처리 스크립트가 기대하는 컬럼 예시 (tasteam_deepfm_data.md / dataPreprocess 기준).

```csv
user_id,anonymous_id,restaurant_id,taste_preferences,visit_time_distribution,is_anonymous,avg_price_tier,primary_category,pref_cat_1,pref_cat_2,pref_cat_3,price_tier,region_gu,region_dong,geohash,day_of_week,time_slot,admin_dong,distance_bucket,weather_bucket,dining_type,first_positive_segment,first_comparison_tag,pref_w_1,pref_w_2,pref_w_3,signal_type,generated_at,recommendation_id
1001,,rest_101,"{""spicy"":0.2,""sweet"":0.5}","{""breakfast"":0.1,""lunch"":0.6}",0,2,한식,한식,중식,,강남구,역삼동,wydm7,,lunch,,,1,2,1,,0.5,0.3,0.2,REVIEW,2026-02-27T10:00:00,rec_001
,anon_002,rest_202,"{""spicy"":0.8}","{""dinner"":0.9}",1,1,중식,중식,,,1,서초구,서초동,wydm6,,dinner,,,2,1,0,,0.6,0.2,0.2,CLICK,2026-02-27T11:00:00,
```

- 실제 컬럼 집합은 파이프라인/전처리 스키마에 따라 다를 수 있음.  
- `time_column`, `group_column` 사용 시 해당 컬럼 필요.

---

### 스코어링 입력: 후보 CSV (candidates_path)

전처리된 feature 벡터 CSV. **컬럼 수 = 해당 run의 feature_sizes 개수**(연속형 + 범주형 인덱스).  
헤더 없음 또는 있음 모두 가능. 행 순서는 meta_path와 동일해야 함.

**예시 (feature_sizes가 12+20 = 32개일 때, 앞 12개 연속·뒤 20개 범주 인덱스):**

```csv
0.2,0.5,0.1,0.0,0.1,0.6,0.0,0.9,0,0.5,0.3,0.2,1,5,2,101,3,1,2,3,1,7,2,42,3,1,2,1,1,0,1,2
0.8,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1,0.6,0.2,0.2,0,8,1,202,2,2,0,0,1,2,1,38,5,2,1,2,0,0,1,1
```

- 위는 연속 12개 + 범주 인덱스 20개(전부 숫자)로, 한 행이 한 후보(user–restaurant 등)에 대응.

---

### 스코어링 입력: 메타 CSV (meta_path, 선택)

후보별 user_id, anonymous_id, restaurant_id, context_snapshot. **행 순서는 candidates_path와 동일.**

```csv
user_id,anonymous_id,restaurant_id,context_snapshot
1001,,rest_101,{}
,anon_002,rest_202,"{""lat"":37.5,""lng"":127.0}"
,anon_003,rest_303,{}
```

- `user_id`·`anonymous_id` 비어 있으면 anonymous_id로 그룹핑.  
- `context_snapshot`은 JSON 문자열 또는 빈 값/`{}`.

---

### 스코어링 출력: recommendation CSV (output_path)

배치 스코어링 응답으로 쓰이는 CSV(S3에 저장). DB recommendation 테이블 INSERT는 호출 측(ETL)에서 수행.

```csv
user_id,anonymous_id,restaurant_id,score,rank,context_snapshot,pipeline_version,generated_at,expires_at
1001,,rest_101,0.892,1,{},deepfm-1.0.20260227120000,2026-02-27T14:00:00.000000+00:00,2026-02-28T14:00:00.000000+00:00
1001,,rest_205,0.654,2,{},deepfm-1.0.20260227120000,2026-02-27T14:00:00.000000+00:00,2026-02-28T14:00:00.000000+00:00
,anon_002,rest_202,0.771,1,"{""lat"":37.5,""lng"":127.0}",deepfm-1.0.20260227120000,2026-02-27T14:00:00.000000+00:00,2026-02-28T14:00:00.000000+00:00
,anon_003,rest_303,0.543,1,{},deepfm-1.0.20260227120000,2026-02-27T14:00:00.000000+00:00,2026-02-28T14:00:00.000000+00:00
```

- `rank`: (user_id 또는 anonymous_id 기준) 그룹 내 점수 순위.  
- `generated_at` / `expires_at`: ISO 8601. `expires_at` = generated_at + ttl_hours.

---

## 공통

- **Content-Type**: `application/json`
- **Base URL**: 서버 배포 주소 (예: `http://localhost:8000`)
- **Health**: `GET /health` → `{"status": "ok"}`
