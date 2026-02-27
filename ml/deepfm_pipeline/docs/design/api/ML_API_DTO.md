# Admin DeepFM API DTO 명세

`api_design.md` 기준 엔드포인트별 **Request / Response DTO** 정의.  
구현: `api/schemas.py`, `api/routers/deepfm.py`.

---

## 1) POST /admin/deepfm/train

학습 트리거. Prefect 학습 플로우 실행 → 모델 artifact + pipeline_version 산출.

### Request body (JSON, 전부 선택)

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `raw_data_dir` | string \| null | (파이프라인 기본) | raw 데이터 디렉터리 (train.csv 등) |
| `processed_data_dir` | string \| null | (파이프라인 기본) | 전처리 결과 저장 경로 |
| `num_train_sample` | integer \| null | null | 학습 샘플 수 상한 |
| `num_test_sample` | integer \| null | null | 테스트 샘플 수 상한 |
| `num_val` | integer \| null | 1000 | 검증 샘플 수 |
| `epochs` | integer \| null | 5 | 에폭 수 |
| `batch_size` | integer \| null | 100 | 배치 크기 |
| `lr` | number \| null | 1e-4 | 학습률 |
| `output_dir` | string \| null | (파이프라인 기본) | 모델/run 산출물 경로 |
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
| `model_path` | string | 저장된 model.pt 절대 경로 |
| `run_manifest_path` | string | run_manifest.json 경로 |
| `metrics` | object \| null | 오프라인 지표 (NDCG@5, NDCG@10, AUC 등). test 없으면 null |

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
| `run_dir` | string \| null | - | null | run 디렉터리 경로. 없으면 pipeline_version으로 output 하위에서 탐색 |
| `candidates_path` | string | O | - | 후보 CSV 경로 (전처리된 feature 열) |
| `output_path` | string | O | - | recommendation CSV 출력 경로 |
| `meta_path` | string \| null | - | null | user_id, anonymous_id, restaurant_id, context_snapshot 메타 CSV |
| `ttl_hours` | number | - | 24 | expires_at TTL(시간) |
| `batch_size` | integer | - | 256 | 추론 배치 크기 |

### Response (200)

| 필드 | 타입 | 설명 |
|------|------|------|
| `pipeline_version` | string | 사용된 pipeline_version |
| `output_path` | string | 출력 CSV 경로 |
| `rows_written` | integer | 출력된 recommendation 행 수 |

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
| `models[].run_dir` | string | run 디렉터리 경로 |
| `models[].created_at` | string \| null | run_manifest 기준 생성 시각 (ISO 8601) |
| `models[].metrics` | object \| null | run_manifest의 metrics (NDCG@K, AUC 등) |
| `active_version` | string \| null | 현재 활성(서빙) pipeline_version. 없으면 null |

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

### Error

- **404**: 해당 pipeline_version의 run이 output 하위에 없음

---

## 공통

- **Content-Type**: `application/json`
- **Base URL**: 서버 배포 주소 (예: `http://localhost:8000`)
- **Health**: `GET /health` → `{"status": "ok"}`
