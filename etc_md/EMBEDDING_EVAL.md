# 임베딩 모델 Precision@k 비교 평가

여러 임베딩 모델에 대해 Precision@k 평가를 수행하고 결과를 비교합니다.

## 지원 모델

- `jhgan/ko-sbert-multitask`
- `dragonkue/BGE-m3-ko`
- `upskyy/bge-m3-korean`

## 사용 방법

### 1. Ground Truth 준비

Ground Truth 파일이 필요합니다: `scripts/Ground_truth_vector_search.json`

```json
{
  "queries": [
    {
      "query": "맛있다 좋다 만족",
      "restaurant_id": 1,
      "relevant_review_ids": [1, 2, 3, 5, 7]
    },
    {
      "query": "맛없다 별로 불만",
      "restaurant_id": 1,
      "relevant_review_ids": [4, 6, 8]
    }
  ]
}
```

### 2. 서버 실행 (각 모델마다 재시작 필요)

각 모델 평가 전에 서버를 재시작하고 `EMBEDDING_MODEL` 환경 변수를 설정해야 합니다.

```bash
# 첫 번째 모델로 서버 시작
export EMBEDDING_MODEL="jhgan/ko-sbert-multitask"
python app.py
```

### 3. 평가 실행

별도 터미널에서 평가 스크립트 실행:

```bash
python scripts/compare_embedding_models.py \
    --models "jhgan/ko-sbert-multitask" "dragonkue/BGE-m3-ko" "upskyy/bge-m3-korean" \
    --ground-truth scripts/Ground_truth_vector_search.json \
    --base-url http://localhost:8000 \
    --k-values 1 3 5 10 \
    --output embedding_comparison.json
```

### 4. 모델 변경 (각 모델 평가 전)

스크립트가 대기 시간을 제공하므로, 그 동안:

1. 서버를 중지 (Ctrl+C)
2. 새로운 모델로 환경 변수 설정:
   ```bash
   export EMBEDDING_MODEL="dragonkue/BGE-m3-ko"
   ```
3. 서버 재시작:
   ```bash
   python app.py
   ```
4. 스크립트가 자동으로 다음 모델 평가 진행

## 옵션 설명

- `--models`: 비교할 임베딩 모델명 리스트 (필수)
- `--ground-truth`: Ground Truth JSON 파일 경로 (필수)
- `--base-url`: API 서버 URL (기본값: http://localhost:8000)
- `--k-values`: 평가할 k 값 리스트 (기본값: 1 3 5 10)
- `--limit`: 검색할 최대 개수 (기본값: 10)
- `--min-score`: 최소 유사도 점수 (기본값: 0.0)
- `--wait-between-models`: 모델 간 대기 시간 (초, 기본값: 10)
- `--output`: 결과 저장 파일 경로 (선택사항)

## 출력 결과

### 콘솔 출력

```
================================================================================
임베딩 모델 비교 결과
================================================================================

총 평가 모델 수: 3
평가 k 값: [1, 3, 5, 10]

--------------------------------------------------------------------------------
모델별 평균 Precision@k:
--------------------------------------------------------------------------------

[jhgan/ko-sbert-multitask]
  상태: completed
  P@1: 0.8500 (85.00%)
  P@3: 0.8000 (80.00%)
  P@5: 0.7500 (75.00%)
  P@10: 0.7000 (70.00%)

[dragonkue/BGE-m3-ko]
  상태: completed
  P@1: 0.9000 (90.00%)
  P@3: 0.8500 (85.00%)
  P@5: 0.8000 (80.00%)
  P@10: 0.7500 (75.00%)

--------------------------------------------------------------------------------
k 값별 최고 성능 모델:
--------------------------------------------------------------------------------
  P@1: dragonkue/BGE-m3-ko (Precision: 0.9000)
  P@3: dragonkue/BGE-m3-ko (Precision: 0.8500)
  P@5: dragonkue/BGE-m3-ko (Precision: 0.8000)
  P@10: dragonkue/BGE-m3-ko (Precision: 0.7500)
================================================================================
```

### JSON 결과 파일

```json
{
  "timestamp": "2026-01-18T12:00:00",
  "total_models": 3,
  "k_values": [1, 3, 5, 10],
  "models": [
    {
      "model_name": "jhgan/ko-sbert-multitask",
      "status": "completed",
      "timestamp": "2026-01-18T12:00:00",
      "total_queries": 10,
      "evaluated_queries": 10,
      "average_precisions": {
        "P@1": 0.85,
        "P@3": 0.80,
        "P@5": 0.75,
        "P@10": 0.70
      },
      "query_results": [...]
    },
    ...
  ],
  "comparison": {
    "average_precisions": {
      "P@1": {
        "jhgan/ko-sbert-multitask": 0.85,
        "dragonkue/BGE-m3-ko": 0.90,
        "upskyy/bge-m3-korean": 0.88
      },
      ...
    },
    "best_model_per_k": {
      "P@1": {
        "model": "dragonkue/BGE-m3-ko",
        "precision": 0.90
      },
      ...
    }
  }
}
```

## Precision@k란?

Precision@k = (상위 k개 검색 결과 중 관련 있는 문서 수) / k

- **P@1**: 상위 1개 결과 중 관련 있는 문서 비율
- **P@3**: 상위 3개 결과 중 관련 있는 문서 비율
- **P@5**: 상위 5개 결과 중 관련 있는 문서 비율
- **P@10**: 상위 10개 결과 중 관련 있는 문서 비율

값이 높을수록 검색 품질이 좋습니다.

## 주의사항

1. **서버 재시작 필요**: 각 모델 평가 전에 서버를 재시작하고 `EMBEDDING_MODEL` 환경 변수를 설정해야 합니다.
2. **데이터 업로드**: 각 모델 변경 후 벡터 데이터를 다시 업로드해야 할 수 있습니다.
3. **평가 시간**: 모델 수와 쿼리 수에 따라 평가 시간이 달라질 수 있습니다.
