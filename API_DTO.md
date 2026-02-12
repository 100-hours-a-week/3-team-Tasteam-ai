# API DTO 명세

**API 요청/응답 스키마(DTO)** 와 **엔드포인트·입출력 예시**만 정리한 문서입니다.  
파이프라인 동작·흐름·Config는 **PIPELINE/PIPELINE_OVERVIEW.md** 를 참고하세요.

- **정의 위치**: `src/models.py`
- **라우터**: `src/api/routers/llm.py`, `sentiment.py`, `vector.py`

---

## 목차

| 섹션 | 내용 |
|------|------|
| [1. API 구조](#1-api-구조) | prefix, 엔드포인트 목록, 문서 링크 |
| [2. 엔드포인트·DTO 요약](#2-엔드포인트dto-요약) | API별 Request/Response DTO 링크 |
| [3. 업로드·검색 데이터 요구 형태](#3-업로드검색-데이터-요구-형태) | Vector upload / Sentiment 리뷰 필수 필드 |
| [4. API 입출력 JSON](#4-api-입출력-json) | 요청/응답 예시 (Comparison, Summary, Sentiment, Vector) |
| [5. API 요청/응답 DTO](#5-api-요청응답-dto) | DTO별 필드 정의, 공통 에러, 선택 필드 설명 |

---

## 1. API 구조

### 1.1 엔드포인트

| Method | path | Request DTO | Response DTO |
|--------|------|-------------|--------------|
| POST | `/api/v1/llm/comparison` | ComparisonRequest | ComparisonResponse |
| POST | `/api/v1/llm/comparison/batch` | ComparisonBatchRequest | ComparisonBatchResponse |
| POST | `/api/v1/llm/summarize` | SummaryRequest | SummaryDisplayResponse |
| POST | `/api/v1/llm/summarize/batch` | SummaryBatchRequest | SummaryBatchResponse |
| POST | `/api/v1/sentiment/analyze` | SentimentAnalysisRequest | SentimentAnalysisResponse (debug 시 상세) |
| POST | `/api/v1/sentiment/analyze/batch` | SentimentAnalysisBatchRequest | SentimentAnalysisBatchResponse |
| POST | `/api/v1/vector/search/similar` | VectorSearchRequest | VectorSearchResponse |
| POST | `/api/v1/vector/upload` | VectorUploadRequest | VectorUploadResponse |

### 1.2 공통 엔드포인트

| Method | path | 설명 |
|--------|------|------|
| GET | `/` | 앱 정보·버전·docs·health 링크 |
| GET | `/health` | 헬스 체크 |
| GET | `/ready` | Readiness (warm-up 완료 시 200, 미완료 시 503) |
| GET | `/metrics` | Prometheus 메트릭 (설치 시) |

### 1.3 API 문서 (Swagger / OpenAPI)

| 문서 | 경로 |
|------|------|
| Swagger UI | `{BASE_URL}/docs` |
| ReDoc | `{BASE_URL}/redoc` |
| OpenAPI JSON | `{BASE_URL}/openapi.json` |

---

## 2. 엔드포인트·DTO 요약

- **Comparison**: [4.1](#41-comparison), [5.2](#52-comparison-dto)
- **Summary**: [4.2](#42-summary), [5.3](#53-summary-dto)
- **Sentiment**: [4.3](#43-sentiment), [5.4](#54-sentiment-dto)
- **Vector**: [4.4](#44-vector), [5.5](#55-vector-dto)
- **공통**: [5.6](#56-공통-dto), [5.7](#57-공통-에러-응답-포맷), [5.8](#58-선택-필드-역할)

---

## 3. 업로드·검색 데이터 요구 형태

**Vector upload / Sentiment** 요청에 넣는 리뷰는 아래 필드가 필요합니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| id | int | O | 리뷰 식별자 |
| restaurant_id | int | O | 레스토랑 식별자 |
| content | str | O | 리뷰 본문 |
| created_at | datetime (ISO 8601) | O | 작성 시각 |

**Vector upload** 시 레스토랑은 선택: `restaurants`: `[{ id?, name }]`. 리뷰만으로 업로드 가능.

---

## 4. API 입출력 JSON

### 4.1 Comparison

**`POST /api/v1/llm/comparison`**

```json
// 요청 (restaurant_id만 사용)
{
  "restaurant_id": 1
}

// 응답 (debug=false)
{
  "restaurant_id": 1,
  "comparisons": [
    {"category": "service", "lift_percentage": 18.5},
    {"category": "price", "lift_percentage": 12.2}
  ],
  "total_candidates": 120,
  "validated_count": 2,
  "category_lift": {"service": 18.5, "price": 12.2},
  "comparison_display": ["서비스 만족도는 평균보다 약 19% 높아요. 전반적으로 서비스 평가가 좋은 편입니다.", "가격 만족도는 평균보다 약 12% 높아요. 전반적으로 가격 평가가 좋은 편입니다."]
}
```

**배치 `POST /api/v1/llm/comparison/batch`**

- 요청: **restaurants** 배열(각 항목: `restaurant_id` 필수). **all_average_data_path**: 선택, 전체 평균 데이터 파일 경로(미지정 시 Config 사용).
- 응답: **`results` 배열**. 각 요소는 단일 Comparison 응답과 동일(**ComparisonResponse**).
- **배치 (Config)**: `COMPARISON_BATCH_ASYNC=true`면 레스토랑별 compare를 `asyncio.gather`로 병렬, false면 순차.

```json
// 요청 (restaurants + all_average_data_path 선택)
{
  "restaurants": [
    {"restaurant_id": 1},
    {"restaurant_id": 2}
  ],
  "all_average_data_path": null
}

// 응답 — { results: [ ComparisonResponse, ... ] }
{
  "results": [
    {
      "restaurant_id": 1,
      "comparisons": [{"category": "service", "lift_percentage": 18.5}, {"category": "price", "lift_percentage": 12.2}],
      "total_candidates": 120,
      "validated_count": 2,
      "category_lift": {"service": 18.5, "price": 12.2},
      "comparison_display": ["서비스 만족도는 평균보다 약 19% 높아요.", "가격 만족도는 평균보다 약 12% 높아요."]
    },
    {
      "restaurant_id": 2,
      "comparisons": [{"category": "service", "lift_percentage": 5.1}],
      "total_candidates": 80,
      "validated_count": 2,
      "category_lift": {"service": 5.1, "price": 0},
      "comparison_display": ["서비스 만족도는 평균보다 약 5% 높아요."]
    }
  ]
}
```

---

### 4.2 Summary

단일과 배치는 **요청·응답 구조가 다름**.

| | 단일 `POST /summarize` | 배치 `POST /summarize/batch` |
|--|--|--|
| **요청** | **평탄 객체** `{ restaurant_id, limit }` | **래핑 객체** `{ restaurants: [ { restaurant_id, limit? } ] }` |
| **응답** | **루트가 요약 1개** (results 없음). `X-Debug: true`면 SummaryResponse, 아니면 SummaryDisplayResponse(필드 적음) | **`{ results: [ ... ] }`** 배열. 각 요소는 **SummaryDisplayResponse** 형식(단일 debug=false와 동일, positive_reviews 등 없음) |

---

**단일 `POST /api/v1/llm/summarize`**

- 요청: 평탄. `restaurant_id`, **limit**: 카테고리(service/price/food)당 검색·요약에 쓸 최대 리뷰 수 (1~100, 기본 10).

```json
// 요청 (평탄 객체, 레스토랑 1개. limit: 1~100, 기본 10)
{
  "restaurant_id": 1,
  "limit": 10
}

// 응답 (debug=false) — 루트가 요약 1개. SummaryDisplayResponse (positive_reviews, positive_count 등 없음)
{
  "restaurant_id": 1,
  "overall_summary": "이 음식점은 서비스·가격·음식 전반에서 만족도가 높으며, 분위기와 맛을 꼽는 리뷰가 많습니다.",
  "categories": {
    "service": {
      "summary": "직원과 사장님이 친절하고 응대가 빠르다.",
      "bullets": ["친절한 서비스", "빠른 응대", "사장님 인사"],
      "evidence": [{"review_id": "r1", "snippet": "직원분이 친절해요", "rank": 0},...]
    },
    "price": {
      "summary": "가격 대비 만족도가 높은 편이다.",
      "bullets": ["합리적인 가격", "무한 리필", "가성비 좋음"],
      "evidence": [{"review_id": "r2", "snippet": "가격 대비 괜찮아요", "rank": 1},...]
    },
    "food": {
      "summary": "음식 맛과 분위기를 칭찬하는 리뷰가 많다.",
      "bullets": ["맛있음", "분위기 좋음", "타르트 인기"],
      "evidence": [{"review_id": "r3", "snippet": "타르트가 맛있어요", "rank": 0},...]
    }
  }
}

// 응답 (debug=true) — SummaryDisplayResponse에 debug 필드 추가. (positive_reviews, negative_reviews 등은 미사용·미노출)
```

---

**배치 `POST /api/v1/llm/summarize/batch`**

- 요청: `restaurants` 배열(각 항목: `restaurant_id`), **limit**: 카테고리당 검색·요약에 쓸 최대 리뷰 수 (1~100, 기본 10). 선택, 전체 공통.
- 응답: **`results` 배열**. 각 요소는 **SummaryDisplayResponse** 형식(단일 debug=false와 동일).
- **배치 (Config)**: `SUMMARY_SEARCH_ASYNC`=aspect(service/price/food) 서치 병렬, `SUMMARY_RESTAURANT_ASYNC`=음식점 간 병렬. 둘 다 false면 완전 순차. `SUMMARY_LLM_ASYNC`=LLM 호출 비동기(to_thread vs AsyncOpenAI). 동시성: `BATCH_SEARCH_CONCURRENCY`, `BATCH_LLM_CONCURRENCY`. 자세한 내용은 [BATCH_SUMMARY_MODE.md](BATCH_SUMMARY_MODE.md) 참고.

```json
// 요청 (restaurants + limit 전체 공통. limit: 1~100, 기본 10)
{
  "restaurants": [
    {"restaurant_id": 1},
    {"restaurant_id": 2}
  ],
  "limit": 10
}

// 응답 — { results: [ SummaryDisplayResponse, ... ] }. 단일(debug=false)과 동일한 필드; results로만 감쌈
{
  "results": [
    {
      "restaurant_id": 1,
      "overall_summary": "이 음식점은...",
      "categories": { "service": {...}, "price": {...}, "food": {...} }
    },
    {
      "restaurant_id": 2,
      "overall_summary": "...",
      "categories": { "service": {...}, "price": {...}, "food": {...} }
    }
  ]
}
```

---

### 4.3 Sentiment

**`POST /api/v1/sentiment/analyze`**

- 리뷰는 **벡터 DB에서 조회**하여 사용. 요청에는 `restaurant_id`만 전달.

```json
// 요청 (restaurant_id만 사용)
{
  "restaurant_id": 1
}

// 응답 (debug=false, SentimentAnalysisDisplayResponse)
{
  "restaurant_id": 1,
  "positive_ratio": 75,
  "negative_ratio": 25
}

// 응답 (debug=true, SentimentAnalysisResponse)
{
  "restaurant_id": 1,
  "positive_count": 75,
  "negative_count": 25,
  "neutral_count": 0,
  "total_count": 100,
  "positive_ratio": 75,
  "negative_ratio": 25,
  "neutral_ratio": 0,
  "debug": {"request_id": "...", "processing_time_ms": 120.5, "tokens_used": null, "model_version": null, "warnings": null}
}
```

**`POST /api/v1/sentiment/analyze/batch`**

- 각 레스토랑 리뷰는 **벡터 DB에서 조회**. 요청에는 `restaurant_id`(및 선택 `restaurant_name`)만 전달.

```json
// 요청 (restaurant_id만 사용)
{
  "restaurants": [
    {"restaurant_id": 1},
    {"restaurant_id": 2},
    {"restaurant_id": 3}
  ]
}

// 응답
{
  "results": [
    {"restaurant_id": 1, "positive_count": 2, "negative_count": 0, "neutral_count": 0, "total_count": 2, "positive_ratio": 100, "negative_ratio": 0, "neutral_ratio": 0},
    {"restaurant_id": 2, "positive_count": 1, "negative_count": 0, "neutral_count": 0, "total_count": 1, "positive_ratio": 100, "negative_ratio": 0, "neutral_ratio": 0},
    {"restaurant_id": 3, "positive_count": 1, "negative_count": 0, "neutral_count": 0, "total_count": 1, "positive_ratio": 100, "negative_ratio": 0, "neutral_ratio": 0}
  ]
}
```

---

### 4.4 Vector

**`POST /api/v1/vector/search/similar`**

- 검색: **하이브리드 통일** (Dense prefetch **dense_prefetch_limit**(기본 200), Sparse prefetch **sparse_prefetch_limit**(기본 300) → RRF → 최종 limit개).
- **폴백**(단일 벡터/Sparse 실패 등): Dense만 사용. **fallback_min_score**: 폴백 경로 최소 유사도 (0.0~1.0, 기본 0.2).
- 요청: **limit**: 반환할 최대 개수 (1~100, 기본 3). **dense_prefetch_limit**: Dense prefetch 개수 (1~2000, 기본 200). **sparse_prefetch_limit**: Sparse prefetch 개수 (1~2000, 기본 300). **fallback_min_score**: 폴백 경로에서만 적용.

```json
// 요청 (limit: 1~100 기본 3, dense/sparse_prefetch_limit: 1~2000 기본 200/300, fallback_min_score: 폴백만)
{
  "query_text": "분위기 좋고 데이트하기 좋은",
  "restaurant_id": 1,
  "limit": 5,
  "dense_prefetch_limit": 200,
  "sparse_prefetch_limit": 300,
  "fallback_min_score": 0.2
}

// 응답
{
  "results": [
    {"review": {"id": 1, "restaurant_id": 1, "content": "분위기 좋고 데이트하기 딱이에요", ...}, "score": 0.89},
    {"review": {"id": 2, "restaurant_id": 1, "content": "연인과 오기 좋은 분위기", ...}, "score": 0.85}
  ],
  "total": 2
}
```

**`POST /api/v1/vector/upload`**

```json
// 요청 (reviews: id, restaurant_id, content, created_at 필수 / restaurants: id 선택, name, reviews)
{
  "reviews": [
    {"id": 1, "restaurant_id": 1, "content": "맛있어요", "created_at": "2025-02-17T17:02:45.366789"}
  ],
  "restaurants": [
    {"id": 1, "name": "테스트 음식점"}
  ]
}

// 응답
{
  "message": "데이터 업로드 완료",
  "points_count": 100,
  "collection_name": "reviews_collection"
}
```

---

## 5. API 요청/응답 DTO

API별 요청/응답에 사용되는 Pydantic 모델(DTO)과 필드를 정리합니다. 정의 위치: `src/models.py`.  
*(라우터: `src/api/routers/llm.py`, `sentiment.py`, `vector.py` 기준으로 검증됨.)*

### 5.1 API → DTO 매핑

| API | Request DTO | Response DTO |
|-----|-------------|--------------|
| `POST /api/v1/llm/comparison` | `ComparisonRequest` | `ComparisonResponse` (또는 Dict, 에러/스킵 시) |
| `POST /api/v1/llm/comparison/batch` | `ComparisonBatchRequest` | `ComparisonBatchResponse` |
| `POST /api/v1/llm/summarize` | `SummaryRequest` | `SummaryDisplayResponse` (debug 필드 선택) |
| `POST /api/v1/llm/summarize/batch` | `SummaryBatchRequest` | `SummaryBatchResponse` |
| `POST /api/v1/sentiment/analyze` | `SentimentAnalysisRequest` | `SentimentAnalysisDisplayResponse` (debug=false) / `SentimentAnalysisResponse` (debug=true) |
| `POST /api/v1/sentiment/analyze/batch` | `SentimentAnalysisBatchRequest` | `SentimentAnalysisBatchResponse` |
| `POST /api/v1/vector/search/similar` | `VectorSearchRequest` | `VectorSearchResponse` |
| `POST /api/v1/vector/upload` | `VectorUploadRequest` | `VectorUploadResponse` |

### 5.2 Comparison DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **ComparisonRequest** | restaurant_id | int | 타겟 레스토랑 ID |
| | restaurant_name | str? | 레스토랑 이름 (응답에 그대로 반환, 선택) |
| **ComparisonDetail** | category | str | "service" \| "price" |
| | lift_percentage | float | (단일−전체)/전체×100 |
| **ComparisonResponse** | restaurant_id | int | 레스토랑 ID |
| | restaurant_name | str? | 레스토랑 이름 (payload 또는 요청에서, 선택) |
| | comparisons | List[ComparisonDetail] | 비교 항목 리스트 |
| | total_candidates | int | 근거 후보 총 개수 |
| | validated_count | int | 검증 통과 개수 |
| | category_lift | Dict[str, float]? | service/price lift |
| | comparison_display | List[str]? | 표시 문장 리스트 |
| | debug | DebugInfo? | 디버그 정보 |
| **ComparisonBatchRequest** | restaurants | List[Dict] | [{ restaurant_id }, ...]. 각 항목에 restaurant_id 필수 |
| | all_average_data_path | str? | 전체 평균·표본용 데이터 파일 경로. 미지정 시 Config 사용 |
| **ComparisonBatchResponse** | results | List[ComparisonResponse] | 레스토랑별 비교 결과 |

### 5.3 Summary DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **SummaryRequest** | restaurant_id | int | 레스토랑 ID |
| | restaurant_name | str? | 레스토랑 이름 (응답에 그대로 반환, 선택) |
| | limit | int | 카테고리(service/price/food)당 검색·요약 최대 리뷰 수 (1~100, 기본 10) |
| **SummaryBatchRequest** | restaurants | List[Dict] | [{ restaurant_id }, ...] |
| | limit | int | 카테고리당 검색·요약 최대 리뷰 수, 전체 공통 (1~100, 기본 10) |
| **CategorySummary** | summary | str | 카테고리 요약 |
| | bullets | List[str] | 핵심 포인트 |
| | evidence | List[Dict] | [{ review_id, snippet, rank }] |
| **SummaryDisplayResponse** | restaurant_id | int | 레스토랑 ID |
| | restaurant_name | str? | 레스토랑 이름 (payload 또는 요청에서, 선택) |
| | overall_summary | str | 전체 요약 |
| | categories | Dict[str, CategorySummary]? | service/price/food 요약 |
| | debug | DebugInfo? | X-Debug: true 시에만 |
| **SummaryBatchResponse** | results | List[SummaryDisplayResponse] | 레스토랑별 요약 |

### 5.4 Sentiment DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **SentimentAnalysisRequest** | restaurant_id | int | 레스토랑 ID (벡터 DB에서 리뷰 조회) |
| | restaurant_name | str? | 레스토랑 이름 (응답에 그대로 반환) |
| **SentimentAnalysisDisplayResponse** | restaurant_id | int | 레스토랑 ID |
| | restaurant_name | str? | 레스토랑 이름 |
| | positive_ratio | int | 긍정 비율 (%) |
| | negative_ratio | int | 부정 비율 (%) |
| **SentimentAnalysisResponse** | (Display 필드) + | | |
| | restaurant_name | str? | 레스토랑 이름 |
| | positive_count, negative_count, neutral_count | int | 개수 |
| | total_count | int | 전체 리뷰 수 |
| | neutral_ratio | int | 중립 비율 (%) |
| | debug | DebugInfo? | 디버그 정보 |
| **SentimentRestaurantBatchInput** | restaurant_id | int | 레스토랑 ID (벡터 DB에서 리뷰 조회) |
| | restaurant_name | str? | 레스토랑 이름 (응답에 그대로 반환) |
| **SentimentAnalysisBatchRequest** | restaurants | List[SentimentRestaurantBatchInput] | 레스토랑 ID 리스트 (각 항목 리뷰는 벡터 DB에서 조회) |
| **SentimentAnalysisBatchResponse** | results | List[SentimentAnalysisResponse] | 레스토랑별 결과 |

### 5.5 Vector DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **VectorSearchRequest** | query_text | str | 검색 쿼리 |
| | restaurant_id | int? | 레스토랑 필터 |
| | limit | int | 반환할 최대 개수 (1~100, 기본 3). 하이브리드: RRF 후 이 개수만 반환. |
| | fallback_min_score | float | 폴백(Dense만) 경로에서만 적용 (0.0~1.0, 기본 0.2). 하이브리드 경로에는 미적용. |
| | dense_prefetch_limit | int | 하이브리드 Dense prefetch 개수 (1~2000, 기본 200). |
| | sparse_prefetch_limit | int | 하이브리드 Sparse prefetch 개수 (1~2000, 기본 300). |
| **VectorSearchResult** | review | ReviewModel | 리뷰 payload |
| | score | float | 유사도 점수 |
| **VectorSearchResponse** | results | List[VectorSearchResult] | 검색 결과 |
| | total | int | 총 개수 |
| **VectorUploadReviewInput** | id | int | 리뷰 ID (필수) |
| | restaurant_id | int | 레스토랑 ID |
| | content | str | 리뷰 내용 |
| | created_at | datetime | 리뷰 작성 시각 (ISO 8601, 필수) |
| **VectorUploadRestaurantInput** | id | int? | 레스토랑 ID (선택) |
| | name | str | 레스토랑 이름 |
| | reviews | List[VectorUploadReviewInput] | 중첩 리뷰 (선택) |
| **VectorUploadRequest** | reviews | List[VectorUploadReviewInput] | 리뷰 리스트 |
| | restaurants | List[VectorUploadRestaurantInput]? | 레스토랑 리스트 |
| **VectorUploadResponse** | message | str | 메시지 |
| | points_count | int | 업로드된 포인트 수 |
| | collection_name | str | 컬렉션 이름 |

### 5.6 공통 DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **ErrorResponse** | code | int | HTTP status code |
| | message | str | 에러 메시지(요약) |
| | details | Any? | 에러 상세(검증 오류 배열/추가 정보 등) |
| | request_id | str | 요청 추적용 ID (`X-Request-Id`와 동일) |
| **DebugInfo** | request_id | str? | 요청 ID |
| | processing_time_ms | float? | 처리 시간 (ms) |
| | tokens_used | int? | 사용 토큰 수 |
| | model_version | str? | 모델 버전 |
| | warnings | List[str]? | 경고 메시지 |
| **ReviewModel** | id | int | 리뷰 ID (필수) |
| | restaurant_id | int | 레스토랑 ID |
| | content | str | 리뷰 내용 |

### 5.7 공통 에러 응답 포맷

모든 API의 4xx/5xx 및 validation 오류는 아래 포맷으로 반환됩니다.

```json
{
  "code": 422,
  "message": "Validation error",
  "details": [
    {"loc": ["body", "field"], "msg": "...", "type": "..."}
  ],
  "request_id": "..."
}
```

- **request_id**
  - 요청 헤더에 `X-Request-Id`(또는 `X-Request-ID`)를 주면 해당 값 사용
  - 없으면 서버가 UUID를 생성
  - 응답 헤더 `X-Request-Id`에도 동일 값이 포함됨
- **Validation 오류(422)**: `message="Validation error"`, `details=errors[]`
- **HTTP 오류(4xx/5xx)**: `message`는 `detail` 문자열, `details`는 원본 `detail`
- **Unhandled(500)**: `message="Internal server error"`, `details=null`

### 5.8 선택 필드 역할

선택(optional) 필드는 **없어도 요청/응답이 성립**하지만, **있으면 그 값이 동작에 반영**됩니다.

**요청(Request) 쪽**

| DTO | 필드 | 역할 |
|-----|------|------|
| **VectorSearchRequest** | restaurant_id | **검색 범위 제한.** 값을 넣으면 해당 레스토랑 리뷰만 검색, None이면 전체 검색. |
| **VectorUploadReviewInput** | id | **필수.** 업로드 시 리뷰 식별·upsert 시 "어느 포인트를 갱신할지" 구분. |
| **VectorUploadRestaurantInput** | id | 레스토랑 식별·대표 벡터 매핑용. 없으면 이름만으로 처리. |
| **VectorUploadRequest** | restaurants | **레스토랑 메타 전달.** 리뷰만 올릴 수도 있고, 넣으면 레스토랑 대표 벡터·이름 매핑에 사용. |
| **ErrorResponse** | details | 422 등 검증/에러 시 상세(필드별 오류 등). 없으면 null. |

**응답(Response) 쪽**

| DTO | 필드 | 역할 |
|-----|------|------|
| **DebugInfo** (전부 선택) | request_id, processing_time_ms, tokens_used, model_version, warnings | **디버그/운영용.** `X-Debug: true` 또는 `debug=true`일 때만 응답에 포함. 요청 추적·지연·토큰·경고 확인용. |
| **SentimentAnalysisResponse** | debug | 감성 분석 응답에 디버그 블록 추가. |
| **SummaryDisplayResponse** | categories | 카테고리별 요약(service/price/food). 파이프라인 결과가 없으면 None. |
| **SummaryDisplayResponse** | debug | 요약 응답에 디버그 블록 추가. |
| **ComparisonResponse** | category_lift | service/price별 lift 퍼센트. 통계·LLM 설명용. 없으면 None. |
| **ComparisonResponse** | comparison_display | 표시용 문장(퍼센트+해석, 표본 수별 톤). 없으면 None. |
| **ComparisonResponse** | debug | 비교 응답에 디버그 블록 추가. |

---

파이프라인 동작·흐름·Config는 **PIPELINE/PIPELINE_OVERVIEW.md** 를 참고하세요.
