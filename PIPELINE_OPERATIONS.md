# 파이프라인 동작 문서 (API 기반)

**Comparison**, **Summary**, **Sentiment**, **Vector** 파이프라인의 API·입력·출력·처리 단계를 정리한 문서입니다.

- **범위**: `src/` 기반 HTTP API. `hybrid_search` / `final_pipeline` 미포함.
- **테스트**: `test_all_task.py` — API 호출만 사용.

---

## 목차 (섹션별 이동)

| 섹션 | 링크 |
|------|------|
| **1. 개요** | [1. 개요](#1-개요) · [1.1 API 구조](#11-api-구조) · [1.1.1 API 문서 (Swagger)](#111-api-문서-swagger--openapi) · [1.2 파이프라인 흐름 (개략)](#12-파이프라인-흐름-개략) |
| **2. Comparison** | [2. Comparison (비교)](#2-comparison-비교) · [2.1 위치](#21-위치) · [2.2 흐름](#22-흐름) · [2.3 출력](#23-출력-api) · [2.4 공통](#24-공통) |
| **3. Summary** | [3. Summary (요약)](#3-summary-요약) · [3.1 위치](#31-위치) · [3.2 입력·시드](#32-입력시드) · [3.3 흐름](#33-흐름) · [3.4 출력](#34-출력-api) |
| **4. Sentiment** | [4. Sentiment (감성 분석)](#4-sentiment-감성-분석) · [4.1 위치](#41-위치) · [4.2 흐름](#42-흐름) · [4.3 출력](#43-출력-api) · [4.4 공통](#44-공통) |
| **5. Vector** | [5. Vector (벡터 검색·업로드·리뷰)](#5-vector-벡터-검색업로드리뷰) · [5.1 위치](#51-위치) · [5.2 컬렉션·벡터](#52-컬렉션벡터-형식) · [5.3 검색 파이프라인](#53-검색-파이프라인) · [5.4 API 엔드포인트](#54-api-엔드포인트) · [5.5 업로드·포인트](#55-업로드포인트) · [5.6 원천데이터](#56-원천데이터-업로드-전-요구-형태) |
| **6. 공통·의존성** | [6. 공통·의존성](#6-공통의존성) · [6.1 의존성](#61-의존성-dependencies) · [6.2 락·SKIP](#62-락skip) · [6.3 Config 요약](#63-config-요약) |
| **7. API 매핑** | [7. API ↔ 파이프라인 매핑](#7-api--파이프라인-매핑) |
| **8. API 입출력 JSON** | [8. API 입출력 JSON](#8-api-입출력-json) · [8.1 Comparison](#81-comparison) · [8.2 Summary](#82-summary) · [8.3 Sentiment](#83-sentiment) · [8.4 Vector](#84-vector) |
| **9. API DTO** | [9. API 요청/응답 DTO](#9-api-요청응답-dto) · [9.1 API → DTO 매핑](#91-api--dto-매핑) · [9.2 Comparison DTO](#92-comparison-dto) · [9.3 Summary DTO](#93-summary-dto) · [9.4 Sentiment DTO](#94-sentiment-dto) · [9.5 Vector DTO](#95-vector-dto) · [9.6 공통 DTO](#96-공통-dto) · [9.7 공통 에러 응답](#97-공통-에러-응답-포맷) · [9.8 선택 필드 역할](#98-선택-필드-역할) |
| **10. 참고** | [10. 참고](#10-참고) |

---

## 1. 개요

### 1.1 API 구조

| prefix | 라우터 | 용도 |
|--------|--------|------|
| `/api/v1/llm` | `llm` | Summary, Comparison |
| `/api/v1/sentiment` | `sentiment` | 감성 분석 |
| `/api/v1/vector` | `vector` | 벡터 검색·업로드 (search/similar, upload) |
| `/api/v1/test` | `test` | 테스트용 |

- **헬스**: `/health`

### 1.1.1 API 문서 (Swagger / OpenAPI)

서버 실행 시 **모든 API**에 대한 인터랙티브 문서가 제공됩니다. (FastAPI 자동 생성)

| 문서 | 경로 | 설명 |
|------|------|------|
| **Swagger UI** | `{BASE_URL}/docs` | 모든 엔드포인트 조회·실행 (Try it out) |
| **ReDoc** | `{BASE_URL}/redoc` | 읽기용 API 레퍼런스 |
| **OpenAPI 스키마** | `{BASE_URL}/openapi.json` | OpenAPI 3.0 JSON (자동화·코드생성용) |

예: 로컬 서버가 `http://localhost:8000` 이면  
- Swagger: **http://localhost:8000/docs**  
- ReDoc: **http://localhost:8000/redoc**

### 1.2 파이프라인 흐름 (개략)

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                     Qdrant (리뷰 컬렉션)                  │
                    │  named: dense(768) + sparse (BM25)  or  단일 벡터        │
                    └─────────────────────────────────────────────────────────┘
                                          ▲
         ┌────────────────────────────────┼────────────────────────────────┐
         │                                │                                │
         ▼                                ▼                                ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│     Vector      │            │     Summary     │            │   Comparison    │
│ /search/similar │            │   /summarize    │            │ /comparison     │
│ /upload         │            │   /summarize/   │            │                 │
│                 │            │   batch         │            │                 │
│                 │            │  aspect_seeds   │            │  Kiwi bigram    │
│                 │            │  → hybrid검색   │            │  → service/     │
│                 │            │  → LLM 요약     │            │  price 비율     │
└─────────────────┘            └─────────────────┘            │  → lift, display│
         │                                │                    └─────────────────┘
         │                                │                                │
         │                     ┌──────────┴──────────┐                     │
         │                     │      Sentiment      │                     │
         │                     │  /analyze           │  (reviews 직접 입력  │
         │                     │  /analyze/batch     │   또는 vector_search)│
         │                     │  HF 1차 + LLM 2차   │                     │
         │                     └─────────────────────┘                     │
         │                                                                │
         └───────────────────────  LLMUtils (Qwen/vLLM/OpenAI)  ──────────┘
```

---

## 2. Comparison (비교)

### 2.1 위치

- `src/comparison.py` — `ComparisonPipeline.compare`
- `src/comparison_pipeline.py` — `calculate_single_restaurant_ratios`, `calculate_comparison_lift`, `format_comparison_display`, `calculate_all_average_ratios_from_*`
- `POST /api/v1/llm/comparison` (`src/api/routers/llm.py`)

### 2.2 흐름

1. **리뷰 조회**  
   - `vector_search.get_restaurant_reviews(restaurant_id)` → `content`/`text`만 `review_texts`로.

2. **단일 음식점 비율**  
   - `comparison_pipeline.calculate_single_restaurant_ratios(review_texts, stopwords)`  
   - Kiwi NNG/NNP bigram + `SERVICE_KW`/`PRICE_KW`/`SERVICE_POSITIVE_KW`/`PRICE_POSITIVE_KW` → `service`, `price` 긍정 비율.

3. **전체 평균 (all_average_ratios)**  
   - ① `ALL_AVERAGE_ASPECT_DATA_PATH` 파일(Spark/TSV·JSON)  
   - ② `get_all_reviews_for_all_average(5000)` (Qdrant)  
   - ③ `Config.ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO`

4. **Lift**  
   - `calculate_comparison_lift(single_ratios, all_average_ratios)`  
   - `lift = (단일 − 전체) / 전체 × 100`

5. **comparison_display**
   - `format_comparison_display(lift_service, lift_price, n_reviews)`
   - 수치(lift %)는 코드 고정 생성. 해석은 표본 수별 톤 적용.
   - `COMPARISON_ASYNC=true`면 service/price LLM 병렬(asyncio.gather), `false`(기본)면 순차 호출.  
   - 퍼센트 + 해석: `"서비스 만족도는 평균보다 약 N% 높아요. 전반적으로 서비스 평가가 {tone}입니다."`  
   - 표본 톤: n≥50 → "좋은 편", 20≤n<50 → "상대적으로 좋은 편(표본 중간)", n<20 → "경향이 보이나 표본이 적어요"

6. **강점 리스트**  
   - `lift > 0` 인 service/price만 `comparisons`에 추가  
   - `{ "category": "service"|"price", "lift_percentage": float }`  
   - `lift_percentage` 내림차순 후 `top_k`개.

### 2.3 출력 (API)

- `restaurant_id`, `comparisons`, `total_candidates`, `validated_count`
- `category_lift`, `comparison_display`, `processing_time_ms`
- **comparisons**: `[{"category":"service","lift_percentage":20.0}, ...]`

### 2.4 공통

- **Spark 로그 억제**: Comparison(comparison_pipeline)은 Spark 사용. 로그 억제는 [trouble_shooting/SPARK_LOG_NOISE.md](trouble_shooting/SPARK_LOG_NOISE.md) 참고.
- **Kiwi**: NNG/NNP bigram.  
- **키워드**: service `친절, 서비스, 응대, 직원, 사장, 불친절`; price `가격, 가성비, 대비, 리필, 무한, 할인, 쿠폰`; 긍정 시드 별도.  
- **불용어**: `data/stopwords-ko.txt` 우선.  
- **comparison_display**: 수치(lift, % 차이)는 코드 고정 생성. 해석은 표본 수별 톤 적용. LLM 붙일 때 안전장치: (1) 숫자 계산 금지 — lift·delta_pct를 입력으로 주고 문장만 생성, (2) 과장 금지 — "최고·압도적·완벽" 등 금지, (3) 표본 작으면 톤 다운 — n≥50 / 20≤n<50 / n<20 분기 결과를 LLM에 전달해 자연스럽게 말만 정리. 근거(리뷰 수·키워드)를 같이 넣어 과장 방지.

---

## 3. Summary (요약)

### 3.1 위치

- `src/summary_pipeline.py` — `summarize_aspects_new`
- `src/aspect_seeds.py` — `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS`
- `src/api/routers/llm.py` — `POST /api/v1/llm/summarize`, `POST /api/v1/llm/summarize/batch`
- `src/vector_search.py` — `query_hybrid_search`

### 3.2 입력·시드

- **입력**: `restaurant_id`(필수), `limit`, `min_score` 사용.
- **하이브리드 검색 쿼리**: **기본 시드만** 사용 (DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS).
- `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` (aspect_seeds 상수)를 service/price/food별 쿼리로 사용.
- `seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]`, `name_list = ["service","price","food"]`.
- `ASPECT_SEEDS_FILE`·`load_aspect_seeds()` 미사용.

### 3.3 흐름

1. **Aspect Seed**  
   - 기본 시드만 사용: `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` (위 참고).

2. **카테고리별 하이브리드 검색**  
   - service/price/food 시드마다:  
     - `query_text = " ".join(seeds[:10])`  
     - `vector_search.query_hybrid_search(query_text, restaurant_id, limit, min_score=0.0)`  
   - Dense + Sparse RRF, `restaurant_id` 필터.  
   - `hits_dict`, `hits_data_dict` (review_id, snippet, rank).

3. **요약**  
   - `summarize_aspects_new(..., llm_utils, per_category_max)`  
   - `_clip`으로 카테고리당 `per_category_max`개.  
   - LLM: system(instructions) + user(JSON payload).  
   - 출력 JSON: service/price/food → `{summary, bullets, evidence: [int]}`; `overall_summary`.  
   - evidence 인덱스 → `{review_id, snippet, rank}` 치환.  
   - **Price 게이트**: price 신호 없으면 summary/bullets 고정 문장.

4. **응답 변환**  
   - `overall_summary`, `categories` (debug일 때만 `positive_count`/`negative_count`=0 등).

### 3.4 출력 (API)

- `restaurant_id`, `overall_summary`, `categories` (service/price/food: summary, bullets, evidence). (debug) `positive_reviews`, `negative_reviews`, `positive_count`, `negative_count`, `debug`

---

## 4. Sentiment (감성 분석)

### 4.1 위치

- `src/sentiment_analysis.py` — `SentimentAnalyzer.analyze`, `_classify_contents`
- `src/api/routers/sentiment.py` — `POST /api/v1/sentiment/analyze`, `POST /api/v1/sentiment/analyze/batch`

### 4.2 흐름

1. **리뷰**  
   - `reviews`가 요청에 있으면 사용.  
   - 없으면(또는 빈 리스트) `vector_search`에서 조회:  
     - `ENABLE_SENTIMENT_SAMPLING=true` → `get_recent_restaurant_reviews(limit=SENTIMENT_RECENT_TOP_K)`  
     - `ENABLE_SENTIMENT_SAMPLING=false` → `get_restaurant_reviews` (전체 리뷰)

2. **1차 분류 (HuggingFace)**  
   - `Dilwolf/Kakao_app-kr_sentiment`, `return_all_scores=True`  
   - `positive_score > 0.8` → `positive`  
   - 그 외 → `negative` 후보, LLM 재판정 대상.

3. **2차 분류 (LLM, negative 후보만)**  
   - `id\tcontent` 라인으로 묶어 `gpt-4o-mini` 등 전달.  
   - 출력: `[{"id":..., "sentiment":"positive"|"negative"|"neutral"}]`  
   - id 매핑으로 라벨 보정, 카운트 재계산.

4. **비율**  
   - `positive_ratio`, `negative_ratio` = `*_count / total_with_sentiment * 100`  
   - `neutral_ratio` = `neutral_count / total_count * 100`

### 4.3 출력 (API)

- `restaurant_id`, `positive_count`, `negative_count`, `neutral_count`, `total_count`  
- `positive_ratio`, `negative_ratio`, `neutral_ratio`

### 4.4 공통

- **모델**: `Dilwolf/Kakao_app-kr_sentiment`; `Config.USE_GPU`·`torch.cuda`로 device.  
- **재판정**: `gpt-4o-mini` 등, `id\tcontent`, JSON `[{"id","sentiment"}]`.

---

## 5. Vector (벡터 검색·업로드·리뷰)

### 5.1 위치

- `src/vector_search.py` — `VectorSearch`
- `src/api/routers/vector.py` — `/search/similar`, `/upload` (업로드 형식: {reviews, restaurants?})

### 5.2 컬렉션·벡터 형식

- **Dense**: FastEmbed `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768차원).  
- **Sparse**: FastEmbed `Qdrant/bm25`.  
- **컬렉션**  
  - **named**: `dense`, `sparse` (신규 생성 시 기본).  
  - **단일 벡터**: 기존 마이그레이션.  
- **query_points**  
  - **named** 컬렉션: Dense 단독 검색 시 `using="dense"` 필수.  
  - **단일** 컬렉션: `using` 생략.

### 5.3 검색 파이프라인

| 메서드 | 용도 | 비고 |
|--------|------|------|
| `query_similar_reviews` | Dense (또는 Dense만) 검색 | `using="dense"` (named일 때) |
| `query_hybrid_search` | Dense + Sparse RRF | `FusionQuery(RRF)`, `Prefetch` dense/sparse. 단일 벡터면 Dense로 폴백 |

### 5.4 API 엔드포인트 (현재)

| Method | path | 설명 |
|--------|------|------|
| POST | `/api/v1/vector/search/similar` | 유사 리뷰 검색 (query_text, restaurant_id, limit, min_score) |
| POST | `/api/v1/vector/upload` | 리뷰·레스토랑 업로드, `restaurant_vectors` 자동 생성 |

### 5.5 업로드·포인트

- `prepare_points`  
  - 컬렉션이 **named**이면 `vector={"dense":..., "sparse":...}`.  
  - **단일**이면 `vector=[...]` (dense만).  
- `upload` 시 `restaurant_vectors` 컬렉션에 레스토랑 대표 벡터 생성 (비교군·유사 레스토랑 검색용).

### 5.6 원천데이터 (업로드 전 요구 형태)

벡터 업로드(`POST /api/v1/vector/upload`) 또는 upsert 전에, **원천데이터**가 아래 형태를 만족하면 됩니다. 파일 형식(JSON 배열, TSV, CSV, DB export 등)은 자유롭고, 최종적으로 API에 넘길 때만 아래 필드로 맞추면 됩니다.

**리뷰(한 건당)**

| 구분 | 필드 | 타입 | 설명 |
|------|------|------|------|
| 필수 | 리뷰 식별자 | id 등 | 한 건을 구분하는 고유값 (API에서는 `id`, 필수) |
| 필수 | 레스토랑 식별자 | restaurant_id | 어느 가게 리뷰인지 (API에서는 `restaurant_id`) |
| 필수 | 리뷰 텍스트 | content 등 | 리뷰 본문 (API에서는 `content`) |
| 필수 | 작성 시각 | created_at | 리뷰 작성 시각 (API에서는 `created_at`, ISO 8601) |

**API 스키마**  
업로드·감성 분석 요청에는 `id`, `restaurant_id`, `content`, `created_at` 모두 필수. 검색 응답에는 `id`, `restaurant_id`, `content` 포함.  

**레스토랑(선택)**

| 구분 | 필드 | 타입 | 설명 |
|------|------|------|------|
| 권장 | 레스토랑 식별자 | id | API에서는 `id` |
| 권장 | 이름 | name | API에서는 `name` (대표 벡터·표시용) |

레스토랑 정보가 없어도 리뷰만으로 업로드는 가능합니다. 레스토랑 정보가 있으면 `restaurants` 배열에 넣어 함께 보내면 됩니다.

**넣는 순서**

1. 원천데이터에서 위 필드만 추출해 `{ "reviews": [ ... ], "restaurants": [ ... ] }` 형태로 만든다.  
2. `POST /api/v1/vector/upload`로 전달한다. (리뷰·레스토랑 추가·갱신은 동일 upload로 upsert 가능.)

**참고**  
- `data/test_data_sample.json`: 리뷰 배열 JSON 예시 (필드 많음 → upload 시 `id`, `restaurant_id`, `content`, `created_at` 사용).  
- `kr3.tsv` + `POST /api/v1/test/generate`: TSV 원천을 배치 요청 형식으로 변환한 뒤, 그 결과에서 `reviews`/`restaurants`를 뽑아 upload 형식으로 재가공 가능.

---

## 6. 공통·의존성

### 6.1 의존성 (dependencies)

- `get_qdrant_client`: Qdrant — `:memory:`, `http(s)://`, on-disk 경로.  
- `get_vector_search`: `VectorSearch` (FastEmbed Dense+Sparse, `Config.COLLECTION_NAME`).  
- `get_llm_utils`: `LLMUtils` (`Config.LLM_MODEL`).  
- `get_sentiment_analyzer`: `SentimentAnalyzer` (HF pipeline).  
- `get_metrics_collector`: `MetricsCollector`.  
- `get_debug_mode`: `X-Debug` 헤더, `debug` 쿼리, `DEBUG_MODE` env.

### 6.2 락·SKIP

- **`acquire_lock(restaurant_id, analysis_type, ttl=3600)`** (Redis)  
  - `sentiment`, `summary`, `comparison` 진입 시 중복 실행 방지.  
- **SKIP**  
  - `metrics.metrics_db.should_skip_analysis(restaurant_id, analysis_type, min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS)`  
  - 최근 성공 이력이 있으면 스킵, `last_success_at` 반환.

### 6.3 Config 요약

| 항목 | 용도 |
|------|------|
| `COLLECTION_NAME` | 리뷰 컬렉션 이름 |
| `QDRANT_URL` | `:memory:`, `http://...`, on-disk 경로 |
| `ALL_AVERAGE_ASPECT_DATA_PATH` | Comparison 전체 평균용 파일 (TSV/JSON) |
| `COMPARISON_ASYNC` | Comparison LLM: true면 service/price 병렬(asyncio.gather), false면 순차(기본값) |
| `SUMMARY_SEARCH_ASYNC` | Summary 배치: aspect(service/price/food) 서치 병렬 |
| `SUMMARY_RESTAURANT_ASYNC` | Summary 배치: 음식점 간 병렬 |
| `SUMMARY_LLM_ASYNC` | Summary 배치: LLM 호출. true=AsyncOpenAI/httpx, false=to_thread |
| `SENTIMENT_CLASSIFIER_USE_THREAD` | Sentiment: true면 HF 분류기 asyncio.to_thread(블로킹 격리), false면 메인 스레드(기본값) |
| `SENTIMENT_LLM_ASYNC` | Sentiment: true면 LLM 재판정 AsyncOpenAI, false면 동기(기본값) |
| `ENABLE_SENTIMENT_SAMPLING` | Sentiment: true면 최근 리뷰 샘플링, false면 전체 리뷰 |
| `SENTIMENT_RECENT_TOP_K` | Sentiment 샘플링 시 최근 리뷰 수 (기본 100) |
| `BATCH_SEARCH_CONCURRENCY` | Summary 배치 검색 동시성 (기본 50) |
| `BATCH_LLM_CONCURRENCY` | Summary 배치 LLM 동시성 (기본 8) |
| `OPENAI_MODEL` | Sentiment LLM 재판정·폴백 모델 (gpt-4o-mini 등) |
| `ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO` | 전체 평균 폴백 |
| `ASPECT_SEEDS_FILE` | Summary aspect seed JSON (선택) |
| `SKIP_MIN_INTERVAL_SECONDS` | SKIP 최소 간격(초) |
| `SENTIMENT_MODEL`, `EMBEDDING_MODEL`, `LLM_MODEL` | 모델명 |
| `LLM_PROVIDER`, `USE_RUNPOD`, `USE_POD_VLLM` | LLM 백엔드 |

---

## 7. API ↔ 파이프라인 매핑

| API | 파이프라인 | 핵심 (src) |
|-----|------------|------------|
| `POST /api/v1/llm/comparison` | Comparison | `ComparisonPipeline.compare` → `comparison_pipeline` (비율, lift, format_comparison_display). `comparisons`: `[{category, lift_percentage}]` |
| `POST /api/v1/llm/summarize` | Summary | **기본 시드만** (`DEFAULT_*_SEEDS`) → `query_hybrid_search` → `summarize_aspects_new` |
| `POST /api/v1/llm/summarize/batch` | Summary (배치) | SUMMARY_SEARCH_ASYNC(aspect 병렬), SUMMARY_RESTAURANT_ASYNC(음식점 간 병렬). 시드 1회 로드. 상세: [BATCH_SUMMARY_MODE.md](BATCH_SUMMARY_MODE.md) |
| `POST /api/v1/sentiment/analyze` | Sentiment | `analyze_async` → `_classify_with_hf_only` + LLM 재판정. SENTIMENT_CLASSIFIER_USE_THREAD(to_thread), SENTIMENT_LLM_ASYNC 분기 |
| `POST /api/v1/sentiment/analyze/batch` | Sentiment (배치) | `analyze_multiple_restaurants_async` → `analyze` |
| `POST /api/v1/vector/search/similar` | Vector | `query_similar_reviews` (Dense, named일 때 `using="dense"`) |
| `POST /api/v1/vector/upload` | Vector | `prepare_points`, `upload_collection`, `upsert_restaurant_vector` |

---

## 8. API 입출력 JSON

### 8.1 Comparison

**`POST /api/v1/llm/comparison`**

```json
// 요청 (restaurant_id, top_k만 사용)
{
  "restaurant_id": 1,
  "top_k": 10
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

---

### 8.2 Summary

단일과 배치는 **요청·응답 구조가 다름**.

| | 단일 `POST /summarize` | 배치 `POST /summarize/batch` |
|--|--|--|
| **요청** | **평탄 객체** `{ restaurant_id, limit, min_score }` | **래핑 객체** `{ restaurants: [ { restaurant_id, limit? } ] }` |
| **응답** | **루트가 요약 1개** (results 없음). `X-Debug: true`면 SummaryResponse, 아니면 SummaryDisplayResponse(필드 적음) | **`{ results: [ ... ] }`** 배열. 각 요소는 **SummaryDisplayResponse** 형식(단일 debug=false와 동일, positive_reviews 등 없음) |

---

**단일 `POST /api/v1/llm/summarize`**

- 요청: 평탄. 사용: `restaurant_id`, `limit`, `min_score`.

```json
// 요청 (평탄 객체, 레스토랑 1개)
{
  "restaurant_id": 1,
  "limit": 10,
  "min_score": 0.0
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

- 요청: `restaurants` 배열(각 항목: `restaurant_id`), **`limit`**(선택, 전체 공통, 기본 10), **`min_score`**(선택, 전체 공통, 기본 0.0).
- 응답: **`results` 배열**. 각 요소는 **SummaryDisplayResponse** 형식(단일 debug=false와 동일).
- **배치 (Config)**: `SUMMARY_SEARCH_ASYNC`=aspect(service/price/food) 서치 병렬, `SUMMARY_RESTAURANT_ASYNC`=음식점 간 병렬. 둘 다 false면 완전 순차. `SUMMARY_LLM_ASYNC`=LLM 호출 비동기(to_thread vs AsyncOpenAI). 동시성: `BATCH_SEARCH_CONCURRENCY`, `BATCH_LLM_CONCURRENCY`. 자세한 내용은 [BATCH_SUMMARY_MODE.md](BATCH_SUMMARY_MODE.md) 참고.

```json
// 요청 (restaurants + limit/min_score는 전체 레스토랑에 일괄 적용)
{
  "restaurants": [
    {"restaurant_id": 1},
    {"restaurant_id": 2}
  ],
  "limit": 10,
  "min_score": 0.0
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

### 8.3 Sentiment

**`POST /api/v1/sentiment/analyze`**

```json
// 요청 (리뷰: id, restaurant_id, content, created_at 필수)
{
  "restaurant_id": 1,
  "reviews": [
    {"id": 1, "restaurant_id": 1, "content": "맛있어요! 서비스도 친절해요.", "created_at": "2025-02-17T17:02:45.366789"},
    {"id": 2, "restaurant_id": 1, "content": "맛있어요", "created_at": "2025-11-14T17:02:45.367453"}
  ]
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

```json
// 요청 (리뷰: id, restaurant_id, content, created_at 필수)
{
  "restaurants": [
    {"restaurant_id": 1, "reviews": [{"id": 1, "restaurant_id": 1, "content": "맛있어요! 서비스도 친절해요.", "created_at": "2025-02-17T17:02:45.366789"}, {"id": 2, "restaurant_id": 1, "content": "맛있어요", "created_at": "2025-11-14T17:02:45.367453"}]},
    {"restaurant_id": 2, "reviews": [{"id": 10, "restaurant_id": 2, "content": "맛있어요", "created_at": "2025-11-14T17:02:45.367453"}]},
    {"restaurant_id": 3, "reviews": [{"id": 20, "restaurant_id": 3, "content": "맛있어요", "created_at": "2025-11-14T17:02:45.367454"}]}
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

### 8.4 Vector

**`POST /api/v1/vector/search/similar`**

```json
// 요청
{
  "query_text": "분위기 좋고 데이트하기 좋은",
  "restaurant_id": 1,
  "limit": 5,
  "min_score": 0.0
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

## 9. API 요청/응답 DTO

API별 요청/응답에 사용되는 Pydantic 모델(DTO)과 필드를 정리합니다. 정의 위치: `src/models.py`.  
*(라우터: `src/api/routers/llm.py`, `sentiment.py`, `vector.py` 기준으로 검증됨.)*

### 9.1 API → DTO 매핑

| API | Request DTO | Response DTO |
|-----|-------------|--------------|
| `POST /api/v1/llm/comparison` | `ComparisonRequest` | `ComparisonResponse` (또는 Dict, 에러/스킵 시) |
| `POST /api/v1/llm/summarize` | `SummaryRequest` | `SummaryDisplayResponse` (debug 필드 선택) |
| `POST /api/v1/llm/summarize/batch` | `SummaryBatchRequest` | `SummaryBatchResponse` |
| `POST /api/v1/sentiment/analyze` | `SentimentAnalysisRequest` | `SentimentAnalysisDisplayResponse` (debug=false) / `SentimentAnalysisResponse` (debug=true) |
| `POST /api/v1/sentiment/analyze/batch` | `SentimentAnalysisBatchRequest` | `SentimentAnalysisBatchResponse` |
| `POST /api/v1/vector/search/similar` | `VectorSearchRequest` | `VectorSearchResponse` |
| `POST /api/v1/vector/upload` | `VectorUploadRequest` | `VectorUploadResponse` |

### 9.2 Comparison DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **ComparisonRequest** | restaurant_id | int | 타겟 레스토랑 ID |
| | top_k | int | 반환 최대 개수 (1~50, 기본 10) |
| **ComparisonDetail** | category | str | "service" \| "price" |
| | lift_percentage | float | (단일−전체)/전체×100 |
| **ComparisonResponse** | restaurant_id | int | 레스토랑 ID |
| | comparisons | List[ComparisonDetail] | 비교 항목 리스트 |
| | total_candidates | int | 근거 후보 총 개수 |
| | validated_count | int | 검증 통과 개수 |
| | category_lift | Dict[str, float]? | service/price lift |
| | comparison_display | List[str]? | 표시 문장 리스트 |
| | debug | DebugInfo? | 디버그 정보 |

### 9.3 Summary DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **SummaryRequest** | restaurant_id | int | 레스토랑 ID |
| | limit | int | 카테고리당 검색 최대 개수 (1~100, 기본 10) |
| | min_score | float | 최소 유사도 (0.0~1.0) |
| **SummaryBatchRequest** | restaurants | List[Dict] | [{ restaurant_id }, ...] |
| | limit | int | 카테고리당 검색 최대 개수, 전체 공통 (1~100, 기본 10) |
| | min_score | float | 최소 유사도, 전체 공통 (0.0~1.0) |
| **CategorySummary** | summary | str | 카테고리 요약 |
| | bullets | List[str] | 핵심 포인트 |
| | evidence | List[Dict] | [{ review_id, snippet, rank }] |
| **SummaryDisplayResponse** | restaurant_id | int | 레스토랑 ID |
| | overall_summary | str | 전체 요약 |
| | categories | Dict[str, CategorySummary]? | service/price/food 요약 |
| | debug | DebugInfo? | X-Debug: true 시에만 |
| **SummaryBatchResponse** | results | List[SummaryDisplayResponse] | 레스토랑별 요약 |

### 9.4 Sentiment DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **SentimentReviewInput** | id | int | 리뷰 ID (필수, LLM 재판정 매핑용) |
| | restaurant_id | int | 레스토랑 ID |
| | content | str | 리뷰 내용 |
| | created_at | datetime | 리뷰 작성 시각 (ISO 8601, 필수) |
| **SentimentAnalysisRequest** | restaurant_id | int | 레스토랑 ID |
| | reviews | List[SentimentReviewInput] | 리뷰 리스트 |
| **SentimentAnalysisDisplayResponse** | restaurant_id | int | 레스토랑 ID |
| | positive_ratio | int | 긍정 비율 (%) |
| | negative_ratio | int | 부정 비율 (%) |
| **SentimentAnalysisResponse** | (Display 필드) + | | |
| | positive_count, negative_count, neutral_count | int | 개수 |
| | total_count | int | 전체 리뷰 수 |
| | neutral_ratio | int | 중립 비율 (%) |
| | debug | DebugInfo? | 디버그 정보 |
| **SentimentRestaurantBatchInput** | restaurant_id | int | 레스토랑 ID |
| | reviews | List[SentimentReviewInput] | 리뷰 리스트 |
| **SentimentAnalysisBatchRequest** | restaurants | List[SentimentRestaurantBatchInput] | 레스토랑별 리뷰 |
| **SentimentAnalysisBatchResponse** | results | List[SentimentAnalysisResponse] | 레스토랑별 결과 |

### 9.5 Vector DTO

| DTO | 필드 | 타입 | 설명 |
|-----|------|------|------|
| **VectorSearchRequest** | query_text | str | 검색 쿼리 |
| | restaurant_id | int? | 레스토랑 필터 |
| | limit | int | 최대 개수 (1~100, 기본 3) |
| | min_score | float | 최소 유사도 (0.0~1.0) |
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

### 9.6 공통 DTO

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

### 9.7 공통 에러 응답 포맷

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

### 9.8 선택 필드 역할

선택(optional) 필드는 **없어도 요청/응답이 성립**하지만, **있으면 그 값이 동작에 반영**됩니다.

**요청(Request) 쪽**

| DTO | 필드 | 역할 |
|-----|------|------|
| **SentimentReviewInput** | id | **필수.** LLM 재판정 시 매핑용. 1차 분류 후 negative만 LLM 재판정할 때, LLM 출력 `[{"id": ..., "sentiment": ...}]`와 매칭. |
| **SentimentReviewInput** | created_at | **필수.** 리뷰 작성 시각 (ISO 8601). ENABLE_SENTIMENT_SAMPLING 시 get_recent_restaurant_reviews 정렬에 사용. |
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

## 10. 참고

- **Comparison**: 현재 API는 Kiwi+lift 경로만 사용. 요청은 `restaurant_id`, `top_k`만. `comparisons`: `{category, lift_percentage}`. Spark 사용(comparison_pipeline), 로그 억제: [trouble_shooting/SPARK_LOG_NOISE.md](trouble_shooting/SPARK_LOG_NOISE.md).  
- **Summary**: **기본 시드만 사용** (`DEFAULT_*_SEEDS` 직접, `load_aspect_seeds`·파일 미사용) + `query_hybrid_search` (Dense+Sparse RRF) → `summarize_aspects_new`.  
- **Vector**: named 컬렉션에서 Dense 단독 검색 시 `using="dense"` 필요. 단일 벡터 컬렉션은 `using` 없음.  
- **Sentiment**: `SentimentAnalyzer` (HF 1차 + LLM 2차 재판정). `ENABLE_SENTIMENT_SAMPLING`에 따라 전체/샘플링 분기.
