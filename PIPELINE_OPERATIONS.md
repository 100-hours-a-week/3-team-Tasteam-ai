# 파이프라인 동작 문서 (src API 기반)

**Strength**, **Summary**, **Sentiment**, **Vector** 파이프라인의 API·입력·출력·처리 단계를 정리한 문서입니다.

- **범위**: `src/` 기반 HTTP API. `hybrid_search` / `final_pipeline` 미포함.
- **테스트**: `test_all_task.py` — API 호출만 사용.

---

## 1. 개요

### 1.1 API 구조

| prefix | 라우터 | 용도 |
|--------|--------|------|
| `/api/v1/llm` | `llm` | Summary, Strength |
| `/api/v1/sentiment` | `sentiment` | 감성 분석 |
| `/api/v1/vector` | `vector` | 벡터 검색·업로드·리뷰 CRUD |
| `/api/v1/restaurants` | `restaurant` | (미사용) |
| `/api/v1/test` | `test` | 테스트용 |

- **문서**: `/docs`, **헬스**: `/health`

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
│     Vector      │            │     Summary     │            │    Strength     │
│ /search/similar │            │   /summarize    │            │ /extract/       │
│ /search/review- │            │   /summarize/   │            │   strengths     │
│   images        │            │   batch         │            │                 │
│ /upload         │            │                 │            │  Kiwi bigram    │
│ /reviews/upsert │            │  aspect_seeds   │            │  → service/     │
│ /reviews/delete │            │  → hybrid검색   │            │  price 비율     │
└─────────────────┘            │  → LLM 요약     │            │  → lift, display│
         │                     └─────────────────┘            └─────────────────┘
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

## 2. Strength (강점 추출)

### 2.1 위치

- `src/strength_extraction.py` — `StrengthExtractionPipeline.extract_strengths`
- `src/strength_pipeline.py` — `calculate_single_restaurant_ratios`, `calculate_strength_lift`, `format_strength_display`, `calculate_all_average_ratios_from_*`
- `POST /api/v1/llm/extract/strengths` (`src/api/routers/llm.py`)

### 2.2 흐름

1. **리뷰 조회**  
   - `vector_search.get_restaurant_reviews(restaurant_id)` → `content`/`text`만 `review_texts`로.

2. **단일 음식점 비율**  
   - `strength_pipeline.calculate_single_restaurant_ratios(review_texts, stopwords)`  
   - Kiwi NNG/NNP bigram + `SERVICE_KW`/`PRICE_KW`/`SERVICE_POSITIVE_KW`/`PRICE_POSITIVE_KW` → `service`, `price` 긍정 비율.

3. **전체 평균 (all_average_ratios)**  
   - ① `ALL_AVERAGE_ASPECT_DATA_PATH` 파일(Spark/TSV·JSON)  
   - ② `get_all_reviews_for_all_average(5000)` (Qdrant)  
   - ③ `Config.ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO`

4. **Lift**  
   - `calculate_strength_lift(single_ratios, all_average_ratios)`  
   - `lift = (단일 − 전체) / 전체 × 100`

5. **strength_display**  
   - `format_strength_display(lift_service, lift_price)`  
   - `multiple = 1 + lift/100` → `"이 음식점의 서비스 만족도는 판교 평균의 {multiple:.2f}배 수준입니다."` 등.

6. **강점 리스트**  
   - `lift > 0` 인 service/price만 `strengths`에 추가  
   - `{ "category": "service"|"price", "lift_percentage": float }`  
   - `lift_percentage` 내림차순 후 `top_k`개.

### 2.3 출력 (API)

- `restaurant_id`, `strength_type`, `strengths`, `total_candidates`, `validated_count`  
- `category_lift`, `strength_display`, `processing_time_ms`  
- **strengths**: `[{"category":"service","lift_percentage":20.0}, ...]`

### 2.4 공통

- **Kiwi**: NNG/NNP bigram.  
- **키워드**: service `친절, 서비스, 응대, 직원, 사장, 불친절`; price `가격, 가성비, 대비, 리필, 무한, 할인, 쿠폰`; 긍정 시드 별도.  
- **불용어**: `data/stopwords-ko.txt` 우선.

---

## 3. Summary (요약)

### 3.1 위치

- `src/summary_pipeline.py` — `summarize_aspects_new`
- `src/aspect_seeds.py` — `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS`
- `src/api/routers/llm.py` — `POST /api/v1/llm/summarize`, `POST /api/v1/llm/summarize/batch`
- `src/vector_search.py` — `query_hybrid_search`

### 3.2 시드: 기본 시드만 사용

- **기본 시드만 사용.** `ASPECT_SEEDS_FILE`·`load_aspect_seeds()` 미사용.
- `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` (aspect_seeds 모듈 상수) 직접 사용.
- `seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]`, `name_list = ["service","price","food"]`.

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
   - `overall_summary`, `categories`, `positive_aspects`/`negative_aspects`=[], `positive_count`/`negative_count`=0.

### 3.4 출력 (API)

- `restaurant_id`, `overall_summary`, `categories` (service/price/food: summary, bullets, evidence)  
- `positive_aspects`/`negative_aspects`=[], `positive_count`/`negative_count`=0, (debug) `debug`

---

## 4. Sentiment (감성 분석)

### 4.1 위치

- `src/sentiment_analysis.py` — `SentimentAnalyzer.analyze`, `_classify_contents`
- `src/api/routers/sentiment.py` — `POST /api/v1/sentiment/analyze`, `POST /api/v1/sentiment/analyze/batch`

### 4.2 흐름

1. **리뷰**  
   - `reviews`가 요청에 있으면 사용.  
   - 없으면 `vector_search.get_restaurant_reviews` 등 (구성에 따라 다름).

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
- `src/api/routers/vector.py` — `/search/similar`, `/search/review-images`, `/upload`, `/restaurants/{id}/reviews`, `/reviews/upsert`, `/reviews/upsert/batch`, `/reviews/delete`, `/reviews/delete/batch`

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
| `query_similar_reviews_with_expansion` | 쿼리 확장 후 검색 | `_should_expand_query` 또는 `expand_query`로 확장 여부 결정, `query_similar_reviews` 호출 |
| `get_reviews_with_images` | 이미지 있는 리뷰만 | `query_similar_reviews_with_expansion` 후 `image_urls` 필터 |

### 5.4 API 엔드포인트

| Method | path | 설명 |
|--------|------|------|
| POST | `/api/v1/vector/search/similar` | 유사 리뷰 검색 (query_text, restaurant_id, limit, min_score, expand_query) |
| POST | `/api/v1/vector/search/review-images` | 리뷰 이미지 검색 (query, restaurant_id, limit, min_score, expand_query) |
| GET | `/api/v1/vector/restaurants/{restaurant_id}/reviews` | 레스토랑별 리뷰 목록 |
| POST | `/api/v1/vector/upload` | 리뷰·레스토랑 업로드, `restaurant_vectors` 자동 생성 |
| POST | `/api/v1/vector/reviews/upsert` | 리뷰 1건 upsert (낙관적 잠금 지원) |
| POST | `/api/v1/vector/reviews/upsert/batch` | 리뷰 배치 upsert |
| DELETE | `/api/v1/vector/reviews/delete` | 리뷰 1건 삭제 |
| DELETE | `/api/v1/vector/reviews/delete/batch` | 리뷰 배치 삭제 |

### 5.5 업로드·포인트

- `prepare_points`  
  - 컬렉션이 **named**이면 `vector={"dense":..., "sparse":...}`.  
  - **단일**이면 `vector=[...]` (dense만).  
- `upload` 시 `restaurant_vectors` 컬렉션에 레스토랑 대표 벡터 생성 (비교군·유사 레스토랑 검색용).

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
  - `sentiment`, `summary`, `strength` 진입 시 중복 실행 방지.  
- **SKIP**  
  - `metrics.metrics_db.should_skip_analysis(restaurant_id, analysis_type, min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS)`  
  - 최근 성공 이력이 있으면 스킵, `last_success_at` 반환.

### 6.3 Config 요약

| 항목 | 용도 |
|------|------|
| `COLLECTION_NAME` | 리뷰 컬렉션 이름 |
| `QDRANT_URL` | `:memory:`, `http://...`, on-disk 경로 |
| `ALL_AVERAGE_ASPECT_DATA_PATH` | Strength 전체 평균용 파일 (TSV/JSON) |
| `ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO` | 전체 평균 폴백 |
| `ASPECT_SEEDS_FILE` | Summary aspect seed JSON (선택) |
| `SKIP_MIN_INTERVAL_SECONDS` | SKIP 최소 간격(초) |
| `SENTIMENT_MODEL`, `EMBEDDING_MODEL`, `LLM_MODEL` | 모델명 |
| `LLM_PROVIDER`, `USE_RUNPOD`, `USE_POD_VLLM` | LLM 백엔드 |

---

## 7. API ↔ 파이프라인 매핑

| API | 파이프라인 | 핵심 (src) |
|-----|------------|------------|
| `POST /api/v1/llm/extract/strengths` | Strength | `StrengthExtractionPipeline.extract_strengths` → `strength_pipeline` (비율, lift, format_strength_display). `strengths`: `[{category, lift_percentage}]` |
| `POST /api/v1/llm/summarize` | Summary | **기본 시드만** (`DEFAULT_*_SEEDS`) → `query_hybrid_search` → `summarize_aspects_new` |
| `POST /api/v1/llm/summarize/batch` | Summary (배치) | 동일, 레스토랑별 반복, 시드 1회 로드 |
| `POST /api/v1/sentiment/analyze` | Sentiment | `SentimentAnalyzer.analyze` → `_classify_contents` (HF + LLM 재판정) |
| `POST /api/v1/sentiment/analyze/batch` | Sentiment (배치) | `analyze_multiple_restaurants_async` → `analyze` |
| `POST /api/v1/vector/search/similar` | Vector | `query_similar_reviews_with_expansion` → `query_similar_reviews` (Dense, named일 때 `using="dense"`) |
| `POST /api/v1/vector/search/review-images` | Vector | `get_reviews_with_images` → `query_similar_reviews_with_expansion`, `image_urls` 필터 |
| `GET /api/v1/vector/restaurants/{id}/reviews` | Vector | `get_restaurant_reviews` |
| `POST /api/v1/vector/upload` | Vector | `prepare_points`, `upload_collection`, `upsert_restaurant_vector` |
| `POST /api/v1/vector/reviews/upsert` | Vector | `upsert_review` |
| `POST /api/v1/vector/reviews/upsert/batch` | Vector | `upsert_reviews_batch` |
| `DELETE /api/v1/vector/reviews/delete` | Vector | `delete_review` |
| `DELETE /api/v1/vector/reviews/delete/batch` | Vector | `delete_reviews_batch` |

---

## 8. 참고

- **Strength**: 현재 API는 Kiwi+lift 경로만 사용. `strengths`: `{category, lift_percentage}`.  
- **Summary**: **기본 시드만 사용** (`DEFAULT_*_SEEDS` 직접, `load_aspect_seeds`·파일 미사용) + `query_hybrid_search` (Dense+Sparse RRF) → `summarize_aspects_new`.  
- **Vector**: named 컬렉션에서 Dense 단독 검색 시 `using="dense"` 필요. 단일 벡터 컬렉션은 `using` 없음.  
- **Sentiment**: `SentimentAnalyzer` (HF 1차 + LLM 2차 재판정).
