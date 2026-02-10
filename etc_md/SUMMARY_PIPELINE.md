# Summarize 파이프라인 문서

레스토랑 리뷰를 **service / price / food** 세 카테고리로 검색·요약하는 파이프라인의 전체 구조, 단계, 입출력, 설정을 정리한 문서입니다.

---

## 1. 목적과 범위

| 항목 | 내용 |
|------|------|
| **목적** | 레스토랑별로 카테고리별 요약(summary, bullets, evidence)과 전체 요약(overall_summary) 생성 |
| **카테고리** | service(서비스/친절), price(가격/가성비), food(음식/메뉴) |
| **말투** | 출력은 모두 **"~해요" 체** (예: 좋아요, 있어요, 없어요) |
| **비고** | 긍정/부정 비율은 세지 않음 → Sentiment 파이프라인과 분리 |

---

## 2. API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/summarize` | 단일 레스토랑 요약 |
| POST | `/summarize/batch` | 여러 레스토랑 배치 요약 |

**단일 요청 예시**

- Body: `{ "restaurant_id": 123, "limit": 10 }`  
- `limit`: 카테고리당 검색·요약에 사용할 최대 리뷰 수 (기본 10)

**배치 요청 예시**

- Body: `{ "restaurants": [{ "restaurant_id": 123 }, ...], "limit": 10 }`  
- `limit`: 모든 레스토랑에 공통 적용

**공통 동작**

- **락**: `restaurant_id` + `analysis_type="summary"` 로 Redis 락(TTL 1시간). 실패 시 409 Conflict.
- **SKIP**: 동일 레스토랑이 `SKIP_MIN_INTERVAL_SECONDS` 이내에 성공한 적 있으면 재계산 생략 후 빈/스킵 응답.
- **응답**: `SummaryDisplayResponse` — `restaurant_id`, `restaurant_name`, `overall_summary`, `categories`(service/price/food 각각 `summary`, `bullets`, `evidence`), (디버그 시) `debug`.

---

## 3. 파이프라인 흐름 개요

```
[요청] → 락 획득 → SKIP 여부 → 시드 결정 → 카테고리별 검색 → LLM 요약 → 후처리 → 응답
```

- **시드**: `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` 만 사용 (파일/ASPECT_SEEDS_FILE 미사용).
- **검색**: 카테고리별로 `_retrieve_category_hits_accuracy_first` 호출 → 1차 Dense → 조건부 2차 Hybrid → 부족 시 넓은 쿼리 → 최근 리뷰 폴백.
- **요약**: 검색된 리뷰를 `summarize_aspects_new`(또는 배치 시 `summarize_aspects_new_async`)로 한 번에 3카테고리 + overall_summary 생성.
- **후처리**: evidence 인덱스 → 실제 객체 치환, price 게이트(가격 키워드 없을 때 고정 문구).

---

## 4. 단계별 상세

### 4.1 시드(쿼리) 결정

- **위치**: `src/aspect_seeds.py`
- **사용 값**: `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` (예: 직원 친절, 가격 대비, 가락 국수 등).
- **쿼리 생성**: 카테고리마다 시드 **최대 10개**를 공백으로 이어 붙인 문자열을 검색 쿼리로 사용.

### 4.2 카테고리별 검색: `_retrieve_category_hits_accuracy_first`

**위치**: `src/api/routers/llm.py`

**입력**: `vector_search`, `category_name`, `query_text`, `restaurant_id`, `final_limit`

**단계**:

1. **1차 Dense-only**  
   - `_query_dense_only(..., limit=max(final_limit*8, 50), min_score=0.0)` 로 후보 확보.

2. **2차 Hybrid(RRF) 조건**  
   - 아래 중 하나라도 만족하면 Hybrid 수행:  
     - `len(dense_hits) < k_min` (k_min = min(3, final_limit))  
     - 상위 8개 텍스트의 카테고리 키워드 적합도 < 0.25  
     - 최고 점수 < 0.25  
     - 상위 5개 점수 차이(top1−top5) < 0.02 (평평함)  
   - Dense vs Hybrid 중 키워드 적합도·결과 수 기준으로 선택 후 `review_id` 기준 dedup.

3. **3차 넓은 쿼리**  
   - 여전히 `len(best_hits) < k_min` 이면 카테고리별 `_BROAD_QUERY`를 붙여 Hybrid 1회 추가 후 merge·dedup.

4. **4차 최근 리뷰**  
   - 그래도 부족하면 `get_recent_restaurant_reviews` 로 해당 레스토랑 최근 리뷰를 가져와 hit 형태로 합친 뒤 dedup.

**출력**: `[{ "payload": { "content", "review_id", ... }, "score": ... }, ...]` 형태의 리스트.

### 4.3 LLM 요약: `summarize_aspects_new` / `summarize_aspects_new_async`

**위치**: `src/summary_pipeline.py`

**입력**:

- `service_reviews`, `price_reviews`, `food_reviews`: 카테고리별 리뷰 텍스트 리스트.
- `service_evidence_data`, `price_evidence_data`, `food_evidence_data`: 카테고리별 `[{ "review_id", "snippet", "rank" }, ...]`.
- `llm_utils`, `per_category_max`: LLM 유틸과 카테고리당 최대 리뷰 수(클리핑용, 기본 8).

**처리**:

- 각 카테고리 리스트를 `per_category_max` 개로 클리핑한 JSON을 LLM에 전달.
- **프롬프트 규칙**: 말투 "~해요" 체, summary 1문장, bullets 3~5개, evidence는 0-based 인덱스, overall_summary 2~3문장, 근거 없으면 "언급이 적어요" 등 해요체로 표현.

**LLM 출력 스키마**:

```json
{
  "service": { "summary": "", "bullets": [], "evidence": [0,1,...] },
  "price":   { "summary": "", "bullets": [], "evidence": [] },
  "food":    { "summary": "", "bullets": [], "evidence": [] },
  "overall_summary": { "summary": "" }
}
```

**후처리**:

- **Evidence 변환**: LLM이 준 인덱스를 해당 카테고리의 evidence 데이터(`review_id`, `snippet`, `rank`) 객체로 치환.
- **Price 게이트**: price evidence 리뷰에 가격 관련 키워드(PRICE_HINTS)가 전혀 없으면  
  - summary: "가격 관련 언급이 많지 않아요. 전반적인 만족감이나 구성(양 등) 중심으로만 해석 가능해요."  
  - bullets: "가격을 직접 언급한 리뷰가 많지 않아요.", "대신 만족/구성/양(푸짐함) 관련 표현이 간접적으로 나타나요."

**실패 시**: 카테고리별 빈 summary/bullets/evidence, overall_summary "요약 생성에 실패했어요." 또는 라우터에서 "요약할 리뷰가 없어요." 등으로 처리.

### 4.4 응답 조립

- 파이프라인 결과를 `SummaryDisplayResponse` 형식으로 변환.
- `overall_summary`가 비어 있으면 카테고리별 summary를 공백으로 이어서 사용하고, 그래도 없으면 "요약할 리뷰가 없어요." 사용.
- 메트릭 수집 후 반환.

---

## 5. 배치 모드 동작

| 설정 | 의미 | 기본값 |
|------|------|--------|
| `SUMMARY_SEARCH_ASYNC` | 카테고리(service/price/food) 검색 병렬 여부 | true |
| `SUMMARY_RESTAURANT_ASYNC` | 레스토랑 간 병렬 여부 | true |
| `SUMMARY_LLM_ASYNC` | LLM 비동기 호출 여부 (`summarize_aspects_new_async` 사용) | true |

- `SUMMARY_SEARCH_ASYNC=true`: 한 레스토랑 내에서 3개 카테고리 검색을 `asyncio.gather`로 병렬 수행.
- `SUMMARY_RESTAURANT_ASYNC=true`: 레스토랑 단위를 `asyncio.gather`로 병렬 수행.
- `SUMMARY_LLM_ASYNC=true`: 배치에서 `summarize_aspects_new_async` 사용, false면 `asyncio.to_thread(summarize_aspects_new)`.
- **세마포어**: `BATCH_SEARCH_CONCURRENCY`, `BATCH_LLM_CONCURRENCY` 로 검색/LLM 동시 실행 수 제한.
- 둘 다 false면 레스토랑·카테고리 모두 순차 처리.

---

## 6. 설정(Config) 요약

| 설정 | 의미 | 기본값 |
|------|------|--------|
| `SUMMARY_SEARCH_ASYNC` | 배치 시 카테고리별 검색 병렬 | true |
| `SUMMARY_RESTAURANT_ASYNC` | 배치 시 레스토랑 간 병렬 | true |
| `SUMMARY_LLM_ASYNC` | 배치 시 LLM 비동기 호출 | true |
| `BATCH_SEARCH_CONCURRENCY` | 배치 검색 동시 실행 수 | 50 |
| `BATCH_LLM_CONCURRENCY` | 배치 LLM 동시 실행 수 | 8 |
| `DENSE_PREFETCH_LIMIT` | Hybrid 검색 Dense prefetch | 200 |
| `SPARSE_PREFETCH_LIMIT` | Hybrid 검색 Sparse prefetch | 300 |
| `FALLBACK_MIN_SCORE` | Hybrid fallback min score | 0.2 |
| `SKIP_MIN_INTERVAL_SECONDS` | 동일 레스토랑 summary 재실행 스킵 간격(초) | 3600 |

---

## 7. 관련 파일

| 파일 | 역할 |
|------|------|
| `src/api/routers/llm.py` | `/summarize`, `/summarize/batch`, 락/SKIP, `_retrieve_category_hits_accuracy_first`, 배치 오케스트레이션 |
| `src/summary_pipeline.py` | `summarize_aspects_new`, `summarize_aspects_new_async`, 프롬프트, JSON 파싱, evidence 치환, price 게이트 |
| `src/aspect_seeds.py` | `DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS` |
| `src/vector_search.py` | `_query_dense_only`, `query_hybrid_search`, `get_recent_restaurant_reviews` |
| `src/config.py` | Summary/배치 관련 Config |
| `src/models.py` | `SummaryRequest`, `SummaryBatchRequest`, `SummaryDisplayResponse`, `CategorySummary` |

---

이 문서는 현재 코드 기준으로 작성되었습니다. 파이프라인 변경 시 함께 갱신하는 것을 권장합니다.
