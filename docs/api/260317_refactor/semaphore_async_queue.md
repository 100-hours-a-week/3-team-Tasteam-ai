
코드 기준으로 정리한 현황입니다.

---

## Sentiment

| 항목 | 적용 여부 | 설명 |
|------|-----------|------|
| **비동기 처리** | ✅ | 단일: `analyze_sentiment` → `analyzer.analyze_async` (async). 배치: `analyze_sentiment_batch` → `analyze_multiple_restaurants_async`. HF 분류기는 `SENTIMENT_CLASSIFIER_USE_THREAD=true`(기본)일 때 `asyncio.to_thread(_classify_with_hf_only, ...)`로 블로킹 격리. 음식점 간 병렬은 `SENTIMENT_RESTAURANT_ASYNC=true`(기본)일 때 `asyncio.gather(*tasks)`. |
| **비동기 큐** | ❌ | asyncio.Queue / 작업 큐 없음. (배치 job 넣을 때만 RQ 사용.) |
| **세마포** | ❌ | 동시성 제한용 세마포 없음. |

- Config: `SENTIMENT_CLASSIFIER_USE_THREAD`, `SENTIMENT_RESTAURANT_ASYNC` (기본 true).  
- LLM 재판정 제거 후, 애매 구간은 중립 처리만 하므로 `SENTIMENT_LLM_ASYNC`는 사용되지 않음.

---

## Summary

| 항목 | 적용 여부 | 설명 |
|------|-----------|------|
| **비동기 처리** | ✅ | 단일: `summarize_reviews` async, 내부에서 `asyncio.to_thread`(seed 생성·검색 등). 배치: `summarize_reviews_batch` → `_batch_summarize_async`. 레스토랑당 `_process_one_restaurant_async`에서 검색은 `SUMMARY_SEARCH_ASYNC`일 때 `asyncio.gather(do_one_search...)`, 요약은 `SUMMARY_LLM_ASYNC`일 때 `summarize_aspects_new_async`, 아니면 `asyncio.to_thread(summarize_aspects_new, ...)`. Distill 사용 시 `summarize_aspects_new_async` 안에서 `asyncio.to_thread(generate_summary_sync, payload)`. 레스토랑 간 병렬은 `SUMMARY_RESTAURANT_ASYNC`(기본 true)일 때 `asyncio.gather(*(one(rd)...))`. |
| **비동기 큐** | ❌ | asyncio.Queue 없음. (배치 job은 RQ로 enqueue.) |
| **세마포** | ✅ | **검색**: `search_sem = asyncio.Semaphore(Config.BATCH_SEARCH_CONCURRENCY)` (기본 50). `do_one_search`에서 `async with search_sem:` 후 `asyncio.to_thread(_retrieve_category_hits_...)`. **LLM/Distill**: `llm_sem = asyncio.Semaphore(Config.BATCH_LLM_CONCURRENCY)` (기본 8). `_process_one_restaurant_async`에서 `async with llm_sem:` 안에서 요약 호출. |

- 적용 위치: `src/api/routers/llm.py`의 `_batch_summarize_async` / `_process_one_restaurant_async` (310–311, 246, 276라인).
- Config: `BATCH_SEARCH_CONCURRENCY`, `BATCH_LLM_CONCURRENCY`, `SUMMARY_SEARCH_ASYNC`, `SUMMARY_RESTAURANT_ASYNC`, `SUMMARY_LLM_ASYNC`.

---

## Comparison

| 항목 | 적용 여부 | 설명 |
|------|-----------|------|
| **비동기 처리** | ✅ | 단일: `compare` async (수치·템플릿만 사용, LLM 없음). 배치: `compare_batch`에서 `COMPARISON_BATCH_ASYNC=true`(기본)일 때 `asyncio.gather(*tasks)`로 음식점별 `compare` 병렬, false면 순차. |
| **비동기 큐** | ❌ | asyncio.Queue 없음. |
| **세마포** | ❌ | 동시성 제한용 세마포 없음. |

- Config: `COMPARISON_BATCH_ASYNC` (기본 true).  
- (과거 `COMPARISON_ASYNC`는 service/price LLM 병렬용이었고, 현재는 템플릿만 쓰므로 사용처 없음.)

---

## 공통·기타

- **app_queue_depth**: `src/api/main.py`의 `track_queue_depth` 미들웨어에서 전역 in-flight 요청 수만 Prometheus로 집계. “비동기 큐” 구현이 아니라 메트릭 이름만 큐처럼 사용.
- **RQ(Redis Queue)**: `src/api/routers/batch.py`의 `POST /enqueue`에서 sentiment/summary/comparison 배치 작업을 RQ에 넣을 때만 사용. API 내부의 asyncio 비동기 흐름과는 별개.

---

## 요약 표

| 구간 | 비동기 처리 | 비동기 큐(asyncio.Queue) | 세마포 |
|------|-------------|---------------------------|--------|
| **Sentiment** | ✅ (to_thread + gather) | ❌ | ❌ |
| **Summary** | ✅ (to_thread + gather) | ❌ | ✅ search_sem(50), llm_sem(8) |
| **Comparison** | ✅ (gather 배치) | ❌ | ❌ |