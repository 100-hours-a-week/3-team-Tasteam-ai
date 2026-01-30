# 배치 요약 (search_async / restaurant_async)

`POST /api/v1/llm/summarize/batch` 엔드포인트의 처리 방식을 설정으로 전환할 수 있도록 구현한 내용을 정리한 문서입니다.

---

## 1. 개요: search_async vs restaurant_async

| 설정 | 의미 | true일 때 | false(기본)일 때 |
|------|------|-----------|------------------|
| **search_async** (BATCH_SEARCH_ASYNC) | **aspect(service/price/food) 서치** 병렬 여부 | 레스토랑 1개 안에서 서치 3개를 `gather`로 병렬 | 서치 3개 순차 (service → price → food) |
| **restaurant_async** (BATCH_RESTAURANT_ASYNC) | **음식점 간** 병렬 여부 | 여러 레스토랑을 `gather`로 동시 처리 | 레스토랑을 for 루프로 순차 처리 |

- **둘 다 false**: 완전 순차 (기존 sync for 루프).
- **둘 중 하나라도 true**: 비동기 경로 사용 (세마포어로 동시성 제한).

---

## 2. 설정 (Config / 환경 변수)

`src/config.py` 및 환경 변수:

| 설정 | 환경 변수 | 기본값 | 설명 |
|------|-----------|--------|------|
| **BATCH_SEARCH_ASYNC** (search_async) | `BATCH_SEARCH_ASYNC` | `false` | aspect(service/price/food) 서치 병렬 |
| **BATCH_RESTAURANT_ASYNC** (restaurant_async) | `BATCH_RESTAURANT_ASYNC` | `false` | 음식점 간 병렬 |
| **BATCH_SEARCH_CONCURRENCY** | `BATCH_SEARCH_CONCURRENCY` | `50` | 검색 동시 실행 상한 (비동기 경로에서 사용) |
| **BATCH_LLM_CONCURRENCY** | `BATCH_LLM_CONCURRENCY` | `8` | LLM 동시 실행 상한 (비동기 경로에서 사용) |
| **LLM_ASYNC** (llm_async) | `LLM_ASYNC` | `false` | `true`면 배치에서 LLM을 **진짜 비동기**로 호출 (httpx.AsyncClient / AsyncOpenAI). `false`면 `to_thread`로 동기 래핑 |

예시 (`.env` 또는 환경 변수):

```bash
# 완전 순차 (기본)
BATCH_SEARCH_ASYNC=false
BATCH_RESTAURANT_ASYNC=false

# aspect만 병렬 (음식점은 순차)
BATCH_SEARCH_ASYNC=true
BATCH_RESTAURANT_ASYNC=false

# 음식점만 병렬 (aspect는 순차)
BATCH_SEARCH_ASYNC=false
BATCH_RESTAURANT_ASYNC=true

# 둘 다 병렬
BATCH_SEARCH_ASYNC=true
BATCH_RESTAURANT_ASYNC=true
BATCH_SEARCH_CONCURRENCY=50
BATCH_LLM_CONCURRENCY=8

# LLM도 비동기(llm_async) 사용
LLM_ASYNC=true
```

---

## 3. 4단계 비동기 적용: 한계와 단계별 개선

배치 요약 파이프라인을 **음식점 간 / 벡터 검색 / LLM** 세 축으로 나누어, 각 단계에서의 한계와 다음 단계로 비동기를 넓혀 갈 때의 이점을 정리했습니다.

| 단계 | 음식점 간 | 벡터 검색 | LLM | 설정 예 |
|------|-----------|-----------|-----|--------|
| **1** | 동기 | 동기 | 동기 | `BATCH_SEARCH_ASYNC=false`, `BATCH_RESTAURANT_ASYNC=false`, `LLM_ASYNC=false` |
| **2** | 동기 | 비동기 | 동기 | `BATCH_SEARCH_ASYNC=true`, `BATCH_RESTAURANT_ASYNC=false`, `LLM_ASYNC=false` |
| **3** | 동기 | 비동기 | 비동기 | `BATCH_SEARCH_ASYNC=true`, `BATCH_RESTAURANT_ASYNC=false`, `LLM_ASYNC=true` |
| **4** | 비동기 | 비동기 | 비동기 | `BATCH_SEARCH_ASYNC=true`, `BATCH_RESTAURANT_ASYNC=true`, `LLM_ASYNC=true` |

---

### 단계 1: 음식점 간 동기, 벡터 검색 동기, LLM 동기

- **의미**: 레스토랑을 하나씩 순차 처리하고, 레스토랑당 서치 3회(service/price/food)도 순차, 요약 LLM도 동기 호출(블로킹).
- **한계**
  - **벡터 검색**: Qdrant 호출이 동기이므로 서치 3회가 끝날 때까지 스레드가 블로킹되고, I/O 대기 시간만큼 CPU가 놀게 됨. 레스토랑 N개 × 3회 서치가 모두 직렬이라 Qdrant 지연이 그대로 총 시간에 더해짐.
  - **LLM**: RunPod/OpenAI 등 HTTP 호출이 동기(requests 등)이면 한 레스토랑 요약이 끝날 때까지 다음 레스토랑을 시작할 수 없음. 네트워크·원격 추론 대기 시간이 전체 지연을 키움.
  - **음식점 간**: 한 레스토랑이 완료되어야 다음 레스토랑을 시작하므로, 여러 음식점을 동시에 처리할 수 없음. 배치 크기가 커질수록 총 소요 시간이 선형으로 증가.
- **다음 단계(2)로 갈 때 좋아지는 점**: 벡터 검색을 비동기(또는 to_thread + gather)로 바꾸면 **레스토랑 1개 안에서** 서치 3개를 동시에 날릴 수 있어, Qdrant I/O 대기 시간을 줄일 수 있음. 동일 레스토랑 수·동일 Qdrant 지연이라도 서치 구간이 “3배 직렬”에서 “병렬”로 바뀌어 레스토랑당 검색 구간 시간이 짧아짐.

---

### 단계 2: 음식점 간 동기, 벡터 검색 비동기, LLM 동기

- **의미**: 레스토랑은 여전히 순차이지만, 레스토랑당 서치 3개는 비동기(gather + to_thread)로 병렬. LLM은 여전히 동기(또는 to_thread로 스레드에서 실행).
- **한계**
  - **벡터 검색**: to_thread로 감싼 동기 Qdrant 호출이라 “이벤트 루프는 블로킹하지 않음”일 뿐, 실제 네트워크 대기는 스레드 풀에서 발생. 그래도 aspect 3개를 동시에 요청하므로 서치 구간은 단축됨.
  - **LLM**: LLM이 동기(또는 to_thread)이면 한 레스토랑의 요약이 끝날 때까지 해당 코루틴이 대기. 여러 레스토랑을 동시에 처리하지 않으므로, LLM 호출이 진짜 비동기(httpx/AsyncOpenAI)가 아니면 I/O 대기 동안 다른 작업을 거의 하지 못함. GPU/원격 서버 지연이 그대로 총 시간에 반영됨.
  - **음식점 간**: 여전히 한 번에 한 레스토랑만 처리하므로, N개 레스토랑이면 “(서치 병렬 + LLM 1회) × N” 순차. 처리량 상한이 레스토랑당 1개로 고정됨.
- **다음 단계(3)로 갈 때 좋아지는 점**: LLM을 비동기(httpx.AsyncClient, AsyncOpenAI)로 바꾸면 **같은 레스토랑 내**에서라도 이벤트 루프가 블로킹되지 않고, 나중에 restaurant_async를 켰을 때 여러 레스토랑의 LLM 요청을 동시에 날리기 좋은 토대가 됨. 뿐만 아니라 RunPod/OpenAI 같은 HTTP I/O 대기 시 스레드를 쓰지 않고 await로 대기하므로 스레드 수·컨텍스트 스위칭 부담이 줄고, 단일 레스토랑 처리에서도 “진짜 비동기”의 이점을 얻을 수 있음.

---

### 단계 3: 음식점 간 동기, 벡터 검색 비동기, LLM 비동기

- **의미**: 레스토랑은 여전히 순차(한 번에 한 레스토랑). 레스토랑당 서치 3개는 비동기로 병렬, LLM은 httpx/AsyncOpenAI 등으로 비동기 호출.
- **한계**
  - **음식점 간**: “한 레스토랑 처리 → 다음 레스토랑” 순서가 유지되므로, 레스토랑 10개면 10번의 “서치(병렬) + LLM(비동기)”가 차례로 실행됨. 여러 음식점의 서치·LLM을 동시에 진행하지 못해, Qdrant·LLM 서버의 동시 처리 능력을 충분히 쓰지 못함. 배치 크기가 크면 총 시간이 여전히 레스토랑 수에 비례해 길어짐.
  - **확장성**: 단일 요청(한 배치) 안에서 처리량을 높이려면 “동시에 처리하는 레스토랑 수”를 늘려야 하는데, 이 단계에서는 그게 불가능함.
- **다음 단계(4)로 갈 때 좋아지는 점**: 음식점 간을 비동기(gather)로 바꾸면 **여러 레스토랑을 동시에** 처리할 수 있음. 서치·LLM이 이미 비동기이므로, N개 레스토랑을 동시에 시작해도 이벤트 루프 하나로 처리 가능. 세마포어로 검색/LLM 동시성을 제한하면 Qdrant·LLM 서버 과부하를 막으면서도, 배치 전체 소요 시간을 “가장 느린 레스토랑 몇 개” 수준으로 줄일 수 있음. 처리량(레스토랑/초)이 크게 늘어남.

---

### 단계 4: 음식점 간 비동기, 벡터 검색 비동기, LLM 비동기

- **의미**: 레스토랑별 태스크를 `gather`로 동시에 실행하고, 레스토랑당 서치 3개도 비동기 병렬, LLM도 비동기(httpx/AsyncOpenAI). 세마포어로 검색·LLM 동시성만 캡.
- **한계**
  - **동시성 상한**: BATCH_SEARCH_CONCURRENCY, BATCH_LLM_CONCURRENCY를 너무 크게 잡으면 Qdrant/LLM 서버나 네트워크에서 지연·에러가 늘어날 수 있음. 환경에 맞게 튜닝 필요.
  - **메모리·타임아웃**: 동시에 많은 레스토랑을 처리하면 메모리와 장시간 요청이 늘어나므로, 매우 큰 배치는 청킹(예: 200~1000개 단위)을 권장.
- **장점**: 위 세 단계의 한계를 모두 완화. 서치 I/O·LLM I/O·음식점 간 처리를 모두 비동기로 활용하므로, 동일 하드웨어·동일 서버에서 배치당 소요 시간과 처리량이 가장 유리함.

---

## 4. search_async=false, restaurant_async=false (완전 순차)

- `Config.BATCH_SEARCH_ASYNC`와 `Config.BATCH_RESTAURANT_ASYNC`가 **둘 다 False**일 때 이 경로로 동작합니다.
- `request.restaurants`를 `for` 루프로 순회합니다.
- 각 레스토랑에 대해:
  1. service → price → food 순으로 `query_hybrid_search` 3회 **순차** 호출
  2. `summarize_aspects_new` 1회 호출 (카테고리별 요약 + overall_summary)
  3. 응답용 dict 생성 후 `results`에 추가
- API 요청/응답 형식은 단일 요약과 동일하게, `limit`/`min_score`는 요청 상위에서 전체 레스토랑 공통 적용.

---

## 5. search_async 또는 restaurant_async 중 하나라도 true (비동기 경로)

- **진입 조건**: `Config.BATCH_SEARCH_ASYNC or Config.BATCH_RESTAURANT_ASYNC`이면 `_batch_summarize_async` 사용.
- **restaurant_async**: `True`면 `asyncio.gather(*(one(rd) for rd in restaurants))`로 음식점 병렬, `False`면 `for rd in restaurants: await one(rd)`로 음식점 순차.
- **레스토랑 1개 처리** (`_process_one_restaurant_async`):
  1. **서치 3개**: **search_async=true**면 `asyncio.gather(do_one_search(service), do_one_search(price), do_one_search(food))`. **search_async=false**면 `for (seeds, name): await do_one_search(seeds, name)` 순차.
  2. **요약 1회**: `Config.LLM_ASYNC`에 따라 분기.
     - **LLM_ASYNC=true (llm_async)**: `await summarize_aspects_new_async(...)` — RunPod은 `httpx.AsyncClient`, OpenAI는 `AsyncOpenAI`로 **진짜 비동기** 호출.
     - **LLM_ASYNC=false (기본)**: `asyncio.to_thread(summarize_aspects_new, ...)` — 동기 함수를 스레드에서 실행.
  3. `_build_category_result`로 응답용 dict 생성 후 반환.
- Qdrant 호출은 동기이므로 `asyncio.to_thread`로 실행. LLM은 **llm_async** 켜면 httpx/AsyncOpenAI, 끄면 to_thread.
- **llm_async 지원**: RunPod(httpx.AsyncClient), OpenAI(AsyncOpenAI). vLLM 직접/로컬은 llm_async 시 NotImplementedError.
- 구현 위치: `src/api/routers/llm.py`, `src/llm_utils.py`, `src/summary_pipeline.py`.

---

## 6. 요청/응답 (공통)

모드와 관계없이 배치 요청·응답 형식은 동일합니다.

- **요청**: `restaurants`(각 항목 `restaurant_id`), 상위 `limit`, `min_score` (전체 레스토랑 공통).
- **응답**: `results`: `SummaryDisplayResponse` 리스트 (단일 `debug=false`와 동일 필드).

자세한 JSON 예시는 `PIPELINE_OPERATIONS.md` §8.2 Summary 배치 섹션을 참고하세요.

---

## 7. 관련 파일

- `src/config.py`: `BATCH_SEARCH_ASYNC`, `BATCH_RESTAURANT_ASYNC`, `BATCH_SEARCH_CONCURRENCY`, `BATCH_LLM_CONCURRENCY`, `LLM_ASYNC`
- `src/api/routers/llm.py`: 배치 엔드포인트 분기, `_batch_summarize_async`, `_process_one_restaurant_async`, `_build_category_result` (LLM_ASYNC 분기)
- `src/llm_utils.py`: `_call_runpod_async`, `_generate_with_local_queue_async`, `_generate_response_async` (httpx / AsyncOpenAI)
- `src/summary_pipeline.py`: `summarize_aspects_new_async`
- `.env.example`: 배치·LLM_ASYNC 관련 환경 변수 예시 주석

---

## 8. llm_async (LLM_ASYNC) 요약

- **off (기본)**: 배치에서 요약은 `asyncio.to_thread(summarize_aspects_new, ...)` — 동기 LLM을 스레드에서 실행.
- **on**: 배치에서 요약은 `await summarize_aspects_new_async(...)` — `_generate_response_async`가 RunPod이면 `httpx.AsyncClient`, OpenAI면 `AsyncOpenAI`로 비동기 호출. vLLM 직접/로컬은 미지원(NotImplementedError).

---

## 9. 참고: async_batch.md

설계 배경(서치/LLM 세마포어 분리, `to_thread` 사용, 음식점 간 병렬)은 `async_batch.md`에 정리되어 있습니다.  
현재 구현은 **search_async**(BATCH_SEARCH_ASYNC)로 aspect 서치 병렬/순차, **restaurant_async**(BATCH_RESTAURANT_ASYNC)로 음식점 간 병렬/순차를 각각 제어하고, 요약은 `LLM_ASYNC`에 따라 **llm_async**(httpx/AsyncOpenAI) 또는 **to_thread(동기)** 로 선택합니다.
