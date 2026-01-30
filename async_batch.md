이건 작업 성격이 딱 나뉘어요:

* **하이브리드 서치 3번/음식점**: 보통 **네트워크 I/O(Qdrant HTTP/gRPC)** → `asyncio`가 제일 잘 먹힘
* **LLM 요약(4개: aspect 3 + 총요약 1)**:

  * vLLM/RunPod 같은 **HTTP 호출이면 I/O** → `asyncio`
  * 로컬에서 **동기 함수로 추론**(예: transformers generate 직접 호출) → CPU/GPU 블로킹 → `to_thread`(임시) 또는 **작업 큐/워커**로 분리

그래서 추천 구조는 **“음식점 단위로는 순차(파이프라인), 음식점들 간에는 병렬”** + **세마포어로 동시성 제한**이에요.

---

## 1) 내가 추천하는 선택: `asyncio.gather()` + `Semaphore`

네 배치 처리에는 이게 정답에 가깝습니다.

* 음식점 N개를 동시에 처리하되
* “서치 동시성”, “LLM 동시성”을 **각각 따로 제한** (중요)

### 왜?

* Qdrant는 동시 요청 과하면 tail latency 튀고
* LLM은 동시 요청 과하면 큐가 밀리고 토큰 생성이 느려짐 (GPU 병목)

---

## 2) 구조(파이프라인) 추천

### 음식점 1개 처리 흐름

1. aspect 3개 서치 → **동시에** 날림 (`gather`)
2. aspect 3개 요약 → **동시에**(또는 제한된 동시성으로) 실행
3. 총요약 → 마지막에 1번 실행

즉, **음식점 내부는 “부분 병렬 + 단계적”**이 좋고,
**음식점들끼리는 병렬**이 핵심이에요.

---

## 3) 가장 실용적인 코드 스켈레톤 (FastAPI/배치 둘 다 잘 맞음)

```python
import asyncio
from dataclasses import dataclass

ASPECTS = ["service", "price", "food"]

@dataclass
class RestaurantTask:
    restaurant_id: int
    query_seed: str  # 예: 음식점명/설명/대표리뷰 등

# --- 동시성 제한: 각각 따로 잡기 ---
SEARCH_CONCURRENCY = 50   # Qdrant/검색 IO 동시성 (환경에 맞춰 조절)
LLM_CONCURRENCY = 8       # LLM 동시성 (GPU 1장 기준 보통 2~16 사이)

search_sem = asyncio.Semaphore(SEARCH_CONCURRENCY)
llm_sem = asyncio.Semaphore(LLM_CONCURRENCY)

async def hybrid_search(restaurant: RestaurantTask, aspect: str):
    async with search_sem:
        # TODO: Qdrant hybrid search async client 호출
        # return topk docs
        ...

async def llm_summarize(aspect: str, docs: list[str]):
    async with llm_sem:
        # TODO: vLLM/RunPod/LLM 서버 async 호출
        # return summary
        ...

async def llm_total_summary(aspect_summaries: dict):
    async with llm_sem:
        # TODO: 총요약 호출
        ...

async def process_one_restaurant(r: RestaurantTask):
    # 1) aspect별 search 3개 병렬
    search_results = await asyncio.gather(
        *(hybrid_search(r, a) for a in ASPECTS),
        return_exceptions=False
    )
    aspect_docs = dict(zip(ASPECTS, search_results))

    # 2) aspect별 요약 병렬
    aspect_summaries_list = await asyncio.gather(
        *(llm_summarize(a, aspect_docs[a]) for a in ASPECTS),
        return_exceptions=False
    )
    aspect_summaries = dict(zip(ASPECTS, aspect_summaries_list))

    # 3) 총요약 (단일)
    total = await llm_total_summary(aspect_summaries)

    return {
        "restaurant_id": r.restaurant_id,
        "aspect_summaries": aspect_summaries,
        "total_summary": total,
    }

async def batch_process(restaurants: list[RestaurantTask]):
    # 음식점들 간 병렬 (단, 내부 sem이 전체 동시성을 제어함)
    results = await asyncio.gather(
        *(process_one_restaurant(r) for r in restaurants),
        return_exceptions=True
    )

    ok, failed = [], []
    for r, res in zip(restaurants, results):
        if isinstance(res, Exception):
            failed.append((r.restaurant_id, str(res)))
        else:
            ok.append(res)

    return ok, failed
```

이 형태가 **`asyncio.gather()`를 써야 하는 전형적인 배치**예요.

---

## 4) ThreadPoolExecutor는 언제 끼냐?

딱 한 경우: **LLM 호출이나 hybrid search 호출 코드가 “동기(블로킹)”**일 때.

예: `requests.post(...)` / 동기 Qdrant client / 동기 vLLM 클라이언트

그럼 아래 중 하나로 감싸면 됩니다:

* 가장 쉬운 해법: `asyncio.to_thread(sync_func, ...)`  ✅ (ThreadPoolExecutor 내부 사용)
* 또는 직접 ThreadPoolExecutor 만들기

핵심은: **블로킹 IO를 event loop에서 돌리면 전체가 멈춘다**는 점.

---

## 5) 운영에서 중요한 디테일 (이거 안 하면 병렬이 오히려 느려짐)

### ✅ 세마포어를 “서치/LLM 각각 따로”

* 검색은 50 동시 가능해도
* LLM은 8만 넘어도 GPU가 밀려서 전체가 느려질 수 있어요.

### ✅ 배치 사이즈를 “음식점 개수”로 쪼개기

너무 많은 음식점을 한 번에 gather에 넣으면 메모리/큐가 커져서 역효과.

* 예: 200~1000개 단위로 chunking해서 돌리기

### ✅ 재시도/타임아웃은 단계별로

* 검색 실패는 재시도 1~3회
* LLM 실패는 idempotent하게 재시도하되, 입력 크기 과다/429면 backoff

---

## 6) 네 케이스에 대한 최종 선택

* **기본 선택:** `asyncio.gather()` + (검색 sem, LLM sem)
* **단, 호출이 동기면:** 그 부분만 `asyncio.to_thread()`로 감싸서 Thread로 보완
* **CPU-heavy 전처리가 진짜 병목이면:** 그 전처리만 Process로 따로 (하지만 지금 플로우는 보통 IO/LLM 병목일 확률 큼)

---

원하면, 네가 쓰는 게

1. Qdrant 호출이 **async**인지 **sync**인지
2. LLM 호출이 **HTTP**인지(비동기 가능) vs **로컬 generate 동기**인지
   이 두 가지만 기준으로, 위 스켈레톤을 **네 코드 스타일(FastAPI/RunPod/vLLM/Qdrant)**에 맞춰 바로 “실제 사용 가능한 형태”로 바꿔줄게.
