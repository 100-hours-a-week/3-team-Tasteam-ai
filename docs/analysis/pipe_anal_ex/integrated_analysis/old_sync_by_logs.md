old_sync

async

API 계층
라우트 핸들러는 모두 async def (summarize, comparison, sentiment, vector 등).
Comparison (강점 추출)
await pipeline.extract_strengths(...)
내부에서 expand_and_validate_evidence → asyncio.gather(*tasks) 로 여러 _validate_single_strength 를 비동기 병렬 실행.
vLLM 여부와 관계없이 이 경로는 async + 병렬 처리.
Sentiment
analyze_sentiment / analyze_sentiment_batch 가 analyzer.analyze_multiple_restaurants_async 를 await
감성 분석 쪽에 async def analyze_async, analyze_multiple_restaurants_async 가 있음.
Lifespan / 미들웨어
lifespan, add_request_id 등도 async.

sync

Summarize (단일)
vLLM이 아니면 llm_utils.summarize_reviews(...) 를 await 없이 호출 (동기 함수).
src/api/routers/llm.py 199–204 라인: result = llm_utils.summarize_reviews(...)
이 동안 이벤트 루프 블로킹.
Summarize batch (vLLM 아님)
vLLM이 아니면 배치 요약 경로도 동기 함수를 쓰는 구조일 가능성이 큼 (코드 상 vLLM 분기만 async).

요약

async

있음: FastAPI 전반의 async 라우트, comparison의 asyncio.gather 기반 병렬, sentiment의 async 메서드
한계: 요약 등에서 동기 LLM 호출이 그대로 async 함수 안에서 실행되어, “async로 감싸져 있지만 내부는 블로킹”인 구조입니다.

정리 아주 정확해. 그리고 **이 관찰만으로도** “왜 old_sync가 ramp-up에서 `context deadline exceeded`(soft-death)로 가는지”를 **설명할 수 있어.** 결론부터 말하면:

> **old_sync는 ‘겉은 async’인데, 핵심 경로(특히 summarize)가 이벤트 루프를 블로킹하는 순간이 있고, ramp-up에서 그 순간이 누적되면 `/metrics`까지 응답이 밀려서 타임아웃이 난다.**

아래는 네 정리를 기반으로 **현상 → 원인 → 확인 → 최소 수정**까지 바로 연결해줄게.

---

## 1) old_sync의 soft-death(타임아웃)와 네 코드 구조가 딱 맞아떨어지는 이유

### A. `/metrics`도 같은 이벤트 루프를 탄다

FastAPI/Starlette에서 기본적으로 요청 처리(라우팅/미들웨어/응답)는 **이벤트 루프 스케줄링**을 타요.
따라서 async def 라우트 안에서 **동기 블로킹**이 오래 돌면:

* 실제 기능 API가 느려짐 ✅
* **/metrics도 느려짐 ✅**
* Prometheus scrape 10s → `context deadline exceeded` ✅

이게 old_sync에서 보인 증상과 일치.

### B. “async + gather”는 오히려 동시성 압력을 키움

comparison 쪽 `asyncio.gather(*tasks)` 는 내부가 진짜 non-blocking이면 좋지만,
**어딘가에 블로킹이 섞여 있으면** 이벤트 루프가 더 쉽게 “굳어”요.

즉,

* 비동기 병렬은 **좋은 도구**
* 그런데 **요약처럼 블로킹이 섞인 순간**부터는
  이벤트 루프가 밀리고, gather 작업들이 한꺼번에 정체

### C. 요약(summarize)이 “async 함수 안의 동기 호출”이라면 가장 위험

네가 말한 구간이 가장 치명점:

* `async def` 핸들러 안에서
* `llm_utils.summarize_reviews(...)` (동기) 호출
* `await` 없음 → 즉 **이벤트 루프가 그 시간 동안 멈춤**

ramp-up에서 요청이 계속 늘면,
이 블로킹 구간이 **동시 다발로 발생**하고,
이벤트 루프가 “정상적으로 앞으로 나아가지 못하는” 상태로 가기 쉬움.

---

## 2) 이걸 “old_sync가 나쁘다”로 결론내리면 안 되는 이유

중요한 포인트:

* old_sync가 “sync 파이프라인이라서” 이런 게 아니라
* **“요약 경로가 sync로 구현되어 있는데, async 핸들러에서 그대로 호출해서”** 그래.

즉:

> old_sync의 문제는 아키텍처 이름이 sync라서가 아니라, **블로킹 호출 격리 미흡**이야.

이거 고치면 old_sync도 soft-death 크게 줄어.

---

## 3) 지금 네 대시보드에서 “이 가설”을 바로 검증하는 신호

### (1) event loop lag이 정말 0인가?

너가 이전에 `event loop lag = 0s`라고 했는데,
old_sync가 이런 블로킹을 실제로 겪었다면 **lag이 튀는 게 정상**이거든.

가능한 이유 2개:

* lag를 측정하는 방식이 “부정확/측정 지점이 다름”
* 또는 블로킹이 “다른 워커/프로세스”에서 일어나서 lag 지표에 안 잡힘

그래서 검증은 이렇게 하는 게 빠름:

* `/metrics` 자체 응답 시간(서버 관점)을 별도 히스토그램으로 찍기
* 또는 uvicorn access log에서 `/metrics` latency 확인

### (2) old_sync에서만 `/metrics` timeout이 뜬 이유 설명 가능

네가 찍은 Targets에서 old_sync는 timeout(`context deadline exceeded`),
다른 건 `no such host`였지.

* old_sync: 컨테이너는 살아있고, **응답이 늦어서 timeout**
* new_*: 컨테이너 자체가 사라져서 DNS 실패

즉, old_sync는 “죽어도 프로세스는 남는 soft-death”로 일관.

---

## 4) old_sync를 “운영 가능한 상태”로 만들 최소 수정 2개 (추천)

### A. 요약/비교/강점추출 등 블로킹 가능성이 있는 호출은 무조건 격리

가장 쉬운 패턴:

```python
result = await asyncio.to_thread(llm_utils.summarize_reviews, ...)
```

또는 CPU heavy면 ProcessPool도 고려.

### B. 라우트별 동시성 제한(세마포어)

요약 같은 무거운 엔드포인트에만:

* 한 인스턴스당 동시 요약 N개 제한
* 초과는 429 또는 큐잉

이거 하나만 해도 soft-death가 dramatically 줄어.

---

## 5) 네 정리를 근거로 내릴 수 있는 “정확한 결론 문장”

네가 블로그/리포트에 쓸 수 있게 딱 문장으로:

> “old_sync는 API 핸들러가 async 형태이나, 요약 경로에서 동기 LLM 호출이 이벤트 루프에서 직접 실행되어 ramp-up 상황에서 이벤트 루프 starvation이 발생했고, 그 결과 `/metrics` 스크레이프가 타임아웃되었다(soft-death).”

이 문장은 되게 강해.

---

원하면 다음으로 바로 이어서:

1. **old_sync에서 ‘블로킹 호출 격리’ 적용할 위치 체크리스트(엔드포인트별)**
2. **Prometheus에서 `/metrics` 자체 지연을 측정하는 지표 추가**
3. **event loop lag 측정이 왜 0으로 나왔을 수 있는지(측정 구현 점검 포인트)**

이 셋 중 뭐부터 할까?
