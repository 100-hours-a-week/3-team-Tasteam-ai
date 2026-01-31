[1 tool called]


문서 내용을 바탕으로, 현재 코드베이스의 Qdrant/LLM 호출 방식을 확인 중입니다.


[3 tools called]


`async_batch.md`에 대한 정리와, 현재 코드와 맞춰 본 생각입니다.

---

## 잘 맞는 부분

1. **작업 성격 구분**  
   서치 3회/음식점 = I/O, LLM 4회 = I/O(HTTP) 또는 블로킹(로컬 추론)이라는 구분이 맞고, 그에 따라 asyncio vs `to_thread` 제안도 타당합니다.

2. **세마포어를 서치/LLM 따로 두기**  
   Qdrant 동시성과 LLM 동시성을 분리하는 설계가 중요합니다. 지금 코드도 `qdrant_client`(동기), LLM은 `requests` 또는 동기 호출을 쓰는 경로가 있어서, 병렬화 시 **검색 쪽 concurrency**와 **LLM 쪽 concurrency**를 각각 캡으로 두는 게 맞습니다.

3. **“음식점 단위는 파이프라인, 음식점들끼리는 병렬”**  
   한 음식점 안에서는 (서치 3개 → 요약 3개 → 총요약 1개) 순서가 있고, 음식점 간에는 `gather`로 병렬이라는 구조가 현재 배치 로직과도 잘 맞습니다.

4. **동기 클라이언트일 때 `to_thread`**  
   현재 `vector_search.query_hybrid_search`는 **동기**이고, `summarize_aspects_new` 안의 LLM 호출도 **동기**일 가능성이 큽니다. 문서대로라면 이 둘을 `asyncio.to_thread(...)`로 감싼 뒤, 그 위에 `gather` + 세마포어를 두는 방식이 현실적인 선택입니다.

---

## 현재 코드와의 차이

- **문서 스켈레톤**: `hybrid_search(restaurant, aspect)` 3개를 `gather`로 한 번에 호출하는 **aspect 단위 병렬**을 가정합니다.
- **현재 구현**: 한 음식점에 대해 `service` → `price` → `food` 순으로 **for 루프로 순차** 호출하고, 그 다음 `summarize_aspects_new` 한 번에 (서비스/가격/음식 리뷰를 한꺼번에) 넘겨 카테고리별 요약 + overall 한 번에 뽑습니다.

그래서 “음식점 내부”를 문서처럼 바꾸려면:

- 서치만 aspect별로 나누어 `gather(hybrid_search(service), hybrid_search(price), hybrid_search(food))`로 바꾸고,
- 그 결과를 모아서 **지금처럼** `summarize_aspects_new`에 한 번에 넘기는 식으로 하면, 문서의 “서치 3개 병렬” 이점을 살리면서도 기존 요약 파이프라인을 최대한 유지할 수 있습니다.  
  (요약을 aspect별로 완전히 쪼개서 병렬 호출하는 건, 지금 `summarize_aspects_new`가 한 번의 LLM 호출로 3 aspect + overall을 내는 구조라면 리팩터가 더 필요합니다.)

---

## 추가로 생각해볼 점

- **배치 청킹(200~1000개 단위)**  
  음식점 수가 크면 `gather`에 한 번에 다 넣지 말고, 문서처럼 청크 단위로 나누는 게 메모리·타임아웃 측면에서 안전합니다.

- **return_exceptions=True + ok/failed 분리**  
  문서의 `batch_process`처럼 `return_exceptions=True`로 받고, 성공/실패를 나눠서 응답에 넣으면, 한 두 개 실패해도 나머지 결과는 그대로 돌려줄 수 있어서 운영에 유리합니다.

- **실제 적용 시 확인할 것**  
  문서 마지막에 적어둔 것처럼,  
  - Qdrant 호출이 **sync인지 async인지**(현재는 sync),  
  - LLM이 **HTTP 비동기 가능한지 vs 로컬 동기인지**  
  이 두 가지만 정리하면, 문서 스켈레톤을 “서치만 `to_thread`로 감싼 gather + 기존 `summarize_aspects_new`”에 맞게 구체 코드로 잘 녹일 수 있습니다.

---

**한 줄 요약**: 설계 방향(음식점 간 병렬, 서치/LLM 세마포어 분리, 동기일 때 `to_thread`)이 현재 구조와 잘 맞고, 실제 적용 시에는 “서치 3개만 aspect 단위 병렬”로 끌어오고 요약은 기존 파이프라인을 유지하는 식으로 단계적으로 가져가면 좋겠다는 생각입니다.