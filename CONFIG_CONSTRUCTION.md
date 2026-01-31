Config로 빼면 좋은 것 (대부분 여기에 해당)

1) 환경별로 바뀌는 것

DB/Redis/Qdrant URL, credentials(단, 비밀은 .env/Secret)

모델 경로/이름, device(cpu/gpu), worker 수

로깅 레벨, observability endpoint

rate limit, timeout, retry 횟수

2) 성능/비용 트레이드오프 튜닝 값

batch_size, max_concurrency, semaphore 크기

to_thread on/off, worker 모드 선택

cache TTL, top_k, rerank_k

GPU 사용 여부, quantization on/off

3) 실험 플래그/기능 토글

feature flags (A/B, 단계적 롤아웃)

“sync vs async vs thread isolation” 같은 실행 모드

코드에 두는 게 좋은 것
1) 의미/정합성/안전이 강한 규칙

“이 값이면 반드시 이렇게 동작해야 한다” 같은 불변 로직

보안상 위험한 옵션(예: auth off)을 config로 쉽게 켜지 못하게

API 스키마/필드 구조 같은 “계약” 요소

2) 너무 많아서 관리 비용이 폭증하는 미세 파라미터

거의 안 바꾸는데 config에 넣으면 오히려 “설정 지옥” 됨

“모든 숫자 상수”를 config로 빼는 건 대개 유지보수 악화

3) 개발자만 건드려야 하는 내부 상수

디버그용 해킹 옵션들

임시 실험값(나중에 제거할 것)

---

그리고 config는 “다 모으되”, 한 덩어리로 뭉치지 말고 도메인별로 나눠:

server, inference, retrieval, cache, spark, observability …

---

## 적용 현황 (src/config.py)

위 원칙에 따라 `src/config.py`가 도메인별로 분리되어 있음:

| 도메인 | 클래스 | 포함 항목 |
|--------|--------|-----------|
| Server | `_ServerConfig` | (FastAPI 등은 app 레벨) |
| Inference | `_InferenceConfig` | 모델, device, GPU, LLM 백엔드, batch/concurrency, retry/timeout, 실행 모드(SUMMARY_LLM_ASYNC, SUMMARY_SEARCH_ASYNC, SUMMARY_RESTAURANT_ASYNC, COMPARISON_ASYNC, SENTIMENT_*) |
| Retrieval | `_RetrievalConfig` | Qdrant URL, collection, embedding, top_k, aspect seed |
| Cache | `_CacheConfig` | SKIP 최소 간격 |
| Spark | `_SparkConfig` | 전체 평균 데이터 경로, 비율 |
| Observability | `_ObservabilityConfig` | 메트릭, 로깅, CPU/GPU 모니터링, Watchdog |

- 기존 `Config.USE_GPU`, `Config.QDRANT_URL` 등 **호환 유지** (다중 상속)
- 도메인별 접근: `Config.Inference.USE_GPU`, `Config.Retrieval.QDRANT_URL` 등