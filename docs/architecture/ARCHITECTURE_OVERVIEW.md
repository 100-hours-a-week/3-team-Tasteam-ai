# 프로젝트 아키텍처 전체 정리

이 문서는 **배포 구조**, **애플리케이션 계층**, **외부 의존성**, **관측 가능성**, **파이프라인 상세**를 한곳에 정리한 아키텍처 개요입니다.

---

## Part I. 전체 아키텍처

### 1. 아키텍처 개요

| 항목 | 내용 |
|------|------|
| **프레임워크** | FastAPI (Python) |
| **역할** | 레스토랑 리뷰 감성 분석·벡터 검색·LLM 요약·비교 API |
| **진입점** | `app.py` → uvicorn `src.api.main:app` (기본 포트 8001) |
| **문서** | `/docs` (Swagger), `/redoc` |

**전체 구성 (개념)**

```
[클라이언트]
    │
    ▼
[new_async:8001]  FastAPI 앱  ─┬─ [Qdrant]  벡터/리뷰 저장소
    │                          ├─ [Redis]   캐시·분산 락·RQ 큐
    │                          ├─ [OpenAI | RunPod Pod vLLM]  LLM (인프로세스 vLLM 제거. Serverless 미사용)
    │                          └─ [spark-service:8002]  Spark MSA (전체 평균·recall seeds). SPARK_SERVICE_URL로 HTTP 호출
    │
    ├─ POST /api/v1/batch/enqueue  (BATCH_USE_QUEUE=true 시)
    │      └─► [batch-worker]  RQ 워커가 배치 작업 소비 (재시도 → DLQ). 워커도 Spark는 spark-service 호출
    │
    ├─ [trigger_offline_batch]  오프라인 트리거 (cron/EventBridge에서 enqueue 호출)
    │
    ├─ Prometheus 스크래프 (/metrics)
    ▼
[Prometheus] ──► [Grafana] 대시보드
[Alertmanager] 알림
[node-exporter] 호스트 메트릭
[jobmgr:1041]   커스텀 메트릭
```

### 2. 배포 구조 (Docker Compose)

| 서비스 | 이미지/빌드 | 포트 | 역할 |
|--------|-------------|------|------|
| **redis** | redis:7-alpine | 6379 | 캐시(CacheManager)·분산 락(RedisLock)·**RQ 큐**. AOF 영속(redis_data 볼륨). |
| **new_async** | tasteam-new-async:latest | 8001 | FastAPI API. Redis·**spark-service** 의존(healthy 후 기동). `SPARK_SERVICE_URL=http://spark-service:8002` 로 Spark MSA 호출. |
| **spark-service** | dockerfile | 8002 | **Spark 마이크로서비스**. 전체 평균·recall seeds 계산(`scripts/spark_service.py`). new_async·batch-worker가 HTTP로만 호출. |
| **batch-worker** | dockerfile | - | RQ 워커. `BATCH_USE_QUEUE=true` 시 배치 작업 소비(재시도·DLQ). Redis·spark-service 의존. `SPARK_SERVICE_URL` 로 Spark MSA 호출. |
| **jobmgr** | Dockerfile.metrics | 1041 | 메트릭 수집/노출 (선택). |
| **node-exporter** | prom/node-exporter:v1.7.0 | 9100 | 호스트 메트릭. |
| **prometheus** | prom/prometheus:v2.42.0 | 9090 | 스크래프·알림 규칙. new_async, jobmgr, node-exporter, alertmanager 의존. |
| **alertmanager** | prom/alertmanager:v0.26.0 | 9093 | 알림 라우팅. |
| **grafana** | grafana/grafana:9.3.6 | 3000 | 대시보드. Prometheus 데이터소스. |

- **볼륨**: `redis_data` (Redis 영속).
- **환경**: new_async는 `.env` + `REDIS_HOST=redis`, `REDIS_PORT=6379`, `REDIS_DB=0`, `SPARK_SERVICE_URL=http://spark-service:8002` 등. batch-worker도 동일하게 `SPARK_SERVICE_URL` 설정.
- **오프라인 배치**: `trigger_offline_batch.py`로 enqueue 호출 → RQ 워커가 처리. `docs/batch/offline_batch_processing.md`, `docs/batch/offline_batch_strategy.md` 참고.
- **배포와 MSA**: Compose에 API(new_async), Spark(spark-service), 배치(batch-worker)가 각각 별도 서비스로 정의되어 있으며, API·워커는 Spark를 인프로세스가 아닌 HTTP로만 사용하므로 **배포까지 MSA**가 적용된 구성이다.

### 3. 애플리케이션 계층

| 계층 | 내용 |
|------|------|
| **진입점** | `app.py`: uvicorn.run("src.api.main:app", host="0.0.0.0", port=PORT). |
| **앱 생성** | `src/api/main.py`: FastAPI(lifespan=lifespan), 라우터 등록(sentiment, vector, llm, test, batch). |
| **Lifespan** | CPU 모니터 시작(설정 시) → warm-up(임베딩·감성 파이프라인) → **Event loop lag 주기 측정 태스크** 시작 → yield → lag 태스크 취소 → CPU 모니터 정지. |
| **미들웨어** | X-Request-Id 부여; Queue depth (app_queue_depth_inc/dec); CORS. |
| **예외 처리** | RequestValidationError→422, StarletteHTTPException→해당 status, 그 외→500. code/message/details/request_id 포함. |
| **종료 처리** | atexit·SIGTERM/SIGINT 핸들러에서 "shutdown" 로그; excepthook으로 미처리 예외 저장 후 atexit에서 traceback 출력. |

- **엔드포인트**: `GET /`, `GET /health`, `GET /ready`(warm-up 완료 시 200), `GET /metrics`(Prometheus).

### 4. 외부 의존성

| 의존성 | 용도 | 설정(환경 변수) |
|--------|------|------------------|
| **Qdrant** | 벡터 저장·하이브리드 검색, 리뷰 조회. 로컬 경로 또는 원격 URL. | `QDRANT_URL`, `COLLECTION_NAME`, `QDRANT_VECTORS_ON_DISK` |
| **Redis** | 캐시(LLM·감성·임베딩 결과), 분산 락, **RQ 작업 큐**(`BATCH_USE_QUEUE=true` 시). 미연결 시 캐시·락·큐 비활성. | `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` |
| **OpenAI** | 요약·감성 재판정·비교 해석용 LLM (gpt-4o-mini 등). 429/5xx 시 벤더 API 페일오버(Gemini) 가능(`GEMINI_API_KEY`). | `OPENAI_API_KEY`, `OPENAI_MODEL`, `GEMINI_API_KEY` |
| **LLM 제공자** | `LLM_PROVIDER`: `openai`(API 전용), `runpod`(Pod vLLM), `local`(로컬 모델). 미설정 시 기본 `openai`. RunPod 사용 시 `VLLM_POD_BASE_URL`로 직접 호출. | `LLM_PROVIDER`, `VLLM_POD_BASE_URL` |
| **RunPod Pod vLLM** | LLM 1차 백엔드(Provider=runpod 시). `VLLM_POD_BASE_URL`로 직접 호출 (OpenAI 호환 /v1). 기본 예: `http://213.173.108.70:17517/v1`. **Pod만 사용** (Serverless 미사용). | `VLLM_POD_BASE_URL` |

**RunPod Serverless 미사용 이유** (`docs/runpod/why_dont_use_runpod_serverless.md`): Serverless는 요청 없을 때 워커가 종료되어 Ephemeral이며, Prometheus pull 기반 스크래핑 대상이 불안정함. Pod는 항상 떠 있어 고정 IP·지속적 메트릭 수집에 적합.

**RunPod Pod 운영 전략** (`docs/runpod/why_dont_use_runpod_serverless.md`):
- **켜기/끄기**: AWS EventBridge로 작업이 필요한 시간에 Pod 기동, 필요 없을 때 종료. 제어는 RunPod **CLI** 또는 **REST API(GraphQL)** 사용.
- **Prometheus**: Pod **공인 주소**의 `/metrics`를 스크래핑.
- **포트 변경 대응**: **프록시**를 두어 Pod 포트가 바뀌는 문제를 흡수 (앱은 프록시 고정 주소로 `VLLM_POD_BASE_URL` 설정).

#### 4.1 프로세스 분리와 장점 — 모놀리식(Monolithic) vs 마이크로서비스(MSA)

본 프로젝트는 **모놀리식(Monolithic)** — API 한 프로세스에 LLM·Spark·배치 실행을 모두 넣는 방식 — 대신, **마이크로서비스(MSA)** 적 분리를 채택합니다. LLM은 RunPod Pod, Spark는 별도 서비스, 배치는 RQ 워커로 분리해 리소스·장애·스케일을 격리합니다.

| 구분 | 모놀리식 (Monolithic) | 마이크로서비스 (MSA) — 본 프로젝트 |
|------|------------------------|-----------------------------------|
| **구조** | API 프로세스 안에 LLM 로드·Spark(JVM)·배치 실행이 공존 | API는 HTTP 클라이언트만. LLM→Pod, Spark→Spark 서비스, 배치→RQ 워커가 각각 별도 프로세스/컨테이너 |
| **리소스** | 한 프로세스가 CPU·메모리·GPU·JVM 모두 사용 → 메모리·기동 부담 큼 | 서비스별로 CPU/메모리/GPU/JVM 분리 → 경량 API·독립 스케일 |
| **장애** | LLM OOM·Spark/JVM 오류가 API 프로세스까지 다운시킬 수 있음 | 한 서비스 장애가 다른 서비스로 전파되지 않음 (장애 격리) |
| **배포** | LLM·Spark 버전 올리려면 API 이미지 전체 재배포 | 서비스별 독립 배포·스케일 가능 |

아래는 분리된 각 구성요소별 요약입니다.

**LLM — RunPod Pod 컨테이너에서 별도 프로세스(마이크로서비스)로 분리**

| 구분 | 내용 |
|------|------|
| **구성** | 메인 앱(new_async)은 LLM을 인프로세스로 로드하지 않고, **RunPod Pod**에서 동작하는 vLLM을 `VLLM_POD_BASE_URL`로 HTTP 호출. |
| **장점** | (1) **리소스 격리**: API 서버는 CPU/메모리만 사용하고, GPU·대형 모델은 Pod에서만 사용. (2) **독립 스케일**: API와 LLM을 각각 스케일·버전 업데이트 가능. (3) **비용 제어**: Pod를 작업 시간대에만 기동·종료(EventBridge 등)해 유휴 비용 절감. (4) **안정성**: LLM 크래시/OOM이 API 프로세스에 영향을 주지 않음. (5) **Prometheus**: Pod 고정 주소로 메트릭 스크래핑 가능(Serverless는 Ephemeral이라 불리함). |
| **관련 문서** | `docs/runpod/why_dont_use_runpod_serverless.md`, `etc_md/RUNPOD_POD_VLLM_GUIDE.md`, `docs/runpod/lambda_runpod_pod.md` |

**Spark — 별도 마이크로서비스(MSA)로 분리**

| 구분 | 내용 |
|------|------|
| **구성** | Comparison의 전체 평균·recall seeds 계산에 Spark 사용. **Spark 마이크로서비스**(`scripts/spark_service.py`)를 별도 프로세스(컨테이너)로 두고, `SPARK_SERVICE_URL` 설정 시 메인 앱/워커는 해당 URL로 HTTP 요청만 보냄. **Docker Compose**에 `spark-service`가 정의되어 있으며, new_async·batch-worker는 `SPARK_SERVICE_URL=http://spark-service:8002` 로 의존·호출. (모놀리식이면 API 프로세스에 JVM/pyspark 로드 필요.) |
| **장점** | (1) **JVM 미로드**: 메인 API·RQ 워커 프로세스에서 pyspark/JVM을 로드하지 않아 기동·메모리 부담 감소. (2) **Docker/경량 환경**: JVM 없는 이미지에서 API·워커만 배포 가능. (3) **장애 격리**: Spark/JVM 오류가 메인 프로세스를 죽이지 않음. (4) **리소스 분리**: 대용량 파일 처리(64만 건 등)는 Spark 전용 서버에서만 수행. (5) **배포 MSA**: Compose 기동만으로 API·Spark·워커가 서비스 단위로 분리된 배포 구성. |
| **관련 문서** | `docs/spark/SPARK_SERVICE.md` |

**RQ — 배치 실행 주체 분리**

| 구분 | 내용 |
|------|------|
| **구성** | `BATCH_USE_QUEUE=true` 시 API는 배치 작업을 **Redis RQ 큐**에 넣고, **batch-worker**(또는 RunPod Pod 내 `rq worker`)가 작업 소비. |
| **장점** | (1) **API 과부하 방지**: 무거운 배치가 API 프로세스를 블로킹하지 않음. (2) **재시도·DLQ**: RQ 재시도·FailedJobRegistry로 실패 작업 관리. (3) **오프라인 배치**: EventBridge → Lambda → Pod 기동 → enqueue → 워커 처리 → terminate 흐름으로 스케줄 배치 가능. |
| **관련 문서** | §8.5, `docs/batch/offline_batch_strategy.md`, `docs/batch/offline_batch_processing.md`, `docs/runpod/lambda_runpod_pod.md` |

### 5. 관측 가능성

| 구성요소 | 역할 |
|----------|------|
| **Prometheus** | new_async:8001 `/metrics`, jobmgr:1041, node-exporter:9100 스크래프. job명 `fastapi`(라벨 pipeline=new_async), `jobmgr`, `node`. |
| **Grafana** | Prometheus 데이터소스, 대시보드(prometheus_overview 등). |
| **Alertmanager** | Prometheus 알림 수신·라우팅. |

**앱이 노출하는 주요 Prometheus 메트릭** (`src/metrics_collector.py`, `/metrics`)

| 메트릭명 | 타입 | 설명 |
|----------|------|------|
| `app_queue_depth` | Gauge | 동시 처리 중인 요청 수 (in-flight). |
| `app_worker_busy` | Gauge | 워커 유휴(0)/바쁨(1). |
| `event_loop_lag_seconds` | Gauge | 이벤트 루프 지연(초). call_soon 콜백 실행까지 걸린 시간. |
| `analysis_requests_total` | Counter | 분석 요청 수 (analysis_type, status). |
| `analysis_processing_time_seconds` | Histogram | API 처리 시간. |
| `llm_ttft_seconds` | Histogram | Time to first token (초). |

- 그 외 `prometheus_fastapi_instrumentator`로 요청 수·지연 등 자동 수집.

### 6. 파이프라인 관계 요약

```
[클라이언트]
    │
    ├─ POST /api/v1/vector/upload           → 리뷰 벡터 적재 (Dense+Sparse). 유사 검색 API는 미제공(Summary/Sentiment/Comparison 내부에서 VectorSearch 사용).
    │
    ├─ POST /api/v1/sentiment/analyze       → 감성 분류 (HF 1차 + LLM 재판정) → 비율
    ├─ POST /api/v1/sentiment/analyze/batch
    │
    ├─ POST /api/v1/llm/summarize           → 카테고리별 검색 + LLM 요약 (해요체)
    ├─ POST /api/v1/llm/summarize/batch
    │
    ├─ POST /api/v1/llm/comparison          → Kiwi+Spark 비율 → lift → LLM 해석
    ├─ POST /api/v1/llm/comparison/batch
    │
    ├─ POST /api/v1/batch/enqueue           → 배치 작업 큐에 넣기 (BATCH_USE_QUEUE=true 시)
    └─ GET  /api/v1/batch/status/{job_id}   → 작업 상태 조회
```

- **Vector** 업로드 결과는 Summary 검색·Comparison 리뷰 조회에서 사용.
- **Summary**는 요약만, **Sentiment**는 긍정/부정/중립 비율.
- **Comparison**은 Vector 리뷰 + Kiwi(±Spark) 비율 → 전체 평균 대비 lift → LLM 해석.

**DeepFM ML 파이프라인** (별도 서비스, §16): `ml/deepfm_pipeline` — 학습 트리거·배치 스코어링·모델/버전 조회·활성화 Admin API. 포트 8000, 메인 앱(8001)과 독립 배포.

---

## Part II. 파이프라인 상세

서비스에 포함된 **모든 파이프라인**을 단계·입출력·설정·예외 처리까지 상세히 정리한 부분입니다.

---

## 7. 시스템 개요 (앱·라우터·엔드포인트)

### 7.1 애플리케이션

| 항목 | 내용 |
|------|------|
| **프레임워크** | FastAPI |
| **역할** | 레스토랑 리뷰 감성 분석·벡터 검색·LLM 요약·다른 음식점과 비교 API |
| **문서** | `/docs` (Swagger), `/redoc` |

### 7.2 라우터 prefix

| prefix | 태그 | 담당 모듈 |
|--------|------|-----------|
| `/api/v1/sentiment` | sentiment | `src/api/routers/sentiment.py` |
| `/api/v1/vector` | vector | `src/api/routers/vector.py` |
| `/api/v1/llm` | llm | `src/api/routers/llm.py` |
| `/api/v1/batch` | batch | `src/api/routers/batch.py` |
| `/api/v1/test` | test | `src/api/routers/test.py` |

### 7.3 공통 미들웨어·엔드포인트

- **X-Request-Id**: 요청 헤더 `X-Request-Id` 또는 `X-Request-ID`가 없으면 UUID 생성, 응답 헤더에 동일 값 설정.
- **Queue depth**: Prometheus용 in-flight 요청 수 증가/감소 (`app_queue_depth_inc` / `app_queue_depth_dec`).
- **CORS**: `CORSMiddleware` (allow_origins=`*`, allow_credentials=True, allow_methods/headers=`*`).
- **예외 처리**: `RequestValidationError` → 422 JSON, `StarletteHTTPException` → 해당 status + JSON, 그 외 `Exception` → 500. 모든 에러 응답에 `code`, `message`, `details`, `request_id` 포함.
- **엔드포인트**:
  - `GET /`: 앱명·버전·docs·health 링크.
  - `GET /health`: `{"status": "healthy", "version": "1.0.0"}`.
  - `GET /ready`: warm-up 완료 시 `{"status": "ready"}`, 미완료 시 **503** + `{"status": "not ready"}`.
  - `GET /metrics`: Prometheus 메트릭 (prometheus_fastapi_instrumentator 설치 시).

### 7.4 Lifespan·Warm-up

- **lifespan**: 앱 시작 시 CPU 모니터 시작(Config에 따라), **warm-up**을 `asyncio.to_thread`로 실행 후 `app.state.ready = True`.
- **Warm-up 내용**:
  1. `get_qdrant_client()` → `get_vector_search(client)` → `encoder.encode(["warmup"])` (VectorSearch/임베딩).
  2. `get_sentiment_analyzer(vector_search)` → `_get_sentiment_pipeline()` → `pl("웜업")` (Sentiment pipeline).
- Warm-up 실패 시 로그 경고만 하고 `/ready`는 503 유지. 정상 완료 시 "서비스 warm-up 완료, readiness=True" 로그.

---

## 8. 공통 인프라

### 8.1 락 (Redis)

| 항목 | 내용 |
|------|------|
| **용도** | 동일 `restaurant_id` + 동일 `analysis_type`에 대한 중복 실행 방지 |
| **키** | `restaurant_id` + `analysis_type` (summary / sentiment / comparison) 조합 |
| **TTL** | 3600초(1시간) 고정 (코드 내 하드코딩) |
| **실패 시** | `RuntimeError`("중복 실행 방지" 메시지) → API에서 **409 Conflict** 반환 |
| **적용 엔드포인트** | `/sentiment/analyze`, `/sentiment/analyze/batch`(레스토랑별), `/llm/summarize`, `/llm/summarize/batch`(레스토랑별), `/llm/comparison`, `/llm/comparison/batch`(레스토랑별) |
| **모듈** | `src/cache.py`의 `acquire_lock` (context manager) |

### 8.2 SKIP (재실행 생략)

| 항목 | 내용 |
|------|------|
| **조건** | `metrics_collector.metrics_db`가 존재하고, 해당 `restaurant_id` + `analysis_type`에 대해 **마지막 성공 시각**이 `SKIP_MIN_INTERVAL_SECONDS` 초 이내일 때. 버전별 SKIP 시 `model_version`, `prompt_version` 인자로 동일 버전 결과만 대상. |
| **동작** | 실제 파이프라인 실행 없이 "스킵" 응답 반환. 메트릭에는 `skipped: true`, `reason: "recent_success"`, `last_success_at` 기록 |
| **설정** | `SKIP_MIN_INTERVAL_SECONDS` (기본 **3600**) |
| **스킵 응답** | Summary: `overall_summary=""` 등 빈 요약. Sentiment: count/ratio=0 등. Comparison: comparisons=[] 등. 디버그 시 `processing_time_ms`, `request_id` 포함 |

### 8.3 메트릭

- **수집**: 요청별 `metrics.collect_metrics(restaurant_id, analysis_type, start_time, batch_size, ...)`. 에러 시 `error_count`, `additional_info` 포함.
- **TTFUR**: `metrics.record_llm_ttft(analysis_type, ttft_ms)` — 요청 수신 시각(t0)부터 응답 반환 직전(t1)까지 밀리초.
- **Prometheus**: `prometheus_fastapi_instrumentator`로 요청 수·지연 등 자동 수집. `/metrics` 노출.

### 8.4 배치·비동기 동작 (Sentiment / Summary / Comparison)

**Sentiment, Summary, Comparison 세 파이프라인은 배치 API에서 다음 두 수준의 비동기로 동작합니다.** 기본 Config 값 기준으로 모두 활성화되어 있습니다.

| 구분 | 설명 | 기본값 |
|------|------|--------|
| **음식점 간 비동기** | 배치 요청 시 여러 레스토랑을 **동시에** 처리 (예: `asyncio.gather`로 레스토랑별 태스크 병렬 실행). | 세 파이프라인 모두 **true** |
| **파이프라인 내 비동기** | 한 레스토랑 처리 시 **파이프라인 내부** 단계를 비동기/병렬로 수행 (분류기 스레드 격리, LLM 비동기 호출, 카테고리별 검색 병렬 등). | 세 파이프라인 모두 **true** |

**파이프라인별 요약**

| 파이프라인 | 음식점 간 비동기 (배치) | 파이프라인 내 비동기 |
|------------|-------------------------|----------------------|
| **Sentiment** | `SENTIMENT_RESTAURANT_ASYNC=true` → 레스토랑별 `analyze_async`를 `asyncio.gather`로 병렬 | `SENTIMENT_CLASSIFIER_USE_THREAD=true`(HF 분류를 `to_thread`로 격리), `SENTIMENT_LLM_ASYNC=true`(LLM 재판정 AsyncOpenAI) |
| **Summary** | `SUMMARY_RESTAURANT_ASYNC=true` → 레스토랑별 처리를 `asyncio.gather`로 병렬 | `SUMMARY_SEARCH_ASYNC=true`(한 레스토랑 내 service/price/food 검색 병렬), `SUMMARY_LLM_ASYNC=true`(LLM 비동기 호출) |
| **Comparison** | `COMPARISON_BATCH_ASYNC=true` → 레스토랑별 `compare`를 `asyncio.gather`로 병렬 | `COMPARISON_ASYNC=true`(서비스/가격 LLM 해석 2개를 `asyncio.gather`로 병렬) |

위 플래그를 false로 두면 해당 수준은 순차 실행됩니다. 단일 요청(레스토랑 1개) API에서는 “음식점 간”은 해당 없고, 파이프라인 내 비동기만 적용될 수 있습니다.

### 8.5 배치 실행 주체 분리 (오프라인 배치)

배치 작업이 API 프로세스를 과부하시키지 않도록, **실행 주체를 분리**합니다. **RQ 방식만** 사용합니다. 배치 처리 시 LLM 추론에는 **훈련이 끝난(서빙용) 모델**을 사용하며, 해당 vLLM Pod는 **GPU A40**을 사용하도록 명시한다(§6.1 GPU 용도별 지정 참고).

| 방식 | 설명 | 사용 |
|------|------|------|
| **RQ 워커** | `BATCH_USE_QUEUE=true` 시 API가 작업을 큐에 넣고 `job_id` 반환. batch-worker가 소비. | `POST /api/v1/batch/enqueue`, `GET /api/v1/batch/status/{job_id}` |
| **단일 레스토랑 스크립트** | RQ 미사용. 동기 API(sentiment/summarize/comparison)를 직접 호출. 디버깅용. | `python scripts/run_single_restaurant.py -r 1 --base-url http://localhost:8001` |

- **작업 큐·재시도·DLQ**: RQ `retry=BATCH_JOB_MAX_RETRIES`. 실패 시 FailedJobRegistry(DLQ)에 자동 등록.
- **관련 문서**: `docs/batch/offline_batch_strategy.md`, `docs/batch/offline_batch_processing.md`

**오프라인 배치 흐름:**
```
[외부: cron/EventBridge] → trigger_offline_batch.py -i restaurants.json -t all
    ↓
POST /api/v1/batch/enqueue → Redis 큐 → RQ 워커 → queue_tasks.run_all_batch_job (sentiment→summary→comparison 순차)
    ↓
GET /api/v1/batch/status/{job_id} → 결과 조회
```

### 8.6 결과 저장 & 버전 관리

- **키**: `restaurant_id` + `analysis_type` + `model_version` + `prompt_version` + `created_at`
- **analysis_metrics**: `prompt_version` 컬럼 추가. `should_skip_analysis`에 `model_version`, `prompt_version` 인자로 버전별 SKIP 가능.
- **결과 저장**: 미사용 (analysis_results 테이블 제거됨). 배치 결과는 API 응답으로만 반환.
- **Config**: `PROMPT_VERSION` (기본 `v1`).

**락 vs SKIP (역할 차이):**
| 구분 | **Redis 락** | **SQLite SKIP** (`should_skip_analysis`) |
|------|-------------|------------------------------------------|
| 목적 | 동시 실행 방지 | 재실행 생략 |
| 시점 | 처리 중 | 이미 완료된 요청 기준 |
| 실패 시 | 409 Conflict | 스킵 응답(빈 요약 등) 반환 |

### 8.7 오프라인 배치 사용 방법

| 용도 | 방법 |
|------|------|
| **트리거** | `python scripts/trigger_offline_batch.py -i data/restaurants.json -t all --base-url http://localhost:8001` |
| **ingestion (벡터 업로드)** | `curl -X POST http://localhost:8001/api/v1/vector/upload -H "Content-Type: application/json" -d @tasteam_app_data.json` (또는 `run_all_restaurants_api.py --upload-only`) |
| **단일 레스토랑 디버깅** | `python scripts/run_single_restaurant.py -r 1 --base-url http://localhost:8001` (RQ 미사용, 동기 API 직접 호출) |
| **enqueue 직접 호출** | `curl -X POST http://localhost:8001/api/v1/batch/enqueue -H "Content-Type: application/json" -d '{"job_type":"all","restaurants":[{"restaurant_id":1},{"restaurant_id":2}],"limit":10}'` |

- **RQ 동작**: enqueue 호출 시 Redis 큐에 job 등록. **RQ 워커(batch-worker)가 실행 중이어야** 실제 작업이 수행됨.
- **job_type**: `sentiment` | `summary` | `comparison` | `all` (all = sentiment→summary→comparison 순차)
- 상세: `docs/batch/offline_batch_processing.md`

---

## 9. Vector 파이프라인

**라우터**: `src/api/routers/vector.py`  
**prefix**: `/api/v1/vector`  
**핵심 클래스**: `src/vector_search.py`의 `VectorSearch`

**현재 제공 API**: `POST /api/v1/vector/upload` 만 제공. 유사 리뷰 검색(`/vector/search/similar`)은 제거됨. Summary/Sentiment/Comparison은 내부에서 `VectorSearch`(query_hybrid_search, get_restaurant_reviews 등)를 사용.

### 9.1 업로드: `POST /vector/upload`

#### 9.1.1 역할

리뷰(및 선택적 레스토랑) 데이터를 벡터 DB에 적재. **Dense + Sparse** 이중 벡터를 배치로 인코딩한 뒤 Qdrant에 upsert합니다.

#### 9.1.2 요청 Body (VectorUploadRequest)

- `reviews`: 리스트. 각 항목에 `id`(또는 review_id), `restaurant_id`, `content`, `created_at`(또는 datetime) 필수. `validate_review_data` 검증.
- `restaurants`: 선택. 각 항목에 `reviews` 등 포함 가능. 있으면 그 안의 리뷰도 병합하여 사용.

#### 9.1.3 처리 단계 (prepare_points → upload_collection)

1. **리뷰 수집**: `data["reviews"]`가 있으면 그대로 사용. 없으면 `data["restaurants"]`에서 `reviews`를 꺼내 병합. 각 리뷰는 `validate_review_data` 통과 필수.
2. **메타데이터 구성**: 리뷰별로 `id`, `restaurant_id`, `content`, `created_at`, `review_id`, `restaurant_name`, `image_urls`, `version` 등을 payload용 dict로 구성.
3. **배치 인코딩**: `batch_size`(Config.get_optimal_batch_size("embedding")) 단위로:
   - Dense: `encoder.encode(batch_texts)`.
   - Sparse: `_sparse_model.embed([text])` per text (실패 시 해당 텍스트는 Sparse None, Dense만 사용).
4. **PointStruct 생성**: 각 리뷰에 대해 Dense 벡터 + (있으면) Sparse 벡터 + payload를 묶어 PointStruct 리스트 생성. point_id는 해시 또는 UUID.
5. **업로드**: `upload_collection(points)` — Qdrant upsert.
6. **restaurant_vectors**: 업로드된 리뷰에서 restaurant_id를 수집하고, 별도 컬렉션 `restaurant_vectors`에 레스토랑 메타데이터 등이 필요하면 생성/갱신.

#### 9.1.4 응답 (VectorUploadResponse)

- `message`: 성공/경고 메시지 (포인트 0개면 "경고: 생성된 포인트가 없습니다").
- `points_count`: 업로드된 포인트 수.
- `collection_name`: 사용한 컬렉션 이름.

### 9.2 Vector 관련 Config·상수

| 설정 | 기본값 | 설명 |
|------|--------|------|
| EMBEDDING_MODEL | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Dense 모델명 |
| EMBEDDING_DIM | 768 | Dense 차원 |
| SPARSE_EMBEDDING_MODEL | Qdrant/bm25 | Sparse 모델명 |
| DENSE_PREFETCH_LIMIT | 200 | Hybrid Dense prefetch |
| SPARSE_PREFETCH_LIMIT | 300 | Hybrid Sparse prefetch |
| FALLBACK_MIN_SCORE | 0.2 | Dense 폴백 시 min_score |
| COLLECTION_NAME | reviews_collection | 컬렉션 이름 |
| QDRANT_URL | ./qdrant_data | Qdrant URL 또는 로컬 경로 |
| EMBEDDING_CACHE_DIR | (없음) | FastEmbed 캐시 경로 (설정 시 FASTEMBED_CACHE_PATH 사용) |

---

## 10. Sentiment 파이프라인

**라우터**: `src/api/routers/sentiment.py`  
**prefix**: `/api/v1/sentiment`  
**핵심 클래스**: `src/sentiment_analysis.py`의 `SentimentAnalyzer`

### 10.1 단일: `POST /sentiment/analyze`

#### 10.1.1 역할

단일 레스토랑의 **전체 리뷰**에 대해 긍정/부정/중립 개수와 비율을 산출합니다. **1차 HuggingFace sentiment 분류** 후, **1차 부정으로 나온 리뷰 중 일부를 LLM으로 재판정**해 최종 집계에 반영합니다.

#### 10.1.2 요청 Body (SentimentAnalysisRequest)

- `reviews`: 리뷰 리스트 (id, restaurant_id, content, created_at 등). Pydantic/딕셔너리 모두 가능.
- `restaurant_id`: 레스토랑 ID (BIGINT FK).

#### 10.1.3 처리 단계

1. **락 획득**: `acquire_lock(restaurant_id, "sentiment", ttl=3600)`.
2. **SKIP 여부**: `metrics_db.should_skip_analysis(restaurant_id, "sentiment", SKIP_MIN_INTERVAL_SECONDS)` → true면 스킵 응답 반환.
3. **content 추출**: `extract_content_list(reviews)` → `content`(또는 `review`) 필드만 리스트로. 비어 있으면 count/ratio 0으로 조기 반환.
4. **1차 분류 (_classify_with_hf_only)**:
   - `_get_sentiment_pipeline()`으로 전역 싱글톤 HF pipeline 로드 (sentiment-analysis, Config.SENTIMENT_MODEL, device=CPU 강제 옵션 가능).
   - 리뷰를 `LLM_BATCH_SIZE`(기본 10) 단위로 나누어 `pipe(batch, top_k=None)` 호출.
   - 각 레이블에 대해: **positive score > 0.8**이면 긍정, 아니면 부정으로 간주하고 해당 리뷰를 **LLM 재판정 대상(negative_reviews_for_llm)**에 추가.
   - 반환: (positive_count, negative_count, neutral_count, total_count, sentiment_labels, negative_reviews_for_llm).
5. **LLM 재판정** (negative_reviews_for_llm이 비어 있지 않을 때):
   - 입력: `id\tcontent` 형식의 줄 단위 텍스트.
   - OpenAI(또는 Config에 따른 백엔드)로 "sentiment classification, JSON array [{\"id\":..., \"sentiment\":\"positive|negative|neutral\"}]" 요청.
   - 응답에서 id별 sentiment를 읽어, 기존 negative로 잡힌 것 중 positive/neutral로 바뀐 만큼 개수 보정하고 labels 갱신.
   - 재판정 실패 시 1차 분류 결과를 그대로 사용.
6. **비율 계산**:
   - `total_with_sentiment = positive_count + negative_count`.
   - `positive_ratio = round((positive_count / total_with_sentiment) * 100)` (total_with_sentiment가 0이면 0).
   - `negative_ratio` 동일.
   - `neutral_ratio = round((neutral_count / total_count) * 100)`.
7. **메트릭 수집** 후 `SentimentAnalysisResponse` 반환.

#### 10.1.4 응답 (SentimentAnalysisResponse)

- `restaurant_id`, `restaurant_name`, `positive_count`, `negative_count`, `neutral_count`, `total_count`, `positive_ratio`, `negative_ratio`, `neutral_ratio`.
- 디버그 시 `debug`: `request_id`, `processing_time_ms`, `tokens_used`, `model_version`.

#### 10.1.5 Sentiment 관련 Config

| 설정 | 기본값 | 설명 |
|------|--------|------|
| SENTIMENT_MODEL | Dilwolf/Kakao_app-kr_sentiment | HuggingFace sentiment 모델 |
| SENTIMENT_FORCE_CPU | true | True면 device=-1 (CPU만 사용, meta tensor 오류 회피) |
| SENTIMENT_CLASSIFIER_USE_THREAD | true | True면 HF 분류를 asyncio.to_thread로 실행 (이벤트 루프 격리) |
| SENTIMENT_LLM_ASYNC | true | True면 LLM 재판정을 AsyncOpenAI로 호출 |
| SENTIMENT_RESTAURANT_ASYNC | true | 배치 시 레스토랑 간 병렬 |
| LLM_BATCH_SIZE | 10 | 1차 분류 배치 크기·LLM 입력 배치 참고 |
| OPENAI_MODEL | gpt-4o-mini | LLM 재판정에 사용하는 모델 |

### 10.2 배치: `POST /sentiment/analyze/batch`

- 요청: `SentimentAnalysisBatchRequest` — `restaurants` 리스트 (각 항목에 reviews, restaurant_id 등).
- `SENTIMENT_RESTAURANT_ASYNC=true`면 `asyncio.gather`로 레스토랑별 `analyze_async` 병렬 실행.
- 반환: `SentimentAnalysisBatchResponse` — 레스토랑별 `SentimentAnalysisResponse` 리스트.

---

## 11. Summary 파이프라인

**라우터**: `src/api/routers/llm.py`  
**prefix**: `/api/v1/llm`  
**핵심**: `_retrieve_category_hits_accuracy_first`(llm.py), `summarize_aspects_new` / `summarize_aspects_new_async`(summary_pipeline.py), `aspect_seeds.py`

### 11.1 단일: `POST /llm/summarize`

#### 11.1.1 역할

레스토랑 리뷰를 **service / price / food** 세 카테고리로 검색한 뒤, 카테고리별 요약(summary, bullets, evidence)과 **overall_summary**를 생성합니다. 출력 말투는 **"~해요" 체**로 통일합니다. 긍정/부정 비율은 세지 않습니다.

#### 11.1.2 요청 Body (SummaryRequest)

- `restaurant_id`: 레스토랑 ID.
- `limit`: 카테고리당 검색·요약에 사용할 최대 리뷰 수 (기본 10).

#### 11.1.3 처리 단계

1. **락** → **SKIP** (§8).
2. **시드 결정**: 해당 음식점 리뷰로 **recall seed**를 생성해 카테고리별 시드 사용(실패·리뷰 부족 시 `DEFAULT_SERVICE_SEEDS` 등 기본 시드 폴백). 카테고리별 시드 **최대 10개**를 공백으로 이어 쿼리 문자열 생성. 상세: `docs/spark/SUMMARY_RECALL_SEEDS.md`.
3. **카테고리별 검색** (service → price → food 순, 단일 요청은 순차): 각 카테고리에 대해 `_retrieve_category_hits_accuracy_first` 호출.

##### _retrieve_category_hits_accuracy_first 상세

- **입력**: vector_search, category_name, query_text, restaurant_id, final_limit.
- **상수**: `k_min = min(3, max(1, final_limit))`, `dense_candidate_limit = max(final_limit * 8, 50)`.

**1차 — Dense-only**

- `vector_search._query_dense_only(query_text, restaurant_id, limit=dense_candidate_limit, min_score=0.0)`.
- 결과를 `dense_hits`로 둠.

**2차 — Hybrid 필요 조건 및 실행**

- 키워드: `_CATEGORY_KEYWORDS[category_name]` (예: service → 서비스, 친절, 응대, 직원, …).
- `dense_texts = _hits_to_texts(dense_hits)`, `dense_ratio = _text_keyword_hit_ratio(dense_texts, keywords, top_n=8)` (상위 8개 중 키워드 1개라도 포함된 비율).
- `dense_top_score`: dense_hits[0].score.
- `dense_flat`: (dense_hits 5개 이상일 때) (top1 score - top5 score) < 0.02 이면 True.
- **need_hybrid** = `(len(dense_hits) < k_min) OR (dense_ratio < 0.25) OR (dense_top_score < 0.25) OR dense_flat`.
- need_hybrid이면 `query_hybrid_search(query_text, restaurant_id, limit=max(final_limit*3, 30), ...)` 호출 → hybrid_hits.
- **선택**: Dense vs Hybrid 중 `hybrid_ratio > dense_ratio` 이거나 `(len(dense_hits) < k_min and len(hybrid_hits) >= len(dense_hits))` 이면 best_hits = hybrid_hits, 아니면 best_hits = dense_hits.
- `best_hits = _dedup_hits_by_review_id(best_hits)`.

**3차 — 넓은 쿼리**

- `len(best_hits) < k_min` 이면 `_BROAD_QUERY[category_name]` 문자열을 쿼리에 붙여 `broad_query`로 Hybrid 1회 추가 검색. 기존 best_hits와 merge 후 dedup.

**4차 — 최근 리뷰**

- 여전히 `len(best_hits) < k_min` 이면 `vector_search.get_recent_restaurant_reviews(restaurant_id, limit=...)`로 최근 리뷰를 가져와 hit 형태로 변환 후 merge·dedup.

- **반환**: hit 리스트 (각 항목 payload, score).

4. **hits_dict / hits_data_dict 구성**: 카테고리별로 hit의 payload.content, review_id, rank를 리스트로 정리. restaurant_name은 payload에서 채움.
5. **LLM 요약**: `summarize_aspects_new(service_reviews, price_reviews, food_reviews, service_evidence_data, price_evidence_data, food_evidence_data, llm_utils, per_category_max=request.limit)`.

##### summarize_aspects_new 상세

- **입력**: 카테고리별 리뷰 텍스트 리스트, 카테고리별 evidence 데이터 [{ review_id, snippet, rank }], llm_utils, per_category_max(기본 8).
- **클리핑**: 각 카테고리를 `per_category_max`개로 자른 뒤 JSON payload로 LLM에 전달.
- **프롬프트 규칙**: 말투 "~해요" 체, summary 1문장, bullets 3~5개, evidence는 0-based 인덱스 배열, overall_summary 2~3문장, 근거 없으면 "언급이 적어요" 등.
- **LLM 출력 스키마**: `{ "service": { summary, bullets, evidence: [int] }, "price": ..., "food": ..., "overall_summary": { summary } }`.
- **후처리**: evidence 인덱스를 해당 카테고리의 evidence 객체(review_id, snippet, rank) 리스트로 치환. **Price 게이트**: price의 evidence 리뷰에 PRICE_HINTS(가격, 가성비, …)가 전혀 없으면 summary·bullets를 고정 문구로 덮음 ("가격 관련 언급이 많지 않아요. …", "가격을 직접 언급한 리뷰가 많지 않아요." 등).
- **실패 시**: JSON 파싱 재시도 후에도 실패하면 카테고리별 빈 summary/bullets/evidence, overall_summary "요약 생성에 실패했어요." 반환.

6. **응답 조립**: 파이프라인 결과에서 overall_summary 추출. 비어 있으면 카테고리별 summary를 공백으로 이어 붙이고, 그래도 없으면 "요약할 리뷰가 없어요." 사용. `SummaryDisplayResponse` 생성 후 메트릭 수집·반환.

#### 11.1.4 응답 (SummaryDisplayResponse)

- `restaurant_id`, `restaurant_name`, `overall_summary`, `categories`: { service, price, food } 각각 `CategorySummary`(summary, bullets, evidence).
- 디버그 시 `debug`: request_id, processing_time_ms 등.

### 11.2 배치: `POST /llm/summarize/batch`

- 요청: `SummaryBatchRequest` — `restaurants` 리스트, `limit`(공통).
- **시드**: 레스토랑마다 해당 리뷰로 recall seed 생성(실패 시 기본 시드). `docs/spark/SUMMARY_RECALL_SEEDS.md` 참고.
- **SUMMARY_SEARCH_ASYNC=true**: 한 레스토랑 내에서 service/price/food 검색을 `asyncio.gather`로 병렬.
- **SUMMARY_RESTAURANT_ASYNC=true**: 레스토랑 단위를 `asyncio.gather`로 병렬.
- **SUMMARY_LLM_ASYNC=true**: `summarize_aspects_new_async` 사용(비동기 LLM), false면 `asyncio.to_thread(summarize_aspects_new)`.
- **세마포어**: `BATCH_SEARCH_CONCURRENCY`(기본 50), `BATCH_LLM_CONCURRENCY`(기본 8)로 검색·LLM 동시 실행 수 제한.
- 둘 다 false면 레스토랑·카테고리 모두 순차.

### 11.3 Summary 관련 Config

| 설정 | 기본값 | 설명 |
|------|--------|------|
| SUMMARY_SEARCH_ASYNC | true | 배치 시 카테고리별 검색 병렬 |
| SUMMARY_RESTAURANT_ASYNC | true | 배치 시 레스토랑 간 병렬 |
| SUMMARY_LLM_ASYNC | true | 배치 시 LLM 비동기 호출 |
| BATCH_SEARCH_CONCURRENCY | 50 | 배치 검색 세마포어 |
| BATCH_LLM_CONCURRENCY | 8 | 배치 LLM 세마포어 |
| RECALL_SEEDS_SPARK_THRESHOLD | 2000 | 이 리뷰 수 미만이면 recall seed 계산에 Python(Kiwi), 이상이면 Spark 사용. `docs/spark/SUMMARY_RECALL_SEEDS.md` 참고. |
| DENSE_PREFETCH_LIMIT, SPARSE_PREFETCH_LIMIT, FALLBACK_MIN_SCORE | (Vector와 동일) | Hybrid/폴백 검색 |

---

## 12. Comparison 파이프라인

**라우터**: `src/api/routers/llm.py`  
**prefix**: `/api/v1/llm`  
**핵심**: `src/comparison.py`의 `ComparisonPipeline`, `src/comparison_pipeline.py`의 Kiwi+Spark 비율·lift·전체 평균

### 12.1 단일: `POST /llm/comparison`

#### 12.1.1 역할

단일 레스토랑을 **전체 평균**과 비교해 **service / price** 만족도 **lift(%)**를 산출하고, LLM으로 자연어 해석 문장을 생성합니다. lift > 0인 카테고리만 comparisons에 포함됩니다.

#### 12.1.2 요청 Body (ComparisonRequest)

- `restaurant_id`: 타겟 레스토랑 ID.

#### 12.1.3 처리 단계

1. **락** → **SKIP** (§8).
2. **전체 평균(all_average_ratios) 계산** — 우선순위:
   - **① 파일**: `all_average_data_path` 또는 `Config.ALL_AVERAGE_ASPECT_DATA_PATH`가 있으면 `calculate_all_average_ratios_from_file(path, stopwords, project_root)` 호출. `SPARK_SERVICE_URL` 설정 시 해당 서비스 `POST /all-average-from-file` 호출, 미설정 시 로컬 Spark로 TSV/JSON 리뷰를 읽어 Kiwi bigram → service/price 긍정 비율 계산. 성공 시 해당 비율 사용.
   - **② Qdrant**: ① 실패 또는 없으면 `vector_search.get_all_reviews_for_all_average(limit=5000)`로 리뷰 샘플 조회 후 `calculate_all_average_ratios_from_reviews(all_reviews, stopwords)` (Kiwi만 또는 Spark)로 비율 계산.
   - **③ Config fallback**: 그래도 없으면 `ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO` (기본 0.60, 0.55) 사용.
3. **단일 레스토랑 리뷰 조회**: `vector_search.get_restaurant_reviews(str(restaurant_id))`. 없으면 comparisons=[], category_lift=0, comparison_display 템플릿으로 조기 반환.
4. **단일 레스토랑 비율**: `calculate_single_restaurant_ratios(review_texts, stopwords)`.

##### calculate_single_restaurant_ratios 상세

- **Spark 경로**(SPARK_AVAILABLE이고 DISABLE_SPARK가 아닐 때): RDD로 텍스트 병렬화 → `_spark_calculate_ratios(rdd, stopwords)`.
  - Kiwi 토크나이저로 NNG/NNP만, len≥2, 불용어 제외, bigram (a b) 생성 → (phrase, 1) emit → reduceByKey → takeOrdered(2000) → is_noise 제거.
  - phrase를 **SERVICE_KW** / **PRICE_KW**로 분류: SERVICE_KW = {친절, 서비스, 응대, 직원, 사장, 불친절}, PRICE_KW = {가격, 가성비, 대비, 리필, 무한, 할인, 쿠폰}.
  - service 긍정: phrase에 **SERVICE_POSITIVE_KW**(친절) 포함 개수 / service 총 개수. price 긍정: **PRICE_POSITIVE_KW**(가성비, 가격 합리, 가격 만족, 무한 리필, 리필 가능) 포함 개수 / price 총 개수.
- **Python 폴백**(Spark 비활성 또는 Py4J/Spark 오류 시): `_python_calculate_ratios(texts, stopwords)` — Kiwi만으로 동일 bigram·분류·비율 계산.
- 반환: `{"service": 0.xx, "price": 0.xx}` (소수 둘째자리 반올림).

5. **Lift 계산**: `calculate_comparison_lift(single_restaurant_ratios, all_average_ratios)`.
   - **식**: `lift[category] = ((single_ratio - all_ratio) / all_ratio) * 100` (all_ratio > 0일 때). 반올림 정수.
   - 반환: `{"service": 20, "price": 18}` 형태.
6. **LLM 해석**: 서비스·가격 각각에 대해 `generate_comparison_interpretation_async(label, lift, tone, n_reviews)` 호출. lift≤0이면 "평균과 비슷합니다." 고정. `COMPARISON_ASYNC=true`면 `asyncio.gather`로 2개 병렬.
   - LLM 실패 시 **템플릿 폴백**: lift≥30 → "약 N% 높아, 차이가 큰 편", lift<10 → "N% 높아, 차이는 크지 않은 편", 그 외 "약 N% 높아, 차이가 어느 정도 있습니다."
7. **comparisons 리스트**: lift_dict에서 **lift > 0**인 카테고리만 항목으로 넣음 (category, lift_percentage, comparison_display 문장 등).
8. **응답**: `ComparisonResponse`(comparisons, category_lift, comparison_display, total_candidates, validated_count, processing_time_ms 등).

#### 12.1.4 응답 (ComparisonResponse)

- `restaurant_id`, `restaurant_name`, `comparisons`(lift>0인 카테고리별 항목), `total_candidates`, `validated_count`, `category_lift`(service/price 수치), `comparison_display`(서비스/가격 해석 문장 리스트).
- 디버그 시 `debug` 포함.

### 12.2 배치: `POST /llm/comparison/batch`

- 요청: `ComparisonBatchRequest` — `restaurants` 리스트, `all_average_data_path`(선택).
- `COMPARISON_BATCH_ASYNC=true`면 레스토랑별 compare를 `asyncio.gather`로 병렬.
- 반환: `ComparisonBatchResponse` — 레스토랑별 `ComparisonResponse` 리스트.

### 12.3 Comparison 관련 Config·상수

| 설정 | 기본값 | 설명 |
|------|--------|------|
| COMPARISON_ASYNC | true | 서비스/가격 LLM 해석 병렬 |
| COMPARISON_BATCH_ASYNC | true | 배치 시 레스토랑 간 병렬 |
| SPARK_SERVICE_URL | (없음) | 설정 시 전체 평균을 Spark 마이크로서비스(MSA) HTTP 호출로 계산. 미설정 시 로컬 Spark 또는 Python 폴백. `docs/spark/SPARK_SERVICE.md` 참고. |
| ALL_AVERAGE_ASPECT_DATA_PATH | data/test_data_sample.json | 전체 평균용 파일 경로 |
| ALL_AVERAGE_SERVICE_RATIO | 0.60 | 전체 평균 fallback (service) |
| ALL_AVERAGE_PRICE_RATIO | 0.55 | 전체 평균 fallback (price) |
| DISABLE_SPARK | false | true면 Spark 비활성, Kiwi만 사용 |

- **comparison_pipeline.py 상수**: SERVICE_KW, PRICE_KW, SERVICE_POSITIVE_KW, PRICE_POSITIVE_KW (위 6.1.3 참고).

---

## 13. 파이프라인 관계 요약

```
[클라이언트]
    │
    ├─ POST /api/v1/vector/upload           → 리뷰 벡터 적재 (Dense+Sparse). 유사 검색 API는 미제공(내부 VectorSearch 사용).
    │
    ├─ POST /api/v1/sentiment/analyze       → 감성 분류 (HF 1차 + LLM 재판정) → 비율
    ├─ POST /api/v1/sentiment/analyze/batch
    │
    ├─ POST /api/v1/llm/summarize           → 카테고리별 검색 + LLM 요약 (해요체)
    ├─ POST /api/v1/llm/summarize/batch
    │
    ├─ POST /api/v1/llm/comparison          → Kiwi+Spark 비율 → lift → LLM 해석
    ├─ POST /api/v1/llm/comparison/batch
    │
    ├─ POST /api/v1/batch/enqueue           → 배치 작업 큐에 넣기 (BATCH_USE_QUEUE=true 시)
    └─ GET  /api/v1/batch/status/{job_id}   → 작업 상태 조회
```

- **Vector** 업로드 결과는 Summary의 카테고리별 검색·Comparison의 레스토랑 리뷰 조회에서 사용됩니다.
- **Summary**는 긍정/부정 비율을 세지 않고 **요약만** 생성하며, **Sentiment**가 비율을 담당합니다.
- **Comparison**은 Vector에 적재된 리뷰를 조회한 뒤 Kiwi(+Spark)로 service/price 긍정 비율을 구하고, 전체 평균과 비교해 lift와 LLM 설명을 만듭니다.

### 6.1 디스틸·학습 파이프라인 (RunPod, API 외부)

요약 KD·QLoRA 학습은 **Prefect flows**와 **RunPod Pod**로 실행되며, 메인 FastAPI API와는 별도 스크립트입니다.

| 항목 | 내용 |
|------|------|
| **오케스트레이션** | Prefect (`scripts/distill_flows.py`): **all** = build_dataset → labeling(Pod) → train_student(Pod, 단일 run) → evaluate → merge. **all_sweep** = 동일하되 학습은 Pod에서 sweep(여러 run) 실행 후 best adapter는 wandb artifact에서 수급. §6.1.1·6.1.2 참고. `docs/easydistill/distill_with_prefect.md` |
| **학습 실행** | RunPod Pod에서 Docker 이미지(`Dockerfile.train-llm`, `jinsoo1218/train-llm`)로 `train_qlora.py` 실행. Network Volume `/workspace`에 데이터·어댑터 저장. |
| **스토리지** | RunPod Network Volume. Pod 없이 파일 접근은 **S3 호환 API**(`aws s3 --endpoint-url https://s3api-eu-ro-1.runpod.io`) 사용. 학습용 볼륨 ID 예: `4rlm64f9lv`, vLLM용: `2kn4qj6rql`. |
| **관련 문서** | `docs/runpod_cli/runpod_net_vol_s3_api.md`, `docs/runpod_cli/distill_train_net_vol.md`, `docs/runpod_cli/vllm_net_vol.md`, `docs/easydistill/distill_strategy.md` |

**GPU 용도별 지정**

| 용도 | GPU | 비고 |
|------|-----|------|
| **훈련** (train Pod, sweep Pod) | **RTX 4090** | QLoRA 등 학습 전용 Pod. |
| **라벨링** (labeling Pod) | **A40** | Teacher vLLM 추론. |
| **배치 처리** (vLLM 추론 Pod) | **A40** | 배치 작업 시 사용하는 LLM은 **훈련이 끝난(서빙용) 모델**이다. |

#### 6.1.1 파이프라인 종류: all vs all_sweep

| 플로우 | 단계 | 학습 실행 위치 |
|--------|------|----------------|
| **all** | build_dataset → labeling(Pod) → **train_student_with_pod**(단일 run) → evaluate → merge | 학습용 Pod에서 **한 번** QLoRA 학습 (`train_qlora.py`). adapter는 Pod 볼륨에서 다운로드. |
| **all_sweep** | build_dataset → labeling(Pod) → **run_sweep_and_evaluate**(use_pod=True) → merge | 학습용 Pod에서 **sweep** 실행. 아래 §6.1.2 참고. |

실행 예: `python scripts/distill_flows.py all`, `python scripts/distill_flows.py all_sweep [--sweep-id ...]`.

#### 6.1.2 Sweep on Pod (all_sweep)

**Sweep**이란 wandb에서 정의한 하이퍼파라미터 조합들에 대해 **여러 번의 학습(run)**을 수행하는 단위입니다. **Pod에서 sweep을 한다** = **한 Pod 안에서 wandb agent가 run 1 → run 2 → … 순서대로 여러 번 학습**을 진행하는 것을 의미합니다.

| 항목 | 내용 |
|------|------|
| **등록** | `register_sweep_task(sweep_yaml)`로 wandb에 sweep 등록 후 `sweep_id` 획득. |
| **실행** | `run_sweep_on_pod_task(sweep_id, labeled_path, output_dir)`: 라벨 디렉터리를 학습용 Network Volume에 업로드 → 학습용 이미지로 Pod 생성 (`dockerEntrypoint`: `python /app/scripts/run_qlora_sweep.py`, `dockerStartCmd`: `[sweep_id]`) → Pod RUNNING 대기 → **Pod가 종료될 때까지** 대기(`wait_until_stopped`, 기본 4시간) → Pod 삭제. |
| **Pod 내 동작** | 컨테이너 진입점은 `run_qlora_sweep.py`(wandb agent). 동일 Pod에서 sweep에 할당된 여러 run을 **순차 실행**. 각 run은 QLoRA 학습 후 adapter를 wandb artifact로 업로드. |
| **Best adapter** | Pod 디스크가 아닌 **wandb artifact**에서 수급. `get_best_adapter_from_artifact_task(sweep_id, download_dir, metric_name="train/loss")`로 sweep의 best run을 조회한 뒤 해당 run의 `qlora-adapter-{run_id}` artifact를 다운로드. |
| **이어지는 단계** | `run_sweep_and_evaluate_flow(use_pod=True)`는 sweep(Pod) 완료 후 위 task로 best adapter를 받아 로컬에서 `evaluate_flow` 실행. `distill_pipeline_all_sweep`은 이 flow를 `use_pod=True`로 호출한 뒤, best adapter가 있으면 `merge_for_serving_flow`까지 수행. |

요약: **all_sweep**은 로컬이 아닌 **한 대의 학습용 Pod**에서 sweep 전체(여러 run)를 실행하고, best adapter는 wandb artifact로 받아 로컬에서 평가·merge 합니다.

### 6.2 DeepFM 파이프라인 (추천/CTR, API 외부)

**CTR 예측·개인화 랭킹**용 DeepFM 학습은 메인 FastAPI API·디스틸 파이프라인과 **독립**적으로 동작합니다. 생성형 LLM이 아니며 EasyDistill KD와는 부적합합니다.

| 항목 | 내용 |
|------|------|
| **역할** | 사용자·아이템·컨텍스트 피처 → **스칼라 점수(클릭률 등)** 예측. 주변 맛집 후보를 **개인화 랭킹**하는 스코어러로 사용. |
| **오케스트레이션** | Prefect (`deepfm_training/training_flow.py`): `deepfm_training_flow` — 전처리 → 학습 → 모델 저장. |
| **모델** | PyTorch DeepFM (FM + DNN). Criteo 형식 데이터. `deepfm_training/model/DeepFM.py`. |
| **데이터** | `deepfm_training/data/raw/` (train.txt, test.txt). 전처리 결과는 `data/`에 생성. 학습 결과는 `output/<run_id>/model.pt`. |
| **실행** | 로컬: `python deepfm_training/training_flow.py`. Docker: `Dockerfile.deepfm` → `deepfm-training` 이미지. |
| **관련 문서** | `docs/prefect/deepfm_training_pipeline.md`, `docs/prefect/prefect_designe.md`, `deepfm_training/deepfm_tasteam.md`(tasteam 서비스 관점·피처 설계·랭킹 역할), `docs/easydistill/learning_method_fit.md`(DeepFM은 LLM 증류 부적합 명시). |

---

## 14. 설정(Config) 전체 요약

| 구분 | 설정 | 기본값 | 설명 |
|------|------|--------|------|
| **공통** | SKIP_MIN_INTERVAL_SECONDS | 3600 | SKIP 판단 간격(초) |
| **Vector** | EMBEDDING_MODEL | paraphrase-multilingual-mpnet-base-v2 | Dense 모델 |
| | EMBEDDING_DIM | 768 | Dense 차원 |
| | SPARSE_EMBEDDING_MODEL | Qdrant/bm25 | Sparse 모델 |
| | DENSE_PREFETCH_LIMIT | 200 | Hybrid Dense prefetch |
| | SPARSE_PREFETCH_LIMIT | 300 | Hybrid Sparse prefetch |
| | FALLBACK_MIN_SCORE | 0.2 | Dense 폴백 min_score |
| | COLLECTION_NAME | reviews_collection | Qdrant 컬렉션 |
| | QDRANT_URL | ./qdrant_data | Qdrant 주소 |
| **Sentiment** | SENTIMENT_MODEL | Dilwolf/Kakao_app-kr_sentiment | HF 모델 |
| | SENTIMENT_FORCE_CPU | true | CPU만 사용 |
| | SENTIMENT_CLASSIFIER_USE_THREAD | true | HF를 to_thread로 실행 |
| | SENTIMENT_LLM_ASYNC | true | LLM 재판정 비동기 |
| | SENTIMENT_RESTAURANT_ASYNC | true | 배치 레스토랑 병렬 |
| **Summary** | SUMMARY_SEARCH_ASYNC | true | 배치 카테고리 검색 병렬 |
| | SUMMARY_RESTAURANT_ASYNC | true | 배치 레스토랑 병렬 |
| | SUMMARY_LLM_ASYNC | true | 배치 LLM 비동기 |
| | BATCH_SEARCH_CONCURRENCY | 50 | 배치 검색 동시 수 |
| | BATCH_LLM_CONCURRENCY | 8 | 배치 LLM 동시 수 |
| | RECALL_SEEDS_SPARK_THRESHOLD | 2000 | recall seed 계산 시 Spark 사용 리뷰 수 기준. `docs/spark/SUMMARY_RECALL_SEEDS.md` 참고. |
| **Comparison** | COMPARISON_ASYNC | true | 서비스/가격 LLM 병렬 |
| | COMPARISON_BATCH_ASYNC | true | 배치 레스토랑 병렬 |
| | SPARK_SERVICE_URL | (없음) | 설정 시 Spark 서비스로 전체 평균 계산. `docs/spark/SPARK_SERVICE.md` 참고. |
| | ALL_AVERAGE_ASPECT_DATA_PATH | data/test_data_sample.json | 전체 평균 파일 |
| | ALL_AVERAGE_SERVICE_RATIO | 0.60 | 전체 평균 fallback service |
| | ALL_AVERAGE_PRICE_RATIO | 0.55 | 전체 평균 fallback price |
| | DISABLE_SPARK | false | Spark 비활성 시 Kiwi만 |
| **배치·큐** | BATCH_USE_QUEUE | false | true면 배치 API가 RQ 큐에 enqueue, false면 동기 실행 |
| | RQ_QUEUE_NAME | batch | RQ 큐 이름 |
| | BATCH_JOB_MAX_RETRIES | 3 | RQ job 재시도 횟수 |
| | REDIS_URL | (env) | RQ/락용 Redis URL |
| | PROMPT_VERSION | v1 | 결과 버전 관리용 (analysis_metrics, SKIP 판단) |

---

## 15. 관련 파일 맵 (상세)

| 파이프라인 | 라우터 | 핵심 함수·클래스 | 의존 모듈 |
|------------|--------|------------------|-----------|
| **Vector** | api/routers/vector.py | prepare_points, upload_collection (업로드 API만 제공). query_hybrid_search·get_restaurant_reviews 등은 Summary/Sentiment/Comparison 내부 사용 | vector_search.py, config, models |
| **Sentiment** | api/routers/sentiment.py | SentimentAnalyzer._get_sentiment_pipeline, _classify_with_hf_only, _classify_contents, _apply_llm_reclassify_*, analyze, analyze_async | sentiment_analysis.py, review_utils(extract_content_list), config, models |
| **Summary** | api/routers/llm.py | _get_seed_list_for_restaurant(음식점별 recall seed), _retrieve_category_hits_accuracy_first, _process_one_restaurant_async, _batch_summarize_async | summary_pipeline.py(summarize_aspects_new, _async), comparison_pipeline(compute_recall_seeds_from_reviews, recall_seeds_to_seed_lists), vector_search, config, models |
| **Comparison** | api/routers/llm.py | ComparisonPipeline.compare, compare_batch | comparison.py, comparison_pipeline.py(calculate_single_restaurant_ratios, calculate_comparison_lift, calculate_all_average_ratios_from_file/from_reviews, format_comparison_display, _spark_calculate_ratios, _python_calculate_ratios), vector_search, llm_utils, config, models |
| **Batch (큐)** | api/routers/batch.py | POST /batch/enqueue, GET /batch/status/{job_id} | queue_tasks.py(run_*_batch_job, run_all_batch_job), scripts/rq_worker.py, scripts/trigger_offline_batch.py |
| **DeepFM (API 외부)** | — | Prefect flow. `deepfm_training/training_flow.py`(deepfm_training_flow), `main.py` | model/DeepFM.py, data/raw, output. §6.2 참고. |

**공통**: api/main.py, api/dependencies.py, config.py, models.py, cache.py(acquire_lock), metrics_collector.py, metrics_db.py.

---

## 16. DeepFM ML 파이프라인

**위치**: `ml/deepfm_pipeline/` — 메인 FastAPI 앱(new_async)과 **별도 서비스**로, 추천(DeepFM) 학습·배치 스코어링·모델/버전 관리를 담당합니다.

### 16.1 역할·개요

| 항목 | 내용 |
|------|------|
| **프레임워크** | FastAPI (Python 3.11) |
| **역할** | Admin API: 학습 트리거, 배치 스코어링/추천 생성, 모델 목록 조회, 서빙용 pipeline_version 활성화 |
| **진입점** | `uvicorn api.main:app --host 0.0.0.0 --port 8000` (기본 8000) |
| **문서** | `ml/deepfm_pipeline/docs/design/api/api_design.md`, `ML_API_DTO.md` |

**전체 흐름 (개념)**

```
[Admin/ETL]
    │
    ▼
[DeepFM API :8000]  FastAPI  ─┬─ POST /admin/deepfm/train        → Prefect 플로우(전처리→학습) → output/<run>/ model.pt, run_manifest.json, pipeline_version
    │                          ├─ POST /admin/deepfm/score-batch  → run_dir + 후보 CSV → recommendation CSV (INSERT는 호출 측)
    │                          ├─ GET  /admin/deepfm/models      → output/ 하위 run 목록 + active_version
    │                          └─ POST /admin/deepfm/activate     → output/active_pipeline_version.txt 갱신
    │
    ▼
[output/]  run 디렉터리별 model.pt, feature_sizes.txt, pipeline_version.txt, run_manifest.json
```

### 16.2 디렉터리 구조

| 경로 | 설명 |
|------|------|
| `api/main.py` | FastAPI 앱, CORS, 라우터 등록, `GET /health` |
| `api/schemas.py` | TrainRequestDto, TrainResponseDto, ScoreBatchRequestDto/ResponseDto, ModelInfoDto, ModelsResponseDto, ActivateRequestDto/ResponseDto |
| `api/routers/deepfm.py` | `/admin/deepfm` 하위 엔드포인트 구현 |
| `training_flow.py` | Prefect flow: 전처리 → 학습 → run_manifest·pipeline_version 산출 |
| `model/`, `data/`, `utils/` | DeepFM 모델, 데이터셋, 전처리·score_batch·wandb 등 |
| `output/` | run별 디렉터리(`model.pt`, `feature_sizes.txt`, `pipeline_version.txt`, `run_manifest.json`), `active_pipeline_version.txt` |

### 16.3 API 엔드포인트

| 메서드 | 경로 | 역할 |
|--------|------|------|
| POST | `/admin/deepfm/train` | 학습 트리거. Prefect `deepfm_training_flow` 동기 실행 → pipeline_version, model_path, run_manifest_path, metrics 반환 |
| POST | `/admin/deepfm/score-batch` | 배치 스코어링. `pipeline_version`(또는 run_dir) + candidates_path → recommendation CSV 출력. DB INSERT는 호출 측(ETL) |
| GET | `/admin/deepfm/models` | output 하위 run 목록 + 현재 활성 `active_version` |
| POST | `/admin/deepfm/activate` | 서빙용 pipeline_version 활성화 → `output/active_pipeline_version.txt` 기록 |

- **Health**: `GET /health` → `{"status": "ok"}`  
- DTO 상세: `ml/deepfm_pipeline/docs/design/api/ML_API_DTO.md`

### 16.4 학습 플로우

1. **전처리** (`preprocess_task`): raw_data_dir → processed_data_dir (train/valid/test split, feature_sizes 등). 선택 시 W&B artifact 로깅.
2. **학습** (`train_task`): DeepFM 학습 → `output/<run>/model.pt`, `feature_sizes.txt` 저장.
3. **run 산출**: `pipeline_version` 발급(예: `deepfm-1.0.YYYYMMDDHHMMSS`), `run_manifest.json`(pipeline_version, model_path, metrics, timestamp_utc) 기록.
4. 학습은 **동기** 실행(같은 프로세스에서 `training_flow.deepfm_training_flow` 호출).

### 16.5 스코어링

- **진입**: `POST /admin/deepfm/score-batch` (pipeline_version, candidates_path, output_path 필수; run_dir 없으면 pipeline_version으로 output 하위에서 run 디렉터리 탐색).
- **실행**: `utils.score_batch.run(run_dir, candidates_path, output_path, meta_path, ttl_hours, batch_size)` — run_dir에서 model.pt·feature_sizes 로드 후 배치 추론 → recommendation 형식 CSV 출력.
- **recommendation 테이블 INSERT**는 API가 하지 않으며, ETL/DB 측에서 수행.

### 16.6 활성 버전

- **활성 pipeline_version**: `output/active_pipeline_version.txt`에 한 줄로 저장.
- `GET /admin/deepfm/models`의 `active_version`, 추천 API 등에서 이 값을 참조해 서빙 모델로 사용.

### 16.7 Docker

- **이미지**: `ml/deepfm_pipeline/Dockerfile` — `python:3.11-slim`, `ml-requirements.txt` 설치, `PYTHONPATH=/app`, `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- **빌드 컨텍스트**: `ml/deepfm_pipeline/`. `.dockerignore`로 output/data 등 제외 권장.

### 16.8 참고 문서

| 문서 | 설명 |
|------|------|
| `ml/deepfm_pipeline/docs/design/api/api_design.md` | API 설계(필수 엔드포인트) |
| `ml/deepfm_pipeline/docs/design/api/ML_API_DTO.md` | Request/Response DTO 명세 |
| `ml/deepfm_pipeline/docs/design/deepfm/deepfm_design.md` | DeepFM 파이프라인 설계 |

---

이 문서는 현재 코드 기준으로 작성되었습니다. 파이프라인 변경 시 함께 갱신하는 것을 권장합니다.  
카테고리별 요약 검색 조건의 트레이드오프는 `docs/reference/pipe_anal_ex.md/second_stage_search_tradeoffs.md`, Summary 단계별 상세는 `etc_md/SUMMARY_PIPELINE.md` 등을 참고하면 됩니다.

- **CPU/GPU 서버 분리**: [GPU_PLATFORM_SELECTION.md](../runpod/GPU_PLATFORM_SELECTION.md) — GPU 플랫폼 선택 참고.
- **RunPod Pod vs Serverless**: [why_dont_use_runpod_serverless.md](../runpod/why_dont_use_runpod_serverless.md) — Serverless 미사용 이유 (Ephemeral, Prometheus 스크래핑 불안정 등). Pod만 사용.
- **오프라인 배치 전략**: [offline_batch_strategy.md](../batch/offline_batch_strategy.md) — RQ 워커, 작업 큐·재시도·DLQ, 버전별 SKIP.
- **오프라인 배치 사용법**: [offline_batch_processing.md](../batch/offline_batch_processing.md) — 트리거, ingestion, 단일 레스토랑, enqueue 예시.
- **DeepFM 학습 파이프라인**: [deepfm_training_pipeline.md](../prefect/deepfm_training_pipeline.md) — Prefect flow, 실행·배포. [deepfm_tasteam.md](../../deepfm_training/deepfm_tasteam.md) — tasteam 서비스 관점·랭킹·피처 설계.
