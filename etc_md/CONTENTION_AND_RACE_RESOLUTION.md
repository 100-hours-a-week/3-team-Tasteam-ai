# 경합·레이스 해결 정리

프로젝트에서 **경합(contention)·레이스(race condition) 해결**이 어떻게 다뤄지는지, 관련 문서와 코드를 한곳에 정리한 문서입니다.  
내용은 여러 문서에 나뉘어 있으므로, 이 문서는 **인덱스·요약** 역할을 합니다.

---

## 1. 개요

경합/레이스는 **세 가지 관점**으로 나뉘어 설명·구현되어 있습니다.

| 관점 | 대상 | 목적 | 상세 문서 |
|------|------|------|------------|
| **요청 단위** | 같은 레스토랑·같은 분석 타입의 동시 실행 | 중복 실행 방지 → 409 반환 | §2 |
| **모델/초기화·추론** | HF·VectorSearch 등 싱글톤 초기화·동시 추론 | 초기화 race 방지, (선택) 추론 동시성 제한 | §3 |
| **램프업·캐시** | 동시 로딩·캐시 접근 | NoSuchFile·중복 로드 방지 | §4 |

---

## 2. 요청 단위 경합 — Redis 락 (중복 실행 방지)

### 2.1 목적

동일한 `restaurant_id` + 동일한 `analysis_type`(summary / sentiment / comparison)에 대해 **동시에 한 번만** 실행되도록 합니다.  
같은 음식점에 대한 요약/감성/비교가 **동시에 두 번 이상** 돌지 않게 하는 것이 목표입니다.

### 2.2 방식

| 항목 | 내용 |
|------|------|
| **수단** | Redis 분산 락 |
| **키** | `restaurant_id` + `analysis_type` 조합 |
| **TTL** | 3600초(1시간) — 코드 내 고정 |
| **획득** | `src/cache.py`의 `acquire_lock(restaurant_id, analysis_type, ttl=3600)` (context manager) |
| **원자적 설정** | Redis `SET NX EX` 사용 |
| **실패 시** | `RuntimeError`("중복 실행 방지" 메시지) → API에서 **409 Conflict** 반환 |
| **Redis 미연결** | 락 없이 진행 (개발 환경 지원) |

### 2.3 적용 엔드포인트

- `POST /api/v1/sentiment/analyze`, `POST /api/v1/sentiment/analyze/batch` (레스토랑별)
- `POST /api/v1/llm/summarize`, `POST /api/v1/llm/summarize/batch` (레스토랑별)
- `POST /api/v1/llm/comparison`, `POST /api/v1/llm/comparison/batch` (레스토랑별)

### 2.4 참고 문서

- **PIPELINE/PIPELINE_OVERVIEW.md** — §2.1 락 (Redis)
- **etc_md/WHAT_IS_REDIS.md** — Redis 분산 락 정의·캐시와 구분
- **etc_md/OPERATION_STRATEGY.md** — Redis 락 구현·409 반환
- **etc_md/PRODUCTION_ISSUES_AND_IMPROVEMENTS.md** — Redis 락 구현 완료 내역
- **LLM_SERVICE_STEP/PRODUCTION_INFRASTRUCTURE.md** — 레이어 3: Redis 락(동시 중복 차단)

---

## 3. 모델/초기화·추론 경합 — HF·VectorSearch

### 3.1 목적

- **초기화 race**: 여러 스레드/코루틴이 동시에 싱글톤(HF pipeline, VectorSearch 등)을 생성·로드하지 않도록 함.
- **추론 동시성**: (선택) 같은 파이프라인 인스턴스에 대한 동시 호출을 제한해 불안정·크래시를 줄임.

### 3.2 HF Sentiment 파이프라인

| 대상 | 방식 | 비고 |
|------|------|------|
| **파이프라인 초기화 race** | `threading.Lock` + **Double-Checked Locking(DCL)** | lock 밖·안에서 각각 한 번 체크 후 생성 |
| **이벤트 루프 블로킹** | `SENTIMENT_CLASSIFIER_USE_THREAD=true` → `asyncio.to_thread`로 HF 호출 | 블로킹 격리 |
| **캐시/로딩 race** | warm-up + 캐시 경로 고정 권장 | 문서에 권장, 구현 여부는 환경별 확인 |
| **동시 추론** | 별도 lock 없음 | 필요 시 inference lock·세마포어 추가 권장 |

- 구현: `src/sentiment_analysis.py` — `_shared_lock`, `_get_sentiment_pipeline()`.

### 3.3 VectorSearch(임베딩) 초기화

- **목적**: 여러 요청이 동시에 VectorSearch/임베딩 모델을 초기화·로드하지 않도록 함.
- **방식**: Sentiment와 동일하게 **싱글톤 + DCL** 적용 (dependencies 등에서 초기화 시 락 사용).
- **캐시**: `EMBEDDING_CACHE_DIR` 등으로 캐시 경로를 `/tmp`가 아닌 고정 경로 사용 권장.

### 3.4 참고 문서

- **hf_model_race_condition.md** (프로젝트 루트) — HF 초기화 DCL, 추론 동시 접근, 권장 사항 요약
- **PIPELINE/PIPELINE_OVERVIEW.md** — warm-up·의존성

---

## 4. 램프업·캐시·VectorSearch 경합 (중복 로드, NoSuchFile)

### 4.1 목적

- ramp-up 또는 트래픽 급증 시 **여러 요청이 동시에** 모델/임베딩 로드를 트리거하는 상황을 줄임.
- 캐시 다운로드·쓰기가 끝나기 전에 접근하거나, **동일 리소스를 동시에 초기화**해서 NoSuchFile·크래시가 나는 것을 방지.

### 4.2 권장/적용 사항

| 항목 | 내용 |
|------|------|
| **모델 warm-up** | 서버 시작 시(또는 lifespan) 모델·임베딩 preload. 첫 요청에서 로딩하지 않음. |
| **캐시 경로** | ONNX/HF 캐시를 컨테이너 볼륨 등 **고정 경로**로 두어 동시 접근·휘발 위험 감소. |
| **VectorSearch DCL** | Sentiment와 동일하게 초기화 구간에 **동시 진입을 막는 락** 적용. |
| **Readiness** | warm-up 완료 후 `/ready` 200 반환. 미완료 시 503. |

### 4.3 참고 문서

- **pipe_anal_ex.md/ramp-up-logs-analysis.md** — ramp-up·캐시 race, warm-up·캐시 경로 권장
- **pipe_anal_ex.md/new_sync_async_by_logs.md** — new_sync/new_async에서의 동시 초기화·중복 로드, VectorSearch DCL·warm-up 반영
- **pipe_anal_ex.md/why_only_old_sync_vector_upload_and_142.md** — 3개 포트 동시 요청 시 동작 차이

---

## 5. 요약 표

| 구분 | 경합 유형 | 해결/권장 방식 | 코드/설정 |
|------|-----------|----------------|-----------|
| **요청 단위** | 동일 (restaurant_id, analysis_type) 동시 실행 | Redis 락, 실패 시 409 | `src/cache.py` `acquire_lock` |
| **HF 초기화** | 파이프라인 싱글톤 동시 생성 | DCL (`threading.Lock`) | `sentiment_analysis.py` `_shared_lock` |
| **HF 추론** | 같은 pipe 동시 호출 | (현재) 제한 없음 / (권장) inference lock·세마포어 | Config: `SENTIMENT_CLASSIFIER_USE_THREAD` |
| **VectorSearch 초기화** | 임베딩/모델 동시 로드 | DCL + warm-up + 캐시 경로 고정 | `dependencies.py`, `main.py` lifespan |
| **캐시/로딩** | 다운로드·캐시 미완료 시 접근 | warm-up, 캐시 경로 볼륨·고정 | `EMBEDDING_CACHE_DIR`, warm-up |

---

## 6. 관련 문서 목록 (경로)

- **PIPELINE/PIPELINE_OVERVIEW.md** — §2.1 락, warm-up
- **etc_md/WHAT_IS_REDIS.md** — Redis 분산 락
- **etc_md/OPERATION_STRATEGY.md** — Redis 락 구현
- **etc_md/PRODUCTION_ISSUES_AND_IMPROVEMENTS.md** — Redis 락·409
- **LLM_SERVICE_STEP/PRODUCTION_INFRASTRUCTURE.md** — 레이어 3 Redis 락
- **hf_model_race_condition.md** — HF race·DCL·추론 동시성
- **pipe_anal_ex.md/ramp-up-logs-analysis.md** — ramp-up·캐시 race
- **pipe_anal_ex.md/new_sync_async_by_logs.md** — VectorSearch DCL·warm-up

이 문서는 위 문서들의 요약·인덱스이며, 상세는 각 문서를 참고하면 됩니다.
