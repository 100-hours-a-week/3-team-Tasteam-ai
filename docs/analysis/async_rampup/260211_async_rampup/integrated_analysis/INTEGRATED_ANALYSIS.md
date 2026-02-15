# 260211_final_rampup 통합 분석

앱 로그(`load_test_async_1000_final.log`)와 Prometheus/Grafana 메트릭(md·csv)을 묶어서 정리한 통합 분석입니다.

---

## 1. 실험 개요

| 항목 | 내용 |
|------|------|
| **대상** | new_async (FastAPI, 포트 8001) |
| **앱 로그** | `load_test_async_1000_final.log` (73,643줄) |
| **메트릭** | Grafana 패널 export(md) + CSV, Prometheus 스크래프 |
| **로그 기간 (UTC)** | 2026-02-11 06:43:25 ~ 10:24:15 (약 3시간 41분) |

---

## 2. 앱 로그 타임라인

### 2.1 기동·Warm-up

| 시점 (UTC) | 이벤트 |
|------------|--------|
| 06:43:25 | FastAPI 애플리케이션 시작 |
| 06:43:26~06:45:09 | 임베딩 모델(xenova/paraphrase-multilingual-mpnet-base-v2) 다운로드·로드 |
| 06:45:09 | Dense 벡터 모델 로드 완료, 하이브리드 컬렉션 생성 |
| 06:45:10~06:45:11 | Sparse(BM25) 모델 로드 |
| 06:45:11 | VectorSearch warm-up 완료 |
| 06:45:11~06:46:18 | Sentiment 모델(Dilwolf/Kakao_app-kr_sentiment) 로드 |
| **06:46:18** | **서비스 warm-up 완료, readiness=True** |
| 06:46:21~ | /ready 200, /metrics 스크래프, /health 200 |

**Warm-up 소요:** 약 **2분 53초** (06:43:25 → 06:46:18).

### 2.2 벡터 업로드

| 시점 (UTC) | 이벤트 |
|------------|--------|
| 06:46:45 | POST /health 200 후, 포인트 준비 시작: **리뷰 1000개, 레스토랑 50개** |
| 06:48:26 | 1000개 포인트 생성·업로드 완료, restaurant_vectors·대표 벡터 생성 |
| 06:48:27 | **POST /api/v1/vector/upload 200 OK** |

업로드 구간 약 **1분 42초**.

### 2.3 부하 구간 요청 결과 (로그 기준)

| API | 총 요청 수 | 200 OK | 500 Error |
|-----|------------|--------|------------|
| **POST /api/v1/vector/upload** | 1 | 1 | 0 |
| **POST /api/v1/sentiment/analyze/batch** | 500 | 0 | **500** |
| **POST /api/v1/llm/summarize/batch** | 500 | **500** | 0 |
| **POST /api/v1/llm/comparison/batch** | 11 (로그 상) | 11 | 0 |

- **감성 배치:** 전 구간 **전부 500** → 성공 0건. 원인은 아래 §3.
- **요약 배치:** 전부 200 OK.
- **비교 배치:** 로그에 찍힌 건 11건, 모두 200 OK (실제 부하는 클라이언트 설정에 따름).

---

## 3. 감성 배치 500 원인 (앱 로그)

첫 감성 요청 직후 반복되는 예외:

```
RuntimeError: The size of tensor a (540) must match the size of tensor b (512) at non-singleton dimension 1
```

- **의미:** BERT 기반 감성 모델(Kakao_app-kr_sentiment)의 **최대 시퀀스 길이 512**를 넘기는 입력(540 토큰)이 들어옴.
- **위치:** `transformers/models/bert/modeling_bert.py` → `embeddings + position_embeddings`.
- **결과:** sentiment batch 500건 모두 이 오류로 500 응답. **Analysis Success Rate / Completed jobs/s 는 성공 건이 0이라 Prometheus에서도 시리즈가 비었음.**

요약·비교는 정상 200이므로, **감성 입력 토큰 길이 제한(truncation/max_length)** 만 보완하면 됨.

---

## 4. Prometheus/Grafana 메트릭 요약

### 4.1 데이터가 있는 메트릭 (디렉토리 md·csv와 대응)

| 메트릭 | 쿼리/소스 | 파일 | 해석 |
|--------|------------|------|------|
| **TTFUR P95** | `llm_ttft_seconds` (P95) | TTFUR_p95.md, TTFUR P95-*.csv | **new_async - comparison** 만 시리즈 있음. 약 **5s** 구간 후 많은 NaN (comparison 호출이 일부 구간에만 존재). |
| **Worker utilization** | `app_worker_busy` | worker_utilization.md, Worker utilization-*.csv | **new_async 100%** 지속 → 부하 구간 내 워커 풀가동. |
| **CPU (API process)** | `process_cpu_seconds_total` | cpu(API_process).md, CPU-*.csv | **new_async:8001** 약 **420%~765%** (멀티코어). |
| **Mem (API process)** | `process_resident_memory_bytes` | mem(API_process).md, Mem-*.csv | 약 **4.07~4.42 GiB** 구간 유지. |
| **Backlog / Lag (queue depth)** | `app_queue_depth` | backlog_lag(queue_depth).md, Backlog _ Lag (queue depth)-*.csv | 동시 처리 중인 요청 수(in-flight). CSV 기준 724구간, 값 0 위주(유휴 시). |

### 4.2 Event loop lag — 대시보드 데이터 없음 / export 값 전부 0

- **대시보드:** Event loop lag 패널에 **실제 데이터가 없었음** (또는 유의미한 값 없음).
- **원인:** 앱이 **`event_loop_lag_seconds` 메트릭을 노출하지 않음.** Prometheus에 해당 시리즈가 없어서 집계된 데이터가 없음.
- **export (event_loop_lag.md, Event loop lag-*.csv):** Time은 시간 범위대로 채워지지만 **Value는 전부 0**으로 나옴.  
  - 대시보드 쿼리가 `event_loop_lag_seconds{job="fastapi"} or vector(0)` 이기 때문에, **실제 시리즈가 없을 때 `vector(0)`가 구간을 0으로 채움.**  
  - 따라서 “측정값이 0”이 아니라 **원본 메트릭이 없어서 0으로 채워진 것**으로 해석해야 함.
- **필요 시:** event loop lag를 쓰려면 앱에서 해당 메트릭을 측정·노출하도록 구현해야 함.

### 4.3 쿼리는 있으나 시리즈 없음 (집계된 데이터 없음)

아래는 Grafana 패널 export(md)에는 있지만 **`"series": []`** 이거나 해당 기간에 **한 건도 집계되지 않은 상태**입니다.

| 메트릭 | 쿼리 | 원인 |
|--------|------|------|
| Analysis Success Rate | `rate(analysis_requests_total{status="success"}) / rate(analysis_requests_total)` | `analysis_requests_total` 미기록 (호출부 미구현). |
| Completed jobs/s | `rate(analysis_requests_total{status="success"})` | 동일. |
| Completion time P95 | `histogram_quantile(0.95, rate(analysis_processing_time_seconds_bucket))` | `analysis_processing_time_seconds` 미기록. |

시리즈가 비어 있던 직접 원인은 **호출부 미구현**(`collect_metrics` 미호출)입니다. 또한 **감성 배치가 전부 500**이라, 설령 호출하더라도 success 카운트는 0이었을 것입니다.

---

## 5. 앱 로그와 메트릭의 대응

- **Worker 100% + CPU 420%~765%:** 앱이 부하 구간 내 내내 바쁨. 감성 500으로 실패해도 요청은 처리 시도했고, 요약 500건은 성공 처리.
- **TTFUR P95 ≈ 5s (comparison):** comparison 배치가 일부 구간에만 있어서 P95가 일부 구간에서만 값이 있고, 나머지는 NaN.
- **메모리 4.07~4.42 GiB:** 안정 구간 유지, OOM 없음.
- **Event loop lag:** 앱이 메트릭을 노출하지 않아 대시보드에 실제 데이터 없음. export의 Value 전부 0은 쿼리 `or vector(0)` 채움(§4.2).
- **Backlog/Lag (queue depth):** `app_queue_depth` export 추가됨. 동시 처리 요청 수 추이와 로그 타임라인 대조 가능.
- **감성 전부 500:** `analysis_requests_total`/`analysis_processing_time_seconds`가 비어 있는 것과 일치(성공 건 0 + 호출부 미구현 가능성).

---

## 6. 앱 종료 시점 및 원인

### 6.1 마지막 로그·상황

- **마지막 로그 (UTC 09:26:51 ~ 09:27:10):** OpenAI API **429 Too Many Requests** 반복.
  - **RPD(일일 요청):** Limit 10,000, Used 10,000.
  - **TPM(분당 토큰):** Limit 200,000, Used 200,000.
- `src.summary_pipeline - ERROR - LLM 비동기 호출 실패: Error code: 429 - rate_limit_exceeded` 다수.
- 로그는 **429 재시도 중**에서 끊김. 앱이 스스로 종료했다는 메시지나 예외 스택은 **없음**.
- **Prometheus:** 해당 구간 전후로 FastAPI 타겟이 **down**으로 기록됨 → 스크래프 실패 또는 응답 불가 상태.

정리하면, **rate limit으로 유의미한 진행이 막린 뒤 FastAPI가 down 된 상황**으로 보는 것이 맞음.

### 6.2 종료 원인에 대한 결론

- **로그만으로는 직접 원인 확정 불가.** 가능성만 정리할 수 있음:
  1. **프로세스 크래시** – 429 재시도/에러 처리 중 미처리 예외로 종료.
  2. **컨테이너 종료** – OOM, 또는 운영자가 `docker stop` 등으로 중단.
  3. **앱 멈춤(hang)** – 재시도·부하로 이벤트 루프 블로킹 → `/metrics` 무응답 → Prometheus가 down 판단.

### 6.3 사후 확인 시도 (docker inspect)

- 실험 후 컨테이너를 내리지 않은 상태에서 `docker inspect` 수행:
  - **결과:** `Status: running`, `ExitCode: 0`, `OOMKilled: false`, `Error: (empty)`.
- **해석:** 현재 컨테이너는 **재시작 정책으로 다시 올라온 인스턴스**일 가능성이 큼. `inspect`의 State는 **현재(또는 마지막) 실행만** 보여 주므로, **당시 down을 일으킨 실행의 ExitCode/OOMKilled는 이미 소실된 상태**. 따라서 이번 down의 원인을 이 방법으로는 더 이상 복원할 수 없음.

### 6.4 향후 종료 원인 파악을 위한 권장

| 조치 | 목적 |
|------|------|
| **종료 시 로그** | `atexit` 또는 SIGTERM/SIGINT 핸들러에서 shutdown·마지막 예외 로그 출력. → **구현됨:** `src/api/main.py` (excepthook + atexit + signal handler, shutdown 로그 및 `traceback.print_exception`). |
| **Exit code 보관** | down 직후 `docker inspect --format '{{.State.ExitCode}} {{.State.OOMKilled}}' <container>` 결과를 실험 로그에 기록 (137 → OOM 등 추정 가능). |
| **재시작 정책** | 원인 분석이 필요할 때는 재시작 정책을 잠시 끄고 한 번 죽었을 때 inspect로 확인. |
| **OOM/이벤트** | 실험 후 `dmesg \| grep -i oom`, Docker/K8s 이벤트 로그 확인. |

---

## 7. 결론 및 권장 사항

| 구분 | 요약 |
|------|------|
| **성공한 부분** | 벡터 업로드(1건), 요약 배치(500건 200 OK), 비교 배치(로그 상 11건 200 OK). Worker 100%, CPU·메모리·TTFUR(comparison)·**Backlog/Lag(queue depth)** 메트릭 수집·export 됨. |
| **실패한 부분** | **감성 배치 500건 전부 500** → BERT max length 512 초과(540 토큰). |
| **앱 종료** | 실험 후반 OpenAI 429 rate limit → Prometheus에서 FastAPI down. 로그·inspect로는 직접 원인 확정 불가(§6 참고). |
| **메트릭 공백** | Analysis Success Rate, Completed jobs/s, Completion time P95 → 호출부 미기록. **Event loop lag** → 앱이 `event_loop_lag_seconds` 미노출, 대시보드 실제 데이터 없음(export Value 전부 0은 `or vector(0)` 채움). |

**권장 사항**

1. **감성 파이프라인:** 입력 토큰을 **max_length=512 이하로 truncate** (또는 모델/토크나이저 설정으로 512 고정). 재현 테스트 후 로그에서 500 사라지는지 확인.
2. **E2E 메트릭:** `collect_metrics()` 를 분석 완료 시점에 호출하도록 구현해, Success Rate / Completed jobs/s / Completion time P95 가 쌓이게 하기. (구현 반영됨.)
3. **Backlog/Lag (queue depth):** `backlog_lag(queue_depth).md`, `Backlog _ Lag (queue depth)-*.csv` 로 export 추가됨. 동시 처리량·로그 타임라인 대조 분석에 활용 가능.
4. **종료 원인 파악:** 앱에는 atexit/signal shutdown 로그 및 미처리 예외 출력이 구현됨(`src/api/main.py`). 다음 실험부터 exit code 기록·OOM·이벤트 확인(§6.4)까지 적용 시 down 원인 추적 가능.

---

*문서 생성: 260211_final_rampup 디렉토리 앱 로그 + Prometheus/Grafana md·csv 기준 통합 분석.*
