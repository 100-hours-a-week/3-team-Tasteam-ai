## Prometheus + Grafana 사용 가이드 (현재 수집 범위 포함)

이 문서는 이 레포의 `docker-compose.yml` 기준으로 **Prometheus + Grafana**를 띄우고, **현재 무엇을 수집/시각화할 수 있는지**를 정리합니다.

---

## 1. 빠른 시작 (로컬)

### 1.1 실행

프로젝트 루트에서:

```bash
docker compose up -d
```

### 1.1-1 이미지/컨테이너 다시 띄우기 (재빌드 상황)

프로젝트 루트에서:

```bash
docker compose build api # api 빌드
docker compose up -d api # api 컨테이너 띄우기
```

### 1.2 접속 URL

| 구성요소 | URL | 비고 |
|---|---|---|
| **FastAPI** | `http://localhost:8001` | API 서버 |
| **FastAPI Metrics** | `http://localhost:8001/metrics` | Prometheus 스크래핑 대상 |
| **Job Metrics (jobmgr)** | `http://localhost:1041/metrics` | Prometheus 스크래핑 대상 |
| **Node Exporter** | `http://localhost:9100/metrics` | 호스트/컨테이너 OS 메트릭 |
| **Prometheus UI** | `http://localhost:9090` | Targets 확인 가능 |
| **Grafana UI** | `http://localhost:3000` | 익명 Viewer 허용(설정) |

### 1.3 Docker에서 API(요약/비교/감성)가 실패할 때

로컬에서는 통과하는데 **Docker Compose로 띄우면** 요약/비교/감성 API가 500 또는 타임아웃이 나는 경우, 다음을 확인하세요.

- **`OPENAI_API_KEY`**  
  컨테이너에는 호스트의 셸 환경변수가 그대로 넘어가지 않습니다.  
  프로젝트 루트에 **`.env`** 파일을 만들고 `OPENAI_API_KEY=sk-...` 를 넣어 두면, `docker-compose.yml`의 `api` 서비스가 `env_file: .env` 로 읽어 갑니다.
- **`.env` 예시**  
  `OPENAI_API_KEY=sk-...`  
  `LLM_PROVIDER=openai`  
  `OPENAI_MODEL=gpt-4o-mini`
- **Qdrant**  
  현재 Compose 기본값은 `QDRANT_URL=:memory:` 입니다. 컨테이너를 내렸다 올리면 데이터가 비어 있으므로, 테스트 시 **Qdrant 업로드 → 감성/요약/비교/벡터검색** 순서로 한 번에 실행해야 합니다.
- **비교(comparison) 500 / JAVA_GATEWAY_EXITED**  
  비교 파이프라인은 PySpark(Java JVM)를 사용합니다. Docker 이미지에는 **OpenJDK 17**이 포함되어 있어 기본적으로 Spark를 사용합니다.  
  이미지 크기를 줄이고 Spark를 쓰지 않으려면 `api` 서비스에 **`DISABLE_SPARK: "true"`** 를 넣으면, 비교는 Kiwi만 사용하는 Python 경로로 동작합니다.

---

## 2. 구성 파일/역할

### 2.1 `docker-compose.yml`

서비스 구성:

- **`api`**: FastAPI 서버 (포트 8001)
- **`jobmgr`**: `prometheus.py` 기반의 간단한 Prometheus exporter (포트 1041)
- **`node-exporter`**: Node Exporter (포트 9100, OS/호스트 메트릭)
- **`prometheus`**: Prometheus 서버 (포트 9090)
- **`grafana`**: Grafana (포트 3000)

### 2.2 `configs/prometheus/prometheus.yml`

스크래핑 타겟:

- `jobmgr:1041` (job_name: `jobmgr`)
- `api:8001/metrics` (job_name: `fastapi`, metrics_path: `/metrics`)
- `node-exporter:9100` (job_name: `node`)

### 2.3 `configs/grafana/*`

- **datasource**: `configs/grafana/datasources/datasource.yml`
  - Prometheus datasource UID: `prometheus`
- **dashboards**: `configs/grafana/dashboards/`
  - `prometheus_overview.json` (기본 대시보드 예시)
- **ini**: `configs/grafana/ini/grafana.ini`
  - 익명 접속 Viewer 허용

---

## 3. 현재 “수집 가능한 메트릭” 범위

중요: 여기서 “수집 가능”은 **Prometheus가 주기적으로 스크래핑하여 시각화 가능한 범위**를 의미합니다.

### 3.1 FastAPI (`/metrics`) — HTTP 레벨 메트릭

FastAPI는 `prometheus-fastapi-instrumentator`의 기본 메트릭을 노출합니다.

- **`http_requests_total`** (Counter)  
  - 라벨: `handler`, `status`, `method`
- **`http_request_size_bytes`** (Summary)  
  - 라벨: `handler`
- **`http_response_size_bytes`** (Summary)  
  - 라벨: `handler`
- **`http_request_duration_seconds`** (Histogram)  
  - 라벨: `handler`, `method` (bucket 수를 줄여 카디널리티를 낮춘 기본 구성)
- **`http_request_duration_highr_seconds`** (Histogram)  
  - 라벨 없음(기본값), bucket 수가 많음(상세)

즉, 현재 Grafana/Prometheus로 바로 볼 수 있는 것은:

- 엔드포인트별 요청 수/응답코드 분포
- 엔드포인트별 지연시간 분포(P50/P95 같은 계산은 Grafana에서 쿼리로)
- 요청/응답 바이트 규모

### 3.2 jobmgr (`prometheus.py` /metrics) — 프로세스/파이썬 런타임 메트릭 + 커스텀 Info

`prometheus_client` 기본 레지스트리에서 다음이 노출됩니다(대표 예시):

- `process_cpu_seconds_total`
- `process_resident_memory_bytes`
- `process_virtual_memory_bytes`
- `python_gc_*`

추가로 이 레포의 `prometheus.py`는:

- `metrics_agent_info` (Info)  
  - 예: `{service="jobmgr", version="1.0"}`

### 3.3 Node Exporter (`node-exporter:9100/metrics`) — OS/호스트 메트릭

- **`node_*`**: CPU, 메모리, 디스크, 네트워크 등 (표준 node_exporter 메트릭)
- Docker Compose 환경에서는 **컨테이너** 기준 메트릭이 수집됨. Linux 호스트 전체를 보려면 호스트에 node_exporter를 직접 설치하거나 `network_mode: host` 등 추가 설정이 필요할 수 있음.

### 3.4 MetricsCollector → Prometheus (FastAPI `/metrics`에 포함)

`MetricsCollector.collect_metrics()` / `collect_vllm_metrics()` 호출 시 **동일 기본 레지스트리**에 갱신되므로, FastAPI의 `/metrics`에서 함께 노출됩니다. (prometheus_client 설치 시)

- **`analysis_processing_time_seconds`** (Histogram) — 라벨: `analysis_type`, `status`
- **`analysis_requests_total`** (Counter) — 라벨: `analysis_type`, `status`
- **`analysis_tokens_used_total`** (Counter) — 라벨: `analysis_type`
- **`llm_ttft_seconds`** (Histogram) — 라벨: `analysis_type`
- **`llm_tps`** (Histogram) — 라벨: `analysis_type`
- **`llm_tokens_total`** (Counter) — 라벨: `analysis_type`

즉, 감성/요약/비교 API가 호출될 때마다 위 메트릭이 갱신되며, Prometheus가 `api:8001/metrics`를 스크래핑하면 **전부 수집**됩니다.

---

## 4. 참고 사항 및 한계

### 4.1 Node Exporter (호스트 vs 컨테이너)

Node Exporter는 **docker-compose에 포함**되어 있으며 `node-exporter:9100`으로 스크래핑됩니다.

- Docker Compose 환경에서는 **컨테이너** 기준 OS 메트릭이 수집됩니다.
- Linux에서 **호스트 전체** CPU/메모리/디스크를 보려면 호스트에 node_exporter를 직접 설치하거나 `network_mode: host` 등 추가 설정을 고려하세요.

### 4.2 MetricsCollector — Prometheus 노출 완료

**파이프라인/LLM 지표**는 **이미 Prometheus에 노출**되어 있습니다. (위 **§3.4** 참고)

- `analysis_processing_time_seconds`, `analysis_requests_total`, `analysis_tokens_used_total`
- `llm_ttft_seconds`, `llm_tps`, `llm_tokens_total`

동일 데이터가 SQLite(`metrics.db`)와 로그에도 저장됩니다.  
`batch_size`, `cache_hit`, `error_count`/`warning_count` 등 **일부 상세 필드**는 아직 Prometheus 라벨로만 노출되지 않을 수 있으며, 필요 시 `metrics_collector.py`에서 추가 갱신하면 됩니다.

---

## 5. 확인/디버깅 방법

### 5.1 Prometheus Targets 확인

Prometheus UI:

- `http://localhost:9090/targets`

여기에서 `jobmgr`, `fastapi`, `node`가 **UP**인지 확인합니다.

### 5.2 수동 확인(curl)

```bash
# FastAPI 메트릭 노출 확인
curl -s "http://localhost:8001/metrics" | head -n 30

# jobmgr 메트릭 노출 확인
curl -s "http://localhost:1041/metrics" | head -n 30

# Node Exporter 메트릭 확인
curl -s "http://localhost:9100/metrics" | head -n 30
```

---

## 6. Grafana에서 바로 보기 (기본 대시보드)

Grafana 접속:

- `http://localhost:3000`

Provisioning으로 들어간 기본 대시보드:

- **Folder**: `Monitor`
- **Dashboard**: `Prometheus Overview`

현재 포함된 패널:

- **Target Up Status** — `up` (jobmgr, fastapi, node 타겟 Up 여부)
- **Process Memory (jobmgr)** — jobmgr 프로세스 RSS
- **FastAPI Request Rate** — `rate(http_requests_total{job="fastapi"})` (엔드포인트·메서드별)
- **FastAPI Request Duration (P95)** — `http_request_duration_seconds` P95
- **Node Exporter Memory** — `node_memory_MemAvailable_bytes` / `MemFree_bytes`
- **Process Memory (API & jobmgr)** — fastapi·jobmgr 프로세스 RSS
- **Analysis Requests** — `analysis_requests_total{job="fastapi"}` (sentiment/summary/comparison 호출 수)

---

## 7. 자주 묻는 질문

### Q1. “FastAPI 내부 처리시간(processing_time_ms)”, “tokens_used”, “TTFT”도 Grafana로 볼 수 있나요?

**예.** `MetricsCollector`가 Prometheus 기본 레지스트리에 갱신하므로, FastAPI `/metrics`를 스크래핑하는 Prometheus·Grafana에서 **바로 볼 수 있습니다.**  
노출되는 메트릭 이름은 **§3.4**를 참고하세요 (`analysis_processing_time_seconds`, `analysis_tokens_used_total`, `llm_ttft_seconds` 등).

### Q2. `tasteam_app` API 호출이 없는데도 메트릭이 보이나요?

Prometheus는 **스크래핑 대상이 살아있으면(up=1)** 메트릭을 수집합니다.
다만 `http_requests_total` 같은 값은 API 호출이 있어야 증가합니다.

