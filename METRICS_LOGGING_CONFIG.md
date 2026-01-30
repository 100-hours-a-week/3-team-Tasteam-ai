# 메트릭·로그 수집 Config 가이드

메트릭 및 구조화 로그 수집은 **상시가 아니라 환경 변수(Config)로 켜고 끕니다.**  
기본값은 **끔(false)** 이며, 수집이 필요할 때만 `METRICS_AND_LOGGING_ENABLE=true`로 활성화합니다.

---

## 1. 개요

| 구분 | 설명 |
|------|------|
| **수집 대상** | API 처리 메트릭(처리 시간, 토큰 수, 배치 크기 등), vLLM 메트릭(TTFT, TPS 등), 구조화 디버그 로그 |
| **저장소** | 로그 파일(`METRICS_LOG_DIR`/debug.log), SQLite(`METRICS_DB_PATH`) |
| **제어 방식** | `METRICS_AND_LOGGING_ENABLE`로 전체 on/off, 활성화 시 `METRICS_ENABLE_LOGGING` / `METRICS_ENABLE_DB`로 로그/DB 저장 여부 선택 |

---

## 2. 환경 변수

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `METRICS_AND_LOGGING_ENABLE` | `false` | **마스터 스위치.** `true`일 때만 메트릭·구조화 로그 수집. `false`면 수집 없음(no-op, 파일/DB 미생성) |
| `METRICS_ENABLE_LOGGING` | `true` | 수집 활성화 시 **로그 파일** 저장 여부 (`METRICS_AND_LOGGING_ENABLE=true`일 때만 적용) |
| `METRICS_ENABLE_DB` | `true` | 수집 활성화 시 **SQLite** 저장 여부 (`METRICS_AND_LOGGING_ENABLE=true`일 때만 적용) |
| `METRICS_DB_PATH` | `metrics.db` | SQLite 파일 경로 |
| `METRICS_LOG_DIR` | `logs` | 구조화 로그 디렉토리 (예: `logs/debug.log`) |

---

## 3. 동작 방식

### 3.1 수집 끔 (기본)

- `METRICS_AND_LOGGING_ENABLE`를 설정하지 않거나 `false`로 두면 수집이 **완전히 비활성화**됩니다.
- `MetricsCollector`는 `enable_logging=False`, `enable_db=False`로 생성되어:
  - 로그 파일을 만들지 않고
  - SQLite에 쓰지 않으며
  - `collect_metrics()` / `collect_vllm_metrics()` 호출은 no-op으로 동작합니다.

### 3.2 수집 켬

- `METRICS_AND_LOGGING_ENABLE=true`로 설정하면 수집이 활성화됩니다.
- 이때 다음이 적용됩니다.
  - `METRICS_ENABLE_LOGGING=true` → 구조화 로그를 `METRICS_LOG_DIR`(기본 `logs`)에 기록.
  - `METRICS_ENABLE_DB=true` → 메트릭을 `METRICS_DB_PATH`(기본 `metrics.db`) SQLite에 저장.
- 로그만 쓰고 DB는 쓰지 않으려면 `METRICS_ENABLE_DB=false`, DB만 쓰고 로그는 쓰지 않으려면 `METRICS_ENABLE_LOGGING=false`로 설정하면 됩니다.

### 3.3 SKIP 로직과의 관계

- SKIP 로직(같은 레스토랑/분류에 대한 최근 성공 실행 여부)은 `metrics_db`의 `get_last_success_at()` / `should_skip_analysis()`를 사용합니다.
- `METRICS_AND_LOGGING_ENABLE=false`이면 `metrics_db`가 `None`이 되며, 라우터에서는 `if metrics.metrics_db`로 감싸져 있어 SKIP 분기는 실행되지 않고 항상 분석을 수행합니다.
- SKIP을 쓰려면 수집을 켜고 `METRICS_ENABLE_DB=true`로 두어야 합니다.

---

## 4. 코드 위치

| 역할 | 파일 | 설명 |
|------|------|------|
| 설정 정의 | `src/config.py` | `METRICS_AND_LOGGING_ENABLE`, `METRICS_ENABLE_LOGGING`, `METRICS_ENABLE_DB` 등 |
| 수집기 주입 | `src/api/dependencies.py` | `get_metrics_collector()`에서 마스터 스위치가 꺼져 있으면 `enable_logging=False`, `enable_db=False`로 생성 |
| 수집·저장 | `src/metrics_collector.py` | `MetricsCollector`: 로그 파일(`StructuredLogger`) 및 SQLite(`MetricsDB`) 연동 |
| 로그 기록 | `src/logger_config.py` | `StructuredLogger`: 구조화 JSON 로그 (디렉토리/파일 생성은 수집 활성화 시에만) |

---

## 5. .env 예시

```bash
# 수집 끔 (기본) — 설정 없음 또는:
# METRICS_AND_LOGGING_ENABLE=false

# 수집 켬 (로그 + DB 모두)
METRICS_AND_LOGGING_ENABLE=true
METRICS_ENABLE_LOGGING=true
METRICS_ENABLE_DB=true
METRICS_DB_PATH=metrics.db
METRICS_LOG_DIR=logs

# 수집 켬 (로그만, DB 없음)
# METRICS_AND_LOGGING_ENABLE=true
# METRICS_ENABLE_LOGGING=true
# METRICS_ENABLE_DB=false
```

---

## 6. 관련 문서

- **`etc_md/METRICS.md`** — 수집 항목, 집계 지표, Goodput 등 메트릭 상세
- **`.env.example`** — 메트릭/로그 관련 환경 변수 예시
- **`CPU_MONITOR.md`** — CPU 실시간 모니터링(별도 Config: `CPU_MONITOR_ENABLE`)
