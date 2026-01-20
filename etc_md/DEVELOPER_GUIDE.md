# 개발자 가이드 (Developer Guide)

## 목차
1. [디버그 모드 개요](#디버그-모드-개요)
2. [디버그 모드 활성화](#디버그-모드-활성화)
3. [응답 모델 차이](#응답-모델-차이)
4. [메트릭 수집 시스템](#메트릭-수집-시스템)
5. [성능 분석](#성능-분석)
6. [환경 변수 설정](#환경-변수-설정)
7. [사용 예시](#사용-예시)

---

## 디버그 모드 개요

디버그 모드는 내부 개발자 및 운영팀을 위한 기능으로, API 응답에 상세한 디버그 정보를 포함하고 모든 메트릭을 수집합니다.

### 주요 특징

- **외부 사용자 모드 (기본)**: 최소 필드만 반환 (`DisplayResponse`)
- **디버그 모드**: 전체 필드 + 디버그 정보 반환 (`Response` + `DebugInfo`)
- **자동 메트릭 수집**: 모든 요청에 대해 성능 메트릭 자동 수집
- **이중 저장**: 로그 파일 (모든 정보) + SQLite (중요 메트릭만)

### 지원 엔드포인트

- `POST /api/v1/sentiment/analyze` (감성 분석)
- `POST /api/v1/llm/summarize` (리뷰 요약)
- `POST /api/v1/llm/extract/strengths` (강점 추출)

---

## 디버그 모드 활성화

디버그 모드는 다음 3가지 방법으로 활성화할 수 있으며, 우선순위는 다음과 같습니다:

1. **X-Debug 헤더** (최우선)
2. **debug 쿼리 파라미터**
3. **환경 변수** (`DEBUG_MODE`)

### 1. X-Debug 헤더 사용

```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -H "X-Debug: true" \
  -d '{"restaurant_id": 1, "reviews": [...]}'
```

헤더 값은 대소문자 구분 없이 `true`, `1`, `yes` 중 하나면 활성화됩니다.

### 2. 쿼리 파라미터 사용

```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze?debug=true" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "reviews": [...]}'
```

### 3. 환경 변수 사용

```bash
export DEBUG_MODE=true
python app.py
```

모든 요청에 대해 디버그 모드가 활성화됩니다.

---

## 응답 모델 차이

### 일반 모드 (외부 사용자)

최소 필드만 반환하여 API 응답 크기를 최소화합니다.

#### 감성 분석 (`SentimentAnalysisDisplayResponse`)

```json
{
  "positive_ratio": 60,
  "negative_ratio": 40
}
```

#### 리뷰 요약 (`SummaryDisplayResponse`)

```json
{
  "overall_summary": "가츠동과 빠른 회전이 장점인 반면, 음식이 다소 짜고 일부 메뉴는 만족스럽지 않다."
}
```

#### 강점 추출 (`StrengthDisplayResponse`)

```json
{
  "target_restaurant_strength": "타겟 레스토랑이 다른 음식점들에 비해 가지는 장점: 음식 맛과 서비스 품질이 뛰어남"
}
```

### 디버그 모드 (내부 개발자)

전체 필드 + 디버그 정보를 반환합니다.

#### 감성 분석 (`SentimentAnalysisResponse` + `DebugInfo`)

```json
{
  "restaurant_id": 1,
  "positive_count": 3,
  "negative_count": 2,
  "total_count": 5,
  "positive_ratio": 60,
  "negative_ratio": 40,
  "debug": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time_ms": 1234.56,
    "tokens_used": 1500,
    "model_version": "Qwen/Qwen2.5-7B-Instruct",
    "warnings": []
  }
}
```

#### 리뷰 요약 (`SummaryResponse` + `DebugInfo`)

```json
{
  "restaurant_id": "res_1234",
  "positive_summary": "가츠동이 괜찮고, 웨이팅이 길지 않고 회전이 빨라 편리하다.",
  "negative_summary": "음식이 짜고 다른 메뉴는 애매하며 점심시간에 붐빈다.",
  "overall_summary": "가츠동과 빠른 회전이 장점인 반면, 음식이 다소 짜고 일부 메뉴는 만족스럽지 않다.",
  "positive_reviews": [...],
  "negative_reviews": [...],
  "positive_count": 3,
  "negative_count": 2,
  "debug": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time_ms": 2345.67,
    "tokens_used": 3200,
    "model_version": "Qwen/Qwen2.5-7B-Instruct",
    "warnings": []
  }
}
```

#### 강점 추출 (`StrengthResponse` + `DebugInfo`)

```json
{
  "restaurant_id": 1,
  "comparison_summary": "비교 대상 레스토랑들의 긍정 리뷰 요약",
  "target_restaurant_strength": "타겟 레스토랑이 다른 음식점들에 비해 가지는 장점: 음식 맛과 서비스 품질이 뛰어남",
  "target_reviews": [...],
  "comparison_reviews": [...],
  "target_count": 1,
  "comparison_count": 1,
  "debug": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time_ms": 3456.78,
    "tokens_used": 4500,
    "model_version": "Qwen/Qwen2.5-7B-Instruct",
    "warnings": []
  }
}
```

### DebugInfo 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `request_id` | string | 요청 ID (UUID) |
| `processing_time_ms` | float | 처리 시간 (밀리초) |
| `tokens_used` | int | 사용된 토큰 수 |
| `model_version` | string | 모델 버전 |
| `warnings` | array[string] | 경고 메시지 리스트 |

---

## 메트릭 수집 시스템

모든 API 요청에 대해 자동으로 메트릭을 수집하며, **이중 저장 방식**을 사용합니다:

1. **로그 파일** (`logs/debug.log`): 모든 디버그 정보를 JSON 형태로 저장
2. **SQLite** (`metrics.db`): 중요한 메트릭만 구조화하여 저장

**주의사항:**
- 메트릭에는 `restaurant_id`만 저장되며, 레스토랑 이름 등 메타데이터는 저장되지 않습니다.
- 비즈니스 데이터 분석 (레스토랑 이름 포함)은 RDB/NoSQL에서 JOIN하여 수행해야 합니다.
- 이 API의 메트릭은 성능/운영 목적 (처리 시간, 토큰 사용량 등)입니다.

### 로그 파일 구조

각 로그 엔트리는 JSON 형태로 저장됩니다:

```json
{
  "timestamp": "2024-01-01T12:00:00.123456",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "restaurant_id": 1,
  "analysis_type": "sentiment",
  "processing_time_ms": 1234.56,
  "tokens_used": 1500,
  "batch_size": 100,
  "cache_hit": false,
  "model_version": "Qwen/Qwen2.5-7B-Instruct",
  "error_count": 0,
  "warning_count": 0
}
```

**특징:**
- 로그 파일은 자동 회전 (최대 10MB, 백업 5개)
- 모든 디버그 정보 포함
- 구조화된 JSON 형태로 저장되어 분석 용이

### SQLite 데이터베이스 구조

`analysis_metrics` 테이블:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `id` | INTEGER | 자동 증가 ID |
| `restaurant_id` | INTEGER | 레스토랑 ID |
| `analysis_type` | TEXT | 분석 타입 ('sentiment', 'summary', 'strength') |
| `model_version` | TEXT | 모델 버전 |
| `processing_time_ms` | REAL | 처리 시간 (밀리초) |
| `tokens_used` | INTEGER | 사용된 토큰 수 |
| `batch_size` | INTEGER | 배치 크기 |
| `cache_hit` | BOOLEAN | 캐시 히트 여부 |
| `error_count` | INTEGER | 에러 개수 |
| `warning_count` | INTEGER | 경고 개수 |
| `created_at` | TIMESTAMP | 생성 시간 |

**인덱스:**
- `restaurant_id`
- `analysis_type`
- `created_at`

**특징:**
- 중요한 메트릭만 저장 (성능 최적화)
- WAL 모드 활성화 (동시성 향상)
- 집계 쿼리 최적화

---

## 성능 분석

### SQLite에서 성능 통계 조회

Python 스크립트를 사용하여 성능 통계를 조회할 수 있습니다:

```python
from src.metrics_collector import MetricsCollector
from src.config import Config

# 메트릭 수집기 초기화
metrics = MetricsCollector(
    enable_logging=Config.METRICS_ENABLE_LOGGING,
    enable_db=Config.METRICS_ENABLE_DB,
    db_path=Config.METRICS_DB_PATH,
    log_dir=Config.METRICS_LOG_DIR,
)

# 최근 7일간 전체 성능 통계
stats = metrics.get_performance_stats(days=7)
print(stats)
# [
#   {
#     "analysis_type": "sentiment",
#     "avg_processing_time_ms": 1234.56,
#     "total_requests": 1000,
#     "total_tokens_used": 1500000,
#     "total_errors": 5,
#     "error_rate": 0.005
#   },
#   ...
# ]

# 특정 분석 타입만 조회
sentiment_stats = metrics.get_performance_stats(
    analysis_type="sentiment",
    days=7
)

# 리소스 정리
metrics.close()
```

### SQL 쿼리로 직접 조회

SQLite CLI 또는 Python `sqlite3` 모듈을 사용하여 직접 쿼리할 수 있습니다:

```python
import sqlite3

conn = sqlite3.connect("metrics.db")
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

# 최근 7일간 감성 분석 평균 처리 시간
cursor.execute("""
    SELECT 
        AVG(processing_time_ms) as avg_time,
        COUNT(*) as total
    FROM analysis_metrics
    WHERE analysis_type = 'sentiment'
      AND created_at >= datetime('now', '-7 days')
""")

result = cursor.fetchone()
print(f"평균 처리 시간: {result['avg_time']:.2f}ms")
print(f"총 요청 수: {result['total']}")

conn.close()
```

### 로그 파일 분석

로그 파일은 JSON Lines 형식이므로, `jq` 또는 Python으로 분석할 수 있습니다:

```bash
# 최근 100줄 조회
tail -n 100 logs/debug.log | jq '.'

# 처리 시간이 2000ms 이상인 로그 필터링
cat logs/debug.log | jq 'select(.processing_time_ms > 2000)'

# 에러가 발생한 로그만 필터링
cat logs/debug.log | jq 'select(.error_count > 0)'
```

```python
import json

# 로그 파일 읽기
with open("logs/debug.log", "r") as f:
    logs = [json.loads(line) for line in f if line.strip()]

# 처리 시간이 2000ms 이상인 로그 필터링
slow_logs = [log for log in logs if log.get("processing_time_ms", 0) > 2000]

# 평균 처리 시간 계산
avg_time = sum(log.get("processing_time_ms", 0) for log in logs) / len(logs)
print(f"평균 처리 시간: {avg_time:.2f}ms")
```

---

## 환경 변수 설정

메트릭 수집 시스템의 동작을 제어하는 환경 변수들:

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `DEBUG_MODE` | `false` | 디버그 모드 전역 활성화 |
| `METRICS_ENABLE_LOGGING` | `true` | 로그 파일 저장 활성화 |
| `METRICS_ENABLE_DB` | `true` | SQLite 저장 활성화 |
| `METRICS_DB_PATH` | `metrics.db` | SQLite 데이터베이스 경로 |
| `METRICS_LOG_DIR` | `logs` | 로그 디렉토리 |

### 설정 예시

```bash
# .env 파일
DEBUG_MODE=false
METRICS_ENABLE_LOGGING=true
METRICS_ENABLE_DB=true
METRICS_DB_PATH=metrics.db
METRICS_LOG_DIR=logs
```

### 비활성화

메트릭 수집을 완전히 비활성화하려면:

```bash
export METRICS_ENABLE_LOGGING=false
export METRICS_ENABLE_DB=false
```

---

## 사용 예시

### Python 클라이언트

```python
import requests

# 일반 모드
response = requests.post(
    "http://localhost:8000/api/v1/sentiment/analyze",
    json={"restaurant_id": 1, "reviews": [...]},
    headers={"Content-Type": "application/json"}
)
print(response.json())
# {"positive_ratio": 60, "negative_ratio": 40}

# 디버그 모드 (헤더)
response = requests.post(
    "http://localhost:8000/api/v1/sentiment/analyze",
    json={"restaurant_id": 1, "reviews": [...]},
    headers={
        "Content-Type": "application/json",
        "X-Debug": "true"
    }
)
print(response.json())
# {
#   "restaurant_id": 1,
#   "positive_count": 3,
#   "negative_count": 2,
#   "total_count": 5,
#   "positive_ratio": 60,
#   "negative_ratio": 40,
#   "debug": {
#     "request_id": "...",
#     "processing_time_ms": 1234.56,
#     "tokens_used": 1500,
#     "model_version": "Qwen/Qwen2.5-7B-Instruct",
#     "warnings": []
#   }
# }

# 디버그 모드 (쿼리 파라미터)
response = requests.post(
    "http://localhost:8000/api/v1/sentiment/analyze?debug=true",
    json={"restaurant_id": 1, "reviews": [...]},
    headers={"Content-Type": "application/json"}
)
```

### cURL

```bash
# 일반 모드
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "reviews": [...]}'

# 디버그 모드 (헤더)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -H "X-Debug: true" \
  -d '{"restaurant_id": 1, "reviews": [...]}'

# 디버그 모드 (쿼리 파라미터)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze?debug=true" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "reviews": [...]}'
```

---

## 참고 사항

### 주의사항

1. **프로덕션 환경**: 디버그 모드는 프로덕션 환경에서 외부 사용자에게 노출하지 않아야 합니다. 필요 시 내부 네트워크에서만 접근 가능하도록 제한하세요.

2. **성능 영향**: 디버그 모드는 응답 크기를 증가시키지만, 메트릭 수집 자체는 비동기적으로 수행되므로 성능 영향은 미미합니다.

3. **디스크 공간**: 로그 파일과 SQLite 데이터베이스는 시간이 지나면서 증가하므로, 주기적으로 정리하는 스크립트를 실행하는 것을 권장합니다.

### 데이터 정리

오래된 메트릭 데이터를 삭제하려면:

```python
from src.metrics_collector import MetricsCollector

metrics = MetricsCollector()
metrics.cleanup_old_data(days=90)  # 90일 이상 된 데이터 삭제
metrics.close()
```

---

## 관련 문서

- [API 명세서](API_SPECIFICATION.md): 외부 사용자를 위한 API 문서
- [아키텍처 문서](ARCHITECTURE.md): 시스템 아키텍처 및 설계 문서
- [README](README.md): 프로젝트 개요 및 사용 가이드

