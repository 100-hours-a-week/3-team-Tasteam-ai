# 정량 측정 가이드

본 문서는 프로젝트의 성능, 품질, 비용 등 정량적 지표를 측정하고 분석하기 위한 가이드입니다. `MODEL_SELECTION.md`에 명시된 모델 후보군과 현재 코드베이스의 측정 도구를 활용하여 체계적인 벤치마크를 수행합니다.

---

## 목차

1. [개요](#개요)
2. [정량 지표 필요 항목](#정량-지표-필요-항목)
3. [현재 코드베이스 기준 측정 절차](#현재-코드베이스-기준-측정-절차)
4. [모델별 비교 측정 방법](#모델별-비교-측정-방법)
5. [측정 도구 및 사용법](#측정-도구-및-사용법)
6. [측정 결과 분석](#측정-결과-분석)
7. [지속적 측정 전략](#지속적-측정-전략)

---

## 개요

### 측정 목적

1. **성능 검증**: 설계 문서의 "예상 효과"를 실제 데이터로 검증
2. **모델 비교**: `MODEL_SELECTION.md`에 명시된 모델 후보군의 성능 비교
3. **최적화 근거**: 데이터 기반 최적화 결정
4. **SLA 준수**: TTFT < 2초 등 SLA 달성 여부 모니터링

### 측정 범위

- **성능 지표**: 처리 시간, 처리량, 지연 시간
- **리소스 사용량**: GPU, CPU, 메모리
- **비용 지표**: 토큰 사용량, 월간 비용
- **품질 지표**: 정확도, 에러율
- **확장성 지표**: 가용성, 확장 효율성

---

## 정량 지표 필요 항목

### 1. 성능 지표

#### 1.1 처리 시간 (Latency)

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **감성 분석 평균 처리 시간** | 1.2초 | `scripts/benchmark.py` | 높음 |
| **감성 분석 P95 처리 시간** | 3.2초 | `scripts/benchmark.py` | 높음 |
| **감성 분석 P99 처리 시간** | 6.8초 | `scripts/benchmark.py` | 중간 |
| **리뷰 요약 평균 처리 시간** | 2.5초 | `scripts/benchmark.py` | 높음 |
| **리뷰 요약 P95 처리 시간** | 4.8초 | `scripts/benchmark.py` | 높음 |
| **강점 추출 평균 처리 시간** | 3.0초 | `scripts/benchmark.py` | 높음 |
| **강점 추출 P95 처리 시간** | 5.5초 | `scripts/benchmark.py` | 높음 |
| **배치 처리 시간 (10개 레스토랑)** | 5-10초 | `scripts/benchmark.py` (batch 모드) | 높음 |

**측정 절차:**
```bash
# 단일 요청 측정
python scripts/benchmark.py --endpoint /api/v1/sentiment/analyze \
  --iterations 50 --warmup 5

# 배치 요청 측정
python scripts/benchmark.py --endpoint /api/v1/sentiment/analyze/batch \
  --iterations 20 --warmup 3 --mode batch
```

#### 1.2 TTFT (Time To First Token)

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **평균 TTFT** | < 2초 (SLA) | `vllm_metrics` 테이블 | 높음 |
| **P95 TTFT** | < 3초 | `vllm_metrics` 테이블 | 높음 |
| **P99 TTFT** | < 5초 | `vllm_metrics` 테이블 | 중간 |

**측정 절차:**
```sql
-- SQLite에서 TTFT 통계 조회
SELECT 
    AVG(ttft_ms) as avg_ttft_ms,
    MIN(ttft_ms) as min_ttft_ms,
    MAX(ttft_ms) as max_ttft_ms,
    (SELECT ttft_ms FROM vllm_metrics 
     ORDER BY ttft_ms LIMIT 1 OFFSET (SELECT COUNT(*) * 0.95 FROM vllm_metrics)) as p95_ttft_ms,
    (SELECT ttft_ms FROM vllm_metrics 
     ORDER BY ttft_ms LIMIT 1 OFFSET (SELECT COUNT(*) * 0.99 FROM vllm_metrics)) as p99_ttft_ms
FROM vllm_metrics
WHERE created_at >= datetime('now', '-7 days')
AND analysis_type = 'sentiment';
```

#### 1.3 처리량 (Throughput)

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **단일 Pod 처리량** | 10-15 req/s | `scripts/benchmark.py` | 높음 |
| **Goodput (SLA 만족 처리량)** | 8-12 req/s | `GoodputTracker` | 높음 |
| **TPS (Tokens Per Second)** | 모델별 측정 | `vllm_metrics` 테이블 | 높음 |
| **TPOT (Time Per Output Token)** | 모델별 측정 | `vllm_metrics` 테이블 | 중간 |

**측정 절차:**
```python
# Python 코드로 Goodput 통계 조회
from src.metrics_collector import MetricsCollector

metrics = MetricsCollector()
goodput_stats = metrics.get_goodput_stats()
print(f"Throughput: {goodput_stats['throughput_tps']:.2f} TPS")
print(f"Goodput: {goodput_stats['goodput_tps']:.2f} TPS")
print(f"SLA Compliance: {goodput_stats['sla_compliance_rate']:.2f}%")
```

### 2. 리소스 사용량

#### 2.1 GPU 메트릭

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **평균 GPU 사용률** | 70-90% | `GPUMonitor` | 높음 |
| **피크 GPU 사용률** | 85-95% | `GPUMonitor` | 높음 |
| **GPU 메모리 사용률** | 75-92% | `GPUMonitor` | 높음 |
| **GPU 온도** | < 85°C | `GPUMonitor` | 중간 |

**측정 절차:**
```python
# Python 코드로 GPU 메트릭 수집
from scripts.gpu_monitor import GPUMonitor

monitor = GPUMonitor(device_index=0)
gpu_metrics = monitor.get_metrics()
print(f"GPU Utilization: {gpu_metrics['gpu_util_percent']:.2f}%")
print(f"Memory Usage: {gpu_metrics['memory_util_percent']:.2f}%")
```

#### 2.2 CPU/메모리

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **평균 CPU 사용률** | 65% | 시스템 모니터링 도구 | 중간 |
| **피크 CPU 사용률** | 82% | 시스템 모니터링 도구 | 중간 |
| **평균 메모리 사용률** | 72% | 시스템 모니터링 도구 | 중간 |
| **피크 메모리 사용률** | 88% | 시스템 모니터링 도구 | 중간 |

### 3. 비용 지표

#### 3.1 토큰 사용량

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **감성 분석 토큰 사용량** | 1,200 tokens/req | `analysis_metrics` 테이블 | 높음 |
| **리뷰 요약 토큰 사용량** | 모델별 측정 | `analysis_metrics` 테이블 | 높음 |
| **강점 추출 토큰 사용량** | 모델별 측정 | `analysis_metrics` 테이블 | 높음 |
| **대표 벡터 TOP-K 효과** | 60-80% 감소 | 모델별 비교 | 높음 |

**측정 절차:**
```sql
-- 토큰 사용량 통계 조회
SELECT 
    analysis_type,
    AVG(tokens_used) as avg_tokens,
    MIN(tokens_used) as min_tokens,
    MAX(tokens_used) as max_tokens,
    COUNT(*) as request_count
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY analysis_type;
```

#### 3.2 월간 비용

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **GPU Pod 비용** | $400-600/월 | RunPod 대시보드 | 중간 |
| **유휴 시간 비용 절감** | 50-70% | Watchdog 로그 분석 | 중간 |
| **토큰 기반 비용** | 모델별 측정 | 토큰 사용량 × 모델별 가격 | 중간 |

### 4. 품질 지표

#### 4.1 정확도

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **요약 품질 (BLEU Score)** | 0.82 | Ground Truth 비교 | 높음 |
| **강점 추출 정확도** | 88% | Ground Truth 비교 | 높음 |
| **감성 분석 정확도** | 모델별 측정 | Ground Truth 비교 | 높음 |

**측정 절차:**
```bash
# Ground Truth 기반 평가
python scripts/evaluate_summary.py
python scripts/evaluate_strength_extraction.py
python scripts/evaluate_sentiment_analysis.py
```

#### 4.2 에러율

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **HTTP 4xx 에러율** | < 1% | `analysis_metrics` 테이블 | 높음 |
| **HTTP 5xx 에러율** | < 0.1% | `analysis_metrics` 테이블 | 높음 |
| **성공률** | > 99% | `analysis_metrics` 테이블 | 높음 |
| **OOM 발생 빈도** | 월 0-1회 | 운영 로그 분석 | 높음 |

**측정 절차:**
```sql
-- 에러율 통계 조회
SELECT 
    analysis_type,
    COUNT(*) as total_requests,
    SUM(error_count) as total_errors,
    (SUM(error_count) * 100.0 / COUNT(*)) as error_rate_percent,
    (COUNT(*) - SUM(error_count)) * 100.0 / COUNT(*) as success_rate_percent
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY analysis_type;
```

### 5. 확장성 지표

#### 5.1 가용성

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **단일 Pod 가용성** | 95% | Health Check 로그 | 중간 |
| **다중 Pod (2개) 가용성** | 99% | Health Check 로그 | 중간 |
| **다중 Pod (5개) 가용성** | 99.9% | Health Check 로그 | 낮음 |

#### 5.2 확장 효율성

**필수 측정 항목:**

| 지표 | 목표값 | 측정 방법 | 우선순위 |
|------|--------|----------|---------|
| **로드 밸런싱 오버헤드** | < 5% | 다중 Pod 벤치마크 | 낮음 |
| **선형 확장 여부** | Pod 수에 비례 | 다중 Pod 벤치마크 | 낮음 |

---

## 현재 코드베이스 기준 측정 절차

### 1. 자동 메트릭 수집

현재 코드베이스는 다음 메트릭을 자동으로 수집합니다:

#### 1.1 `MetricsCollector`를 통한 기본 메트릭

**수집 위치:**
- `src/api/routers/sentiment.py`: 감성 분석 엔드포인트
- `src/api/routers/llm.py`: 요약 및 강점 추출 엔드포인트

**수집 항목:**
- `processing_time_ms`: 전체 처리 시간
- `tokens_used`: 사용된 토큰 수
- `batch_size`: 배치 크기
- `cache_hit`: 캐시 히트 여부
- `error_count`: 에러 개수
- `status`: "success" | "fail" | "skipped"

**저장 위치:**
- SQLite: `metrics.db` → `analysis_metrics` 테이블
- 로그 파일: `logs/debug.log` (JSON 형식)

#### 1.2 `collect_vllm_metrics()`를 통한 vLLM 상세 메트릭

**수집 위치:**
- `src/llm_utils.py`: `_generate_with_vllm()` 메서드

**수집 항목:**
- `prefill_time_ms`: Prefill 시간 (입력 처리)
- `decode_time_ms`: Decode 시간 (토큰 생성)
- `ttft_ms`: Time To First Token
- `tps`: Tokens Per Second
- `tpot_ms`: Time Per Output Token
- `n_tokens`: 생성된 토큰 수

**저장 위치:**
- SQLite: `metrics.db` → `vllm_metrics` 테이블

#### 1.3 `GoodputTracker`를 통한 SLA 기반 처리량

**수집 위치:**
- `src/metrics_collector.py`: `collect_metrics()` 메서드 내부

**수집 항목:**
- `throughput_tps`: 전체 처리량
- `goodput_tps`: SLA 만족 처리량 (TTFT < 2초)
- `sla_compliance_rate`: SLA 준수율

**저장 위치:**
- 메모리 기반 (실시간 조회)

### 2. 수동 벤치마크 측정

#### 2.1 `scripts/benchmark.py` 사용법

**기본 사용법:**
```bash
# 감성 분석 단일 요청 벤치마크
python scripts/benchmark.py \
  --endpoint /api/v1/sentiment/analyze \
  --iterations 50 \
  --warmup 5 \
  --output results/sentiment_single.json

# 감성 분석 배치 요청 벤치마크
python scripts/benchmark.py \
  --endpoint /api/v1/sentiment/analyze/batch \
  --iterations 20 \
  --warmup 3 \
  --mode batch \
  --output results/sentiment_batch.json

# 리뷰 요약 벤치마크
python scripts/benchmark.py \
  --endpoint /api/v1/llm/summarize \
  --iterations 30 \
  --warmup 3 \
  --output results/summary.json

# 강점 추출 벤치마크
python scripts/benchmark.py \
  --endpoint /api/v1/llm/extract/strengths \
  --iterations 20 \
  --warmup 3 \
  --output results/strength.json
```

**출력 형식:**
```json
{
  "num_iterations": 50,
  "success_count": 48,
  "error_count": 2,
  "latency": {
    "avg": 1.234,
    "min": 0.987,
    "max": 2.456,
    "median": 1.189,
    "p95": 2.123,
    "p99": 2.345,
    "std": 0.234
  },
  "throughput": {
    "avg": 8.12,
    "min": 4.56,
    "max": 10.23
  },
  "gpu_metrics": {
    "before": {...},
    "after": {...}
  }
}
```

#### 2.2 GPU 모니터링

**실시간 GPU 메트릭 수집:**
```python
from scripts.gpu_monitor import GPUMonitor
import time

monitor = GPUMonitor(device_index=0)

# 벤치마크 시작 전
gpu_before = monitor.get_metrics()

# 벤치마크 실행
# ... API 호출 ...

# 벤치마크 종료 후
gpu_after = monitor.get_metrics()

print(f"GPU Utilization: {gpu_after['gpu_util_percent']:.2f}%")
print(f"Memory Usage: {gpu_after['memory_util_percent']:.2f}%")
```

### 3. SQLite 쿼리를 통한 통계 분석

#### 3.1 기본 성능 통계

```sql
-- 최근 7일간 평균 처리 시간
SELECT 
    analysis_type,
    AVG(processing_time_ms) as avg_time_ms,
    MIN(processing_time_ms) as min_time_ms,
    MAX(processing_time_ms) as max_time_ms,
    COUNT(*) as request_count
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY analysis_type;
```

#### 3.2 P95/P99 처리 시간

```sql
-- P95 처리 시간 계산
SELECT 
    analysis_type,
    (SELECT processing_time_ms FROM analysis_metrics 
     WHERE analysis_type = am.analysis_type
     ORDER BY processing_time_ms 
     LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.95 AS INTEGER) FROM analysis_metrics WHERE analysis_type = am.analysis_type)
    ) as p95_time_ms,
    (SELECT processing_time_ms FROM analysis_metrics 
     WHERE analysis_type = am.analysis_type
     ORDER BY processing_time_ms 
     LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.99 AS INTEGER) FROM analysis_metrics WHERE analysis_type = am.analysis_type)
    ) as p99_time_ms
FROM analysis_metrics am
WHERE created_at >= datetime('now', '-7 days')
GROUP BY analysis_type;
```

#### 3.3 vLLM 메트릭 분석

```sql
-- Prefill vs Decode 시간 비율 분석
SELECT 
    analysis_type,
    AVG(prefill_time_ms) as avg_prefill_ms,
    AVG(decode_time_ms) as avg_decode_ms,
    AVG(prefill_time_ms) * 100.0 / (AVG(prefill_time_ms) + AVG(decode_time_ms)) as prefill_ratio_percent,
    AVG(ttft_ms) as avg_ttft_ms,
    AVG(tps) as avg_tps,
    AVG(tpot_ms) as avg_tpot_ms
FROM vllm_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY analysis_type;
```

#### 3.4 Goodput 통계

```sql
-- SLA 준수율 분석
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    SUM(CASE WHEN ttft_ms < 2000 THEN 1 ELSE 0 END) as sla_met_requests,
    SUM(n_tokens) as total_tokens,
    SUM(CASE WHEN ttft_ms < 2000 THEN n_tokens ELSE 0 END) as sla_met_tokens,
    (SUM(CASE WHEN ttft_ms < 2000 THEN n_tokens ELSE 0 END) * 100.0 / SUM(n_tokens)) as sla_compliance_rate
FROM vllm_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

---

## 모델별 비교 측정 방법

### 1. 모델 후보군 (`MODEL_SELECTION.md` 참고)

#### 1.1 Sentiment Analysis 모델

**후보 모델:**
- `leekangwon/klue-roberta-sentiment`
- `nlp04/korean_sentiment_analysis_kcelectra`
- KR-ELECTRA 백본 기반 파인튜닝 모델 (향후)

**비교 측정 항목:**
- 정확도 (Ground Truth 기반)
- 처리 시간 (평균, P95, P99)
- 토큰 사용량
- GPU 메모리 사용량

#### 1.2 Embedding 모델

**후보 모델:**
- `BAAI/bge-m3`
- `dragonkue/BGE-m3-ko`
- `upskyy/bge-m3-korean`

**비교 측정 항목:**
- 임베딩 품질 (벡터 검색 정확도)
- 임베딩 생성 시간
- GPU 메모리 사용량
- 벡터 차원 수

#### 1.3 LLM 모델

**후보 모델:**
- `Qwen2.5-7B-Instruct` (현재)
- `Llama 3.1 8B Instruct`
- `Gemma 2 9B IT`
- EEVE-Korean 계열 (한국어 특화)

**비교 측정 항목:**
- TTFT (Time To First Token)
- TPS (Tokens Per Second)
- TPOT (Time Per Output Token)
- 토큰 사용량
- GPU 메모리 사용량
- 한국어 품질 (요약, 강점 추출 정확도)

### 2. 모델별 벤치마크 절차

#### 2.1 모델 변경 방법

**설정 파일 수정:**
```python
# src/config.py 또는 환경 변수
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 기본값
# LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Llama 3.1
# LLM_MODEL = "google/gemma-2-9b-it"  # Gemma 2
```

**환경 변수로 설정:**
```bash
export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python app.py
```

#### 2.2 모델별 벤치마크 스크립트

**모델별 비교 벤치마크 실행:**
```bash
# Qwen2.5-7B-Instruct 벤치마크
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
python scripts/benchmark.py \
  --endpoint /api/v1/sentiment/analyze \
  --iterations 50 \
  --output results/qwen2.5_7b_sentiment.json

# Llama 3.1 8B 벤치마크
export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python scripts/benchmark.py \
  --endpoint /api/v1/sentiment/analyze \
  --iterations 50 \
  --output results/llama3.1_8b_sentiment.json

# Gemma 2 9B 벤치마크
export LLM_MODEL="google/gemma-2-9b-it"
python scripts/benchmark.py \
  --endpoint /api/v1/sentiment/analyze \
  --iterations 50 \
  --output results/gemma2_9b_sentiment.json
```

#### 2.3 모델별 메트릭 수집

**SQLite에서 모델별 통계 조회:**
```sql
-- 모델별 성능 비교
SELECT 
    model_version,
    analysis_type,
    AVG(processing_time_ms) as avg_time_ms,
    AVG(tokens_used) as avg_tokens,
    COUNT(*) as request_count,
    SUM(error_count) as total_errors
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY model_version, analysis_type
ORDER BY model_version, analysis_type;
```

**vLLM 메트릭 모델별 비교:**
```sql
-- 모델별 vLLM 메트릭 비교
SELECT 
    (SELECT model_version FROM analysis_metrics 
     WHERE request_id = vm.request_id LIMIT 1) as model_version,
    AVG(vm.ttft_ms) as avg_ttft_ms,
    AVG(vm.tps) as avg_tps,
    AVG(vm.tpot_ms) as avg_tpot_ms,
    AVG(vm.prefill_time_ms) as avg_prefill_ms,
    AVG(vm.decode_time_ms) as avg_decode_ms
FROM vllm_metrics vm
WHERE vm.created_at >= datetime('now', '-7 days')
GROUP BY model_version;
```

### 3. 모델별 품질 평가

#### 3.1 Ground Truth 기반 평가

**요약 품질 평가:**
```bash
# Qwen2.5-7B-Instruct
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
python scripts/evaluate_summary.py \
  --ground_truth scripts/Ground_truth_summary.json \
  --output results/qwen2.5_7b_summary_eval.json

# Llama 3.1 8B
export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
python scripts/evaluate_summary.py \
  --ground_truth scripts/Ground_truth_summary.json \
  --output results/llama3.1_8b_summary_eval.json
```

**강점 추출 평가:**
```bash
# 모델별 강점 추출 정확도 비교
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
python scripts/evaluate_strength_extraction.py \
  --ground_truth scripts/Ground_truth_strength.json \
  --output results/qwen2.5_7b_strength_eval.json
```

**감성 분석 평가:**
```bash
# 모델별 감성 분석 정확도 비교
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
python scripts/evaluate_sentiment_analysis.py \
  --ground_truth scripts/Ground_truth_sentiment.json \
  --output results/qwen2.5_7b_sentiment_eval.json
```

---

## 측정 도구 및 사용법

### 1. 자동 수집 도구

#### 1.1 `MetricsCollector`

**위치:** `src/metrics_collector.py`

**주요 메서드:**
- `collect_metrics()`: 기본 분석 메트릭 수집
- `collect_vllm_metrics()`: vLLM 상세 메트릭 수집
- `get_performance_stats()`: 성능 통계 조회
- `get_goodput_stats()`: Goodput 통계 조회

**사용 예시:**
```python
from src.metrics_collector import MetricsCollector

metrics = MetricsCollector()

# 성능 통계 조회
stats = metrics.get_performance_stats(
    analysis_type="sentiment",
    days=7
)

# Goodput 통계 조회
goodput = metrics.get_goodput_stats(recent_n=100)
```

#### 1.2 `GoodputTracker`

**위치:** `src/goodput_tracker.py`

**주요 메서드:**
- `add_request()`: 요청 결과 추가
- `calculate_goodput()`: Goodput 계산
- `get_recent_stats()`: 최근 N개 요청 통계

**사용 예시:**
```python
from src.goodput_tracker import GoodputTracker

tracker = GoodputTracker(ttft_sla_ms=2000)

# 요청 추가
tracker.add_request(
    ttft_ms=1500,
    n_tokens=100,
    processing_time_ms=2000
)

# Goodput 계산
stats = tracker.calculate_goodput()
print(f"SLA Compliance: {stats['sla_compliance_rate']:.2f}%")
```

#### 1.3 `GPUMonitor`

**위치:** `scripts/gpu_monitor.py`

**주요 메서드:**
- `get_metrics()`: GPU 메트릭 수집

**사용 예시:**
```python
from scripts.gpu_monitor import GPUMonitor

monitor = GPUMonitor(device_index=0)
gpu_metrics = monitor.get_metrics()
print(f"GPU Utilization: {gpu_metrics['gpu_util_percent']:.2f}%")
```

### 2. 수동 벤치마크 도구

#### 2.1 `scripts/benchmark.py`

**주요 기능:**
- 단일 요청 성능 측정
- 배치 요청 성능 측정
- GPU 메트릭 수집
- 통계 계산 (평균, P95, P99)

**명령줄 옵션:**
```bash
python scripts/benchmark.py \
  --endpoint <API_ENDPOINT> \
  --iterations <N> \
  --warmup <N> \
  --mode <single|batch> \
  --output <OUTPUT_FILE> \
  --base-url <BASE_URL>
```

### 3. 평가 도구

#### 3.1 `scripts/evaluate_summary.py`

**기능:** 요약 품질 평가 (BLEU Score)

**사용법:**
```bash
python scripts/evaluate_summary.py \
  --ground_truth scripts/Ground_truth_summary.json \
  --output results/summary_eval.json
```

#### 3.2 `scripts/evaluate_strength_extraction.py`

**기능:** 강점 추출 정확도 평가

**사용법:**
```bash
python scripts/evaluate_strength_extraction.py \
  --ground_truth scripts/Ground_truth_strength.json \
  --output results/strength_eval.json
```

#### 3.3 `scripts/evaluate_sentiment_analysis.py`

**기능:** 감성 분석 정확도 평가

**사용법:**
```bash
python scripts/evaluate_sentiment_analysis.py \
  --ground_truth scripts/Ground_truth_sentiment.json \
  --output results/sentiment_eval.json
```

---

## 측정 결과 분석

### 1. 성능 분석

#### 1.1 처리 시간 분석

**목표:**
- 평균 처리 시간이 목표값 이하인지 확인
- P95/P99 처리 시간이 SLA를 만족하는지 확인
- 시간대별/요일별 패턴 분석

**분석 방법:**
```sql
-- 시간대별 평균 처리 시간
SELECT 
    strftime('%H', created_at) as hour,
    AVG(processing_time_ms) as avg_time_ms,
    COUNT(*) as request_count
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
AND analysis_type = 'sentiment'
GROUP BY hour
ORDER BY hour;
```

#### 1.2 처리량 분석

**목표:**
- 단일 Pod 처리량이 목표값 이상인지 확인
- Goodput vs Throughput 비교
- SLA 준수율 모니터링

**분석 방법:**
```python
# GoodputTracker를 통한 실시간 분석
from src.metrics_collector import MetricsCollector

metrics = MetricsCollector()
goodput = metrics.get_goodput_stats()

if goodput['sla_compliance_rate'] < 90:
    print("⚠️ SLA 준수율이 90% 미만입니다.")
```

### 2. 리소스 분석

#### 2.1 GPU 활용률 분석

**목표:**
- GPU 활용률이 70-90% 범위인지 확인
- GPU 메모리 사용률 모니터링
- 병목 구간 식별 (Prefill vs Decode)

**분석 방법:**
```sql
-- Prefill vs Decode 시간 비율
SELECT 
    AVG(prefill_time_ms) * 100.0 / (AVG(prefill_time_ms) + AVG(decode_time_ms)) as prefill_ratio_percent,
    AVG(decode_time_ms) * 100.0 / (AVG(prefill_time_ms) + AVG(decode_time_ms)) as decode_ratio_percent
FROM vllm_metrics
WHERE created_at >= datetime('now', '-7 days');
```

### 3. 비용 분석

#### 3.1 토큰 사용량 분석

**목표:**
- 토큰 사용량이 목표값 이하인지 확인
- 대표 벡터 TOP-K 효과 검증 (60-80% 감소)
- 모델별 토큰 사용량 비교

**분석 방법:**
```sql
-- 모델별 토큰 사용량 비교
SELECT 
    model_version,
    AVG(tokens_used) as avg_tokens,
    COUNT(*) as request_count
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY model_version
ORDER BY avg_tokens;
```

### 4. 품질 분석

#### 4.1 정확도 분석

**목표:**
- 요약 품질 (BLEU Score)이 0.82 이상인지 확인
- 강점 추출 정확도가 88% 이상인지 확인
- 모델별 정확도 비교

**분석 방법:**
```bash
# 평가 스크립트 실행 후 결과 확인
cat results/summary_eval.json | jq '.bleu_score'
cat results/strength_eval.json | jq '.accuracy'
```

#### 4.2 에러율 분석

**목표:**
- HTTP 4xx 에러율이 1% 미만인지 확인
- HTTP 5xx 에러율이 0.1% 미만인지 확인
- OOM 발생 빈도 모니터링

**분석 방법:**
```sql
-- 에러율 추이 분석
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    SUM(error_count) as total_errors,
    (SUM(error_count) * 100.0 / COUNT(*)) as error_rate_percent
FROM analysis_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY date
ORDER BY date DESC;
```

---

## 지속적 측정 전략

### 1. 자동화된 측정

#### 1.1 CI/CD 통합

**목표:** 코드 변경 시 자동 벤치마크 실행

**구현 방법:**
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark
on:
  pull_request:
    branches: [main]
jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run Benchmark
        run: |
          python scripts/benchmark.py \
            --endpoint /api/v1/sentiment/analyze \
            --iterations 20 \
            --output results/benchmark.json
```

#### 1.2 정기적 벤치마크

**목표:** 주간/월간 성능 추이 모니터링

**구현 방법:**
```bash
# Cron 작업으로 주간 벤치마크 실행
0 2 * * 0 /path/to/scripts/weekly_benchmark.sh
```

### 2. 대시보드 구축

#### 2.1 SQLite 기반 대시보드

**목표:** 실시간 성능 모니터링

**구현 방법:**
- SQLite 쿼리를 통한 통계 조회
- Python 스크립트로 주기적 리포트 생성

#### 2.2 향후 확장: Prometheus + Grafana

**목표:** 실시간 메트릭 시각화

**구현 계획:**
- Prometheus 메트릭 수집기 통합
- Grafana 대시보드 구축
- 알림 규칙 설정

### 3. 측정 결과 문서화

#### 3.1 벤치마크 리포트

**목표:** 측정 결과를 체계적으로 문서화

**포함 내용:**
- 측정 일시 및 환경
- 측정 항목 및 결과
- 목표값 대비 성과
- 모델별 비교 결과

#### 3.2 성능 회고

**목표:** 정기적 성능 회고 및 개선 계획 수립

**주기:**
- 주간: 주요 지표 리뷰
- 월간: 종합 성능 분석 및 개선 계획

---

## 부록

### A. 측정 체크리스트

**벤치마크 전 확인사항:**
- [ ] API 서버가 정상 실행 중인지 확인
- [ ] GPU가 사용 가능한지 확인
- [ ] 테스트 데이터가 준비되어 있는지 확인
- [ ] 환경 변수 설정 확인 (모델명 등)

**벤치마크 후 확인사항:**
- [ ] 측정 결과 파일 저장
- [ ] SQLite 메트릭 확인
- [ ] GPU 메트릭 확인
- [ ] 에러 로그 확인

### B. 참고 문서

- `METRICS.md`: 메트릭 수집 상세 가이드
- `VLLM_PERFORMANCE_MEASUREMENT.md`: vLLM 성능 측정 가이드
- `MODEL_SELECTION.md`: 모델 후보군 목록
- `LLM_SERVICE_STEP/FINAL_ARCHITECTURE.md`: 설계 문서 (예상 효과 참고)

### C. 문제 해결

**일반적인 문제:**
- GPU 메모리 부족: 배치 크기 감소
- 측정 결과 불안정: 워밍업 횟수 증가
- SQLite 쿼리 느림: 인덱스 확인

**문의:**
- 측정 관련 이슈는 프로젝트 이슈 트래커에 등록
