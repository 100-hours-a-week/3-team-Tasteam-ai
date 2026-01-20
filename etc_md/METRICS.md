## 집계 지표

### 1. 기본 분석 메트릭 (analysis_metrics 테이블)
SQLite에 저장되며 get_performance_stats()로 집계 가능:

**저장 지표**
restaurant_id: 레스토랑 ID
analysis_type: 분석 타입 (sentiment, summary, strength, image_search)
model_version: 모델 버전
processing_time_ms: 전체 처리 시간 (밀리초)
tokens_used: 사용된 토큰 수
batch_size: 배치 크기
cache_hit: 캐시 히트 여부 (Boolean)
error_count: 에러 개수
warning_count: 경고 개수
created_at: 생성 시간 (타임스탬프)

**집계 가능한 통계**
metrics.get_performance_stats(analysis_type="sentiment", days=7)
metrics.get_performance_stats(analysis_type="image_search", days=7)

**반환값**
avg_processing_time_ms: 평균 처리 시간
total_requests: 전체 요청 수
total_tokens_used: 총 사용 토큰 수
total_errors: 총 에러 수
error_rate: 에러율 (에러 수 / 전체 요청 수)

---

### 2. vLLM 상세 메트릭 
vLLM 전용 메트릭으로 collect_vllm_metrics()로 저장:

**저장 지표**
request_id: 요청 ID (UUID)
restaurant_id: 레스토랑 ID
analysis_type: 분석 타입 (sentiment, summary, strength, image_search)
prefill_time_ms: Prefill 시간 (밀리초) - 입력 처리 시간
decode_time_ms: Decode 시간 (밀리초) - 토큰 생성 시간
total_time_ms: 전체 시간 (밀리초)
n_tokens: 생성된 토큰 수
tpot_ms: Time Per Output Token (밀리초) - 토큰당 생성 시간
tps: Tokens Per Second - 초당 토큰 생성 수
ttft_ms: Time To First Token (밀리초) - 첫 토큰 생성까지 시간
created_at: 생성 시간

---

### 3. Goodput 통계 (실시간 메모리 기반)
GoodputTracker로 실시간 집계:

**집계 가능한 지표**
metrics.get_goodput_stats()  # 전체 통계
metrics.get_goodput_stats(recent_n=100)  # 최근 100개 요청

**반환값**
throughput_tps: 전체 처리량 (Tokens Per Second)
goodput_tps: SLA 만족 처리량 (Tokens Per Second)
sla_compliance_rate: SLA 준수율 (%)
total_requests: 전체 요청 수
sla_met_requests: SLA 만족 요청 수 (TTFT < 2초)
avg_ttft_ms: 평균 TTFT (밀리초)
SLA 기준
기본 SLA: TTFT < 2000ms (2초)
Goodput = SLA 만족 요청의 실제 처리량

---

4. GPU 메트릭 (실시간, 선택적)
GPUMonitor로 수집 (실시간 조회, DB 저장 아님):

**수집 가능한 지표**
from scripts.gpu_monitor import GPUMonitormonitor = GPUMonitor()gpu_metrics = monitor.get_metrics()

**반환값**
gpu_util_percent: GPU 사용률 (%)
memory_util_percent: 메모리 사용률 (%)
memory_used_mb: 사용 중인 메모리 (MB)
memory_total_mb: 전체 메모리 (MB)
memory_free_mb: 여유 메모리 (MB)

---

### 집계 가능한 주요 분석

**1. 성능 분석**
평균/최소/최대 처리 시간
P95/P99 처리 시간 (SQL 쿼리로 계산)
처리량 (TPS) 추이
Prefill vs Decode 시간 비율

**2. 품질 분석**
에러율 추이
SLA 준수율 (Goodput)
Throughput vs Goodput 비교

**3. 리소스 분석**
토큰 사용량 추이
배치 크기별 성능
GPU 사용률 (실시간)

**4. 시간대별 분석**
일별/시간대별 통계
피크 시간대 식별
트래픽 패턴 분석

**5. 이미지 검색 쿼리 확장 분석**
쿼리 확장 사용률 (확장된 쿼리 vs 원본 쿼리)
쿼리 확장 성능 (TTFT, TPS, Prefill/Decode 시간)
쿼리 확장 효과 분석 (검색 결과 개수 비교)
additional_info 필드 활용:
- original_query: 원본 쿼리
- expanded_query: 확장된 쿼리
- query_expanded: 확장 여부 (Boolean)
- results_count: 검색 결과 개수

**집계 예시 (SQL)**
```sql
-- 이미지 검색 쿼리 확장 사용률
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_searches,
    SUM(CASE 
        WHEN additional_info LIKE '%"query_expanded": true%' 
        THEN 1 ELSE 0 
    END) as expanded_count,
    AVG(processing_time_ms) as avg_time_ms,
    AVG(tokens_used) as avg_tokens
FROM analysis_metrics
WHERE analysis_type = 'image_search'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- 쿼리 확장 성능 분석
SELECT 
    AVG(ttft_ms) as avg_ttft_ms,
    AVG(tps) as avg_tps,
    AVG(prefill_time_ms) as avg_prefill_ms,
    AVG(decode_time_ms) as avg_decode_ms,
    AVG(n_tokens) as avg_tokens
FROM vllm_metrics
WHERE analysis_type = 'image_search'
AND created_at >= datetime('now', '-7 days');
```