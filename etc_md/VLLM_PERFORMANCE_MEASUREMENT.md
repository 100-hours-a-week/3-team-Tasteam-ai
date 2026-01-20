# vLLM 성능 측정 및 최적화 제안

## 개요

본 문서는 현재 프로젝트의 vLLM 기반 추론 시스템에 대한 **정밀 성능 측정** 및 **최적화 전략**을 제안합니다.

현재 시스템은 기본적인 `processing_time_ms`만 측정하고 있으며, vLLM의 핵심 성능 지표(Prefill/Decode 분리, TTFT, TPS, Goodput 등)를 측정하지 못하고 있습니다.

---

## 1. Prefill / Decode 분리 측정 (최우선 과제)

### 1.1 현재 문제점

**현재 코드** (`src/llm_utils.py:945-953`):
```python
outputs = await loop.run_in_executor(
    self.executor,
    self.llm.generate,
    prompts,
    sampling_params
)
return [output.outputs[0].text for output in outputs]
```

**문제:**
- `llm.generate()` 전체 시간만 측정
- Prefill(입력 처리)과 Decode(토큰 생성) 단계를 구분하지 못함
- 병목 구간 파악 불가

### 1.2 해결 방안

#### Option 1: vLLM RequestOutput 메트릭 활용 (권장)

vLLM의 `RequestOutput` 객체는 다음 정보를 포함:
- `metrics.first_token_time`: Prefill 시간 (TTFT)
- `metrics.finished_time`: 전체 완료 시간
- `metrics.n_tokens`: 생성된 토큰 수

**구현 예시:**

```python
async def _generate_with_vllm(
    self,
    prompts: List[str],
    temperature: float = 0.1,
    max_tokens: int = 100,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    vLLM을 사용하여 비동기로 응답 생성 + 메트릭 수집
    
    Returns:
        (생성된 응답 리스트, 메트릭 딕셔너리)
    """
    if not self.use_pod_vllm:
        raise ValueError("vLLM이 초기화되지 않았습니다.")
    
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # 전체 시작 시간
    start_time = time.time()
    
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        self.executor,
        self.llm.generate,
        prompts,
        sampling_params
    )
    
    total_time = time.time() - start_time
    
    # 메트릭 수집
    metrics = {
        "total_time_ms": total_time * 1000,
        "requests": []
    }
    
    responses = []
    total_prefill_time = 0
    total_decode_time = 0
    total_tokens = 0
    
    for output in outputs:
        text = output.outputs[0].text
        responses.append(text)
        
        # vLLM RequestOutput 메트릭 추출
        if hasattr(output, 'metrics') and output.metrics:
            first_token_time = output.metrics.first_token_time or 0
            finished_time = output.metrics.finished_time or 0
            n_tokens = len(output.outputs[0].token_ids)
            
            prefill_time_ms = first_token_time * 1000
            decode_time_ms = (finished_time - first_token_time) * 1000 if finished_time > first_token_time else 0
            
            total_prefill_time += prefill_time_ms
            total_decode_time += decode_time_ms
            total_tokens += n_tokens
            
            metrics["requests"].append({
                "prefill_time_ms": prefill_time_ms,
                "decode_time_ms": decode_time_ms,
                "total_time_ms": finished_time * 1000,
                "n_tokens": n_tokens,
                "tpot_ms": decode_time_ms / n_tokens if n_tokens > 0 else 0,  # Time Per Output Token
            })
    
    # 평균 메트릭
    n_requests = len(outputs)
    if n_requests > 0:
        metrics["avg_prefill_time_ms"] = total_prefill_time / n_requests
        metrics["avg_decode_time_ms"] = total_decode_time / n_requests
        metrics["avg_tpot_ms"] = (total_decode_time / total_tokens) if total_tokens > 0 else 0
        metrics["total_tokens"] = total_tokens
        metrics["tps"] = total_tokens / total_time if total_time > 0 else 0  # Tokens Per Second
    
    return responses, metrics
```

#### Option 2: LLMEngine.step() 직접 계측

vLLM 내부 `LLMEngine.step()` 메서드에 타이머 삽입 (고급)

**장점:** 더 정확한 Prefill/Decode 구분  
**단점:** vLLM 코드 수정 필요, 버전 업데이트 시 유지보수 어려움

**권장하지 않음** (Option 1로 충분)

### 1.3 메트릭 저장 확장

**`src/metrics_db.py` 테이블 스키마 확장:**

```python
cursor.execute("""
    CREATE TABLE IF NOT EXISTS vllm_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        restaurant_id INTEGER,
        analysis_type TEXT,
        
        -- Prefill/Decode 분리 지표
        prefill_time_ms REAL,
        decode_time_ms REAL,
        total_time_ms REAL,
        
        -- 토큰 관련
        n_tokens INTEGER,
        tpot_ms REAL,  -- Time Per Output Token
        
        -- 처리량
        tps REAL,  -- Tokens Per Second
        
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
```

**`src/metrics_collector.py` 확장:**

```python
def collect_vllm_metrics(
    self,
    request_id: str,
    restaurant_id: Optional[int],
    analysis_type: str,
    vllm_metrics: Dict[str, Any],
):
    """vLLM 전용 메트릭 수집"""
    if self.enable_db and self.metrics_db:
        self.metrics_db.insert_vllm_metric(
            request_id=request_id,
            restaurant_id=restaurant_id,
            analysis_type=analysis_type,
            prefill_time_ms=vllm_metrics.get("avg_prefill_time_ms"),
            decode_time_ms=vllm_metrics.get("avg_decode_time_ms"),
            total_time_ms=vllm_metrics.get("total_time_ms"),
            n_tokens=vllm_metrics.get("total_tokens"),
            tpot_ms=vllm_metrics.get("avg_tpot_ms"),
            tps=vllm_metrics.get("tps"),
        )
```

---

## 2. Continuous Batching 실제 효과 수치화

### 2.1 측정 대상 지표

| 지표 | 설명 | 측정 방법 |
|------|------|-----------|
| **TTFT** (Time To First Token) | 첫 토큰 생성까지 시간 | `output.metrics.first_token_time` |
| **TTFT P95** | 95 백분위수 TTFT | 여러 요청 수집 후 계산 |
| **TPS** (Tokens Per Second) | 초당 생성 토큰 수 | `total_tokens / total_time` |
| **GPU Utilization** | GPU 사용률 (%) | `nvidia-smi` 또는 `pynvml` |
| **Batch Size** | vLLM이 실제 배치 처리한 크기 | vLLM 로그 또는 모니터링 |

### 2.2 Continuous Batching vs Static Batching 비교 실험

**실험 설계:**

1. **동일 워크로드**:
   - 10개 레스토랑, 각 50개 리뷰
   - 총 500개 리뷰 감성 분석

2. **측정 시나리오**:
   - **시나리오 A**: 순차 처리 (배치 크기 1)
   - **시나리오 B**: 정적 배치 (배치 크기 50)
   - **시나리오 C**: vLLM Continuous Batching (자동)

3. **수집 지표**:
   - 전체 처리 시간
   - 평균 TTFT
   - P95 TTFT
   - 총 TPS
   - 평균 GPU 사용률

**예상 결과:**

| 시나리오 | 전체 시간 | 평균 TTFT | TPS | GPU 사용률 |
|----------|-----------|-----------|-----|------------|
| A (순차) | 500초 | 50ms | 50 | 30% |
| B (정적 배치) | 150초 | 100ms | 166 | 70% |
| C (Continuous) | 100초 | 80ms | 250 | 85% |

### 2.3 GPU 사용률 실시간 모니터링

**`scripts/gpu_monitor.py` 추가:**

```python
import pynvml
import time
from typing import Dict

class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """GPU 메트릭 수집"""
        util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
        
        return {
            "gpu_util_percent": util.gpu,
            "memory_util_percent": (mem_info.used / mem_info.total) * 100,
            "memory_used_mb": mem_info.used / (1024 ** 2),
            "memory_total_mb": mem_info.total / (1024 ** 2),
        }
    
    def cleanup(self):
        pynvml.nvmlShutdown()
```

**메트릭 수집에 통합:**

```python
# src/metrics_collector.py에 GPU 메트릭 추가
def collect_metrics(self, ..., gpu_metrics: Optional[Dict] = None):
    if gpu_metrics:
        debug_info["gpu_util_percent"] = gpu_metrics.get("gpu_util_percent")
        debug_info["memory_util_percent"] = gpu_metrics.get("memory_util_percent")
```

---

## 3. KV Cache 효과 "토큰 수 기준" 증명

### 3.1 실험 설계

**가설:** vLLM의 PagedAttention이 KV Cache를 효율적으로 관리하여, 출력 토큰 수가 증가해도 **TPOT (Time Per Output Token)가 일정**하게 유지된다.

**실험:**

1. **동일 입력 프롬프트**
2. **다양한 출력 토큰 수**:
   - 출력 128 tokens
   - 출력 256 tokens
   - 출력 512 tokens
   - 출력 1024 tokens

3. **측정 지표**:
   - TPOT (Decode Time / N Tokens)
   - TBT (Time Between Tokens)
   - 메모리 사용량

**구현 예시:**

```python
async def benchmark_kv_cache_efficiency():
    """KV Cache 효율성 벤치마크"""
    prompt = "음식점 리뷰를 분석하세요: " + "좋은 음식점입니다. " * 50
    
    results = []
    for max_tokens in [128, 256, 512, 1024]:
        responses, metrics = await llm_utils._generate_with_vllm(
            prompts=[prompt],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        
        tpot = metrics["avg_tpot_ms"]
        results.append({
            "max_tokens": max_tokens,
            "actual_tokens": metrics["total_tokens"],
            "tpot_ms": tpot,
            "decode_time_ms": metrics["avg_decode_time_ms"],
        })
    
    return results
```

**기대 결과:**

| 출력 토큰 수 | TPOT (ms) | Decode Time (ms) | 메모리 증가 |
|--------------|-----------|------------------|-------------|
| 128 | 2.5 | 320 | +200MB |
| 256 | 2.6 | 666 | +400MB |
| 512 | 2.7 | 1382 | +800MB |
| 1024 | 2.8 | 2867 | +1600MB |

**결론:** TPOT가 거의 일정 → KV Cache가 효율적으로 작동

---

## 4. GPU Idle 기준 + Watchdog 정책 수치화

### 4.1 현재 Watchdog 설정

**`src/config.py:66-69`:**
```python
IDLE_THRESHOLD: int = 5  # GPU 사용률 5% 미만
CHECK_INTERVAL: int = 60  # 60초마다 체크
IDLE_LIMIT: int = 5  # 5회 연속 (5분)
MIN_RUNTIME: int = 600  # 최소 10분 실행
```

### 4.2 수치화 목표

#### 정책 설정 근거

| 정책 | 현재값 | 근거 | 개선안 |
|------|--------|------|--------|
| **IDLE_THRESHOLD** | 5% | 임의 설정 | 실제 idle 상태 측정 후 결정 (3-7%) |
| **IDLE_LIMIT** | 5회 | 5분 대기 | Cold Start 비용 고려하여 최적화 |
| **MIN_RUNTIME** | 600초 | 10분 | Pod 시작 비용 회수 시간 계산 |

#### Cold Start 비용 분석

**측정 항목:**
1. **Pod 시작 시간**: RunPod Pod 시작 → 모델 로딩 완료
2. **모델 로딩 시간**: vLLM 모델 로드 시간
3. **첫 요청 지연**: Cold Start로 인한 추가 지연

**비용 계산:**

```
Cold Start 비용 = (Pod 시작 시간) × (시간당 요금)
유휴 비용 = (Idle 시간) × (시간당 요금)

최적 IDLE_LIMIT = Cold Start 비용 / 유휴 비용
```

**예시:**
- Pod 시작 시간: 3분
- 시간당 요금: $1.50/hr
- Cold Start 비용: (3/60) × $1.50 = $0.075
- 유휴 비용 (분당): $1.50/60 = $0.025/min
- 최적 Idle 시간: $0.075 / $0.025 = 3분

**결론:** `IDLE_LIMIT = 3` (3회 × 1분 = 3분)이 최적

### 4.3 개선된 Watchdog 정책

```python
# 동적 임계값 계산
class AdaptiveWatchdog(RunPodWatchdog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_usage_history = []
        self.request_count = 0
    
    def calculate_dynamic_threshold(self) -> float:
        """최근 GPU 사용 패턴 기반 동적 임계값 계산"""
        if len(self.gpu_usage_history) < 10:
            return self.idle_threshold
        
        # 최근 10회 평균의 10%를 idle 임계값으로 설정
        recent_avg = np.mean(self.gpu_usage_history[-10:])
        dynamic_threshold = max(3, recent_avg * 0.1)
        
        return dynamic_threshold
    
    def should_shutdown(self, gpu_usage: float) -> bool:
        """종료 여부 판단 (개선된 로직)"""
        self.gpu_usage_history.append(gpu_usage)
        
        # 1. 최소 실행 시간 확인
        if time.time() - self.start_time < self.min_runtime:
            return False
        
        # 2. 동적 임계값 계산
        threshold = self.calculate_dynamic_threshold()
        
        # 3. Idle 판단
        if gpu_usage < threshold:
            self.idle_count += 1
        else:
            self.idle_count = 0
        
        # 4. Cold Start 비용 고려
        expected_idle_cost = self.idle_count * (self.hourly_rate / 60)
        cold_start_cost = self.cold_start_time_min * (self.hourly_rate / 60)
        
        # 유휴 비용이 Cold Start 비용보다 크면 종료
        return expected_idle_cost > cold_start_cost
```

---

## 5. Goodput 개념 실제 적용

### 5.1 Goodput 정의

**Goodput**: SLA를 만족하는 요청의 실제 처리량

```
Goodput = (SLA 만족 요청 수) / (전체 시간)
Throughput = (전체 요청 수) / (전체 시간)
```

### 5.2 SLA 설정

**프로젝트 SLA:**
- **TTFT < 2초**: 사용자가 첫 응답을 2초 이내에 받아야 함
- **전체 TPS: 300**: 초당 300개 토큰 생성 목표

### 5.3 Goodput 측정 구현

```python
class GoodputTracker:
    def __init__(self, ttft_sla_ms: float = 2000):
        self.ttft_sla_ms = ttft_sla_ms
        self.requests = []
    
    def add_request(self, ttft_ms: float, n_tokens: int, processing_time_ms: float):
        """요청 결과 추가"""
        meets_sla = ttft_ms < self.ttft_sla_ms
        self.requests.append({
            "ttft_ms": ttft_ms,
            "n_tokens": n_tokens,
            "processing_time_ms": processing_time_ms,
            "meets_sla": meets_sla,
        })
    
    def calculate_goodput(self) -> Dict[str, float]:
        """Goodput 계산"""
        if not self.requests:
            return {}
        
        total_time_s = sum(r["processing_time_ms"] for r in self.requests) / 1000
        total_tokens = sum(r["n_tokens"] for r in self.requests)
        sla_met_tokens = sum(r["n_tokens"] for r in self.requests if r["meets_sla"])
        
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0
        goodput = sla_met_tokens / total_time_s if total_time_s > 0 else 0
        sla_compliance_rate = (sla_met_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return {
            "throughput_tps": throughput,
            "goodput_tps": goodput,
            "sla_compliance_rate": sla_compliance_rate,
            "total_requests": len(self.requests),
            "sla_met_requests": sum(1 for r in self.requests if r["meets_sla"]),
        }
```

**메트릭 수집 통합:**

```python
# src/metrics_collector.py
def collect_metrics(self, ..., ttft_ms: Optional[float] = None):
    """메트릭 수집 (Goodput 추적 포함)"""
    # 기존 로직...
    
    # Goodput 추적
    if ttft_ms is not None:
        self.goodput_tracker.add_request(
            ttft_ms=ttft_ms,
            n_tokens=tokens_used,
            processing_time_ms=processing_time_ms,
        )
```

### 5.4 Goodput 대시보드

**SQLite 쿼리:**

```sql
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    SUM(CASE WHEN prefill_time_ms < 2000 THEN 1 ELSE 0 END) as sla_met_requests,
    SUM(n_tokens) as total_tokens,
    SUM(CASE WHEN prefill_time_ms < 2000 THEN n_tokens ELSE 0 END) as sla_met_tokens,
    AVG(tps) as avg_tps,
    (SUM(CASE WHEN prefill_time_ms < 2000 THEN n_tokens ELSE 0 END) * 1.0 / SUM(n_tokens) * 100) as sla_compliance_rate
FROM vllm_metrics
WHERE created_at >= datetime('now', '-7 days')
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

**예시 결과:**

| Date | Total Requests | SLA Met | Total Tokens | Goodput TPS | SLA Compliance |
|------|----------------|---------|--------------|-------------|----------------|
| 2026-01-13 | 1000 | 850 | 50000 | 255 | 85% |
| 2026-01-12 | 950 | 800 | 48000 | 240 | 84% |
| 2026-01-11 | 1100 | 900 | 55000 | 270 | 82% |

---

## 6. 구현 우선순위 및 로드맵

### Phase 1: 기본 메트릭 수집 (1-2일)
- [ ] `_generate_with_vllm()` 메서드에 Prefill/Decode 분리 측정 추가
- [ ] `metrics_db.py`에 `vllm_metrics` 테이블 추가
- [ ] `metrics_collector.py`에 `collect_vllm_metrics()` 추가

### Phase 2: Continuous Batching 효과 측정 (2-3일)
- [ ] GPU 모니터링 모듈 추가 (`scripts/gpu_monitor.py`)
- [ ] 벤치마크 스크립트에 TTFT, TPS 측정 추가
- [ ] 배치 크기별 성능 비교 실험

### Phase 3: KV Cache 효율성 증명 (1-2일)
- [ ] TPOT 측정 로직 추가
- [ ] 출력 토큰 수별 벤치마크 실험
- [ ] 결과 문서화

### Phase 4: Watchdog 정책 최적화 (2-3일)
- [ ] Cold Start 비용 측정
- [ ] 동적 임계값 계산 로직 추가
- [ ] 비용 최적화 분석

### Phase 5: Goodput 추적 시스템 (2-3일)
- [ ] `GoodputTracker` 클래스 구현
- [ ] SLA 준수율 대시보드 쿼리 작성
- [ ] 메트릭 수집 통합

**총 예상 기간: 8-13일**

---

## 7. 예상 개선 효과

| 항목 | 현재 | 개선 후 | 효과 |
|------|------|---------|------|
| **측정 정밀도** | 전체 시간만 | Prefill/Decode 분리 | 병목 구간 식별 가능 |
| **성능 가시성** | 낮음 | 높음 | 최적화 근거 마련 |
| **비용 효율성** | 임의 정책 | 데이터 기반 정책 | 10-20% 비용 절감 예상 |
| **SLA 관리** | 없음 | Goodput 추적 | 품질 보장 |
| **운영 효율성** | 수동 모니터링 | 자동 메트릭 수집 | 운영 시간 50% 절감 |

---

## 8. 참고 자료

- [vLLM Performance Tuning](https://docs.vllm.ai/en/latest/models/performance.html)
- [vLLM RequestOutput Metrics](https://discuss.vllm.ai/t/does-llm-generate-differentiate-between-prefill-and-decode-phases)
- [Goodput 개념](https://blog.vllm.ai/2025/12/13/vllm-router-release.html)
- 현재 프로젝트: `BENCHMARK.md`, `INFERENCE_OPTIMIZATION.md`
