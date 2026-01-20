# GPUMonitor vs nvidia-smi

## 개요

GPUMonitor 모듈과 nvidia-smi는 **같은 NVML (NVIDIA Management Library)**을 사용하지만, 접근 방식이 다릅니다.

## 기술적 관계

```
NVML (NVIDIA Management Library)
    ├── nvidia-smi (CLI 도구)
    └── pynvml (Python API 바인딩)
        └── GPUMonitor (프로젝트 모듈)
```

- **nvidia-smi**: NVML을 사용하는 CLI 도구
- **pynvml**: NVML의 Python 바인딩
- **GPUMonitor**: pynvml을 사용하는 Python 모듈

**결론**: GPUMonitor는 nvidia-smi와 별개가 아니라, 같은 NVML을 Python API로 사용하는 방식입니다.

---

## 비교표

| 항목 | GPUMonitor (pynvml) | nvidia-smi (CLI) |
|------|---------------------|------------------|
| **접근 방식** | Python API 직접 호출 | subprocess로 CLI 실행 |
| **성능** | 더 빠름 (프로세스 생성 없음) | 상대적으로 느림 (프로세스 생성 오버헤드) |
| **의존성** | pynvml 패키지 필요 | nvidia-smi 바이너리 필요 |
| **에러 처리** | Python 예외 처리 | subprocess 반환값 처리 |
| **통합성** | Python 코드와 자연스럽게 통합 | 문자열 파싱 필요 |
| **메모리 효율** | 더 효율적 (프로세스 생성 없음) | 프로세스 생성 오버헤드 |

---

## 프로젝트 내 사용 현황

### 변경 전 (nvidia-smi 사용)

프로젝트에는 두 가지 방식이 혼재되어 있었습니다:

#### 방식 1: GPUMonitor (pynvml 사용)
```python
# scripts/gpu_monitor.py
import pynvml
pynvml.nvmlInit()
util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
```

#### 방식 2: nvidia-smi CLI 호출
```python
# scripts/watchdog.py, scripts/benchmark.py
subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", ...])
```

### 변경 후 (GPUMonitor 통일)

모든 GPU 모니터링이 GPUMonitor를 사용하도록 통일되었습니다:

#### watchdog.py
```python
from scripts.gpu_monitor import GPUMonitor

class RunPodWatchdog:
    def __init__(self, ...):
        self.gpu_monitor = GPUMonitor(device_index=0)
    
    def get_gpu_usage(self) -> Optional[float]:
        metrics = self.gpu_monitor.get_metrics()
        if metrics:
            return metrics.get("gpu_util_percent")
        return None
```

#### benchmark.py
```python
from scripts.gpu_monitor import GPUMonitor

class PerformanceBenchmark:
    def __init__(self, ...):
        self.gpu_monitor = GPUMonitor(device_index=0)
    
    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        metrics = self.gpu_monitor.get_metrics()
        if metrics:
            return {
                "gpu_utilization": metrics.get("gpu_util_percent", 0.0),
                "memory_used_mb": metrics.get("memory_used_mb", 0.0),
                "memory_total_mb": metrics.get("memory_total_mb", 0.0),
                "memory_usage_percent": metrics.get("memory_util_percent", 0.0)
            }
        return None
```

---

## 변경 이유

### 1. 성능 개선
- **프로세스 생성 오버헤드 제거**: subprocess 호출 없이 직접 API 호출
- **더 빠른 응답 시간**: 프로세스 생성 및 파싱 단계 제거

### 2. 코드 일관성
- **통일된 접근 방식**: 모든 GPU 모니터링이 동일한 방식 사용
- **유지보수 용이**: 한 곳에서 GPU 모니터링 로직 관리

### 3. 에러 처리 개선
- **Python 예외 처리**: try-except로 자연스러운 에러 처리
- **타입 안정성**: Python 타입 힌팅 지원

### 4. 통합성 향상
- **Python 코드와 자연스러운 통합**: 문자열 파싱 불필요
- **메트릭 딕셔너리 반환**: 구조화된 데이터 반환

---

## GPUMonitor 사용 방법

### 기본 사용법

```python
from scripts.gpu_monitor import GPUMonitor

# GPUMonitor 인스턴스 생성
monitor = GPUMonitor(device_index=0)

# GPU 메트릭 조회
metrics = monitor.get_metrics()

if metrics:
    print(f"GPU 사용률: {metrics['gpu_util_percent']}%")
    print(f"메모리 사용률: {metrics['memory_util_percent']}%")
    print(f"메모리 사용량: {metrics['memory_used_mb']} MB")
    print(f"전체 메모리: {metrics['memory_total_mb']} MB")
    print(f"여유 메모리: {metrics['memory_free_mb']} MB")

# 리소스 정리
monitor.cleanup()
```

### 전역 인스턴스 사용

```python
from scripts.gpu_monitor import get_gpu_monitor

# 전역 인스턴스 반환 (싱글톤 패턴)
monitor = get_gpu_monitor(device_index=0)
metrics = monitor.get_metrics()
```

---

## 반환 메트릭

GPUMonitor의 `get_metrics()` 메서드는 다음 메트릭을 반환합니다:

```python
{
    "gpu_util_percent": float,      # GPU 사용률 (%)
    "memory_util_percent": float,   # 메모리 사용률 (%)
    "memory_used_mb": float,        # 사용 중인 메모리 (MB)
    "memory_total_mb": float,       # 전체 메모리 (MB)
    "memory_free_mb": float         # 여유 메모리 (MB)
}
```

---

## 의존성

GPUMonitor를 사용하려면 `pynvml` 패키지가 필요합니다:

```bash
pip install pynvml>=11.5.0
```

`requirements.txt`에 이미 포함되어 있습니다.

---

## 참고 문서

- **GPUMonitor 구현**: `scripts/gpu_monitor.py`
- **Watchdog 사용 예시**: `scripts/watchdog.py`
- **Benchmark 사용 예시**: `scripts/benchmark.py`
- **vLLM 성능 측정**: `VLLM_PERFORMANCE_MEASUREMENT.md`
