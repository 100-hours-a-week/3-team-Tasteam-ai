# 서버 GPU 실시간 모니터링

벤치마크 시 **서버**에서 GPU 사용량을 주기적으로 샘플링해 `logs/gpu_usage.log`에 기록한다.  
(Config 무관, **X-Enable-GPU-Monitor** 헤더로만 활성화)

---

## 1. 활성화

| 방식 | 설명 |
|------|------|
| **test_all_task.py** | `--benchmark-gpu` 또는 `--benchmark` 사용 시 모든 API 요청에 `X-Enable-GPU-Monitor: true` 전송 |
| **직접 요청** | API 호출 시 헤더 `X-Enable-GPU-Monitor: true` 포함 |

서버는 해당 헤더를 받은 요청 처리 시 `get_or_create_benchmark_gpu_monitor()`를 호출해 백그라운드 GPU 모니터를 시작한다.

---

## 2. 요구사항

- **pynvml** 설치: `pip install pynvml`
- 서버가 **GPU가 있는 머신**에서 실행될 것 (NVML 사용)

---

## 3. 로그 형식

`logs/gpu_usage.log`에 `CPU_MONITOR_INTERVAL`(기본 1초)마다 한 줄씩 JSON:

```json
{"timestamp": "2025-01-27T12:00:00.000000", "device_index": 0, "gpu_util_percent": 45, "memory_util_percent": 62.5, "memory_used_mb": 5120.0, "memory_total_mb": 8192.0}
```

- **gpu_util_percent**: GPU 연산 사용률 (%)
- **memory_util_percent**: VRAM 사용률 (%)
- **memory_used_mb** / **memory_total_mb**: VRAM 사용량(MB) / 전체(MB)

---

## 4. 관련 파일

- `src/gpu_monitor.py`: `ServerGPUMonitor`, `get_or_create_benchmark_gpu_monitor()`
- `src/api/dependencies.py`: `get_metrics_collector()` 내 `X-Enable-GPU-Monitor` 처리
- `test_all_task.py`: `--benchmark-gpu` 시 `BENCHMARK_HEADERS["X-Enable-GPU-Monitor"] = "true"`

---

## 5. CPU 모니터와의 차이

| 구분 | CPU 모니터 | 서버 GPU 모니터 |
|------|-------------|-----------------|
| **Config** | `CPU_MONITOR_ENABLE` (lifespan에서 시작) | 없음 (헤더로만) |
| **벤치마크** | `X-Enable-CPU-Monitor` → `get_or_create_benchmark_cpu_monitor()` | `X-Enable-GPU-Monitor` → `get_or_create_benchmark_gpu_monitor()` |
| **로그** | `logs/cpu_usage.log` | `logs/gpu_usage.log` |
