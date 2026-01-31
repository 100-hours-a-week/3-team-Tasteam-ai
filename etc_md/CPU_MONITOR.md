# CPU 실시간 모니터링

백그라운드 태스크로 주기적 CPU·메모리 샘플링 후 로그 기록.

---

## 1. 설정

| 설정 | 환경 변수 | 기본값 | 설명 |
|------|-----------|--------|------|
| **CPU_MONITOR_ENABLE** | `CPU_MONITOR_ENABLE` | `false` | `true`면 백그라운드 CPU 모니터 시작 |
| **CPU_MONITOR_INTERVAL** | `CPU_MONITOR_INTERVAL` | `1.0` | 샘플링 간격 (초) |

예시 (`.env`):

```bash
# CPU 실시간 추적 활성화
CPU_MONITOR_ENABLE=true
CPU_MONITOR_INTERVAL=1.0
```

---

## 2. 동작

- **시작**: FastAPI 앱 기동 시 (`lifespan` startup) `CPUMonitor.start()` → 백그라운드 `asyncio.Task` 생성.
- **샘플링 루프**: `interval`마다 `psutil.cpu_percent()`, `psutil.Process(pid).cpu_percent()`, 메모리 정보를 수집.
- **로그 저장**: `logs/cpu_usage.log`에 JSON 형식으로 기록 (RotatingFileHandler, 10MB × 3개 백업).
- **종료**: 앱 종료 시 (`lifespan` shutdown) `CPUMonitor.stop()` → 백그라운드 태스크 정리.

---

## 3. 로그 형식

`logs/cpu_usage.log`에 매 `interval`마다 한 줄씩 JSON:

```json
{"timestamp": "2026-01-30T09:45:12.345678", "system_cpu_percent": 23.5, "process_cpu_percent": 8.2, "system_mem_percent": 65.3, "process_mem_mb": 512.4}
{"timestamp": "2026-01-30T09:45:13.345678", "system_cpu_percent": 24.1, "process_cpu_percent": 9.0, "system_mem_percent": 65.4, "process_mem_mb": 515.2}
...
```

- **system_cpu_percent**: 시스템 전체 CPU 사용률(%)
- **process_cpu_percent**: 현재 프로세스(FastAPI) CPU 사용률(%)
- **system_mem_percent**: 시스템 전체 메모리 사용률(%)
- **process_mem_mb**: 현재 프로세스 메모리 사용량(MB)

---

## 4. 분석 방법

- **실시간 곡선**: `logs/cpu_usage.log`를 읽어 `timestamp` × `process_cpu_percent` 그래프 그리기.
- **특정 시간대**: API 요청 시작/종료 시각과 매칭해 "그 구간의 CPU 곡선" 추출.
- **기능별 CPU**: 요청 로그(`logs/debug.log`)의 `timestamp`와 `cpu_usage.log`의 `timestamp`를 매칭하면, "이 요청 처리 중 CPU가 어땠는지" 추정 가능.

---

## 5. 주의사항

- **오버헤드**: `interval`이 너무 짧으면(예: 0.1초) 샘플링 자체가 CPU를 쓸 수 있음. 1초 정도가 실용적.
- **interval=None vs 값**: `psutil.cpu_percent(interval=None)`은 블로킹 없이 즉시 반환하지만 정확도가 떨어질 수 있음. `interval=1.0` 등으로 주면 그 구간 평균을 재지만 1초 `sleep`이 들어감. 백그라운드 태스크에서는 `interval=None`으로 샘플링 후 `asyncio.sleep(interval)`로 대기하는 방식이 일반적.
- **파일 크기**: RotatingFileHandler로 10MB × 3개 백업 설정. 1초 간격이면 JSON 한 줄 약 150바이트 → 10MB면 약 6~7만 샘플(약 18시간 분량). 필요 시 `maxBytes`, `backupCount` 조정.

---

## 6. 관련 파일

- `src/config.py`: `CPU_MONITOR_ENABLE`, `CPU_MONITOR_INTERVAL`
- `src/cpu_monitor.py`: `CPUMonitor` 클래스, `get_cpu_monitor()` 싱글톤
- `src/api/main.py`: `lifespan`에서 `cpu_monitor.start()` / `cpu_monitor.stop()`
- `.env.example`: CPU 모니터 환경 변수 예시

## 7. 서버 GPU 모니터

벤치마크 시 서버 GPU를 로그에 남기려면 **서버 GPU 모니터**를 사용한다.  
- 헤더: `X-Enable-GPU-Monitor: true` (test_all_task.py `--benchmark-gpu` 시 전송)  
- 로그: `logs/gpu_usage.log` (pynvml 기반, JSON 한 줄씩)  
- 상세: [GPU_MONITOR.md](GPU_MONITOR.md)
