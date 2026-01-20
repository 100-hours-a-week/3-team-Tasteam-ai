vllm 시나리오 예시
250109 작성

[시간 흐름]
0.00s  → 음식점 A의 batch1 도착
0.02s  → 음식점 B의 batch1 도착
0.03s  → 음식점 C의 batch1 도착

vLLM 내부:
  → 0.04s 시점에 3개를 하나의 batch로 GPU에 올림 (continuous batch)

vllm 아키텍쳐 추가 버전

[Async Python Layer]
   ├─ process_restaurant("A") → 리뷰배치1,2
   ├─ process_restaurant("B") → 리뷰배치1,2
   ├─ process_restaurant("C") → 리뷰배치1
       ↓
[HTTP/GRPC Calls to vLLM]
       ↓
[vLLM Scheduler]
   ├─ Continuous batching
   ├─ GPU Memory management (PagedAttention)
   ├─ KV cache reuse
       ↓
[GPU]
   └─ 병렬 텐서 연산 (15개 prompt 동시에)
       ↓
[vLLM Response Stream]
       ↓
[Async Python Layer]
   ├─ await 각 음식점 결과
   └─ 음식점별 집계


로컬 vllm

from vllm import LLM, SamplingParams
import asyncio

llm = LLM(model="Qwen2.5-7B-Instruct")

async def analyze_restaurant(restaurant_id, reviews):
    prompts = [f"[{restaurant_id}] 리뷰: {r}\n감성 분석해줘." for r in reviews]
    sampling = SamplingParams(temperature=0)
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]

async def main():
    tasks = [
        analyze_restaurant("A", a_reviews),
        analyze_restaurant("B", b_reviews),
        analyze_restaurant("C", c_reviews),
    ]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())

---

watchdog_healthcheck.md

## 외부 Watchdog/Healthcheck 방식 적용 가능성 검토

### 제안된 방식 분석

```bash
#!/bin/bash
GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [ "$GPU_USAGE" -lt 5 ]; then
    echo "GPU idle, shutting down..."
    docker stop my_vllm_container
fi
```

핵심 개념:
- 서버 내부: vLLM + FastAPI만 실행 (자동 종료 로직 없음)
- 외부 모니터링: GPU 사용량 모니터링 및 Pod 제어
- 분리된 책임: 모니터링과 제어를 외부로 분리

---

### 적용 가능성: 가능하며 권장

#### 장점

1. 관심사 분리
   - 서버는 비즈니스 로직에 집중
   - 모니터링/제어는 외부로 분리
   - 서버 코드 단순화

2. 유연성
   - 모니터링 로직 변경 시 서버 재배포 불필요
   - 다양한 모니터링 방식 적용 가능
   - 여러 Pod를 하나의 스크립트로 관리 가능

3. 안정성
   - 서버 내부 종료 로직으로 인한 예기치 않은 종료 방지
   - 외부 모니터링 실패 시에도 서버는 계속 실행
   - 장애 격리 용이

4. 확장성
   - 여러 Pod를 중앙에서 모니터링
   - 통합 로깅 및 알림
   - 메트릭 수집 용이

---

### RunPod Pod 환경에서의 구현

#### 1. RunPod API를 통한 Pod 제어

```python
# watchdog_script.py
import requests
import time
import subprocess
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodWatchdog:
    """RunPod Pod 모니터링 및 자동 종료"""
    
    def __init__(self, api_key: str, pod_id: str, idle_threshold: int = 5, check_interval: int = 60):
        self.api_key = api_key
        self.pod_id = pod_id
        self.idle_threshold = idle_threshold  # GPU 사용률 임계값 (%)
        self.check_interval = check_interval  # 체크 간격 (초)
        self.idle_count = 0  # 연속 idle 횟수
        self.idle_limit = 5  # 5분간 idle이면 종료 (60초 * 5회)
    
    def get_gpu_usage(self) -> Optional[float]:
        """GPU 사용률 조회 (nvidia-smi)"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"GPU 사용률 조회 실패: {e}")
        return None
    
    def get_pod_status(self) -> Optional[dict]:
        """RunPod Pod 상태 조회"""
        url = f"https://api.runpod.ai/v1/{self.pod_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Pod 상태 조회 실패: {e}")
        return None
    
    def stop_pod(self) -> bool:
        """RunPod Pod 종료"""
        url = f"https://api.runpod.ai/v1/stop/{self.pod_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.post(url, headers=headers, timeout=30)
            if response.status_code == 200:
                logger.info(f"Pod {self.pod_id} 종료 성공")
                return True
            else:
                logger.error(f"Pod 종료 실패: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Pod 종료 중 오류: {e}")
        return False
    
    def check_recent_requests(self) -> bool:
        """최근 요청 기록 확인 (선택사항)"""
        # Prometheus, 로그 파일, 또는 API 엔드포인트를 통해 확인
        # 예: FastAPI의 /metrics 엔드포인트
        try:
            response = requests.get("http://localhost:8000/metrics", timeout=5)
            # 메트릭 파싱 및 최근 요청 확인
            # ...
            return True  # 최근 요청이 있음
        except:
            return False  # 확인 불가
    
    def monitor_loop(self):
        """모니터링 루프"""
        logger.info(f"Watchdog 시작: Pod {self.pod_id}, 임계값 {self.idle_threshold}%")
        
        while True:
            try:
                # GPU 사용률 확인
                gpu_usage = self.get_gpu_usage()
                
                if gpu_usage is None:
                    logger.warning("GPU 사용률 조회 실패, 다음 체크까지 대기")
                    time.sleep(self.check_interval)
                    continue
                
                logger.info(f"GPU 사용률: {gpu_usage}%")
                
                # 최근 요청 확인 (선택사항)
                has_recent_requests = self.check_recent_requests()
                
                # Idle 판단
                if gpu_usage < self.idle_threshold and not has_recent_requests:
                    self.idle_count += 1
                    logger.info(f"Idle 상태 감지 ({self.idle_count}/{self.idle_limit})")
                    
                    if self.idle_count >= self.idle_limit:
                        logger.info("연속 idle 시간 초과, Pod 종료")
                        if self.stop_pod():
                            break  # 종료 성공 시 루프 종료
                        else:
                            # 종료 실패 시 계속 모니터링
                            self.idle_count = 0
                else:
                    # 사용 중이면 idle 카운터 리셋
                    if self.idle_count > 0:
                        logger.info("활성 상태 감지, idle 카운터 리셋")
                    self.idle_count = 0
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Watchdog 중지")
                break
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    import os
    
    watchdog = RunPodWatchdog(
        api_key=os.getenv("RUNPOD_API_KEY"),
        pod_id=os.getenv("RUNPOD_POD_ID"),
        idle_threshold=int(os.getenv("IDLE_THRESHOLD", "5")),
        check_interval=int(os.getenv("CHECK_INTERVAL", "60"))
    )
    
    watchdog.monitor_loop()
```

#### 2. 간단한 Bash 스크립트 버전

```bash
#!/bin/bash
# watchdog.sh

RUNPOD_API_KEY="${RUNPOD_API_KEY}"
POD_ID="${RUNPOD_POD_ID}"
IDLE_THRESHOLD=5
CHECK_INTERVAL=60
IDLE_LIMIT=5
IDLE_COUNT=0

while true; do
    # GPU 사용률 조회
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$GPU_USAGE" ]; then
        echo "GPU 사용률 조회 실패, 다음 체크까지 대기"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "GPU 사용률: ${GPU_USAGE}%"
    
    # 최근 요청 확인 (선택사항)
    # 예: 로그 파일 또는 API 엔드포인트 확인
    HAS_RECENT_REQUESTS=false
    # if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    #     HAS_RECENT_REQUESTS=true
    # fi
    
    # Idle 판단
    if [ "$GPU_USAGE" -lt "$IDLE_THRESHOLD" ] && [ "$HAS_RECENT_REQUESTS" = false ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
        echo "Idle 상태 감지 ($IDLE_COUNT/$IDLE_LIMIT)"
        
        if [ "$IDLE_COUNT" -ge "$IDLE_LIMIT" ]; then
            echo "연속 idle 시간 초과, Pod 종료"
            
            # RunPod API를 통한 Pod 종료
            curl -X POST "https://api.runpod.ai/v1/stop/${POD_ID}" \
                -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
                -H "Content-Type: application/json"
            
            exit 0
        fi
    else
        # 사용 중이면 idle 카운터 리셋
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "활성 상태 감지, idle 카운터 리셋"
        fi
        IDLE_COUNT=0
    fi
    
    sleep $CHECK_INTERVAL
done
```

#### 3. FastAPI 메트릭 엔드포인트 추가 (선택사항)

```python
# src/api/main.py에 추가

from prometheus_client import Counter, Gauge, generate_latest
from starlette.responses import Response

# 메트릭 정의
request_count = Counter('api_requests_total', 'Total API requests')
gpu_usage = Gauge('gpu_usage_percent', 'GPU usage percentage')

@app.get("/metrics")
async def metrics():
    """Prometheus 메트릭 엔드포인트"""
    # GPU 사용률 업데이트 (선택사항)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_usage.set(float(result.stdout.strip()))
    except:
        pass
    
    return Response(content=generate_latest(), media_type="text/plain")
```

---

### 내부 자동 종료 vs 외부 Watchdog 비교

| 항목 | 내부 자동 종료 | 외부 Watchdog |
|------|---------------|--------------|
| **구현 복잡도** | 중간 (비동기 타이머) | 낮음 (독립 스크립트) |
| **서버 코드 영향** | 있음 (미들웨어 추가) | 없음 |
| **유연성** | 낮음 (서버 재배포 필요) | 높음 (스크립트만 수정) |
| **안정성** | 중간 (서버 내부 로직) | 높음 (외부 격리) |
| **확장성** | 낮음 (단일 서버) | 높음 (여러 Pod 관리) |
| **디버깅** | 어려움 (서버 내부) | 쉬움 (독립 실행) |
| **모니터링** | 제한적 | 풍부 (중앙 집중) |

---

### 구현 시 고려사항

#### 1. RunPod Pod ID 관리

```python
# Pod ID는 환경 변수 또는 설정 파일에서 관리
POD_ID = os.getenv("RUNPOD_POD_ID")
# 또는 Pod 메타데이터에서 자동 조회
```

#### 2. 메트릭 수집 방법

옵션 1: nvidia-smi 직접 사용
```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
```

옵션 2: Prometheus 메트릭
```python
# FastAPI에 /metrics 엔드포인트 추가
# watchdog에서 메트릭 조회
```

옵션 3: 로그 파일 분석
```bash
# 최근 요청 로그 확인
tail -n 100 /var/log/app.log | grep "POST /api"
```

#### 3. 안전장치

```python
# 최소 실행 시간 보장
MIN_RUNTIME = 600  # 10분
start_time = time.time()

if time.time() - start_time < MIN_RUNTIME:
    logger.info("최소 실행 시간 미달, 종료하지 않음")
    continue

# 활성 작업 확인
# 예: 진행 중인 요청이 있으면 종료하지 않음
```

#### 4. 알림 및 로깅

```python
# 종료 전 알림 (선택사항)
def send_notification(message: str):
    # Slack, Discord, Email 등
    pass

if self.idle_count >= self.idle_limit:
    send_notification(f"Pod {self.pod_id} 자동 종료 예정 (GPU 사용률: {gpu_usage}%)")
    self.stop_pod()
```

---

### 배포 방법

#### 1. 별도 서버에서 실행

```bash
# 관리 서버에서 cron으로 실행
*/5 * * * * /path/to/watchdog_script.py
```

#### 2. Cloud Function (AWS Lambda, GCP Functions)

```python
# serverless function으로 배포
def watchdog_handler(event, context):
    watchdog = RunPodWatchdog(...)
    watchdog.monitor_loop()
```

#### 3. Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: runpod-watchdog
spec:
  schedule: "*/5 * * * *"  # 5분마다
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: watchdog
            image: watchdog:latest
            env:
            - name: RUNPOD_API_KEY
              valueFrom:
                secretKeyRef:
                  name: runpod-secret
                  key: api-key
```

---

### 결론 및 권장사항

#### 적용 가능: 권장

1. 관심사 분리
   - 서버는 비즈니스 로직에 집중
   - 모니터링은 외부로 분리

2. 유연성과 확장성
   - 모니터링 로직 변경 시 서버 재배포 불필요
   - 여러 Pod를 중앙에서 관리 가능

3. 안정성
   - 서버 내부 종료 로직으로 인한 예기치 않은 종료 방지
   - 장애 격리 용이

#### 구현 우선순위

1. 1단계: 기본 Watchdog 스크립트
   - GPU 사용률 모니터링
   - RunPod API를 통한 Pod 종료

2. 2단계: 고급 기능
   - 최근 요청 확인
   - 메트릭 수집
   - 알림 기능

3. 3단계: 프로덕션 배포
   - Cloud Function 또는 CronJob으로 배포
   - 모니터링 대시보드

이 방식으로 구현을 진행할까요?