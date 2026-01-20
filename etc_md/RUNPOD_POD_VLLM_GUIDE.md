# RunPod Pod + 로컬 vLLM + Watchdog 구현 가이드

## 개요

본 문서는 **RunPod Pod 환경에서 vLLM을 직접 사용하고, 외부 Watchdog 스크립트로 자동 종료**하는 방식의 구현 가이드입니다.

### 아키텍처

```
┌─────────────────────────────────────────┐
│  RunPod Pod (상시 GPU 서버)            │
│  ┌──────────────────────────────────┐  │
│  │  FastAPI + vLLM (로컬)          │  │
│  │  - vLLM 모델 로드                │  │
│  │  - FastAPI 서버 실행             │  │
│  │  - 자동 종료 로직 없음           │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
              ↑
              │
┌─────────────────────────────────────────┐
│  외부 Watchdog 스크립트                 │
│  - GPU 사용률 모니터링                  │
│  - RunPod API로 Pod 제어                │
└─────────────────────────────────────────┘
```

---

## 주요 특징

### 1. vLLM 직접 사용
- **네트워크 오버헤드 없음**: 로컬에서 직접 호출
- **Continuous Batching 자동 활용**: vLLM이 여러 요청을 자동으로 배치 처리
- **높은 성능**: 최고의 추론 성능 달성

### 2. 외부 Watchdog
- **관심사 분리**: 서버는 비즈니스 로직에만 집중
- **유연성**: 모니터링 로직 변경 시 서버 재배포 불필요
- **안정성**: 서버 내부 종료 로직으로 인한 예기치 않은 종료 방지

### 3. 비용 최적화
- **자동 종료**: 유휴 시 Pod 자동 종료로 비용 절감
- **예측 가능한 비용**: 사용 시간 기준 과금

---

## 구현 단계

### 1단계: 환경 변수 설정

#### Pod 내부 (FastAPI + vLLM)

```bash
# vLLM 직접 사용 모드 활성화
export USE_POD_VLLM=true
export USE_RUNPOD=false  # Serverless Endpoint 비활성화

# vLLM 설정
export VLLM_TENSOR_PARALLEL_SIZE=1  # 단일 GPU
export VLLM_MAX_MODEL_LEN=4096  # 최대 모델 길이 (선택사항)

# 모델 설정
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

#### 외부 서버 (Watchdog)

```bash
# RunPod API 설정
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_POD_ID="your_pod_id"

# Watchdog 설정
export IDLE_THRESHOLD=5  # GPU 사용률 임계값 (%)
export CHECK_INTERVAL=60  # 체크 간격 (초)
export IDLE_LIMIT=5  # 연속 idle 횟수 (5분)
export MIN_RUNTIME=600  # 최소 실행 시간 (10분)
```

---

### 2단계: Pod 배포

#### Dockerfile 빌드

```bash
docker build -t review-analysis-api:latest .
```

#### RunPod Pod 생성

1. RunPod 대시보드에서 Pod 생성
2. GPU 타입 선택 (최소 24GB VRAM 권장)
3. Docker 이미지 업로드 또는 레지스트리에서 pull
4. 환경 변수 설정:
   - `USE_POD_VLLM=true`
   - `USE_RUNPOD=false`
   - `VLLM_TENSOR_PARALLEL_SIZE=1`

#### Pod 실행

```bash
# Pod 내부에서 실행
python app.py
```

---

### 3단계: Watchdog 실행

#### 방법 1: 별도 서버에서 실행

```bash
# 환경 변수 설정
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_POD_ID="your_pod_id"

# Watchdog 실행
python scripts/watchdog.py
```

#### 방법 2: Cron으로 주기적 실행

```bash
# crontab -e
*/5 * * * * /path/to/python /path/to/scripts/watchdog.py
```

#### 방법 3: 명령줄 인자 사용

```bash
python scripts/watchdog.py \
    --api-key "your_api_key" \
    --pod-id "your_pod_id" \
    --idle-threshold 5 \
    --check-interval 60 \
    --idle-limit 5 \
    --min-runtime 600
```

---

## 코드 변경사항

### 1. Config 설정 추가

**파일**: `src/config.py`

```python
# vLLM 직접 사용 설정 (RunPod Pod 환경)
USE_POD_VLLM: bool = os.getenv("USE_POD_VLLM", "false").lower() == "true"
VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_MAX_MODEL_LEN: Optional[int] = int(os.getenv("VLLM_MAX_MODEL_LEN")) if os.getenv("VLLM_MAX_MODEL_LEN") else None

# Watchdog 설정
RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
IDLE_THRESHOLD: int = int(os.getenv("IDLE_THRESHOLD", "5"))
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL", "60"))
IDLE_LIMIT: int = int(os.getenv("IDLE_LIMIT", "5"))
MIN_RUNTIME: int = int(os.getenv("MIN_RUNTIME", "600"))
```

### 2. LLMUtils에 vLLM 직접 사용 모드 추가

**파일**: `src/llm_utils.py`

주요 변경사항:
- `_init_vllm()`: vLLM 초기화
- `_generate_with_vllm()`: vLLM 비동기 추론
- `analyze_all_reviews_vllm()`: vLLM을 사용한 배치 분석
- `_create_sentiment_prompt()`: 프롬프트 생성
- `_parse_sentiment_response()`: 응답 파싱

### 3. SentimentAnalyzer 비동기 메서드 추가

**파일**: `src/sentiment_analysis.py`

```python
async def analyze_async(self, reviews, restaurant_id, max_retries):
    """vLLM 직접 사용 모드에서 비동기 분석"""
    if hasattr(self.llm_utils, 'use_pod_vllm') and self.llm_utils.use_pod_vllm:
        result = await self.llm_utils.analyze_all_reviews_vllm(...)
    else:
        result = self.llm_utils.analyze_all_reviews(...)
```

### 4. API 라우터 수정

**파일**: `src/api/routers/sentiment.py`

```python
@router.post("/analyze")
async def analyze_sentiment(...):
    if hasattr(analyzer.llm_utils, 'use_pod_vllm') and analyzer.llm_utils.use_pod_vllm:
        result = await analyzer.analyze_async(...)  # 비동기
    else:
        result = analyzer.analyze(...)  # 동기
```

---

## 사용 방법

### API 호출 (동일)

```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": 1,
    "reviews": [
      {
        "id": 1,
        "restaurant_id": 1,
        "content": "맛있어요!",
        ...
      }
    ]
  }'
```

### Python 코드

```python
# vLLM 직접 사용 모드에서는 자동으로 비동기 처리
from src.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# vLLM 모드에서는 내부적으로 비동기 처리
result = await analyzer.analyze_async(reviews, restaurant_id)
```

---

## 성능 비교

| 항목 | Serverless Endpoint | Pod + vLLM |
|------|-------------------|------------|
| **네트워크 오버헤드** | 있음 (~100-200ms) | 없음 |
| **모델 로딩** | 요청 시 (cold start) | 항상 메모리에 로드 |
| **처리 시간** | 10-15초 (10개 음식점) | 5-10초 (10개 음식점) |
| **비용** | 사용량 기반 | 사용 시간 기반 |
| **관리 복잡도** | 낮음 | 중간 |

---

## Watchdog 동작 방식

### 모니터링 프로세스

1. **GPU 사용률 확인**: `nvidia-smi`를 통해 GPU 사용률 조회
2. **최근 요청 확인**: FastAPI `/health` 엔드포인트 확인 (선택사항)
3. **Idle 판단**: GPU 사용률 < 임계값 && 최근 요청 없음
4. **연속 Idle 카운팅**: 연속 idle 횟수 추적
5. **Pod 종료**: 연속 idle 횟수 >= 제한 시 Pod 종료

### 안전장치

- **최소 실행 시간**: `MIN_RUNTIME` 동안은 종료하지 않음
- **활성 상태 감지**: GPU 사용률이 임계값 이상이면 idle 카운터 리셋
- **종료 실패 처리**: Pod 종료 실패 시 계속 모니터링

---

## 문제 해결

### 1. vLLM 모델 로딩 실패

**증상**: `ImportError: vLLM이 설치되지 않았습니다`

**해결**:
```bash
pip install vllm>=0.3.3
```

### 2. GPU 메모리 부족

**증상**: `CUDA out of memory`

**해결**:
- `VLLM_MAX_MODEL_LEN` 감소
- 더 큰 GPU로 변경
- `VLLM_TENSOR_PARALLEL_SIZE` 조정

### 3. Watchdog가 Pod를 찾을 수 없음

**증상**: `Pod 상태 조회 실패`

**해결**:
- `RUNPOD_POD_ID` 확인
- `RUNPOD_API_KEY` 확인
- RunPod API 권한 확인

### 4. Pod가 예기치 않게 종료됨

**증상**: Pod가 갑자기 종료

**해결**:
- `MIN_RUNTIME` 증가
- `IDLE_LIMIT` 증가
- Watchdog 로그 확인

---

## 모니터링 및 로깅

### Watchdog 로그

```bash
# Watchdog 실행 시 로그 출력
2024-01-01 12:00:00 - INFO - Watchdog 시작: Pod abc123, 임계값 5%, 체크 간격 60초
2024-01-01 12:01:00 - INFO - GPU 사용률: 3%
2024-01-01 12:01:00 - INFO - Idle 상태 감지 (1/5)
2024-01-01 12:05:00 - INFO - 연속 idle 시간 초과, Pod 종료
2024-01-01 12:05:01 - INFO - Pod abc123 종료 성공
```

### FastAPI 로그

```bash
# vLLM 모델 로딩
2024-01-01 12:00:00 - INFO - vLLM 모델 로딩 중: Qwen/Qwen2.5-7B-Instruct
2024-01-01 12:02:00 - INFO - ✅ vLLM 모델 로딩 완료

# 추론 로그
2024-01-01 12:03:00 - INFO - vLLM으로 100개 리뷰를 비동기 배치 분석
2024-01-01 12:03:05 - INFO - ✅ vLLM 배치 분석 완료: 긍정 60개 (60%), 부정 40개 (40%)
```

---

## 비용 최적화 팁

### 1. Watchdog 설정 최적화

```bash
# 더 공격적인 종료 (비용 절감)
IDLE_THRESHOLD=5
IDLE_LIMIT=3  # 3분
MIN_RUNTIME=300  # 5분

# 더 보수적인 종료 (안정성)
IDLE_THRESHOLD=3
IDLE_LIMIT=10  # 10분
MIN_RUNTIME=1800  # 30분
```

### 2. 사용 패턴 분석

- **피크 시간대**: Watchdog 비활성화 또는 `MIN_RUNTIME` 증가
- **비피크 시간대**: 더 공격적인 종료 설정

### 3. 예약 실행

- 특정 시간에만 Pod 실행
- Cron으로 Pod 시작/종료 스케줄링

---

## 참고 문서

- **API 명세서**: [API_SPECIFICATION.md](API_SPECIFICATION.md)
- **아키텍처 문서**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **비동기 vLLM 가이드**: [async_vllm.md](async_vllm.md)
- **Watchdog 상세**: [watchdog_healthcheck.md](watchdog_healthcheck.md)

---

## 결론

RunPod Pod + 로컬 vLLM + Watchdog 방식은 다음과 같은 장점을 제공합니다:

1. **최고 성능**: 네트워크 오버헤드 없이 최고의 추론 성능
2. **비용 최적화**: 자동 종료로 유휴 시간 비용 절감
3. **안정성**: 외부 모니터링으로 서버 안정성 향상
4. **유연성**: 모니터링 로직 변경 시 서버 재배포 불필요

이 방식은 **대규모 처리 및 성능이 중요한 프로덕션 환경**에 적합합니다.

