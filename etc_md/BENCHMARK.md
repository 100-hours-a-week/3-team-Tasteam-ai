## 생성된 파일

**`scripts/benchmark.py`**: 성능 벤치마크 스크립트

## 주요 기능

### 1. 성능 지표 수집
- 지연 시간 (Latency): 평균, 최소, 최대, 중앙값, P95, P99, 표준편차
- 처리량 (Throughput): 초당 처리 리뷰 수
- GPU 메트릭: GPU 사용률, 메모리 사용량 (vLLM 모드)
- 성공/실패 통계

### 2. 단일/배치 모드 측정
- 단일 레스토랑 감성 분석 (`--mode single`)
- 배치 처리 (`--mode batch`)

### 3. 여러 실행 모드 비교 (`--compare`)
- 로컬 Transformers 모드
- RunPod Serverless vLLM 모드
- RunPod Pod + vLLM 모드

### 4. 결과 저장
- JSON 형식으로 결과 저장
- 타임스탬프 포함 파일명 자동 생성

## 사용 방법

### 기본 사용 (배치 모드)

```bash
# 1. 테스트 데이터 생성
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 100 --restaurants 5

# 2. 서버 실행 (로컬 Transformers 모드)
export USE_POD_VLLM="false"
export USE_RUNPOD="false"
python app.py

# 3. 별도 터미널에서 벤치마크 실행
python scripts/benchmark.py --test-data test_data.json --mode batch --iterations 10
```

### RunPod Serverless vLLM 모드 사용

```bash
# 1. 환경 변수 설정
export USE_POD_VLLM="false"
export USE_RUNPOD="true"
#export RUNPOD_API_KEY="your_runpod_api_key"  # 필수
export RUNPOD_ENDPOINT_ID="g09uegksn7h7ed"   # 기본값 또는 본인의 엔드포인트 ID

# 2. 서버 실행
python app.py

# 3. 별도 터미널에서 벤치마크 실행
python scripts/benchmark.py --test-data test_data.json --mode batch --iterations 10
```

**주의사항:**
- `RUNPOD_API_KEY` 환경 변수가 설정되지 않으면 서버 시작 시 오류가 발생합니다
- `RUNPOD_ENDPOINT_ID`는 기본값이 `g09uegksn7h7ed`이며, 본인의 엔드포인트 ID를 사용하려면 변경하세요


### RunPod Pod + vLLM 모드
```bash
   export USE_POD_VLLM="true"
   export USE_RUNPOD="false"
   # 별도 터미널에서 Enter 눌러 진행

    # 2. 서버 실행
    python app.py

   # 3. 별도 터미널에서 벤치마크 실행
    python scripts/benchmark.py --test-data test_data.json --mode batch --iterations 10
    ```

### 여러 모드 비교

`--compare` 옵션을 사용하면 다음 3가지 모드를 순차적으로 측정하고 비교합니다:

```bash
# 비교 모드 실행 (자동으로 3가지 모드를 순차 측정)
python scripts/benchmark.py --test-data test_data.json --compare --mode batch --iterations 5
```

**실행 순서:**

1. **로컬 Transformers 모드**
   ```bash
   export USE_POD_VLLM="false"
   export USE_RUNPOD="false"
   python app.py  # 서버 실행
   # 별도 터미널에서 Enter 눌러 진행
   ```

2. **RunPod Serverless vLLM 모드**
   ```bash
   export USE_POD_VLLM="false"
   export USE_RUNPOD="true"
   export RUNPOD_API_KEY="your_runpod_api_key"  # 필수
   export RUNPOD_ENDPOINT_ID="g09uegksn7h7ed"   # 기본값 또는 본인의 엔드포인트 ID
   python app.py  # 서버 재시작
   # 별도 터미널에서 Enter 눌러 진행
   ```

3. **RunPod Pod + vLLM 모드**
   ```bash
   export USE_POD_VLLM="true"
   export USE_RUNPOD="false"
   python app.py  # 서버 재시작
   # 별도 터미널에서 Enter 눌러 진행
   ```

**주의사항:**
- 각 모드별로 환경 변수를 변경한 후 서버를 재시작해야 합니다
- 비교 모드에서는 각 모드별로 서버 재시작 후 Enter를 입력해야 합니다
- RunPod Serverless vLLM 모드는 `RUNPOD_API_KEY`와 `RUNPOD_ENDPOINT_ID` 환경 변수가 필수입니다

### 옵션 설명

```bash
python scripts/benchmark.py \
  --test-data test_data.json \      # 테스트 데이터 파일 경로
  --base-url http://localhost:8001 \ # API 서버 URL
  --mode batch \                     # single 또는 batch
  --iterations 10 \                  # 측정 반복 횟수
  --warmup 2 \                       # 워밍업 반복 횟수
  --output results.json \            # 결과 저장 파일 경로 (선택)
  --compare                           # 여러 모드 비교 모드
```

## 출력 예시

```
================================================================
성능 벤치마크 결과 요약
================================================================

지연 시간 (Latency):
  평균: 2.3456s
  최소: 2.1234s
  최대: 2.5678s
  중앙값: 2.3400s
  P95: 2.5000s
  P99: 2.5500s
  표준편차: 0.1234s

처리량 (Throughput):
  평균: 42.67 reviews/s
  최소: 38.90 reviews/s
  최대: 47.05 reviews/s
  중앙값: 42.75 reviews/s

GPU 메트릭:
  시작 시 GPU 사용률: 15.3%
  시작 시 메모리 사용: 10240MB / 40960MB (25.0%)
  종료 시 GPU 사용률: 85.7%
  종료 시 메모리 사용: 35840MB / 40960MB (87.5%)

성공: 10/10
실패: 0/10
================================================================
```

## 성능 비교 예시 (`--compare` 모드)

```
================================================================
모드별 성능 비교 결과
================================================================

로컬 Transformers:
  평균 지연 시간: 3.4567s
  평균 처리량: 28.94 reviews/s

RunPod Serverless vLLM:
  평균 지연 시간: 2.1234s
  평균 처리량: 47.11 reviews/s

RunPod Pod + vLLM:
  평균 지연 시간: 2.3456s
  평균 처리량: 42.67 reviews/s

================================================================
성능 개선 분석
================================================================

기준 모드: 로컬 Transformers

RunPod Serverless vLLM vs 로컬 Transformers:
  ✅ 지연 시간 감소: 38.58% (빠름)
  ✅ 처리량 향상: 62.78% (향상)

RunPod Pod + vLLM vs 로컬 Transformers:
  ✅ 지연 시간 감소: 32.14% (빠름)
  ✅ 처리량 향상: 47.46% (향상)
```

## 환경 변수 설정

### RunPod Serverless vLLM 모드 필수 환경 변수

| 환경 변수 | 필수 | 기본값 | 설명 |
|-----------|------|--------|------|
| `USE_POD_VLLM` | 필수 | `false` | RunPod Pod + vLLM 사용 여부 (`false`로 설정) |
| `USE_RUNPOD` | 필수 | `true` | RunPod Serverless vLLM 사용 여부 (`true`로 설정) |
| `RUNPOD_API_KEY` | 필수 | 없음 | RunPod API 키 |
| `RUNPOD_ENDPOINT_ID` | 선택 | `g09uegksn7h7ed` | RunPod 엔드포인트 ID |

**설정 예시:**

```bash
# .env 파일 또는 직접 export
export USE_POD_VLLM="false"
export USE_RUNPOD="true"
export RUNPOD_API_KEY="your_runpod_api_key_here"
export RUNPOD_ENDPOINT_ID="g09uegksn7h7ed"  # 또는 본인의 엔드포인트 ID
```

## 수집 가능한 성능 지표

### 현재 테스트 가이드만으로 불가능했던 것들
1. 서버 측 처리 시간
2. 처리량 (초당 처리 리뷰 수)
3. 통계적 분석 (P95, P99, 표준편차)
4. GPU 사용률 및 메모리 사용량
5. 여러 모드 간 성능 비교

### 이제 가능한 것들
1. 기존 모델 추론의 성능 지표: 지연 시간, 처리량, GPU 사용률
2. 성능 병목 요소: P95/P99로 극값 분석, 표준편차로 안정성 평가
3. 최적화 적용 후 성능 비교: `--compare` 모드로 여러 모드 비교
4. 개선된 수치 측정: 지연 시간 감소율, 처리량 향상률 자동 계산

이 스크립트로 필요한 성능 지표를 수집할 수 있습니다.