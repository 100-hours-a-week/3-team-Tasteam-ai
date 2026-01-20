제안

## 제안에 대한 의견

### 장점

1. SLA 보호 효과
   - 짧은 작업을 먼저 처리하면 평균 대기 시간 감소
   - 작은 요청의 TTFT 개선 가능

2. 이론적 근거
   - Shortest Job First (SJF)와 유사
   - 평균 대기 시간 최소화에 유리

3. 현재 측정 가능
   - Prefill/Decode 비용 측정 중 (`_generate_with_vllm()`)
   - 비용 기반 우선순위 계산 가능

---

### 주의사항 및 제약

#### 1. vLLM Continuous Batching과의 상호작용

현재 구현:
```python
# 모든 배치를 동시에 큐에 넣음
batch_results = await asyncio.gather(*[
    process_single_batch(...) for ... in batch_tasks
])
```

문제점:
- vLLM은 내부적으로 Continuous Batching을 수행
- 여러 요청을 하나의 배치로 묶어 처리
- 클라이언트 측 우선순위가 vLLM 내부 배치 구성에 직접 반영되지 않을 수 있음

#### 2. Prefill/Decode 비용 예측의 어려움

현재 측정:
- Prefill: 입력 토큰 수에 비례
- Decode: 출력 토큰 수에 비례

예측 어려움:
- 출력 토큰 수는 요청 전에 알기 어려움
- `max_tokens`는 상한일 뿐 실제 생성 토큰 수와 다를 수 있음

#### 3. 태스크별 특성 차이

| 태스크 | 입력 토큰 | 출력 토큰 | 총 비용 | 우선순위 |
|--------|----------|----------|--------|---------|
| 감성 분석 | 많음 (100-1000) | 적음 (10-50) | 중간 | 중간 |
| 요약 | 중간 (50-200) | 중간 (50-150) | 중간 | 중간 |
| 강점 추출 | 많음 (200-500) | 많음 (100-300) | 높음 | 낮음 |
| 쿼리 확장 | 적음 (10-30) | 적음 (20-50) | 낮음 | 높음 |

---

### 개선된 제안

#### 옵션 1: Prefill 비용 기반 우선순위 (권장)

이유:
- Prefill 비용은 입력 토큰 수로 예측 가능
- Prefill이 GPU를 더 오래 점유하는 경우가 많음
- TTFT에 직접 영향

구현:
```python
# 입력 토큰 수로 Prefill 비용 추정
def estimate_prefill_cost(prompts: List[str]) -> float:
    total_input_tokens = sum(estimate_tokens(p) for p in prompts)
    return total_input_tokens  # Prefill은 입력 토큰에 비례

# 우선순위 큐 (작은 비용부터)
import heapq
priority_queue = []
for task in tasks:
    cost = estimate_prefill_cost(task.prompts)
    heapq.heappush(priority_queue, (cost, task))
```

#### 옵션 2: 하이브리드 우선순위 큐

구조:
1. 긴급 요청 (SLA 임박): 최우선
2. Prefill 비용이 작은 요청: 우선
3. 나머지: FIFO

#### 옵션 3: 태스크 타입별 우선순위

```python
TASK_PRIORITY = {
    "query_expansion": 1,      # 가장 빠름
    "sentiment_analysis": 2,   # 중간
    "summarization": 3,        # 중간
    "strength_extraction": 4,  # 가장 느림
}
```

---

### 구현 시 고려사항

#### 1. vLLM과의 통합

현재:
- `asyncio.gather()`로 모든 요청을 동시에 전송
- vLLM이 내부적으로 배치 구성

개선:
- 우선순위 큐에서 순차적으로 vLLM에 전송
- 세마포어로 동시 전송 수 제한 유지

#### 2. 메트릭 수집

현재:
- Prefill/Decode 비용 측정 중

추가 필요:
- 태스크별 평균 Prefill/Decode 비용 통계
- 우선순위 변경 전후 비교 데이터

#### 3. 공정성 (Fairness)

문제:
- SJF는 큰 작업을 계속 뒤로 밀 수 있음 (Starvation)

해결:
- Aging: 대기 시간이 길어지면 우선순위 상승
- 최대 대기 시간 제한

---

### 최종 의견

제안은 타당하나, 다음을 권장합니다:

1. 단계적 도입
   - 1단계: Prefill 비용 기반 우선순위 (예측 가능)
   - 2단계: 실제 Prefill/Decode 비용 데이터 수집
   - 3단계: 비용 기반 우선순위로 전환

2. 태스크 타입별 우선순위
   - 쿼리 확장 → 감성 분석/요약 → 강점 추출
   - 구현이 단순하고 효과적

3. vLLM 특성 고려
   - Continuous Batching과의 상호작용 테스트
   - 실제 성능 개선 측정 후 확장

4. 모니터링
   - 우선순위 변경 전후 Goodput 비교
   - SLA 준수율 모니터링

결론: 제안은 유효하지만, Prefill 비용 기반 우선순위나 태스크 타입별 우선순위부터 시작하고, 실제 성능 데이터를 수집한 뒤 정교화하는 것을 권장합니다.

--> 'Prefill 비용 기반 우선순위'으로 결정.

---

적용


## 구현 완료 요약

### 1. Config 설정 추가
- `VLLM_USE_PRIORITY_QUEUE`: 우선순위 큐 사용 여부 (기본값: `true`)
- `VLLM_PRIORITY_BY_PREFILL_COST`: Prefill 비용 기반 우선순위 (기본값: `true`)

### 2. Prefill 비용 계산 함수 추가
- `_estimate_prefill_cost()`: 프롬프트의 토큰 수를 추정하여 Prefill 비용 계산

### 3. 우선순위 큐 적용
다음 메서드에 Prefill 비용 기반 우선순위 큐 적용:
- `analyze_multiple_restaurants_vllm()`: 감성 분석 배치 처리
- `summarize_multiple_restaurants_vllm()`: 요약 배치 처리
- `extract_strengths_multiple_restaurants_vllm()`: 강점 추출 배치 처리
  - comparison_summary 생성 단계
  - target_restaurant_strength 생성 단계
  - 단일 단계 파이프라인

### 동작 방식
1. 각 배치 태스크의 프롬프트를 생성하여 Prefill 비용 추정
2. `heapq`를 사용하여 Prefill 비용이 작은 순서로 정렬
3. 정렬된 순서대로 `asyncio.gather()`에 전달하여 처리
4. 세마포어로 동시 처리 수 제한 유지

### 환경 변수 설정
```bash
# 우선순위 큐 활성화 (기본값: true)
export VLLM_USE_PRIORITY_QUEUE=true

# Prefill 비용 기반 우선순위 (기본값: true)
export VLLM_PRIORITY_BY_PREFILL_COST=true

# 비활성화하려면
export VLLM_USE_PRIORITY_QUEUE=false
```

### 기대 효과
- 작은 요청의 SLA 보호: Prefill 비용이 작은 태스크를 먼저 처리하여 TTFT 개선
- 평균 대기 시간 감소: Shortest Job First 방식으로 평균 대기 시간 최소화
- 하위 호환성: 환경 변수로 기존 FIFO 방식으로 전환 가능

구현이 완료되었습니다. 실제 성능 개선 효과는 운영 환경에서 측정해 확인하세요.