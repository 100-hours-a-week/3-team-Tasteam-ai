세마포어와 동적 배치 크기를 함께 사용하는 이유 확인 중.


[2 tools called]


세마포어와 동적 배치 크기는 서로 다른 레벨에서 OOM을 방지합니다.

## 세마포어와 동적 배치 크기의 역할

### 1. 동적 배치 크기 (Dynamic Batch Size)
역할: 단일 배치 내부의 크기 제어 (세로 방향)

```python
# 동적 배치 크기 계산
dynamic_batch_size = calculate_dynamic_batch_size(content_list, max_tokens_per_batch=4000)
# 예: 리뷰가 길면 batch_size=30, 짧으면 batch_size=80

batches = [
    content_list[i:i + dynamic_batch_size]  # 각 배치 내부 크기 조절
    for i in range(0, len(content_list), dynamic_batch_size)
]
```

목적:
- 배치 하나의 메모리 사용량 제어
- 리뷰 길이에 따라 배치 크기 자동 조정
- 단일 배치가 메모리 한도를 넘지 않도록 방지

예시:
- 긴 리뷰 500개: batch_size=30 (각 배치가 4000 토큰 이하)
- 짧은 리뷰 500개: batch_size=80 (각 배치가 4000 토큰 이하)

### 2. 세마포어 (Semaphore)
역할: 동시 처리되는 배치 개수 제한 (가로 방향)

```python
max_concurrent = Config.VLLM_MAX_CONCURRENT_BATCHES  # 기본값: 20
semaphore = Semaphore(max_concurrent)

async def process_single_batch(batch):
    async with semaphore:  # 최대 20개만 동시 처리
        return await vllm_generate(batch)
```

목적:
- 동시에 처리되는 배치 수 제한
- 전체 메모리 사용량 누적 방지
- 여러 배치가 동시에 큐에 들어가도 실제 처리 수를 제한

예시:
- 100개 배치가 큐에 있음 → 세마포어가 최대 20개만 동시 처리
- 나머지 80개는 대기 중 (처리 완료 시 순차적으로 진행)

## 둘을 함께 사용하는 이유

### 시나리오 1: 동적 배치 크기만 사용하는 경우

```python
# 동적 배치 크기만 사용 (세마포어 없음)
batches = [배치1, 배치2, ..., 배치100]  # 각 배치는 적절한 크기

# 문제: 100개 배치를 모두 동시에 처리 시도
await asyncio.gather(*[process(batch) for batch in batches])
# → GPU 메모리 부족! OOM 발생!
```

문제점:
- 각 배치는 안전하지만, 100개를 동시 처리하면 메모리 누적로 OOM 발생
- 배치 수가 많을수록 위험 증가

### 시나리오 2: 세마포어만 사용하는 경우

```python
# 세마포어만 사용 (고정 배치 크기)
semaphore = Semaphore(20)  # 최대 20개 동시 처리

# 문제: 고정 배치 크기 사용
batches = [content_list[i:i+50] for i in range(0, len(content_list), 50)]
# → 긴 리뷰가 많으면 각 배치가 8000 토큰! 단일 배치에서도 OOM 발생!
```

문제점:
- 동시 처리 수는 제한되지만, 배치 하나가 너무 크면 단일 배치에서도 OOM 발생
- 리뷰 길이에 따라 위험도 변동

### 시나리오 3: 동적 배치 크기 + 세마포어 함께 사용

```python
# 1단계: 동적 배치 크기로 각 배치 내부 크기 조절
dynamic_batch_size = calculate_dynamic_batch_size(content_list)  # 리뷰 길이에 따라 조정
batches = [content_list[i:i+dynamic_batch_size] for i in range(0, len(content_list), dynamic_batch_size)]
# → 각 배치는 안전한 크기 (4000 토큰 이하)

# 2단계: 세마포어로 동시 처리 수 제한
semaphore = Semaphore(20)  # 최대 20개만 동시 처리

async def process_with_limit(batch):
    async with semaphore:  # 20개만 동시 처리
        return await vllm_generate(batch)

# → 각 배치는 안전하고, 동시 처리 수도 제한되어 메모리 안전!
```

효과:
- 배치 단위 메모리 안전성: 동적 배치 크기로 단일 배치의 메모리 사용량 제어
- 전체 메모리 안전성: 세마포어로 동시 처리 수 제한해 메모리 누적 방지

## 시각적 설명

```
[메모리 사용량 관점]

동적 배치 크기만:
배치1 (안전)  |  배치2 (안전)  |  배치3 (안전)  |  ...  |  배치100 (안전)
     ↓              ↓              ↓                        ↓
  모두 동시 처리 → GPU 메모리 누적 → OOM 발생!

세마포어만:
배치1 (8000 토큰) ❌ |  배치2 (8000 토큰) ❌ |  ...  (고정 크기 사용)
     ↓                  ↓
  각 배치가 너무 큼 → 단일 배치에서도 OOM 발생!

동적 배치 크기 + 세마포어:
배치1 (4000 토큰) ✅ |  배치2 (4000 토큰) ✅ |  배치3 (4000 토큰) ✅ |  ...
     ↓                    ↓                    ↓
  최대 20개만 동시 처리 (세마포어) → 안전!
```

## 실제 예시

시나리오: 100개 레스토랑, 각각 500개 리뷰 처리

### 동적 배치 크기 적용 후:
- 레스토랑당 500개 리뷰 → 10개 배치로 분할 (동적 크기: 50)
- 총 1000개 배치 생성 (100개 레스토랑 × 10개 배치)

### 세마포어 없이:
```
1000개 배치를 모두 동시 처리 시도
→ GPU 메모리: 1000 × 2GB = 2000GB 필요!
→ 실제 GPU 메모리: 24GB
→ OOM 발생! ❌
```

### 세마포어 적용 후:
```
1000개 배치 중 최대 20개만 동시 처리
→ GPU 메모리: 20 × 2GB = 40GB (하지만 처리 완료 시 즉시 해제)
→ 실제 사용 메모리: 약 25GB (동적 배치 크기로 각 배치가 작음)
→ 안전하게 처리! ✅
```

## 결론

- 동적 배치 크기: 단일 배치의 메모리 사용량 제어 (배치 내부 크기)
- 세마포어: 동시 처리 배치 수 제한 (동시성 제어)

두 기법을 함께 사용해야 배치 단위와 전체 메모리 모두 안전합니다.

```
안전한 배치 처리 = 동적 배치 크기 (단일 배치 안전) + 세마포어 (전체 안전)
```