프로젝트의 LLM 호출 및 배치 처리 구현을 확인 중입니다.


[4 tools called]


배치 처리 엔드포인트 확인 중입니다.


[2 tools called]


## `async_batch_llm.md` 검토 및 적용 가능성 분석

### 현재 프로젝트 상태

1. LLM 호출 방식: 동기 (`requests` 사용)
   - `_call_runpod()`: 동기 HTTP 요청 + polling
   - `analyze_all_reviews()`: 단일 음식점 처리, 순차 배치

2. 배치 처리: 순차 처리
   - `count_sentiments()`: for loop로 순차 배치 처리
   - 여러 음식점 동시 처리 미지원

3. FastAPI 라우터: 비동기 지원 (`async def`)

---

### `async_batch_llm.md` 제안 내용

1. 비동기 큐 배치 처리
   - 여러 음식점 동시 처리 (`asyncio.gather`)
   - 각 음식점 내부 배치 분할
   - vLLM의 continuous batching 활용

2. 아키텍처
   ```
   process_restaurant(r1) → batch(1~5), batch(6~10), batch(11~15) → async infer
   process_restaurant(r2) → batch(1~5), batch(6~10) → async infer
   → aggregate → result
   ```

---

### 적용 가능성: 가능

#### 장점

1. 성능 향상
- 여러 음식점 동시 처리로 처리 시간 단축
- vLLM의 continuous batching 활용으로 GPU 효율 향상

2. 리소스 효율
- 비동기 I/O로 대기 시간 감소
- vLLM이 여러 요청을 하나의 배치로 처리

3. 확장성
- 대량 음식점 처리 시 효과적

#### 필요한 변경사항

1. 비동기 HTTP 클라이언트 도입
   ```python
   # 현재: requests (동기)
   response = requests.post(url, json=data, headers=headers)
   
   # 변경: httpx 또는 aiohttp (비동기)
   async with httpx.AsyncClient() as client:
       response = await client.post(url, json=data, headers=headers)
   ```

2. RunPod 호출 비동기화
   ```python
   # 현재: _call_runpod() - 동기
   def _call_runpod(self, prompt: str) -> str:
       response = requests.post(url, ...)
       # polling...
   
   # 변경: 비동기 버전
   async def _call_runpod_async(self, prompt: str) -> str:
       async with httpx.AsyncClient() as client:
           response = await client.post(url, ...)
           # 비동기 polling...
   ```

3. 배치 처리 로직 추가
   ```python
   async def process_restaurant(rid, reviews, batch_size=5):
       batches = [reviews[i:i+batch_size] 
                  for i in range(0, len(reviews), batch_size)]
       results = await asyncio.gather(*[
           llm_infer_async(b) for b in batches
       ])
       return aggregate(results)
   ```

4. 배치 엔드포인트 구현
   - API 명세에는 있으나 구현 없음
   - `async_batch_llm.md` 방식으로 구현 가능

---

### 구체적 적용 방안

#### 1단계: 비동기 HTTP 클라이언트 도입

```python
# requirements.txt에 추가
httpx>=0.24.0

# llm_utils.py 수정
import httpx

async def _call_runpod_async(self, prompt: str) -> str:
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 비동기 요청 및 polling
        ...
```

#### 2단계: 비동기 배치 처리 메서드 추가

```python
async def analyze_all_reviews_async(
    self,
    review_list: List[str],
    restaurant_id: Union[int, str],
    batch_size: int = 50,  # context middle lost 방지
) -> Dict[str, Any]:
    # 배치로 나누어 비동기 처리
    batches = [review_list[i:i+batch_size] 
               for i in range(0, len(review_list), batch_size)]
    
    results = await asyncio.gather(*[
        self._analyze_batch_async(batch, restaurant_id)
        for batch in batches
    ])
    
    # 집계
    return aggregate_results(results)
```

#### 3단계: 여러 음식점 동시 처리

```python
async def analyze_multiple_restaurants(
    self,
    restaurant_reviews: Dict[int, List[str]],
) -> Dict[int, Dict[str, Any]]:
    results = await asyncio.gather(*[
        self.analyze_all_reviews_async(reviews, rid)
        for rid, reviews in restaurant_reviews.items()
    ])
    return {rid: result for rid, result in zip(restaurant_reviews.keys(), results)}
```

---

### 주의사항

1. RunPod API 제한
   - 동시 요청 수 제한 확인 필요
   - Rate limiting 고려

2. 메모리 사용
   - 동시 요청 증가 시 메모리 사용 증가
   - Semaphore로 동시성 제어 권장

3. 에러 처리
   - 일부 음식점 실패 시 처리 방식 정의
   - 부분 실패 허용 여부 결정

---

### 결론

적용 가능하며 권장합니다.

- 현재: 동기 순차 처리
- 제안: 비동기 병렬 처리
- 효과: 처리 시간 단축, GPU 효율 향상, 확장성 개선

우선순위:
1. 비동기 HTTP 클라이언트 도입 (httpx)
2. `_call_runpod_async()` 구현
3. 배치 처리 비동기화
4. 여러 음식점 동시 처리 엔드포인트 추가

예상 성능 개선:
- 단일 음식점: 10-20% 개선 (비동기 I/O)
- 여러 음식점: 50-80% 개선 (병렬 처리)

이 방향으로 구현 진행할까요?