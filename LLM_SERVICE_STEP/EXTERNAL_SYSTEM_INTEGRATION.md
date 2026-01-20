# 외부 시스템/도구/서비스 통합 설계 문서

## 목차
1. [개요](#개요)
2. [외부 시스템 통합 현황](#외부-시스템-통합-현황)
3. [MCP 도입 여부 결정](#mcp-도입-여부-결정)
4. [상용 API vs 자체 모델 분석](#상용-api-vs-자체-모델-분석)
5. [현재 인터페이스 설계](#현재-인터페이스-설계)
6. [확장성 계획](#확장성-계획)
7. [비용 효율성 분석](#비용-효율성-분석)
8. [결론](#결론)

---

## 개요

본 문서는 프로젝트에서 사용하는 외부 시스템, 도구, 서비스와의 통합 방식을 설계하고, MCP(Model Context Protocol) 도입 여부, 상용 API 사용 전략, 확장성 계획을 다룹니다.

### 설계 목표
- **확장성**: 새로운 외부 서비스 추가 시 최소한의 코드 변경
- **유지보수성**: 명확한 인터페이스와 모듈화된 구조
- **비용 효율성**: 자체 모델 vs 상용 API 비용 분석 및 최적화
- **안정성**: 예외 처리, 재시도, 속도 제한 제어

---

## 외부 시스템 통합 현황

### 1. 현재 사용 중인 외부 시스템

#### 1.1. RunPod (GPU 인프라 및 vLLM 서빙)

**역할:**
- GPU 인프라 제공 (GPU 서버)
- vLLM 서빙 환경 제공

**통합 방식:**
- **GPU 서버 + 로컬 vLLM** (현재 사용 중, 권장)
  - GPU 서버 환경에서 vLLM을 직접 실행
  - 네트워크 오버헤드 없음
  - Continuous Batching 자동 활용
  - 높은 처리량 (5-10배 향상)
- **RunPod Serverless Endpoint** (이전 사용, 현재 미사용)
  - HTTP API를 통한 비동기 호출
  - 네트워크 오버헤드 존재
  - Cold Start 지연

**호출 목적:**
- LLM 추론 (Qwen2.5-7B-Instruct)
- 감성 분석, 리뷰 요약, 강점 추출

**인터페이스:**
- vLLM Python SDK (로컬 vLLM)
- HTTP REST API (Serverless Endpoint, 미사용)

#### 1.2. Qdrant (벡터 데이터베이스)

**역할:**
- 벡터 인덱싱 및 유사도 검색
- 리뷰 임베딩 저장 및 검색

**통합 방식:**
- **Qdrant on-disk 모드** (현재 사용 중)
  - 로컬 파일 시스템에 저장
  - MMAP 활용으로 메모리 효율성
  - 별도 서버 프로세스 불필요
  - 데이터 영속성 및 안정성

**호출 목적:**
- 리뷰 벡터 검색 (의미 기반 유사도 검색)
- **대표 벡터 기반 TOP-K 리뷰 선택**: 감성 분석, 요약, 강점 추출에서 사용
  - 레스토랑의 대표 벡터 계산 (모든 리뷰의 가중 평균)
  - 대표 벡터 주위에서 TOP-K 리뷰 검색 (기본값: 20개)
  - 토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축
- 레스토랑 대표 벡터 저장 및 검색 (`restaurant_vectors` 컬렉션)
- 강점 임베딩 비교 (Step G: 타겟 vs 비교군 유사도 계산)

**인터페이스:**
- Qdrant Python Client (로컬 파일 시스템 접근)

#### 1.3. SentenceTransformer (임베딩 모델)

**역할:**
- 텍스트를 벡터로 변환
- 한국어 특화 임베딩 생성

**통합 방식:**
- **로컬 실행** (현재 사용 중)
  - 모델: `jhgan/ko-sbert-multitask`
  - GPU + FP16 최적화
  - 배치 인코딩 지원

**호출 목적:**
- 리뷰 텍스트 임베딩 생성
- 쿼리 텍스트 임베딩 생성
- 강점 텍스트 임베딩 생성

**인터페이스:**
- SentenceTransformer Python 라이브러리

### 2. 외부 시스템 통합 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (이 API)                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  API Routers                                                      │  │
│  │  - /api/v1/sentiment/analyze                                     │  │
│  │  - /api/v1/llm/summarize                                         │  │
│  │  - /api/v1/llm/extract/strengths                                │  │
│  │  - /api/v1/vector/search/similar                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│        ┌─────────────────────┼─────────────────────┐                    │
│        │                     │                     │                    │
│        ▼                     ▼                     ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │ LLM Utils    │    │ Vector       │    │ Sentiment    │            │
│  │              │    │ Search       │    │ Analyzer     │            │
│  │ - vLLM       │    │              │    │              │            │
│  │   통합       │    │ - Qdrant     │    │ - LLM Utils  │            │
│  │              │    │   통합       │    │   사용       │            │
│  │ - Sentence   │    │              │    │              │            │
│  │   Transformer│    │ - Sentence   │    │              │            │
│  │   통합       │    │   Transformer│    │              │            │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘            │
│         │                   │                    │                     │
└─────────┼───────────────────┼────────────────────┼─────────────────────┘
          │                   │                    │
          │                   │                    │
          ▼                   ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ GPU 서버        │  │ Qdrant         │  │ Sentence        │
│ + 로컬 vLLM     │  │ (on-disk)      │  │ Transformer     │
│                 │  │                 │  │ (로컬)         │
│ - Qwen2.5-7B-   │  │ - 벡터 인덱싱   │  │ - 임베딩 생성   │
│   Instruct      │  │ - 유사도 검색   │  │ - GPU + FP16    │
│ - Continuous    │  │ - MMAP 활용     │  │ - 배치 처리     │
│   Batching      │  │ - 데이터 영속성 │  │                 │
│ - PagedAttention│  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 3. 외부 시스템별 상세 통합 정보

#### 3.1. RunPod 통합

**통합 모듈:** `src/llm_utils.py`

**통합 방식:**
```python
# GPU 서버 + 로컬 vLLM (현재 사용)
if self.use_pod_vllm:
    from vllm import LLM, SamplingParams
    self.llm = LLM(
        model=model_name,
        tensor_parallel_size=Config.VLLM_TENSOR_PARALLEL_SIZE,
        max_model_len=Config.VLLM_MAX_MODEL_LEN,
    )
    # 비동기 추론
    async def _generate_with_vllm(self, prompts, ...):
        outputs = await loop.run_in_executor(
            self.executor,
            self.llm.generate,
            prompts,
            sampling_params
        )
```

**예외 처리:**
- 재시도 로직 (`max_retries`)
- OOM 방지 (동적 배치 크기, 세마포어)
- 에러 로깅 및 메트릭 수집

**속도 제한:**
- 세마포어 기반 동시 처리 수 제한 (`VLLM_MAX_CONCURRENT_BATCHES`: 20)
- **우선순위 큐 (Prefill 비용 기반)**: 작은 요청 우선 처리
  - Prefill 비용은 입력 토큰 수로 정확히 예측 가능
  - Shortest Job First (SJF) 알고리즘 적용
  - 작은 요청 TTFT 30-40% 개선 (2.5초 → 1.8초)
  - SLA 준수율 85% → 92% 향상

#### 3.2. Qdrant 통합

**통합 모듈:** `src/vector_search.py`

**통합 방식:**
```python
# Qdrant on-disk 모드
if Config.QDRANT_URL == ":memory:":
    client = QdrantClient(location=":memory:")
elif os.path.exists(Config.QDRANT_URL):
    # 로컬 경로 (on-disk)
    client = QdrantClient(path=Config.QDRANT_URL)
else:
    # 원격 서버 (미사용)
    client = QdrantClient(url=Config.QDRANT_URL)
```

**예외 처리:**
- 컬렉션 존재 여부 확인
- 포인트 업로드 실패 시 재시도
- 에러 로깅

**속도 제한:**
- 배치 업로드로 네트워크 오버헤드 최소화 (on-disk 모드에서는 불필요)

#### 3.3. SentenceTransformer 통합

**통합 모듈:** `src/vector_search.py`, `src/api/dependencies.py`

**통합 방식:**
```python
# GPU + FP16 최적화
encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
if Config.USE_GPU and torch.cuda.is_available():
    encoder = encoder.cuda()
    if Config.USE_FP16:
        encoder = encoder.half()
```

**예외 처리:**
- GPU 사용 불가 시 CPU로 자동 전환
- 배치 인코딩 실패 시 개별 처리

---

## MCP 도입 여부 결정

### 1. MCP (Model Context Protocol) 개요

**MCP란:**
- AI 애플리케이션과 외부 도구/서비스를 연결하는 표준 프로토콜
- 클라이언트-서버 아키텍처
- 도구 정의, 권한 모델, 상용 API 호출 지원

### 2. MCP 도입 여부 분석

#### 2.1. 현재 프로젝트 특성

**현재 아키텍처:**
- **직접 통합**: 각 외부 서비스를 Python 라이브러리로 직접 통합
- **명확한 책임**: 각 모듈이 특정 외부 서비스만 담당
- **높은 성능**: 네트워크 오버헤드 최소화 (로컬 실행 위주)

**외부 서비스 특성:**
- **RunPod vLLM**: 로컬 실행 (네트워크 없음)
- **Qdrant**: on-disk 모드 (로컬 파일 시스템)
- **SentenceTransformer**: 로컬 실행

#### 2.2. MCP 도입 시 장단점

**장점:**
- **표준화**: 외부 도구 통합의 표준 인터페이스
- **확장성**: 새로운 도구 추가 시 MCP 서버만 추가
- **권한 관리**: 중앙화된 권한 모델
- **상용 API 통합**: OpenAI, Anthropic 등 상용 API 통합 용이

**단점:**
- **추가 복잡성**: MCP 서버/클라이언트 구조 추가
- **성능 오버헤드**: 프로토콜 계층 추가로 인한 지연
- **로컬 실행과의 불일치**: 현재는 대부분 로컬 실행인데 MCP는 원격 통신 전제
- **학습 곡선**: 팀원들의 MCP 학습 필요

#### 2.3. 결정: MCP 미도입

**결정 이유:**

1. **로컬 실행 중심 아키텍처**
   - 현재 대부분의 외부 서비스가 로컬에서 실행됨
   - MCP는 원격 통신을 전제로 하므로 현재 아키텍처와 맞지 않음

2. **성능 우선순위**
   - 네트워크 오버헤드 최소화가 핵심 목표
   - MCP 프로토콜 계층 추가는 성능 저하 요인

3. **단순성 유지**
   - 현재 직접 통합 방식이 명확하고 유지보수 용이
   - MCP 도입 시 추가 복잡성만 증가

4. **현재 요구사항 충족**
   - 현재 외부 서비스 통합이 잘 작동 중
   - MCP가 필요한 복잡한 도구 체인이 없음

### 3. MCP 도입 고려 시나리오

**향후 MCP 도입을 고려할 경우:**

1. **다양한 상용 API 통합 필요 시**
   - OpenAI, Anthropic, Google 등 여러 LLM API 동시 사용
   - 이미지 생성 API (DALL-E, Midjourney 등)
   - 음성 합성 API (TTS)

2. **복잡한 도구 체인 구성 시**
   - 여러 도구를 순차적으로 호출하는 워크플로우
   - 조건부 도구 선택 로직

3. **권한 관리 강화 필요 시**
   - 세밀한 API 호출 권한 제어
   - 사용자별 도구 접근 제한

**현재는 이러한 요구사항이 없으므로 MCP 미도입 결정**

---

## 상용 API vs 자체 모델 분석

### 1. 현재 사용 중인 서비스 분석

#### 1.1. LLM 서빙: GPU 서버 + 로컬 vLLM vs RunPod Serverless Endpoint

**GPU 서버 + 로컬 vLLM (현재 사용, 권장)**

**장점:**
- **비용 효율성**: GPU 서버 사용 시간만 과금 (Go Watchdog로 idle 시간 최소화)
- **성능**: 네트워크 오버헤드 없음, Continuous Batching 자동 활용
- **처리량**: 5-10배 향상 (2 req/s → 10 req/s)
- **지연 시간**: 낮음 (로컬 실행)
- **데이터 보안**: 데이터가 외부로 전송되지 않음

**단점:**
- **초기 설정**: GPU 서버 환경 구성 필요
- **GPU 메모리**: 모델 로딩에 약 14GB 필요
- **Cold Start**: GPU 서버 시작 시 모델 로딩 시간 (약 1-2분)

**비용 분석:**
- GPU 서버 시간당 비용: 약 $0.5-1.0 (GPU 사양에 따라)
- Go Watchdog으로 idle 시간 최소화: 50-70% 비용 절감
- 예상 월 비용: 트래픽에 따라 $100-500

**RunPod Serverless Endpoint (이전 사용, 현재 미사용)**

**장점:**
- **간편한 설정**: API 키만으로 사용 가능
- **자동 스케일링**: 트래픽에 따라 자동 확장
- **Cold Start 최소화**: 엔드포인트가 항상 활성 상태

**단점:**
- **비용**: 요청당 과금으로 높은 트래픽 시 비용 증가
- **성능**: 네트워크 오버헤드로 지연 시간 증가
- **처리량**: 낮음 (네트워크 병목)
- **데이터 보안**: 요청 데이터가 외부로 전송됨

**비용 분석:**
- 요청당 비용: 약 $0.001-0.01 (모델 및 토큰 수에 따라)
- 높은 트래픽 시 월 비용: $500-2000+

**결정: GPU 서버 + 로컬 vLLM 선택**

**이유:**
- 비용 효율성 (Go Watchdog으로 idle 시간 최소화)
- 성능 우수 (5-10배 향상)
- 데이터 보안 (로컬 실행)

#### 1.2. 벡터 데이터베이스: Qdrant on-disk vs Qdrant Cloud

**Qdrant on-disk (현재 사용)**

**장점:**
- **비용**: 무료 (로컬 저장)
- **데이터 보안**: 데이터가 외부로 전송되지 않음
- **성능**: MMAP 활용으로 실용적 성능
- **데이터 영속성**: 로컬 파일 시스템에 저장

**단점:**
- **확장성**: 단일 서버 제한
- **백업**: 수동 백업 필요

**Qdrant Cloud (미사용)**

**장점:**
- **확장성**: 자동 스케일링
- **관리**: 백업, 모니터링 자동화

**단점:**
- **비용**: 월 $50-500+ (데이터 양에 따라)
- **데이터 보안**: 데이터가 클라우드로 전송됨
- **네트워크 오버헤드**: 원격 접근으로 지연 시간 증가

**결정: Qdrant on-disk 선택**

**이유:**
- 비용 효율성 (무료)
- 데이터 보안 (로컬 저장)
- 현재 데이터 규모에 충분

#### 1.3. 임베딩 모델: 로컬 SentenceTransformer vs 상용 API

**로컬 SentenceTransformer (현재 사용)**

**장점:**
- **비용**: 무료 (모델 다운로드만)
- **성능**: GPU + FP16으로 빠른 처리
- **데이터 보안**: 데이터가 외부로 전송되지 않음
- **커스터마이징**: 모델 파인튜닝 가능

**단점:**
- **GPU 메모리**: 모델 로딩에 약 1-2GB 필요
- **업데이트**: 수동 모델 업데이트 필요

**상용 API (예: OpenAI Embeddings, Cohere)**

**장점:**
- **간편한 사용**: API 키만으로 사용 가능
- **최신 모델**: 자동으로 최신 모델 사용

**단점:**
- **비용**: 요청당 $0.0001-0.001 (높은 트래픽 시 비용 증가)
- **데이터 보안**: 요청 데이터가 외부로 전송됨
- **지연 시간**: 네트워크 오버헤드

**비용 분석:**
- 로컬: 무료 (GPU 전력비 제외)
- 상용 API: 월 $100-1000+ (트래픽에 따라)

**결정: 로컬 SentenceTransformer 선택**

**이유:**
- 비용 효율성 (무료)
- 데이터 보안 (로컬 실행)
- 성능 우수 (GPU 활용)

### 2. 향후 상용 API 도입 고려 사항

#### 2.1. 이미지 생성 API

**시나리오:** 리뷰 이미지 자동 생성, 메뉴 이미지 생성

**후보 API:**
- **OpenAI DALL-E 3**: $0.04-0.12/이미지
- **Midjourney**: $10-60/월 (구독)
- **Stable Diffusion API**: $0.002-0.01/이미지

**비용 분석:**
- 월 1000개 이미지 생성 시: $40-120 (DALL-E) vs $2-10 (Stable Diffusion)

**Fallback 전략:**
- 사전 생성한 템플릿 이미지 활용
- 캐싱: 동일한 프롬프트는 캐시된 이미지 반환

**도입 시 고려사항:**
- API 호출 제한 (Rate Limiting)
- 재시도 로직 (Exponential Backoff)
- 비용 모니터링

#### 2.2. 음성 합성 API (TTS)

**시나리오:** 리뷰 요약 음성 변환

**후보 API:**
- **OpenAI TTS**: $0.015/1000자
- **Google Cloud TTS**: $0.004-0.016/1000자
- **Azure TTS**: $0.01-0.02/1000자

**비용 분석:**
- 월 10,000개 리뷰 요약 (평균 100자): $15-20

**Fallback 전략:**
- 캐싱: 동일한 텍스트는 캐시된 음성 반환
- 선택적 사용: 사용자 요청 시에만 생성

#### 2.3. 영상 분석 API

**시나리오:** 리뷰 영상 자동 분석 (향후 확장 계획)

**후보 API:**
- **Google Cloud Video Intelligence**: $0.10-0.50/분
- **AWS Rekognition Video**: $0.10-0.50/분

**비용 분석:**
- 월 100개 영상 (평균 1분): $10-50

**도입 시 고려사항:**
- 높은 비용으로 인해 선택적 사용 필요
- 사전 분석 결과 캐싱

### 3. 상용 API 통합 설계 (향후 도입 시)

#### 3.1. API 호출 모듈 설계

```python
# src/external_apis/image_generation.py
class ImageGenerationAPI:
    """이미지 생성 API 통합 모듈"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.cache = CacheManager()  # Redis 캐싱
        self.rate_limiter = RateLimiter()
    
    async def generate_image(
        self,
        prompt: str,
        use_cache: bool = True,
    ) -> str:
        """이미지 생성 (캐싱 지원)"""
        # 캐시 확인
        if use_cache:
            cached = await self.cache.get(f"image:{prompt}")
            if cached:
                return cached
        
        # Rate Limiting
        await self.rate_limiter.acquire()
        
        # API 호출 (재시도 로직 포함)
        try:
            image_url = await self._call_api_with_retry(prompt)
            # 캐싱
            if use_cache:
                await self.cache.set(f"image:{prompt}", image_url, ttl=86400)
            return image_url
        except Exception as e:
            # Fallback: 사전 생성한 템플릿 이미지
            return await self._get_fallback_image(prompt)
    
    async def _call_api_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
    ) -> str:
        """재시도 로직 포함 API 호출"""
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return await self._call_openai_dalle(prompt)
                elif self.provider == "stable_diffusion":
                    return await self._call_stable_diffusion(prompt)
            except RateLimitError:
                await asyncio.sleep(2 ** attempt)  # Exponential Backoff
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                if attempt == max_retries - 1:
                    raise
        raise Exception("Image generation failed after retries")
```

#### 3.2. Fallback 전략

**캐싱:**
- 동일한 프롬프트는 캐시된 결과 반환
- TTL 설정 (예: 24시간)

**사전 생성 콘텐츠:**
- 자주 사용되는 템플릿 이미지/음성 사전 생성
- API 실패 시 템플릿 반환

**비용 제한:**
- 일일/월별 API 호출 제한 설정
- 제한 초과 시 Fallback 사용

---

## 현재 인터페이스 설계

### 1. REST API 기반 통합

**현재 아키텍처:**
- FastAPI 기반 REST API
- 각 외부 서비스를 독립적인 모듈로 분리
- 의존성 주입을 통한 느슨한 결합

### 2. 모듈별 인터페이스

#### 2.1. LLM Utils 모듈 (`src/llm_utils.py`)

**인터페이스:**
```python
class LLMUtils:
    """LLM 관련 유틸리티 클래스"""
    
    async def _generate_with_vllm(
        self,
        prompts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 100,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """vLLM 비동기 추론 (메트릭 수집 포함)"""
        # 예외 처리, 재시도, 속도 제한 포함
```

**예외 처리:**
- OOM 방지 (동적 배치 크기, 세마포어)
- 재시도 로직 (`max_retries`)
- 에러 로깅 및 메트릭 수집 (MetricsCollector 통합)

**설계된 최적화:**
- **대표 벡터 TOP-K 방식**: 감성 분석, 요약, 강점 추출에서 컨텍스트 크기 최적화
- **우선순위 큐 (Prefill 비용 기반)**: 작은 요청 우선 처리로 SLA 보호
- **동적 배치 크기**: 리뷰 길이에 따른 최적 배치
- **세마포어 제한**: 동시 처리 수 제한으로 OOM 방지
- **비동기 큐 방식**: 여러 레스토랑 병렬 처리

**속도 제한:**
- 세마포어 기반 동시 처리 수 제한 (`VLLM_MAX_CONCURRENT_BATCHES`: 20)
- **우선순위 큐 (Prefill 비용 기반)**: 작은 요청 우선 처리
  - Prefill 비용은 입력 토큰 수로 정확히 예측 가능
  - Shortest Job First (SJF) 알고리즘 적용
  - 작은 요청 TTFT 30-40% 개선 (2.5초 → 1.8초)
  - SLA 준수율 85% → 92% 향상

#### 2.2. Vector Search 모듈 (`src/vector_search.py`)

**인터페이스:**
```python
class VectorSearch:
    """벡터 검색 클래스"""
    
    def query_similar_reviews(
        self,
        query_text: str,
        restaurant_id: Optional[str] = None,
        limit: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """의미 기반 유사 리뷰 검색"""
        # 예외 처리 포함
```

**예외 처리:**
- 컬렉션 존재 여부 확인
- 검색 실패 시 빈 리스트 반환
- 에러 로깅

#### 2.3. 외부 API 호출 모듈 (향후 확장용)

**설계 원칙:**
- 모든 외부 호출은 예외 처리, 재시도, 속도 제한 제어 포함
- 모듈화된 구조로 새로운 API 추가 용이

**템플릿:**
```python
# src/external_apis/base.py
class BaseExternalAPI:
    """외부 API 호출 기본 클래스"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.retry_config = RetryConfig()
    
    async def call_with_retry(
        self,
        func: Callable,
        max_retries: int = 3,
    ):
        """재시도 로직 포함 API 호출"""
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                return await func()
            except RateLimitError:
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                if attempt == max_retries - 1:
                    raise
        raise Exception("API call failed after retries")
```

### 3. 의존성 주입 구조

**설계:**
```python
# src/api/dependencies.py
@lru_cache()
def get_llm_utils() -> LLMUtils:
    """LLM 유틸리티 싱글톤"""
    return LLMUtils(model_name=Config.LLM_MODEL)

@lru_cache()
def get_vector_search(
    encoder: SentenceTransformer = Depends(get_encoder),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> VectorSearch:
    """벡터 검색 의존성"""
    return VectorSearch(
        encoder=encoder,
        qdrant_client=qdrant_client,
        collection_name=Config.COLLECTION_NAME,
    )
```

**장점:**
- 테스트 용이성 (Mock 객체 주입 가능)
- 모듈 간 느슨한 결합
- 싱글톤 패턴으로 리소스 효율성

---

## 확장성 계획

### 1. 새로운 외부 서비스 추가 시나리오

#### 1.1. 이미지 생성 API 추가

**추가 단계:**
1. `src/external_apis/image_generation.py` 모듈 생성
2. `BaseExternalAPI` 상속하여 구현
3. `src/api/dependencies.py`에 의존성 추가
4. API Router에 엔드포인트 추가

**코드 예시:**
```python
# src/external_apis/image_generation.py
class ImageGenerationAPI(BaseExternalAPI):
    """이미지 생성 API"""
    
    async def generate_image(self, prompt: str) -> str:
        return await self.call_with_retry(
            lambda: self._call_dalle_api(prompt)
        )

# src/api/dependencies.py
@lru_cache()
def get_image_generation_api() -> ImageGenerationAPI:
    return ImageGenerationAPI()

# src/api/routers/llm.py
@router.post("/generate-image")
async def generate_image(
    request: ImageGenerationRequest,
    api: ImageGenerationAPI = Depends(get_image_generation_api),
):
    return await api.generate_image(request.prompt)
```

#### 1.2. 음성 합성 API 추가

**추가 단계:**
1. `src/external_apis/tts.py` 모듈 생성
2. 캐싱 로직 포함
3. API Router에 엔드포인트 추가

#### 1.3. 영상 분석 API 추가

**추가 단계:**
1. `src/external_apis/video_analysis.py` 모듈 생성
2. 비용 제한 로직 포함
3. API Router에 엔드포인트 추가

### 2. 확장성 보장 전략

#### 2.1. 인터페이스 표준화

**BaseExternalAPI 클래스:**
- 모든 외부 API가 상속받는 기본 클래스
- 공통 기능: 재시도, 속도 제한, 예외 처리

#### 2.2. 설정 기반 통합

**환경 변수로 제어:**
```python
# config.py
USE_IMAGE_GENERATION_API: bool = os.getenv("USE_IMAGE_GENERATION_API", "false").lower() == "true"
IMAGE_GENERATION_PROVIDER: str = os.getenv("IMAGE_GENERATION_PROVIDER", "openai")
IMAGE_GENERATION_API_KEY: Optional[str] = os.getenv("IMAGE_GENERATION_API_KEY")
```

#### 2.3. 플러그인 아키텍처 (향후 고려)

**확장 포인트:**
```python
# src/external_apis/registry.py
class ExternalAPIRegistry:
    """외부 API 레지스트리"""
    
    _apis: Dict[str, BaseExternalAPI] = {}
    
    @classmethod
    def register(cls, name: str, api: BaseExternalAPI):
        cls._apis[name] = api
    
    @classmethod
    def get(cls, name: str) -> BaseExternalAPI:
        return cls._apis.get(name)
```

### 3. 모니터링 및 로깅

**모든 외부 호출 모니터링:**
- 호출 횟수, 성공/실패율
- 응답 시간, 비용 추적
- 에러 로깅 및 알림

**설계된 모니터링 도구:**
- **MetricsCollector** (`src/metrics_collector.py`):
  - SQLite + 로그 파일에 메트릭 저장
  - `analysis_metrics` 테이블: 기본 분석 메트릭 (restaurant_id, analysis_type, processing_time_ms, tokens_used 등)
  - `vllm_metrics` 테이블: vLLM 상세 메트릭 (prefill_time_ms, decode_time_ms, ttft_ms, tps, tpot_ms 등)
- **GoodputTracker** (`src/goodput_tracker.py`):
  - SLA 기반 처리량 추적 (TTFT < 2초)
  - `throughput_tps`, `goodput_tps`, `sla_compliance_rate` 계산

**구현 예시 (향후 확장용):**
```python
# src/external_apis/base.py
class BaseExternalAPI:
    def __init__(self):
        self.metrics = MetricsCollector()
    
    async def call_with_retry(self, func, ...):
        start_time = time.time()
        try:
            result = await func()
            self.metrics.record_success(
                api_name=self.__class__.__name__,
                duration=time.time() - start_time
            )
            return result
        except Exception as e:
            self.metrics.record_error(
                api_name=self.__class__.__name__,
                error=str(e)
            )
            raise
```

---

## 비용 효율성 분석

### 1. 현재 아키텍처 비용

#### 1.1. GPU 서버 + 로컬 vLLM

**월 비용 추정:**
- GPU 서버 시간당: $0.5-1.0
- Go Watchdog으로 idle 시간 50-70% 절감
- 예상 월 사용 시간: 200-500시간
- **월 비용: $100-500**

**비용 최적화:**
- Go Watchdog으로 idle GPU 서버 자동 종료 (GPU 사용률 < 5% 시 자동 종료)
- **우선순위 큐 (Prefill 비용 기반)**로 처리량 최대화 (20% 향상)
- **대표 벡터 TOP-K 방식**으로 토큰 사용량 60-80% 감소
- 동적 배치 크기로 GPU 활용률 향상 (20-30% → 70-90%)

#### 1.2. Qdrant on-disk

**월 비용: $0** (로컬 저장)

**비용 최적화:**
- on-disk 모드로 서버 비용 없음
- MMAP 활용으로 메모리 효율성

#### 1.3. SentenceTransformer

**월 비용: $0** (로컬 실행, GPU 전력비 제외)

**비용 최적화:**
- FP16으로 메모리 50% 절감
- 배치 처리로 GPU 활용률 향상

### 2. 상용 API 도입 시 비용 비교

#### 2.1. LLM API (예: OpenAI GPT-4)

**비용:**
- GPT-4: $0.03-0.06/1K 토큰 (입력), $0.06-0.12/1K 토큰 (출력)
- 월 1M 토큰 사용 시: $30-120

**vs GPU 서버:**
- GPU 서버가 더 비용 효율적 (월 $100-500로 무제한 사용)

#### 2.2. 임베딩 API (예: OpenAI Embeddings)

**비용:**
- $0.0001/1K 토큰
- 월 10M 토큰 사용 시: $1

**vs 로컬 SentenceTransformer:**
- 로컬이 더 비용 효율적 (무료)

#### 2.3. 이미지 생성 API

**비용:**
- DALL-E 3: $0.04-0.12/이미지
- 월 1000개 이미지: $40-120

**도입 시 고려사항:**
- 캐싱으로 중복 생성 방지
- Fallback 전략으로 비용 제한

### 3. 비용 최적화 전략

#### 3.1. 캐싱

**적용 대상:**
- 이미지 생성 결과
- 음성 합성 결과
- 자주 사용되는 쿼리 결과

**구현:**
```python
# Redis 캐싱 (src/cache.py)
cache_manager = CacheManager()
cached_result = await cache_manager.get(f"image:{prompt}")
if cached_result:
    return cached_result
```

#### 3.2. Fallback 전략

**사전 생성 콘텐츠:**
- 자주 사용되는 템플릿 이미지/음성 사전 생성
- API 실패 시 템플릿 반환

**비용 제한:**
- 일일/월별 API 호출 제한 설정
- 제한 초과 시 Fallback 사용

#### 3.3. 선택적 사용

**사용자 요청 시에만 생성:**
- 이미지/음성 생성은 사용자 요청 시에만 수행
- 자동 생성은 비용이 높으므로 제한

---

## 결론

### 1. 현재 아키텍처의 적합성

**현재 선택:**
- **MCP 미도입**: 로컬 실행 중심 아키텍처에 적합
- **자체 모델 우선**: 비용 효율성 및 데이터 보안
- **REST API 기반 통합**: 명확하고 유지보수 용이

**효과:**
- **비용 효율성**: 월 $100-500 (GPU 서버만)
- **성능**: 네트워크 오버헤드 최소화, 높은 처리량
- **데이터 보안**: 로컬 실행으로 데이터 외부 전송 없음
- **유지보수성**: 명확한 모듈 구조, 쉬운 확장

### 2. 향후 확장 계획

**단기 (1-3개월):**
- 현재 아키텍처 유지
- 필요 시 이미지 생성/음성 합성 API 추가 (모듈화된 구조로)

**중기 (3-6개월):**
- 상용 API 도입 시 BaseExternalAPI 패턴 활용
- 캐싱 및 Fallback 전략 적용

**장기 (6개월+):**
- 복잡한 도구 체인 필요 시 MCP 도입 재검토
- 플러그인 아키텍처로 확장성 강화

### 3. 선택한 통합 방법의 영향

**확장성:**
- ✅ 모듈화된 구조로 새로운 외부 서비스 추가 용이
- ✅ BaseExternalAPI 패턴으로 일관된 인터페이스

**유지보수성:**
- ✅ 명확한 모듈 책임 분리
- ✅ 의존성 주입으로 테스트 용이
- ✅ 설정 기반으로 환경별 제어 가능

**비용 효율성:**
- ✅ 로컬 실행 중심으로 비용 최소화
- ✅ Go Watchdog으로 idle 시간 절감
- ✅ 캐싱 및 Fallback 전략으로 API 비용 제어

**결론:**
현재 아키텍처는 프로젝트의 요구사항에 적합하며, 향후 확장에도 대비되어 있습니다. MCP 도입은 현재 필요하지 않으나, 복잡한 도구 체인이 필요해지면 재검토할 수 있습니다.

---

## 관련 문서

- [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md): 통합 아키텍처 개요
- [ARCHITECTURE.md](ARCHITECTURE.md): 모듈화 아키텍처 상세
- [LLM_SERVICE_DESIGN.md](LLM_SERVICE_DESIGN.md): LLM 서비스 설계 상세
- [RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md): RAG 아키텍처 상세
- [API_SPECIFICATION.md](API_SPECIFICATION.md): API 인터페이스 명세
- [PRODUCTION_INFRASTRUCTURE.md](PRODUCTION_INFRASTRUCTURE.md): 인프라 및 배포 계획
