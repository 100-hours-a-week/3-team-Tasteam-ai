# 프로덕션 인프라 구조 및 모니터링

## 목차
1. [개요](#개요)
2. [서빙 인프라 설계](#서빙-인프라-설계)
3. [모니터링 도구 설계](#모니터링-도구-설계)
4. [관련 문서](#관련-문서)

---

## 개요

본 문서는 레스토랑 리뷰 분석 API 서비스의 서빙 인프라 구조와 모니터링 도구 설계를 설명합니다.

### 서비스 특성

- **서비스 유형**: 실시간 LLM 기반 텍스트 분석 API
- **주요 워크로드**: 
  - 감성 분석 (대표 벡터 TOP-K 방식)
  - 리뷰 요약 (RAG 패턴, aspect 기반 요약)
  - 강점 추출 (구조화된 파이프라인: Step A~H)
  - 이미지 검색 (Query Expansion 지원)
- **리소스 집약적**: GPU 메모리 14GB+ 필요 (Qwen2.5-7B-Instruct)
- **최적화 기법**:
  - 대표 벡터 TOP-K 방식: 토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축
  - 우선순위 큐 (Prefill 비용 기반): 작은 요청 TTFT 30-40% 개선, SLA 준수율 85% → 92% 향상
  - 동적 배치 크기: 리뷰 길이 기반 자동 조정
  - 세마포어 제한: OOM 방지

---

## 서빙 인프라 설계

### 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Client Layer (API Consumer)                      │
│                    (HTTP/REST API 요청 및 응답)                          │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU 서버 (상시 실행)                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Presentation Layer                                              │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │  FastAPI Application (src/api/)                           │ │  │
│  │  │  ┌────────────────────────────────────────────────────┐ │ │  │
│  │  │  │  API Routers (src/api/routers/)                     │ │ │  │
│  │  │  │  ├── sentiment.py    (감성 분석 엔드포인트)         │ │ │  │
│  │  │  │  ├── llm.py          (LLM 요약/강점 추출)           │ │ │  │
│  │  │  │  └── vector.py       (벡터 검색/관리)               │ │ │  │
│  │  │  └────────────────────────────────────────────────────┘ │ │  │
│  │  │  ┌────────────────────────────────────────────────────┐ │ │  │
│  │  │  │  Dependencies (src/api/dependencies.py)            │ │ │  │
│  │  │  │  - get_llm_utils()      (LLMUtils 싱글톤)          │ │ │  │
│  │  │  │  - get_sentiment_analyzer() (SentimentAnalyzer)    │ │ │  │
│  │  │  │  - get_vector_search()  (VectorSearch)             │ │ │  │
│  │  │  │  - get_encoder()        (SentenceTransformer)      │ │ │  │
│  │  │  │  - get_qdrant_client()  (QdrantClient)             │ │ │  │
│  │  │  └────────────────────────────────────────────────────┘ │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │  Domain/Service Layer                                    │ │  │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │  │
│  │  │  │ Sentiment    │  │ Vector       │  │ LLM          │    │ │  │
│  │  │  │ Analysis     │  │ Search       │  │ Utils        │    │ │  │
│  │  │  │ Module       │  │ Module       │  │ Module       │    │ │  │
│  │  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │ │  │
│  │  │         │                  │                  │            │ │  │
│  │  │         └──────────────────┴──────────────────┘            │ │  │
│  │  │                    │                                         │ │  │
│  │  │                    ▼                                         │ │  │
│  │  │         ┌─────────────────────┐                            │ │  │
│  │  │         │ Review Utils Module  │                            │ │  │
│  │  │         └─────────────────────┘                            │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │  vLLM (로컬)                                              │ │  │
│  │  │  - Qwen2.5-7B-Instruct 모델 로드                        │ │  │
│  │  │  - Continuous Batching 자동 활용                         │ │  │
│  │  │  - 네트워크 오버헤드 없음                                 │ │  │
│  │  │  - 항상 메모리에 로드 (Cold Start 없음)                   │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Sentence     │    │   Qdrant     │    │  외부 Watchdog│
│  Transformer  │    │  (Vector DB) │    │  Watchdog     │
│  (Embedding)  │    │  on-disk     │    │  (Go 바이너리) │
│               │    │  (MMAP 기반) │    │  - GPU 모니터링│
│  jhgan/ko-    │    │              │    │  - 자동 종료   │
│  sbert-       │    │              │    │  - RunPod API │
│  multitask    │    │              │    │    제어       │
│  multitask    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 2. GPU 서버 환경

**구성:**
- **플랫폼**: GPU 서버 (상시 실행)
- **GPU 인스턴스**: RTX 3090, A100 등 (GPU 메모리 14GB+ 필요)
- **컨테이너**: Docker 컨테이너 기반 배포
- **네트워크**: GPU 서버 내부에서 모든 서비스 실행 (네트워크 오버헤드 없음)

**특징:**
- **로컬 vLLM**: GPU 서버 내부에서 vLLM 직접 실행
- **Cold Start 없음**: 모델이 항상 메모리에 로드
- **Continuous Batching**: vLLM의 자동 배치 처리 활용
- **비용 최적화**: 외부 Watchdog (Go 바이너리)로 유휴 시 자동 종료

### 3. FastAPI 애플리케이션

**구성:**
- **프레임워크**: FastAPI
- **포트**: 8001
- **API 엔드포인트**:
  - `/api/v1/sentiment/analyze` - 감성 분석
  - `/api/v1/sentiment/analyze/batch` - 배치 감성 분석
  - `/api/v1/llm/summarize` - 리뷰 요약
  - `/api/v1/llm/summarize/batch` - 배치 리뷰 요약
  - `/api/v1/llm/extract/strengths` - 강점 추출 (구조화된 파이프라인: Step A~H)
  - `/api/v1/vector/search/similar` - 유사 리뷰 검색
  - `/api/v1/vector/reviews/upsert` - 리뷰 Upsert
  - `/api/v1/vector/upload` - 벡터 데이터 업로드

**의존성 주입:**
- `get_llm_utils()`: LLMUtils 싱글톤
- `get_sentiment_analyzer()`: SentimentAnalyzer 팩토리
- `get_vector_search()`: VectorSearch 팩토리
- `get_encoder()`: SentenceTransformer 싱글톤
- `get_qdrant_client()`: QdrantClient 싱글톤

### 4. 로컬 vLLM

**구성:**
- **모델**: Qwen2.5-7B-Instruct
- **엔진**: vLLM (로컬 실행)
- **배치 처리**: Continuous Batching 자동 활용
- **동적 배치 크기**: 리뷰 길이 기반 자동 조정
- **세마포어**: 동시 처리 배치 수 제한 (OOM 방지)
- **우선순위 큐**: Prefill 비용 기반 태스크 스케줄링 (SLA 보호)
  - Prefill 비용은 입력 토큰 수로 정확히 예측 가능
  - Shortest Job First (SJF) 알고리즘 적용
  - 작은 요청 우선 처리로 SLA 보호
  - TTFT 30-40% 개선 (2.5초 → 1.8초)
  - SLA 준수율 85% → 92% 향상

**실제 성능 지표 (Qwen/Qwen2.5-7B-Instruct 기준):**
- **감성 분석**: 평균 0.843초, P95 0.874초, 처리량 1.19 req/s (목표 달성 ✅)
- **리뷰 요약**: 평균 0.629초, P95 0.639초, 처리량 1.59 req/s (목표 달성 ✅)
- **강점 추출**: 평균 0.614초, P95 0.653초, 처리량 1.63 req/s (목표 달성 ✅)
- **배치 감성 분석**: 평균 9.15초 (목표: 5.0-10.0초, 달성 ✅)
- **배치 리뷰 요약**: 평균 83.05초 (목표: 5.0-10.0초, 최적화 필요 ⚠️)

**특징:**
- **네트워크 오버헤드 없음**: GPU 서버 내부에서 직접 실행
- **Cold Start 없음**: 모델이 항상 메모리에 로드
- **높은 처리량**: Continuous Batching으로 처리량 향상
- **대표 벡터 TOP-K 방식**: 감성 분석, 요약, 강점 추출에서 컨텍스트 크기 최적화
  - 레스토랑의 대표 벡터 주위에서 TOP-K 리뷰 검색 (기본값: 20개)
  - 토큰 사용량 60-80% 감소
  - 처리 시간 50-70% 단축

### 5. SentenceTransformer (임베딩)

**구성:**
- **모델**: jhgan/ko-sbert-multitask
- **용도**: 텍스트 벡터 인코딩
- **최적화**: GPU 사용, FP16 양자화

### 6. Qdrant (벡터 데이터베이스)

**구성:**
- **모드**: on-disk (MMAP 기반)
- **용도**: 벡터 검색 및 저장
- **컬렉션**:
  - `reviews_collection`: 리뷰 벡터 저장
  - `restaurant_vectors`: 레스토랑 대표 벡터 저장 (대표 벡터 TOP-K 방식에서 사용)

**특징:**
- **MMAP 기반**: OS 레벨 페이지 캐시 활용
- **메모리 효율성**: RAM 사용 최소화
- **영속성**: 디스크에 데이터 저장
- **비용 최적화**: 별도 서버 프로세스 불필요
- **대표 벡터 활용**: 
  - 레스토랑의 모든 리뷰 임베딩의 가중 평균으로 대표 벡터 계산
  - 대표 벡터 주위에서 TOP-K 리뷰 검색으로 관련성 높은 리뷰만 선택
  - 감성 분석, 요약, 강점 추출에서 컨텍스트 크기 최적화

### 7. 세 레이어 중복 실행 방지 전략

**개요:**
과호출 및 중복 실행을 방지하기 위한 3단계 방어 전략입니다.

#### 레이어 1: 스케줄러 (운영 정책) - 거시적 제어
- **상태**: 외부 구현 필요 (cron, Kubernetes CronJob 등)
- **역할**: tier별 호출 빈도 결정 → 과호출 크게 감소
- **정책 예시**:
  - 0~10 tier: 60분마다 업데이트
  - 40~50 tier: 10분마다 업데이트
- **효과**: 불필요한 호출 자체를 줄임

#### 레이어 2: SKIP 로직 (metrics 기반) - 미세한 중복 흡수
- **모듈**: `src/metrics_db.py` (`get_last_success_at()`, `should_skip_analysis()`)
- **역할**: 최근 처리면 실행 생략 → 남은 잔여 과호출 흡수
- **구현 방식**:
  - `analysis_metrics`에서 `MAX(created_at)` 조회
  - `error_count=0` 중 최신 성공 실행 시간 확인
  - `SKIP_MIN_INTERVAL_SECONDS` (기본값: 3600초 = 1시간) 이내면 SKIP
  - SKIP 시: 메트릭 기록 후 빈 응답 반환 (LLM 실행 없음)
- **적용 엔드포인트**:
  - `/api/v1/sentiment/analyze` - 감성 분석
  - `/api/v1/llm/summarize` - 리뷰 요약
  - `/api/v1/llm/extract/strengths` - 강점 추출
- **효과**: 스케줄러 버그, 재시도, 수동 백필 등으로 인한 잔여 과호출 흡수

#### 레이어 3: Redis 락 (동시성) - 동시 중복 차단
- **모듈**: `src/cache.py` (RedisLock 클래스, `acquire_lock()` Context Manager)
- **역할**: 동시에 2개 들어오면 1개만 실행 → 동시 중복 실행 차단
- **구현 방식**:
  - 락 키 형식: `lock:{restaurant_id}:{analysis_type}` (예: `lock:1:sentiment`)
  - Redis `SET NX EX` 사용: 원자적 락 획득 (TTL: 기본 1시간)
  - 엔드포인트 진입 시 락 획득
  - 자동 해제: Context Manager로 예외 발생 시에도 락 자동 해제
  - 안전한 해제: Lua 스크립트로 값 검증 후 삭제 (다른 프로세스의 락 삭제 방지)
- **적용 엔드포인트**:
  - `/api/v1/sentiment/analyze` - 감성 분석
  - `/api/v1/llm/summarize` - 리뷰 요약
  - `/api/v1/llm/extract/strengths` - 강점 추출
- **에러 처리**:
  - 락 획득 실패 시: `HTTPException(status_code=409, detail="중복 실행 방지: ...")`
  - Redis 미연결 시: 락 없이 진행 (개발 환경 지원)
- **효과**: 동일한 분석의 동시 실행을 차단하여 GPU 비용 절감 및 데이터 정합성 보장

**동작 순서:**
```
1. 엔드포인트 진입
   ↓
2. Redis 락 획득 (레이어 3: 동시 중복 차단)
   ↓
3. SKIP 체크 (레이어 2: 최근 성공 실행이면 SKIP)
   ↓
4. 실제 처리 (SKIP되지 않은 경우)
   ↓
5. 메트릭 저장
```

**세 레이어 전략 요약:**
- **레이어 1 (스케줄러)**: 외부 스케줄러가 tier별 호출 빈도 결정 (예: 0~10 tier는 60분마다, 40~50 tier는 10분마다) → 거시적 제어로 과호출 크게 감소
- **레이어 2 (SKIP 로직)**: `analysis_metrics`에서 최근 성공 실행 시간 조회, interval 이내면 SKIP → 미세한 중복/과호출 흡수
- **레이어 3 (Redis 락)**: 동시에 2개 요청이 들어오면 1개만 실행 → 동시 중복 실행 차단

### 8. 외부 Watchdog (Go 바이너리)

**설계:**
- **구현**: Go로 구현된 단일 바이너리
- **기능**:
  - GPU 사용률 모니터링 (nvidia-smi 호출)
  - 최근 요청 기록 확인
  - 유휴 시 RunPod API를 통한 GPU 서버 자동 종료

**특징:**
- **비용 최적화**: 유휴 시간 비용 절감
- **외부 실행**: GPU 서버 외부에서 실행 가능
- **RunPod API 연동**: RunPod API를 통한 GPU 서버 제어
- **장기 실행 안정성**: Go의 GC와 런타임 관리가 Python보다 예측 가능 (24/7 모니터링에 적합)
- **리소스 효율**: 메모리 사용량 10-15MB (Python 대비 85-90% 감소)
- **배포 편의성**: 단일 바이너리로 컨테이너/Docker/Kubernetes에서 바로 실행
- **독립 실행**: FastAPI 서버와 완전히 분리되어 독립적으로 동작

**구현:**
- Go 표준 라이브러리만으로 구현 (`os/exec`, `net/http`, `encoding/json`)
- 예상 코드량: 300-400줄 수준
- 단일 실행파일 빌드: `go build -o watchdog`


### Control-plane Watchdog 아키텍처 다이어그램
```
+--------------------+        RunPod API        +------------------+
|  Watchdog (Go)     |  -------------------->  |  RunPod Control  |
|  watchdog (binary) |                          |      Plane       |
+--------------------+                          +------------------+
        |
        | GPU usage / recent requests
        v
+--------------------+
|   GPU 서버 / vLLM   |
|  FastAPI + Model   |
+--------------------+
NOTE: Watchdog는 Go로 구현됨 (단일 바이너리, 독립 실행)
```



---

## 모니터링 도구 설계

### 1. MetricsCollector

**구성:**
- **모듈**: `src/metrics_collector.py`
- **저장소**: SQLite (`metrics.db`) + 로그 파일 (`logs/`)

**수집 메트릭:**

#### 1.1. 기본 분석 메트릭 (`analysis_metrics` 테이블)
- `restaurant_id`: 레스토랑 ID
- `analysis_type`: 분석 타입 (sentiment, summary, strength_extraction, image_search)
- `processing_time_ms`: 처리 시간 (밀리초)
- `tokens_used`: 사용된 토큰 수
- `batch_size`: 배치 크기
- `cache_hit`: 캐시 히트 여부
- `error_count`: 에러 개수
- `warning_count`: 경고 개수

**최적화 효과 추적:**
- 대표 벡터 TOP-K 방식으로 토큰 사용량 60-80% 감소 추적
- 처리 시간 50-70% 단축 추적

#### 1.2. vLLM 상세 메트릭 (`vllm_metrics` 테이블)
- `prefill_time_ms`: Prefill 단계 소요 시간 (밀리초)
- `decode_time_ms`: Decode 단계 소요 시간 (밀리초)
- `total_time_ms`: 전체 처리 시간 (밀리초)
- `n_tokens`: 생성된 토큰 수
- `tpot_ms`: Time Per Output Token (밀리초)
- `tps`: Tokens Per Second
- `ttft_ms`: Time To First Token (밀리초)

**우선순위 큐 효과 추적:**
- Prefill 비용 기반 우선순위 큐로 작은 요청 TTFT 30-40% 개선 추적
- SLA 준수율 85% → 92% 향상 추적

**Goodput 추적:**
- `GoodputTracker` 통합: SLA (TTFT < 2초) 기반 실제 처리량 측정
- `throughput_tps`: 전체 처리량
- `goodput_tps`: SLA 만족 처리량
- `sla_compliance_rate`: SLA 준수율

**로그 파일:**
- JSON 형식으로 메트릭 저장
- 구조화된 로거 (`StructuredLogger`) 사용

### 2. 로그 파일

**구성:**
- **디렉토리**: `logs/`
- **형식**: JSON 형식 (구조화된 로거)
- **내용**: 메트릭 데이터, 에러 로그, 경고 로그

**특징:**
- **구조화된 로깅**: JSON 형식으로 파싱 용이
- **중앙 집중화**: 모든 로그를 한 곳에서 관리

### 4. SQLite 데이터베이스

**구성:**
- **파일**: `metrics.db`
- **테이블**:
  - `analysis_metrics`: 기본 분석 메트릭
  - `vllm_metrics`: vLLM 상세 메트릭

**특징:**
- **경량 데이터베이스**: 별도 서버 불필요
- **로컬 저장**: GPU 서버 내부에 저장
- **쿼리 지원**: SQL 쿼리로 메트릭 조회 및 집계 가능

### 5. 메트릭 전략 (Metrics Strategy)

**설계:**

#### 1단계: 초기 설계
- **분리 전략**: 로그 파일 + SQLite (`analysis_metrics`, `vllm_metrics`)
- **조회 방식**: `MAX(created_at)` 조회로 최근 성공 실행 시간 확인
- **SKIP 로직**: 최근 성공 실행 시간(`error_count=0` 중 최신 `created_at`)을 조회하여 interval 이내면 SKIP

**특징:**
- ✅ **단순 구현**: 테이블 하나로 빠른 구현
- ✅ **초기 단순성**: 포트폴리오에서 "초기 단순 구현" 강조 가능
- ⚠️ **확장성 제한**: metrics 테이블이 커지면 조회 비용 증가, 역할 혼합

#### 2단계: 개선 방안 (향후)
- **분리 전략**: `analysis_state` 테이블 추가 (관측 vs 상태 분리)
- **조회 방식**: O(1) 조회로 "마지막 성공 시각" 정확하고 빠르게 조회
- **SKIP/재시도 정책**: 복잡한 정책 지원, 분산/멀티워커 환경 대응

**전환 기준:**
- 음식점 수/실행 횟수가 늘어서 metrics가 커질 때
- 분산/멀티워커 가능성이 있을 때
- "SKIP/재시도 정책"이 복잡해질 예정일 때

**이점:**
- ✅ **조회 성능**: O(1) 조회로 빠른 확인
- ✅ **개념 명확성**: 관측(metrics) vs 상태(state) 분리
- ✅ **확장성**: 분산 환경에서도 효율적

**메트릭 전략 요약:**
- **현재 (1단계)**: `analysis_metrics` 테이블에서 `MAX(created_at)` 조회로 최근 성공 실행 시간 확인 → SKIP 로직에 사용
- **향후 (2단계)**: `analysis_state` 테이블 추가하여 O(1) 조회로 성능 향상, 분산 환경 대응

---

### 6. 운영 전략 (Operation Strategy)

**설계:**

#### 1단계: 피크/비피크 시간 전략 (초기)
- **정책**: 모든 음식점에 대해 피크/비피크 시간 동일한 기준 적용
- **이유**: 각 음식점에 대한 트래픽 데이터가 없으므로 일괄 처리
- **효과**: 잦은 업데이트 방지, 비용 효율성 확보

#### 2단계: 트래픽 데이터 기반 군 단위 업데이트 (향후)
- **정책**: 트래픽 데이터 기반으로 음식점을 군 단위로 분류
- **군 분류 예시**: (40~50), (30~40), (20~30), (10~20), (0~10) 등
- **근거**: 리뷰 트래픽은 전형적인 long-tail 분포, 상위 10~20%가 대부분 트래픽 생성

**이점:**
- ✅ **연속값 이산화**: 완전 연속 제어 대비 튜닝/디버깅/운영 리스크 감소
- ✅ **설명 가능성**: 정책 설명이 쉬움, 롤백 용이, SRE/백엔드와 커뮤니케이션 용이
- ✅ **음식점 단위 최적화**: Long-tail 분포를 고려한 맞춤형 업데이트 주기
- ✅ **GPU/LLM 배치 처리와 궁합**: 비동기 큐 + 배치 + watchdog + vLLM 구조에 최적

**개선 방안:**

**1단계 고도화: 개수 × 변화량**
- `update_score = review_count × novelty_ratio`
- `novelty_ratio`: 새로운 aspect 등장 비율, embedding distance, 감성 분포 변화량 (KL divergence)
- **효과**: 리뷰 수 적어도 의미 변화가 크면 빠른 업데이트, 리뷰 많아도 의미 변화 없으면 SKIP

**2단계: 군 유지 + soft rule**
- 군은 유지하되 내부적으로 변화량에 따라 상위/하위 군처럼 처리
- **효과**: 정책 설명 가능성 유지, 효율만 개선

**3단계 (최종): SLA 기반 역설계**
- "요약이 실제 리뷰 상태와 평균 1시간 이상 어긋나지 않게" 같은 SLA 정의
- SLA를 만족하도록 음식점별 최소 업데이트 빈도 계산

**운영 전략 요약:**
- **1단계 (현재)**: 피크/비피크 시간 일괄 처리 → 모든 음식점에 동일한 기준 적용
- **2단계 (향후)**: 트래픽 데이터 기반 군 단위 분류 → (40~50), (30~40), (20~30), (10~20), (0~10) 등으로 분류하여 각 군별 업데이트 주기 차등 적용
- **고도화**: `update_score = review_count × novelty_ratio` 기반으로 변화량이 큰 경우 우선 업데이트
- **최종**: SLA 기반 역설계로 품질 보장하며 최소 업데이트 빈도 계산

---

## 인프라 개선 방안

### 1. 수평 확장 전략 (GCP MIG / Kubernetes HPA)

**설계:**
- 단일 인스턴스 기반 아키텍처 (GPU 서버)
- GCP 기반 자동 스케일링 설계
- Qdrant가 단일 인스턴스로 운영 (on-disk 모드)

**예상 문제:**
- 단일 장애점 (SPOF: Single Point of Failure)
- 트래픽 증가 시 확장 불가
- Qdrant 단일 노드 제한

**개선방안:**

#### 1.1. GCP Managed Instance Group (MIG) 기반 자동 스케일링

**설계:**
- GCP Managed Instance Group 사용
- Go 에이전트가 수집한 메트릭을 Cloud Monitoring에 푸시
- MIG Autoscaler가 Cloud Monitoring 커스텀 메트릭 기반으로 자동 스케일링

**자동 스케일링 정책:**
- **스케일 아웃 (Scale Out) 조건**:
  - 큐 길이 > 임계값 (예: 100개 이상 대기)
  - p95 TTFT > SLA 임계값 (예: 2초 초과)
  - GPU 사용률 > 임계값 (예: 80% 초과) 지속
- **스케일 인 (Scale In) 조건**:
  - 큐 길이 < 임계값 (예: 10개 이하)
  - p95 TTFT < SLA 임계값 (예: 1초 미만) 지속
  - GPU 사용률 < 임계값 (예: 30% 미만) 지속

**동작 방식:**
1. Go 에이전트가 각 노드의 메트릭 수집 (nvidia-smi, vLLM metrics, 큐 길이 등)
2. Cloud Monitoring에 커스텀 메트릭으로 푸시
3. MIG Autoscaler가 Cloud Monitoring 메트릭을 조회
4. 정책에 따라 인스턴스 수 자동 조정 (scale-out/scale-in)

**예상 효과:**
- 트래픽 증가 시 자동 확장
- 비용 최적화 (낮은 트래픽 시 자동 축소)
- 고가용성 (다중 인스턴스로 장애 격리)

#### 1.2. Kubernetes HPA (Horizontal Pod Autoscaler) 기반 자동 스케일링

**설계:**
- GKE (Google Kubernetes Engine) 사용 시
- Horizontal Pod Autoscaler (HPA) 사용
- Cloud Monitoring 커스텀 메트릭을 HPA가 조회하여 Pod 수 자동 조정

**HPA 정책:**
- **메트릭 소스**: Cloud Monitoring 커스텀 메트릭
- **타겟 메트릭**: 큐 길이, p95 TTFT 등
- **최소/최대 Pod 수**: 최소 1개, 최대 20개 (설정 가능)
- **스케일링 동작**: 메트릭 임계값 기반 자동 조정

**동작 방식:**
1. Go 에이전트가 GPU별 메트릭 수집
2. Cloud Monitoring에 커스텀 메트릭으로 푸시
3. HPA가 Cloud Monitoring 메트릭을 조회
4. 정책에 따라 Pod 수 자동 조정 (scale-out/scale-in)

**예상 효과:**
- 트래픽 증가 시 자동 확장
- 비용 최적화 (낮은 트래픽 시 자동 축소)
- Kubernetes 네이티브 통합

#### 1.3. 로드 밸런서 통합

**설계:**
- GCP Cloud Load Balancing 사용
- 다중 GPU 서버 인스턴스로 트래픽 분산
- Health Check 기반 트래픽 라우팅

#### 1.4. Qdrant 클러스터 모드

**설계:**
- Qdrant를 클러스터 모드로 전환 (향후 확장 계획)
- 복제본(Replica) 설정으로 고가용성 보장
- 샤딩(Sharding)으로 수평 확장

#### 1.5. 서비스 분리 (마이크로서비스)

**설계:**
- 감성 분석 서비스
- 벡터 검색 서비스
- LLM 추론 서비스
- 각 서비스별 독립적 확장 (향후 확장 계획)

**예상 효과:**
- 고가용성 달성
- 수평 확장 가능
- 장애 격리
- 자동 스케일링으로 비용 최적화

---

### 2. Health Check 강화

**설계:**
- 기본적인 `/health` 엔드포인트 설계
- 실제 서비스 상태(DB 연결, GPU 상태, 메모리 사용량 등) 확인은 향후 확장 계획

**예상 문제:**
- Kubernetes Liveness/Readiness Probe가 정확한 상태 확인 불가
- 장애 상황 조기 감지 불가
- 자동 복구(Health Check 기반 재시작) 불가

**개선방안:**
1. **상세 Health Check 엔드포인트**
```python
   @app.get("/health")
   async def health_check():
       checks = {
           "status": "healthy",
           "timestamp": datetime.utcnow().isoformat(),
           "checks": {
               "qdrant": await check_qdrant_connection(),
               "gpu": check_gpu_availability(),
               "memory": check_memory_usage(),
               "disk": check_disk_space(),
               "vllm": check_vllm_health()
           }
       }
       
       # 모든 체크가 정상이면 200, 하나라도 실패하면 503
       if all(checks["checks"].values()):
           return checks
       else:
           raise HTTPException(status_code=503, detail=checks)
```

2. **Liveness vs Readiness 분리**
   - `/health/live`: 서비스가 살아있는지 확인 (간단한 체크)
   - `/health/ready`: 서비스가 요청을 처리할 준비가 되었는지 확인 (상세 체크)

3. **GPU 상태 확인**
   - GPU 장애 시 Health Check 실패 (nvidia-smi 호출)

**예상 효과:**
- 정확한 서비스 상태 모니터링
- 자동 장애 복구
- 운영 효율성 향상

---

### 3. Graceful Shutdown

**설계:**
- 기본적인 `lifespan` 컨텍스트 매니저 설계
- 진행 중인 작업 완료 대기 로직은 향후 확장 계획
- ❌ vLLM 리소스 정리 로직 부재

**예상 문제:**
- 재배포 시 진행 중인 요청 손실
- 메모리 누수 가능성
- 데이터 일관성 문제

**개선방안:**
1. **Graceful Shutdown 구현**
   ```python
   import signal
   import asyncio
   
   shutdown_event = asyncio.Event()
   
   def signal_handler(sig, frame):
       logger.info("Shutdown signal received, starting graceful shutdown...")
       shutdown_event.set()
   
   signal.signal(signal.SIGINT, signal_handler)
   signal.signal(signal.SIGTERM, signal_handler)
   
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # 시작 시 초기화
       logger.info("Application starting...")
       yield
       # 종료 시 정리
       logger.info("Application shutting down...")
       await cleanup_resources()
   ```

2. **진행 중 작업 완료 대기**
   - 종료 신호 수신 시 새 요청 거부
   - 진행 중인 작업 완료 대기 (최대 대기 시간 설정)
   - vLLM 리소스 정리

3. **Health Check 연동**
   - Shutdown 시작 시 Health Check가 503 반환하여 로드 밸런서가 트래픽 차단

**예상 효과:**
- 데이터 손실 방지
- 부드러운 재배포
- 리소스 정리 보장

---

### 4. 백업 및 복구 전략

**설계:**
- Qdrant 데이터 백업 자동화는 향후 확장 계획
- 복구 절차는 향후 확장 계획

**예상 문제:**
- 데이터 손실 시 복구 불가
- 장애 발생 시 서비스 중단

**개선방안:**
1. **정기 백업 자동화**
```bash
   # Qdrant 스냅샷 생성 (cron job)
   0 2 * * * qdrant snapshot create --collection reviews_collection
   ```

2. **백업 검증**
   - 백업 파일 무결성 검증
   - 주기적으로 복구 테스트 수행

3. **백업 스토리지**
   - S3, GCS 등 클라우드 스토리지에 백업 저장
   - 지역별 백업 복제

4. **Disaster Recovery 계획**
   - RTO (Recovery Time Objective) 정의
   - RPO (Recovery Point Objective) 정의
   - 복구 절차 문서화 및 테스트

**예상 효과:**
- 데이터 손실 방지
- 빠른 장애 복구
- 비즈니스 연속성 보장

---

### 5. 모니터링 확장 (GCP Cloud Monitoring)

**설계:**
- MetricsCollector (SQLite + 로그 파일) 설계
- GoodputTracker (SLA 기반) 설계
- Go 에이전트 (메트릭 수집 및 푸시) 설계
- Cloud Monitoring (GCP) 통합 설계
- 커스텀 메트릭 푸시 설계

**예상 문제:**
- 실시간 메트릭 모니터링 불가
- 메트릭 시각화 어려움
- 자동 알림 불가
- 자동 스케일링 불가

**개선방안:**

#### 5.1. Go 에이전트 (메트릭 수집 및 상태 관리)

**역할:**
- GPU 서버 노드별 메트릭 수집 에이전트
- 상태 전이 관리 (drain/warmup)
- Spot/Preemptible 인스턴스 대응

##### 5.1.1. 메트릭 에이전트

**수집 메트릭:**
- **GPU 메트릭**:
  - GPU 사용률 (nvidia-smi 기반)
  - GPU 메모리 사용량 (nvidia-smi 기반)
- **vLLM 메트릭**:
  - TTFT (Time To First Token)
  - TPS (Tokens Per Second)
  - 큐 길이 (처리 대기 중인 요청 수)
  - KV cache 사용량 (Key-Value Cache 메모리 사용량)
- **통계 메트릭**: p50/p95/p99 TTFT, p50/p95/p99 처리 시간
- **리소스 메트릭**: CPU 사용률, 시스템 메모리 사용량

**메트릭 푸시:**
- Cloud Monitoring (GCP) 또는 CloudWatch (AWS)에 커스텀 메트릭으로 푸시
- 주기적으로 메트릭 수집 및 푸시 (기본 60초 간격)

**구현:**
- Go로 구현된 단일 바이너리 (메모리 효율: 10-15MB)
- 각 GPU 서버 노드에서 독립 실행
- nvidia-smi 호출 및 vLLM metrics API 조회

##### 5.1.2. 상태 전이 관리 (Drain / Warmup)

**역할:**
- 인스턴스 스케일 다운/업 시 상태 전이 관리
- Graceful shutdown 및 startup 보장

**Drain 단계 (스케일 다운 시):**
1. **신규 요청 차단**: 로드 밸런서에 health check 실패 신호 (503 반환)
2. **처리 중 요청 완료 대기**: 진행 중인 요청 완료까지 대기 (최대 대기 시간 설정)
3. **리소스 정리**: vLLM 리소스 정리, 연결 종료

**Warmup 단계 (스케일 업 시):**
1. **모델 로딩**: vLLM 모델 로드 및 초기화
2. **모델 로딩 완료 신호**: readiness probe 통과 신호
3. **Health Check 통과**: liveness/readiness probe 정상 응답

**Health Check 통합:**
- **Readiness Probe**: 모델 로딩 완료 후 통과 (처리 준비 완료)
- **Liveness Probe**: 서비스 실행 상태 확인 (장애 감지)

**구현:**
- FastAPI `/health/ready` 엔드포인트 연동
- FastAPI `/health/live` 엔드포인트 연동
- 모델 로딩 상태 추적 및 신호 전달

##### 5.1.3. Spot/Preemptible 대응

**역할:**
- Spot 인스턴스 또는 Preemptible 인스턴스 회수 신호 감지 및 대응
- 갑작스러운 인스턴스 종료 시 품질 저하 최소화

**회수 신호 감지:**
- GCP: Preemptible 인스턴스 종료 신호 (30초 전 경고)
- AWS: Spot 인스턴스 종료 신호 (2분 전 경고)
- 메타데이터 서버에서 종료 신호 감지

**즉시 Drain:**
- 회수 신호 감지 시 즉시 drain 프로세스 시작
- 신규 요청 차단
- 진행 중인 요청 최대한 완료 시도 (시간 제한 내)

**품질 저하 모드:**
- 긴급 종료 시나리오 대비
- 중요한 요청만 처리 (우선순위 기반)
- 비중요 요청 거부 또는 단순 응답

**상태 보고:**
- 종료 사유 및 시간을 Cloud Monitoring에 보고
- 메트릭 기록 (처리 중 요청 수, 완료/실패 통계)

**구현:**
- 메타데이터 서버 폴링 (주기적 체크)
- 신호 감지 시 즉시 drain 프로세스 트리거
- 상태 보고를 Cloud Monitoring에 커스텀 이벤트로 기록

---

#### 5.2. Cloud Monitoring (GCP) 통합

**설계:**
- GCP Cloud Monitoring (Stackdriver) 사용
- Go 에이전트가 커스텀 메트릭으로 푸시
- Cloud Monitoring API를 통한 메트릭 전송

**커스텀 메트릭:**
- `custom.googleapis.com/gpu/usage_percent`: GPU 사용률 (%)
- `custom.googleapis.com/gpu/memory_used_gb`: GPU 메모리 사용량 (GB)
- `custom.googleapis.com/vllm/ttft_p95_ms`: p95 TTFT (밀리초)
- `custom.googleapis.com/vllm/tps`: Tokens Per Second
- `custom.googleapis.com/vllm/tpot_ms`: Time Per Output Token (밀리초)
- `custom.googleapis.com/queue/length`: 큐 길이 (대기 중인 요청 수)
- `custom.googleapis.com/queue/pending_requests`: 처리 대기 중인 요청 수

**시각화:**
- Cloud Monitoring 대시보드에서 실시간 메트릭 시각화
- 알람 정책 설정 (임계값 초과 시 알림)

**예상 효과:**
- 실시간 메트릭 모니터링 가능
- 메트릭 시각화 가능
- 자동 알림 가능
- 자동 스케일링 기반 데이터 제공
- 성능 최적화 방향 명확화

---

## 관련 문서

- [ARCHITECTURE.md](ARCHITECTURE.md): 모듈화 아키텍처 상세
- [LLM_SERVICE_DESIGN.md](LLM_SERVICE_DESIGN.md): LLM 서비스 설계 상세
- [RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md): RAG 아키텍처 상세
- [API_SPECIFICATION.md](API_SPECIFICATION.md): API 인터페이스 명세
- [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md): 통합 아키텍처 개요
- [EXTERNAL_SYSTEM_INTEGRATION.md](EXTERNAL_SYSTEM_INTEGRATION.md): 외부 시스템 통합 설계
- [RUNPOD_POD_VLLM_GUIDE.md](../RUNPOD_POD_VLLM_GUIDE.md): GPU 서버 + vLLM 가이드
- [VLLM_PERFORMANCE_MEASUREMENT.md](../VLLM_PERFORMANCE_MEASUREMENT.md): vLLM 성능 측정
- [METRICS.md](../METRICS.md): 메트릭 수집 상세
