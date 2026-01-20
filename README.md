# Review Analysis API

레스토랑 리뷰의 감성 분석, 벡터 검색, LLM 기반 요약 및 강점 추출을 수행하는 FastAPI 기반 프로젝트입니다.

**역할**: 중간 처리 레이어 (RDB/NoSQL → API → RDB/NoSQL)

**LLM 모델**: 
- **빠른 검증용**: OpenAI API (gpt-4o-mini, gpt-4o 등)
- **프로덕션용**: Qwen2.5-7B-Instruct (RunPod Pod 환경에서 로컬 vLLM 사용)

## 주요 기능

1. **감성 분석** (대표 벡터 TOP-K 방식) → `positive_ratio`, `negative_ratio` 추출
   - **대표 벡터 TOP-K**: 레스토랑의 대표 벡터 주위에서 TOP-K 리뷰 선택 (기본값: 20개)
   - **RunPod Pod + 로컬 vLLM**: 최고 성능, 네트워크 오버헤드 없음
   - 배치 처리 지원: 여러 레스토랑 동시 처리 (동적 배치 크기 + 비동기 큐)
   - **효과**: 토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축
2. **리뷰 요약** (대표 벡터 TOP-K + Aspect 기반) → 긍정/부정/전체 요약 + 메타데이터
   - **대표 벡터 TOP-K**: 레스토랑의 대표 벡터 주위에서 TOP-K 리뷰 선택
   - **Aspect 기반 요약**: LLM이 긍정/부정 aspect를 구조화하여 추출 (`positive_aspects`, `negative_aspects`)
   - 배치 처리 지원: 여러 레스토랑 동시 처리 (동적 배치 크기 + 비동기 큐)
   - **효과**: 토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축
3. **강점 추출** (구조화된 파이프라인: Step A~H) → 다른 리뷰들과 비교하여 강점 추출 + 메타데이터
   - **Step A**: 대표 벡터 TOP-K + 다양성 샘플링으로 타겟 긍정 근거 후보 수집
   - **Step B**: LLM으로 구조화된 강점 후보 생성 (aspect, claim, evidence, 최소 5개 보장)
   - **Step C**: Qdrant 벡터 검색으로 근거 확장 및 검증 (support_count, consistency)
   - **Step D**: Connected Components (Union-Find)로 의미 중복 제거
   - **Step D-1**: Claim 후처리 재생성 (템플릿 보정 15-28자, 맛 claim은 구체명사 포함)
   - **Step E~H**: 비교군 기반 차별 강점 계산 (distinct_score, final_score)
   - 배치 처리 지원: 여러 레스토랑 동시 처리 (동적 배치 크기 + 비동기 큐)
4. **리뷰 Upsert** (포인트 업데이트) → 낙관적 잠금을 지원하는 리뷰 추가/수정
   - 개별 upsert: 낙관적 잠금 지원
   - 배치 upsert: 성능 최적화 (10개 리뷰를 1번의 API 호출로 처리)
5. **이미지 리뷰 검색** (벡터 검색) → 의미 기반 검색으로 이미지가 있는 리뷰 반환 + 메타데이터

**모든 응답은 메타데이터를 포함합니다** (id, restaurant_id, member_id, group_id, subgroup_id, content, is_recommended, created_at, updated_at, deleted_at 등)

## 프로젝트 구조

```
tasteam-project-aicode/
├── src/                      # 소스 코드 모듈
│   ├── __init__.py          # 패키지 초기화
│   ├── config.py            # 설정 관리
│   ├── models.py            # Pydantic 모델 정의
│   ├── review_utils.py      # 리뷰 처리 유틸리티
│   ├── sentiment_analysis.py # 감성 분석
│   ├── vector_search.py     # 벡터 검색
│   ├── llm_utils.py         # LLM 유틸리티
│   ├── strength_extraction.py # 강점 추출 파이프라인 (V2)
│   └── api/                 # FastAPI 애플리케이션
│       ├── main.py          # FastAPI 메인 앱
│       ├── dependencies.py  # 의존성 주입
│       └── routers/         # API 라우터
│           ├── sentiment.py    # 감성 분석 엔드포인트
│           ├── vector.py        # 벡터 검색 엔드포인트
│           ├── llm.py          # LLM 요약/강점 추출 엔드포인트
│           └── restaurant.py   # 레스토랑 관련 엔드포인트
├── scripts/                # 유틸리티 스크립트
│   ├── watchdog.py        # RunPod Pod 모니터링 및 자동 종료
│   ├── convert_kr3_tsv.py # kr3.tsv를 API 형식으로 변환하는 스크립트
│   ├── benchmark.py       # 성능 벤치마크 스크립트
│   └── evaluate_precision_at_k.py # Precision@k 평가 스크립트
├── test_api.ipynb   # API 테스트 노트북 (예제)
├── app.py                  # FastAPI 서버 실행 스크립트
├── requirements.txt        # 패키지 의존성
├── README.md              # 프로젝트 문서
├── API_SPECIFICATION.md   # API 명세서 (엔드포인트 목록, 스키마, 아키텍처)
├── ARCHITECTURE.md        # 시스템 아키텍처 문서
├── RUNPOD_POD_VLLM_GUIDE.md  # RunPod Pod + vLLM 구현 가이드
├── BATCH_PROCESSING_IMPROVEMENT.md  # 배치 처리 개선 사항 문서
└── PRODUCTION_ISSUES_AND_IMPROVEMENTS.md  # 프로덕션 환경 문제점 및 개선방안
```

## 설치

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

**주의사항:**
- RunPod Pod 환경에서 실행: GPU가 있는 RunPod Pod에서 실행해야 합니다
- vLLM 사용 시: Qwen2.5-7B-Instruct 모델은 약 14GB의 GPU 메모리가 필요합니다
- GPU 사용 시 CUDA가 설치되어 있어야 합니다
- 모델 최초 다운로드 시 시간이 걸릴 수 있습니다

3. 환경 변수 설정 (선택사항):
```bash
export QDRANT_URL="./qdrant_storage"  # Qdrant on-disk 저장 경로 (기본값: on-disk 모드)

# LLM 제공자 선택 (openai, runpod, local)
# OpenAI API 사용 (빠른 검증용, 권장)
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"  # 기본값: gpt-4o-mini (빠르고 저렴)

# 또는 RunPod Pod + 로컬 vLLM 사용 (프로덕션용)
export LLM_PROVIDER="runpod"
export USE_POD_VLLM="true"
export USE_RUNPOD="false"
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_MAX_MODEL_LEN=4096  # 선택사항

# 동적 배치 크기 설정 (OOM 방지)
export VLLM_MAX_TOKENS_PER_BATCH=4000  # 배치당 최대 토큰 수
export VLLM_MIN_BATCH_SIZE=10  # 최소 배치 크기
export VLLM_MAX_BATCH_SIZE=100  # 최대 배치 크기
export VLLM_MAX_CONCURRENT_BATCHES=20  # 최대 동시 처리 배치 수 (세마포어)

# 우선순위 큐 설정 (Prefill 비용 기반)
export VLLM_USE_PRIORITY_QUEUE=true  # 우선순위 큐 사용 여부 (기본값: true)
export VLLM_PRIORITY_BY_PREFILL_COST=true  # Prefill 비용 기반 우선순위 (기본값: true)

# Watchdog 설정 (외부 모니터링)
export RUNPOD_POD_ID="your_pod_id"  # Pod ID
export IDLE_THRESHOLD=5  # GPU 사용률 임계값 (%)
export CHECK_INTERVAL=60  # 체크 간격 (초)
export IDLE_LIMIT=5  # 연속 idle 횟수
export MIN_RUNTIME=600  # 최소 실행 시간 (초)
```

## 사용 방법

### FastAPI 서버 실행

1. 환경 변수 설정 (선택사항):
```bash
export QDRANT_URL="./qdrant_storage"  # Qdrant on-disk 저장 경로
```

2. 서버 실행:
```bash
# 방법 1: uvicorn 직접 실행
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 방법 2: app.py 실행
python app.py
```

### API 엔드포인트 목록

| 카테고리 | 메서드 | 엔드포인트 | 기능 |
|---------|--------|-----------|------|
| **감성 분석** | POST | `/api/v1/sentiment/analyze` | 리뷰 감성 비율 추출 (positive_ratio, negative_ratio) |
| | POST | `/api/v1/sentiment/analyze/batch` | 배치 감성 분석 |
| **리뷰 요약/강점** | POST | `/api/v1/llm/summarize` | 리뷰 요약 (긍정/부정/전체) |
| | POST | `/api/v1/llm/summarize/batch` | 배치 리뷰 요약 |
| | POST | `/api/v1/llm/extract/strengths` | 강점 추출 (구조화된 파이프라인: Step A~H) |
| **벡터 검색** | POST | `/api/v1/vector/search/similar` | 의미 기반 리뷰 검색 |
| | POST | `/api/v1/vector/search/review-images` | 이미지가 있는 리뷰 검색 |
| | POST | `/api/v1/vector/upload` | 벡터 데이터 업로드 |
| | GET | `/api/v1/vector/restaurants/{restaurant_id}/reviews` | 레스토랑 ID로 리뷰 조회 |
| **리뷰 관리** | POST | `/api/v1/vector/reviews/upsert` | 리뷰 Upsert (낙관적 잠금 지원) |
| | POST | `/api/v1/vector/reviews/upsert/batch` | 리뷰 배치 Upsert |
| | DELETE | `/api/v1/vector/reviews/delete` | 리뷰 삭제 |
| | DELETE | `/api/v1/vector/reviews/delete/batch` | 리뷰 배치 삭제 |
| **테스트** | POST | `/api/v1/test/generate` | 테스트 데이터 생성 (kr3.tsv 샘플링) |
| **헬스 체크** | GET | `/health` | 서버 상태 확인 |
| | GET | `/` | API 기본 정보 |

**🔗 실행 중인 서버에서 자동 생성된 문서 확인:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 테스트 가이드

### OpenAI 기반 빠른 검증 테스트

OpenAI API를 사용하여 모든 기능을 빠르게 검증할 수 있습니다:

```bash
# 1. 환경 변수 설정
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"  # 빠르고 저렴한 모델

# 2. 서버 실행
python app.py

# 3. 다른 터미널에서 테스트 실행
python test_openai_all.py
```

테스트 항목:
- 감성 분석 (단일/배치)
- 리뷰 요약 (단일/배치)
- 강점 추출
- 벡터 검색

### 로컬 테스트

```bash
# 환경 변수 설정
export USE_POD_VLLM="false"
export USE_RUNPOD="false"
export QDRANT_URL="./qdrant_storage"

# 기본 패키지만 설치 (vLLM 불필요)
pip install -r requirements.txt
# vLLM은 설치하지 않아도 됨 (ImportError 시 자동 fallback)

# 방법 1: Python 스크립트로 테스트 데이터 생성 (서버 실행 전에도 가능)
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 100 --restaurants 5

# 서버 실행
python app.py
# 또는
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 방법 2: API 엔드포인트로 테스트 데이터 생성 (서버 실행 후)
curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&restaurants=5" \
  -H "Content-Type: application/json" \
  -o test_data.json

# 테스트 실행
# 단일 레스토랑 감성 분석 (jq 사용)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d "$(cat test_data.json | jq '{restaurant_id: .restaurants[0].restaurant_id, reviews: .restaurants[0].reviews}')"

# 단일 레스토랑 감성 분석 (jq 없이 - Python 사용)
python -c "
import json
with open('test_data.json') as f:
    data = json.load(f)
    first_rest = data['restaurants'][0]
    print(json.dumps({'restaurant_id': first_rest['restaurant_id'], 'reviews': first_rest['reviews']}))
" | curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d @-

# 배치 감성 분석 (모든 레스토랑)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @test_data.json \
  --max-time 600
```

### RunPod Serverless Endpoint (vLLM 템플릿 사용)

```bash
# 환경 변수 설정
export USE_POD_VLLM="false"
export USE_RUNPOD="true"
export RUNPOD_API_KEY="your_runpod_api_key"
export RUNPOD_ENDPOINT_ID="g09uegksn7h7ed"  # 또는 본인의 엔드포인트 ID
export QDRANT_URL="./qdrant_storage"

# 기본 패키지만 설치 (vLLM 불필요)
pip install -r requirements.txt
# RunPod API 호출을 위한 requests 패키지만 필요

# 방법 1: Python 스크립트로 테스트 데이터 생성
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 100 --restaurants 5

# 서버 실행
python app.py

# 방법 2: API 엔드포인트로 테스트 데이터 생성 (서버 실행 후)
curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&restaurants=5" \
  -H "Content-Type: application/json" \
  -o test_data.json

# 테스트 (자동으로 RunPod 엔드포인트로 요청 전송)
# 배치 감성 분석 (모든 레스토랑)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @test_data.json \
  --max-time 600
```

### RunPod Pod + 로컬 vLLM (프로덕션 권장)

```bash
# 환경 변수 설정
export USE_POD_VLLM="true"
export USE_RUNPOD="false"  # vLLM 사용 시 자동으로 false
export QDRANT_URL="./qdrant_storage"

# vLLM 설정
export VLLM_TENSOR_PARALLEL_SIZE=1  # GPU 개수에 따라 조정
export VLLM_MAX_MODEL_LEN=4096  # 선택사항

# 동적 배치 크기 설정 (OOM 방지)
export VLLM_MAX_TOKENS_PER_BATCH=4000
export VLLM_MIN_BATCH_SIZE=10
export VLLM_MAX_BATCH_SIZE=100
export VLLM_MAX_CONCURRENT_BATCHES=20

# 우선순위 큐 설정 (Prefill 비용 기반)
export VLLM_USE_PRIORITY_QUEUE=true  # 우선순위 큐 사용 여부 (기본값: true)
export VLLM_PRIORITY_BY_PREFILL_COST=true  # Prefill 비용 기반 우선순위 (기본값: true)

# Watchdog 설정 (비용 최적화)
export RUNPOD_POD_ID="your_pod_id"
export IDLE_THRESHOLD=5
export CHECK_INTERVAL=60
export IDLE_LIMIT=5
export MIN_RUNTIME=600

# vLLM 필수 설치
pip install vllm>=0.3.3

# 전체 패키지 설치
pip install -r requirements.txt

# 방법 1: Python 스크립트로 테스트 데이터 생성
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 100 --restaurants 5

# Pod 내에서 서버 실행
python app.py

# 방법 2: API 엔드포인트로 테스트 데이터 생성 (서버 실행 후)
curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&restaurants=5" \
  -H "Content-Type: application/json" \
  -o test_data.json

# 테스트
# 배치 처리 테스트 (동적 배치 크기 + 세마포어 적용)
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @test_data.json \
  --max-time 600
```

## 성능 최적화

### 대용량 리뷰 처리
- **감성 분석,요약,강점 추출**: 배치 처리 (동적 배치 크기)로 대량 리뷰 처리 속도 향상
- **벡터 인코딩**: 배치 처리 (배치 크기: 32)로 벡터 변환 최적화
- **에러 처리**: 배치 실패 시 개별 처리로 폴백하여 안정성 보장

### 배치 처리 최적화 (동적 배치 크기 + 비동기 큐 + 우선순위 큐)
- **동적 배치 크기**: 리뷰 길이에 따라 배치 크기 자동 조정 (OOM 방지)
- **세마포어 제한**: 동시 처리 배치 수 제한으로 메모리 누적 방지
- **vLLM Continuous Batching**: GPU 활용률 극대화
- **비동기 큐 처리**: 여러 레스토랑의 배치를 비동기로 처리하여 처리량 향상
- **우선순위 큐 (Prefill 비용 기반)**: Shortest Job First (SJF) 알고리즘 적용
  - Prefill 비용은 입력 토큰 수로 정확히 예측 가능
  - 작은 요청 우선 처리로 SLA 보호
  - **효과**: 작은 요청 TTFT 30-40% 개선 (2.5초 → 1.8초), SLA 준수율 85% → 92% 향상

자세한 내용은 [BATCH_PROCESSING_IMPROVEMENT.md](BATCH_PROCESSING_IMPROVEMENT.md) 및 [PREFILL_DECODING.md](PREFILL_DECODING.md) 참조

### 대표 벡터 TOP-K 방식
- **감성 분석, 요약, 강점 추출**에서 레스토랑의 대표 벡터 주위에서 TOP-K 리뷰 선택
- 대표 벡터는 모든 리뷰 임베딩의 가중 평균 (최신 리뷰/높은 rating에 가중치)
- 관련성 높은 리뷰만 선택하여 컨텍스트 크기 최적화
- **효과**: 토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축

### 벡터 검색 활용
- 모든 요약 및 강점 추출 기능에서 벡터 검색을 활용하여 관련 리뷰 자동 검색
- 의미 기반 검색으로 정확도 향상
- 메타데이터 자동 포함으로 추가 조회 불필요

### 강점 추출 V2 (구조화된 파이프라인: Step A~H)

- **Step A**: 대표 벡터 TOP-K + 다양성 샘플링으로 타겟 긍정 근거 후보 수집
- **Step B**: LLM으로 구조화된 강점 후보 생성 (aspect, claim, evidence, type, 최소 5개 보장)
- **Step C**: Qdrant 벡터 검색으로 근거 확장 및 검증 (support_count_raw/valid/count, consistency)
- **Step D**: Connected Components (Union-Find)로 의미 중복 제거
  - 이중 임계값 (T_high=0.88, T_low=0.82)
  - Evidence overlap 가드레일 (30%)
  - Aspect type 체크 (다른 타입 병합 방지)
- **Step D-1**: Claim 후처리 재생성
  - 템플릿 기반 보정 (15-28자, 메타 표현 통일)
  - LLM 기반 생성 (맛 claim은 구체명사 포함 필수)
- **Step E~H**: 비교군 기반 차별 강점 계산
  - Step E: 비교군 구성 (category 필터 + 대표 벡터 검색)
  - Step F: 비교군 강점 인덱스 (실시간 계산)
  - Step G: 타겟 vs 비교군 유사도 (distinct = 1 - max_sim)
  - Step H: 최종 점수 계산 (rep * (1 + alpha * distinct))
- **구조화된 출력**: aspect + claim + evidence 스니펫 + support_count + distinct_score
- **환각 방지**: support_count, consistency 검증으로 신뢰도 향상
- **대표/차별 강점 구분**: representative vs distinct vs both

자세한 내용은 [STREGNTH_PIPELINE.md](STREGNTH_PIPELINE.md) 및 [LLM_SERVICE_STEP/LLM_SERVICE_DESIGN.md](LLM_SERVICE_STEP/LLM_SERVICE_DESIGN.md)를 참조하세요.

## 검색 품질 평가 (Precision@k)

벡터 검색 결과의 정확도를 Precision@k 지표로 평가할 수 있습니다.

### Precision@k란?

Precision@k = (상위 k개 검색 결과 중 관련 있는 문서 수) / k

- **P@1**: 상위 1개 결과 중 관련 있는 문서 비율
- **P@3**: 상위 3개 결과 중 관련 있는 문서 비율
- **P@5**: 상위 5개 결과 중 관련 있는 문서 비율
- **P@10**: 상위 10개 결과 중 관련 있는 문서 비율

### Ground Truth 준비

Ground Truth 파일은 다음 형식이어야 합니다:

```json
{
  "queries": [
    {
      "query": "맛있다 좋다 만족",
      "restaurant_id": 1,
      "relevant_review_ids": [1, 2, 3, 5, 7]
    },
    {
      "query": "맛없다 별로 불만",
      "restaurant_id": 1,
      "relevant_review_ids": [4, 6, 8]
    }
  ]
}
```

**필드 설명:**
- `query`: 검색 쿼리 텍스트
- `restaurant_id`: 레스토랑 ID 필터 (선택사항, None이면 전체 검색)
- `relevant_review_ids`: 관련 있는 리뷰 ID 리스트 (Ground Truth)

예시 파일: `scripts/ground_truth_example.json`

### 평가 실행

```bash
# 서버 실행 (평가 전에 서버가 실행 중이어야 함)
python app.py

# 별도 터미널에서 Precision@k 평가 실행
python scripts/evaluate_precision_at_k.py \
  --ground-truth scripts/ground_truth_example.json \
  --k-values 1 3 5 10 \
  --limit 10 \
  --min-score 0.0 \
  --output precision_at_k_results.json
```

**옵션 설명:**
- `--ground-truth`: Ground Truth JSON 파일 경로 (필수)
- `--base-url`: API 서버 URL (기본값: http://localhost:8000)
- `--k-values`: 평가할 k 값 리스트 (기본값: 1 3 5 10)
- `--limit`: 검색할 최대 개수 (기본값: 10)
- `--min-score`: 최소 유사도 점수 (기본값: 0.0)
- `--output`: 결과 저장 파일 경로 (선택사항, 기본값: precision_at_k_results/YYYYMMDD_HHMMSS.json)

### 평가 결과

평가 결과는 다음 형식으로 저장됩니다:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_queries": 4,
  "evaluated_queries": 4,
  "k_values": [1, 3, 5, 10],
  "average_precisions": {
    "P@1": 0.7500,
    "P@3": 0.6667,
    "P@5": 0.6000,
    "P@10": 0.5500
  },
  "query_results": [
    {
      "query": "맛있다 좋다 만족",
      "restaurant_id": 1,
      "retrieved_count": 10,
      "relevant_count": 5,
      "retrieved_ids": [1, 2, 3, 4, 5, ...],
      "precisions": {
        "P@1": 1.0,
        "P@3": 1.0,
        "P@5": 0.8,
        "P@10": 0.5
      }
    }
  ]
}
```

**성능 벤치마크와의 차이:**
- **benchmark.py**: 시스템 성능 측정 (latency, throughput, GPU 사용률)
- **evaluate_precision_at_k.py**: 검색 품질 평가 (검색 결과 정확도)

### 추가 평가 스크립트

다음 평가 스크립트들도 사용할 수 있습니다:

1. **감성 분석 평가** (`scripts/evaluate_sentiment_analysis.py`)
   - 리뷰 단위 정확도, 비율 정확도, 개수 정확도
   - Ground Truth: `scripts/Ground_truth_sentiment.json`

2. **리뷰 요약 평가** (`scripts/evaluate_summary.py`)
   - ROUGE/BLEU 점수, Aspect Coverage
   - Ground Truth: `scripts/Ground_truth_summary.json`
   - 필요 패키지: `pip install rouge-score nltk`

3. **강점 추출 평가** (`scripts/evaluate_strength_extraction.py`)
   - Precision@K, Coverage, False Positive Rate
   - Ground Truth: `scripts/Ground_truth_strength.json`

자세한 내용은 [BENCHMARK.md](BENCHMARK.md) 참조

## 설정

`src/config.py`에서 기본 설정을 변경할 수 있습니다:

- `EMBEDDING_MODEL`: 임베딩 모델 (기본값: "jhgan/ko-sbert-multitask")
- `LLM_MODEL`: LLM 모델 (기본값: "Qwen/Qwen2.5-7B-Instruct")
- `MAX_RETRIES`: LLM 호출 최대 재시도 횟수 (기본값: 3)
- `COLLECTION_NAME`: Qdrant 컬렉션 이름 (기본값: "reviews_collection")
- `USE_POD_VLLM`: RunPod Pod + 로컬 vLLM 사용 여부 (기본값: false)
- `VLLM_TENSOR_PARALLEL_SIZE`: 텐서 병렬 크기 (기본값: 1)
- `VLLM_MAX_MODEL_LEN`: 최대 모델 길이 (선택사항)
- `VLLM_MAX_TOKENS_PER_BATCH`: 배치당 최대 토큰 수 (기본값: 4000)
- `VLLM_MIN_BATCH_SIZE`: 최소 배치 크기 (기본값: 10)
- `VLLM_MAX_BATCH_SIZE`: 최대 배치 크기 (기본값: 100)
- `VLLM_MAX_CONCURRENT_BATCHES`: 최대 동시 처리 배치 수 (기본값: 20)
- `VLLM_USE_PRIORITY_QUEUE`: 우선순위 큐 사용 여부 (기본값: true)
- `VLLM_PRIORITY_BY_PREFILL_COST`: Prefill 비용 기반 우선순위 (기본값: true)

## 프로덕션 환경

프로덕션 환경에서 발생할 수 있는 문제점과 개선방안은 [PRODUCTION_ISSUES_AND_IMPROVEMENTS.md](PRODUCTION_ISSUES_AND_IMPROVEMENTS.md)를 참조하세요.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

