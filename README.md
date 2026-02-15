# Review Analysis API

레스토랑 리뷰의 감성 분석, 벡터 검색, LLM 기반 요약 및 강점 추출을 수행하는 FastAPI 기반 프로젝트입니다.

## 주요 기능

- **감성 분석**: 리뷰 감성 비율 추출 (positive_ratio, negative_ratio)
- **리뷰 요약**: 긍정/부정/전체 요약 + Aspect 추출
- **강점 추출**: 비교군 기반 차별 강점 추출
- **벡터 검색**: 의미 기반 리뷰 검색
- **배치 처리**: 여러 레스토랑 동시 처리 지원

## 설치

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

## 환경 변수 설정

```bash
# LLM 제공자 선택 (openai, runpod, local)
export LLM_PROVIDER="openai"  # 기본값: openai
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"

# Qdrant 설정
export QDRANT_URL="./qdrant_db"  # 기본값: ./qdrant_db (on-disk 모드)
# 또는 메모리 모드: export QDRANT_URL=":memory:"
# 또는 원격 서버: export QDRANT_URL="http://localhost:6333"

# vLLM 설정 (RunPod Pod 환경)
export USE_POD_VLLM="true"
export VLLM_MAX_TOKENS_PER_BATCH=4000
export VLLM_MIN_BATCH_SIZE=10
export VLLM_MAX_BATCH_SIZE=100
```

## 사용 방법

### 서버 실행

```bash
python app.py
# 또는
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload
```

### API 문서

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### 주요 엔드포인트

| 기능 | 메서드 | 엔드포인트 |
|------|--------|-----------|
| 감성 분석 | POST | `/api/v1/sentiment/analyze` |
| 배치 감성 분석 | POST | `/api/v1/sentiment/analyze/batch` |
| 리뷰 요약 | POST | `/api/v1/llm/summarize` |
| 배치 리뷰 요약 | POST | `/api/v1/llm/summarize/batch` |
| 강점 추출 | POST | `/api/v1/llm/extract/strengths` |
| 벡터 검색 | POST | `/api/v1/vector/search/similar` |
| 데이터 업로드 | POST | `/api/v1/vector/upload` |
| 헬스 체크 | GET | `/health` |

## 테스트

```bash
# 테스트 실행
python test_all_task.py

# 성능 측정 모드
python test_all_task.py --benchmark --iterations 5

# 결과 저장
python test_all_task.py --benchmark --save-results results.json
```

## Docker 이미지 빌드 및 Docker Hub 푸시

API 이미지를 빌드한 뒤 Docker Hub에 푸시하려면:

1. **Docker Hub 로그인** (최초 1회)
   ```bash
   docker login
   ```

2. **환경 변수 설정** (선택: `.env`에 넣거나 export)
   ```bash
   export DOCKERHUB_USERNAME=your-dockerhub-username
   # 선택: export DOCKER_IMAGE_NAME=tasteam-review-api
   ```

3. **빌드 및 푸시**
   ```bash
   # CUDA 이미지 (기본), 태그 1.0.0 + latest
   ./scripts/build_and_push.sh 1.0.0 cuda

   # CPU 이미지
   ./scripts/build_and_push.sh 1.0.0 cpu

   # 버전 생략 시 latest 사용
   ./scripts/build_and_push.sh latest cpu
   ```

푸시된 이미지 예: `your-dockerhub-username/tasteam-review-api:1.0.0`, `your-dockerhub-username/tasteam-review-api:latest-cpu`

## 프로젝트 구조

```
├── src/              # 소스 코드
│   ├── api/         # FastAPI 라우터
│   ├── config.py    # 설정 관리
│   └── ...
├── scripts/         # 유틸리티 스크립트
├── data/            # 테스트 데이터
├── app.py           # 서버 실행 스크립트
└── requirements.txt # 패키지 의존성
```

## 참고 문서

- **[docs/README.md](docs/README.md)** — 문서 인덱스 (아키텍처, API, RunPod, 배치, Spark, 트러블슈팅 등 카테고리별 정리)
- [PIPELINE_OPERATIONS.md](PIPELINE_OPERATIONS.md) - 파이프라인 동작 (Strength, Summary, Sentiment, Vector)
- [etc_md/OBSERVABILITY_PROM_GRAFANA.md](etc_md/OBSERVABILITY_PROM_GRAFANA.md) - Prometheus/Grafana 사용법 및 현재 수집 범위
- [.env.example](.env.example) - 환경 변수 예시 (Docker Hub 푸시용 `DOCKERHUB_USERNAME`, `DOCKER_IMAGE_NAME` 포함)
- [LLM_SERVICE_STEP/API_SPECIFICATION.md](LLM_SERVICE_STEP/API_SPECIFICATION.md) - API 명세서
- [LLM_SERVICE_STEP/ARCHITECTURE.md](LLM_SERVICE_STEP/ARCHITECTURE.md) - 시스템 아키텍처
