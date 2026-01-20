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
export QDRANT_URL="./qdrant_storage"  # 기본값: :memory:

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
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 주요 엔드포인트

| 기능 | 메서드 | 엔드포인트 |
|------|--------|-----------|
| 감성 분석 | POST | `/api/v1/sentiment/analyze` |
| 배치 감성 분석 | POST | `/api/v1/sentiment/analyze/batch` |
| 리뷰 요약 | POST | `/api/v1/llm/summarize` |
| 배치 리뷰 요약 | POST | `/api/v1/llm/summarize/batch` |
| 강점 추출 | POST | `/api/v1/llm/extract/strengths` |
| 벡터 검색 | POST | `/api/v1/vector/search/similar` |
| 이미지 리뷰 검색 | POST | `/api/v1/vector/search/review-images` |
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

- [API_SPECIFICATION.md](API_SPECIFICATION.md) - API 명세서
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
