"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from .routers import sentiment, vector, llm, restaurant, test

# 로거 설정 (콘솔 출력)
# basicConfig는 한 번만 실행되므로, root 로거에 직접 핸들러 추가
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

# basicConfig도 시도 (이미 설정되어 있어도 무시됨)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 초기화
    logger.info("FastAPI 애플리케이션 시작")
    yield
    # 종료 시 정리
    logger.info("FastAPI 애플리케이션 종료")


app = FastAPI(
    title="Review Sentiment Analysis API",
    description="레스토랑 리뷰 감성 분석, 벡터 검색, LLM 기반 요약 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["sentiment"])
app.include_router(vector.router, prefix="/api/v1/vector", tags=["vector"])
app.include_router(llm.router, prefix="/api/v1/llm", tags=["llm"])
app.include_router(restaurant.router, prefix="/api/v1/restaurants", tags=["restaurants"])
app.include_router(test.router, prefix="/api/v1/test", tags=["test"])


@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Review Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=dict)
async def health():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "version": "1.0.0",
    }

