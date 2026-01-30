"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
import logging
import sys
import uuid

from .routers import sentiment, vector, llm, restaurant, test
from ..cpu_monitor import get_cpu_monitor

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
    
    # CPU 모니터 시작 (Config.CPU_MONITOR_ENABLE=true일 때만)
    cpu_monitor = get_cpu_monitor()
    if cpu_monitor:
        cpu_monitor.start()
    
    yield
    
    # 종료 시 정리
    if cpu_monitor:
        await cpu_monitor.stop()
    logger.info("FastAPI 애플리케이션 종료")


app = FastAPI(
    title="Review Sentiment Analysis API",
    description="레스토랑 리뷰 감성 분석, 벡터 검색, LLM 기반 요약 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 요청 ID 미들웨어 (응답 헤더에도 포함)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


def _error_payload(*, code: int, message: str, details, request_id: str) -> dict:
    return {"code": code, "message": message, "details": details, "request_id": request_id}


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            code=422,
            message="Validation error",
            details=exc.errors(),
            request_id=request_id,
        ),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    # 404 등 라우팅 단계 HTTP 예외 포함
    detail = getattr(exc, "detail", None)
    message = str(detail) if detail is not None else "HTTP error"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(
            code=exc.status_code,
            message=message,
            details=detail,
            request_id=request_id,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code=500,
            message="Internal server error",
            details=None,
            request_id=request_id,
        ),
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

