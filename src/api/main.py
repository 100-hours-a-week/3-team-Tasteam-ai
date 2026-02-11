"""
FastAPI 메인 애플리케이션
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import asyncio
import atexit
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
import logging
import uuid

from .routers import sentiment, vector, llm, test
from .dependencies import get_qdrant_client, get_vector_search, get_sentiment_analyzer
from ..cpu_monitor import get_cpu_monitor
from ..metrics_collector import app_queue_depth_inc, app_queue_depth_dec, set_event_loop_lag_seconds

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _INSTRUMENTATOR_AVAILABLE = True
except ImportError:
    _INSTRUMENTATOR_AVAILABLE = False

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

# --- 종료 시 로그 / 마지막 예외 출력 (down 원인 파악용) ---
_last_exc_info = None


def _excepthook(typ, val, tb):
    """미처리 예외 시 저장 후 기존 excepthook 호출."""
    global _last_exc_info
    _last_exc_info = (typ, val, tb)
    sys.__excepthook__(typ, val, tb)


def _atexit_shutdown():
    """정상/비정상 종료 시 한 번 실행: shutdown 로그 + 마지막 예외가 있으면 출력."""
    logger.info("shutdown (atexit)")
    if _last_exc_info is not None:
        typ, val, tb = _last_exc_info
        try:
            traceback.print_exception(typ, val, tb, file=sys.stderr)
        except Exception:
            pass


def _signal_shutdown(signum, frame):
    """SIGTERM/SIGINT 수신 시 로그 후 종료."""
    logger.info("shutdown requested (signal %s)", signum)
    sys.exit(0)


sys.excepthook = _excepthook
atexit.register(_atexit_shutdown)
if getattr(signal, "SIGTERM", None) is not None:
    signal.signal(signal.SIGTERM, _signal_shutdown)
if getattr(signal, "SIGINT", None) is not None:
    signal.signal(signal.SIGINT, _signal_shutdown)

# ---


async def _event_loop_lag_reporter(interval_seconds: float = 5.0) -> None:
    """
    주기적으로 이벤트 루프 지연을 측정해 Prometheus event_loop_lag_seconds에 기록.
    call_soon으로 넣은 콜백이 실제 실행되기까지 걸린 시간 = 루프가 바쁜 정도.
    """
    loop = asyncio.get_running_loop()
    while True:
        t0 = time.monotonic()
        fut = loop.create_future()

        def _cb() -> None:
            lag = time.monotonic() - t0
            set_event_loop_lag_seconds(lag)
            if not fut.done():
                fut.set_result(None)

        loop.call_soon(_cb)
        await fut
        await asyncio.sleep(interval_seconds)


def _warm_up_services():
    """모델·토크나이저·첫 inference까지 강제 warm-up (첫 요청에서 로드 방지)."""
    try:
        client = get_qdrant_client()
        vector_search = get_vector_search(client)
        list(vector_search.encoder.encode(["warmup"]))
        logger.info("VectorSearch(임베딩) warm-up 완료")
        analyzer = get_sentiment_analyzer(vector_search)
        pl = analyzer._get_sentiment_pipeline()
        if pl is not None:
            pl("웜업")
        logger.info("Sentiment pipeline warm-up 완료")
    except Exception as e:
        logger.warning("warm-up 중 일부 실패 (첫 요청에서 로드될 수 있음): %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    logger.info("FastAPI 애플리케이션 시작")
    app.state.ready = False

    # CPU 모니터 시작 (Config.CPU_MONITOR_ENABLE=true일 때만)
    cpu_monitor = get_cpu_monitor()
    if cpu_monitor:
        cpu_monitor.start()

    # 모델 + 토크나이저 + 첫 inference warm-up (블로킹이므로 스레드에서 실행)
    try:
        await asyncio.to_thread(_warm_up_services)
        app.state.ready = True
        logger.info("서비스 warm-up 완료, readiness=True")
    except Exception as e:
        logger.warning("warm-up 실패, /ready는 503 반환: %s", e)

    # Event loop lag 주기 측정 (Prometheus event_loop_lag_seconds 노출)
    lag_task = asyncio.create_task(_event_loop_lag_reporter(5.0))
    try:
        yield
    finally:
        lag_task.cancel()
        try:
            await lag_task
        except asyncio.CancelledError:
            pass

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


# Queue depth (in-flight 요청 수) — Prometheus app_queue_depth 집계용
@app.middleware("http")
async def track_queue_depth(request: Request, call_next):
    app_queue_depth_inc()
    try:
        response = await call_next(request)
        return response
    finally:
        app_queue_depth_dec()


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


@app.get("/ready")
async def ready(request: Request):
    """Readiness: warm-up 완료 후 200, 미완료 시 503 (K8s readinessProbe 등)."""
    if getattr(request.app.state, "ready", False):
        return {"status": "ready"}
    from fastapi.responses import Response
    return Response(content='{"status":"not ready"}', status_code=503, media_type="application/json")


# Prometheus 메트릭 (요청 수, 지연 시간 등 자동 수집, 패키지 설치 시에만 노출)
if _INSTRUMENTATOR_AVAILABLE:
    Instrumentator().instrument(app).expose(app)
else:
    import logging
    logging.getLogger(__name__).warning(
        "prometheus_fastapi_instrumentator 미설치: /metrics 비활성화. "
        "설치: pip install prometheus-client prometheus-fastapi-instrumentator"
    )

