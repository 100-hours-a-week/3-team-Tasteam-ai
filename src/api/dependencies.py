"""
FastAPI 의존성 주입
"""

from functools import lru_cache
from fastapi import Depends, Header, Query
from qdrant_client import QdrantClient
from typing import Optional

from ..config import Config
from ..sentiment_analysis import SentimentAnalyzer
from ..vector_search import VectorSearch
from ..llm_utils import LLMUtils
from ..metrics_collector import MetricsCollector
from ..cpu_monitor import get_or_create_benchmark_cpu_monitor
from ..gpu_monitor import get_or_create_benchmark_gpu_monitor

# 의존성 싱글톤 (매 요청 인스턴스 생성 방지 — Encoder/Qdrant는 @lru_cache, 나머지는 모듈 캐시)
_vector_search_singleton: Optional[VectorSearch] = None
_sentiment_analyzer_singleton: Optional[SentimentAnalyzer] = None
_default_metrics_collector: Optional[MetricsCollector] = None
_benchmark_metrics_collector: Optional[MetricsCollector] = None


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 싱글톤"""
    # 메모리 모드
    if Config.QDRANT_URL == ":memory:":
        return QdrantClient(location=":memory:")
    
    # HTTP/HTTPS로 시작하면 원격 서버
    if Config.QDRANT_URL.startswith(("http://", "https://")):
        return QdrantClient(url=Config.QDRANT_URL)
    
    # 그 외는 로컬 파일 경로 (on-disk)
    # 디렉토리가 없으면 자동 생성
    import os
    qdrant_path = Config.QDRANT_URL
    if not os.path.isabs(qdrant_path):
        # 상대 경로인 경우 프로젝트 루트 기준으로 변환
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qdrant_path = os.path.join(project_root, qdrant_path)
    
    # 디렉토리 생성
    os.makedirs(qdrant_path, exist_ok=True)
    
    return QdrantClient(path=qdrant_path)


def get_metrics_collector(
    x_benchmark: Optional[str] = Header(None, alias="X-Benchmark"),
    x_enable_cpu_monitor: Optional[str] = Header(None, alias="X-Enable-CPU-Monitor"),
    x_enable_gpu_monitor: Optional[str] = Header(None, alias="X-Enable-GPU-Monitor"),
) -> MetricsCollector:
    """
    메트릭 수집기 싱글톤.
    - X-Benchmark: true 이면 요청 메트릭 수집 활성화 (logs + metrics.db).
    - X-Enable-CPU-Monitor: true 이면 CPU 모니터링만 활성화 (logs/cpu_usage.log).
    - X-Enable-GPU-Monitor: true 이면 서버 GPU 모니터링만 활성화 (logs/gpu_usage.log).
    - 헤더는 독립적: 메트릭만 / CPU만 / GPU만 / 조합 가능.
    """
    global _default_metrics_collector, _benchmark_metrics_collector

    _truth = ("true", "1", "yes")
    want_cpu = x_enable_cpu_monitor and str(x_enable_cpu_monitor).strip().lower() in _truth
    want_gpu = x_enable_gpu_monitor and str(x_enable_gpu_monitor).strip().lower() in _truth
    want_metrics = x_benchmark and str(x_benchmark).strip().lower() in _truth

    if want_cpu:
        get_or_create_benchmark_cpu_monitor()
    if want_gpu:
        get_or_create_benchmark_gpu_monitor()

    if want_metrics:
        if _benchmark_metrics_collector is None:
            _benchmark_metrics_collector = MetricsCollector(
                enable_logging=True,
                enable_db=True,
                db_path=Config.METRICS_DB_PATH,
                log_dir=Config.METRICS_LOG_DIR,
            )
        return _benchmark_metrics_collector

    if _default_metrics_collector is None:
        if not Config.METRICS_AND_LOGGING_ENABLE:
            _default_metrics_collector = MetricsCollector(
                enable_logging=False,
                enable_db=False,
                db_path=Config.METRICS_DB_PATH,
                log_dir=Config.METRICS_LOG_DIR,
            )
        else:
            _default_metrics_collector = MetricsCollector(
                enable_logging=Config.METRICS_ENABLE_LOGGING,
                enable_db=Config.METRICS_ENABLE_DB,
                db_path=Config.METRICS_DB_PATH,
                log_dir=Config.METRICS_LOG_DIR,
            )
    return _default_metrics_collector


@lru_cache()
def get_llm_utils() -> LLMUtils:
    """LLM 유틸리티 싱글톤 (Qwen 모델)"""
    return LLMUtils(model_name=Config.LLM_MODEL)


def get_debug_mode(
    x_debug: Optional[str] = Header(None, alias="X-Debug"),
    debug: Optional[bool] = Query(None, description="디버그 모드 활성화"),
) -> bool:
    """
    디버그 모드 감지
    1. X-Debug 헤더 우선
    2. debug 쿼리 파라미터
    3. 환경 변수 (DEBUG_MODE)
    
    Args:
        x_debug: X-Debug 헤더 값
        debug: debug 쿼리 파라미터 값
        
    Returns:
        디버그 모드 여부
    """
    import os
    
    # Header 우선
    if x_debug is not None:
        return x_debug.lower() in ("true", "1", "yes")
    
    # Query Parameter
    if debug is not None:
        return debug
    
    # 환경 변수
    return os.getenv("DEBUG_MODE", "false").lower() == "true"


def get_vector_search(
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> VectorSearch:
    """벡터 검색 의존성 (싱글톤 — FastEmbed Dense+Sparse, 초기화 비용 재사용)"""
    global _vector_search_singleton
    if _vector_search_singleton is None:
        _vector_search_singleton = VectorSearch(
            qdrant_client=qdrant_client,
            collection_name=Config.COLLECTION_NAME,
        )
    return _vector_search_singleton


def get_sentiment_analyzer(
    vector_search: VectorSearch = Depends(get_vector_search),
) -> SentimentAnalyzer:
    """감성 분석기 의존성 (싱글톤 — HF pipeline 로딩 비용 재사용)"""
    global _sentiment_analyzer_singleton
    if _sentiment_analyzer_singleton is None:
        _sentiment_analyzer_singleton = SentimentAnalyzer(vector_search=vector_search)
    return _sentiment_analyzer_singleton
