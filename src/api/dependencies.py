"""
FastAPI 의존성 주입
"""

from functools import lru_cache
from fastapi import Depends, Header, Query
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import Optional

from ..config import Config
from ..sentiment_analysis import SentimentAnalyzer
from ..vector_search import VectorSearch
from ..llm_utils import LLMUtils
from ..metrics_collector import MetricsCollector


@lru_cache()
def get_encoder() -> SentenceTransformer:
    """SentenceTransformer 인코더 싱글톤"""
    return SentenceTransformer(Config.EMBEDDING_MODEL)


@lru_cache()
def get_qdrant_client() -> QdrantClient:
    """Qdrant 클라이언트 싱글톤"""
    if Config.QDRANT_URL == ":memory:":
        return QdrantClient(location=":memory:")
    
    # HTTP/HTTPS로 시작하면 원격 서버
    if Config.QDRANT_URL.startswith(("http://", "https://")):
        return QdrantClient(url=Config.QDRANT_URL)
    
    # 그 외는 로컬 파일 경로 (on-disk)
    return QdrantClient(path=Config.QDRANT_URL)


@lru_cache()
def get_metrics_collector() -> MetricsCollector:
    """메트릭 수집기 싱글톤"""
    return MetricsCollector(
        enable_logging=Config.METRICS_ENABLE_LOGGING,
        enable_db=Config.METRICS_ENABLE_DB,
        db_path=Config.METRICS_DB_PATH,
        log_dir=Config.METRICS_LOG_DIR,
    )


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
    encoder: SentenceTransformer = Depends(get_encoder),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> VectorSearch:
    """벡터 검색 의존성"""
    return VectorSearch(
        encoder=encoder,
        qdrant_client=qdrant_client,
        collection_name=Config.COLLECTION_NAME,
    )


def get_sentiment_analyzer(
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
) -> SentimentAnalyzer:
    """감성 분석기 의존성"""
    return SentimentAnalyzer(
        llm_utils=llm_utils,
        vector_search=vector_search,
    )
