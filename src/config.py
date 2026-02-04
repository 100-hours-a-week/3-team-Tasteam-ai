"""
설정 관리 모듈 (CONFIG_CONSTRUCTION.md 반영)

도메인별로 분리: server, inference, retrieval, cache, spark, observability
- 환경별 값(URL, credentials, device, worker 수, retry 등)
- 성능/비용 튜닝(batch_size, concurrency, to_thread, cache TTL 등)
- 실험 플래그/실행 모드(sync vs async, sync vs thread isolation)
"""

import os
from typing import Optional, List

try:
    import torch
except ImportError:
    torch = None

# 기본 상수 (코드 내부 상수, CONFIG_CONSTRUCTION: 거의 안 바꾸는 건 config 밖)
DEFAULT_SENTIMENT_MODEL = "Dilwolf/Kakao_app-kr_sentiment"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_MAX_RETRIES = 3
DEFAULT_COLLECTION_NAME = "reviews_collection"
DEFAULT_LLM_BATCH_SIZE = 10
DEFAULT_LLM_KEYWORDS = ["는데", "지만"]


# --- Server (환경별 URL, worker, rate limit, timeout) ---
class _ServerConfig:
    """서버/런타임: URL, worker, timeout 등"""
    pass  # FastAPI 등 서버 설정은 app 레벨에서 처리


# --- Inference (모델, device, LLM 백엔드, batch, 실행 모드) ---
class _InferenceConfig:
    """
    추론 도메인: 모델 경로/이름, device(cpu/gpu), worker 수
    batch_size, max_concurrency, to_thread on/off, 실행 모드
    """
    # 모델
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", DEFAULT_SENTIMENT_MODEL)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))
    SPARSE_EMBEDDING_MODEL: str = os.getenv("SPARSE_EMBEDDING_MODEL", DEFAULT_SPARSE_EMBEDDING_MODEL)
    LLM_MODEL: str = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # device, GPU, quantization
    USE_GPU: bool = (os.getenv("USE_GPU") or os.getenv("USE_GPU#", "true")).lower() == "true"
    GPU_DEVICE: int = int(os.getenv("GPU_DEVICE", "0"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"

    # retry, timeout
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    LLM_POLISH_TIMEOUT_SECONDS: float = float(os.getenv("LLM_POLISH_TIMEOUT_SECONDS", "2.0"))

    # batch, concurrency (성능 튜닝)
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", str(DEFAULT_LLM_BATCH_SIZE)))
    BATCH_SEARCH_CONCURRENCY: int = int(os.getenv("BATCH_SEARCH_CONCURRENCY", "50"))
    BATCH_LLM_CONCURRENCY: int = int(os.getenv("BATCH_LLM_CONCURRENCY", "8"))
    VLLM_MAX_TOKENS_PER_BATCH: int = int(os.getenv("VLLM_MAX_TOKENS_PER_BATCH", "4000"))
    VLLM_MIN_BATCH_SIZE: int = int(os.getenv("VLLM_MIN_BATCH_SIZE", "10"))
    VLLM_MAX_BATCH_SIZE: int = int(os.getenv("VLLM_MAX_BATCH_SIZE", "100"))
    VLLM_DEFAULT_BATCH_SIZE: int = int(os.getenv("VLLM_DEFAULT_BATCH_SIZE", "50"))
    VLLM_MAX_CONCURRENT_BATCHES: int = int(os.getenv("VLLM_MAX_CONCURRENT_BATCHES", "20"))

    # credentials (비밀은 .env)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ENABLE_OPENAI_FALLBACK: bool = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() == "true"
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "g09uegksn7h7ed")
    USE_RUNPOD: bool = os.getenv("USE_RUNPOD", "true").lower() == "true"
    RUNPOD_POLL_INTERVAL: int = int(os.getenv("RUNPOD_POLL_INTERVAL", "2"))
    RUNPOD_MAX_WAIT_TIME: int = int(os.getenv("RUNPOD_MAX_WAIT_TIME", "300"))
    USE_POD_VLLM: bool = os.getenv("USE_POD_VLLM", "false").lower() == "true"
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    VLLM_MAX_MODEL_LEN: Optional[int] = int(os.getenv("VLLM_MAX_MODEL_LEN")) if os.getenv("VLLM_MAX_MODEL_LEN") else None
    VLLM_USE_PRIORITY_QUEUE: bool = os.getenv("VLLM_USE_PRIORITY_QUEUE", "true").lower() == "true"
    VLLM_PRIORITY_BY_PREFILL_COST: bool = os.getenv("VLLM_PRIORITY_BY_PREFILL_COST", "true").lower() == "true"

    # 실행 모드: sync vs async, sync vs thread isolation (실험 플래그)
    
    # sync vs thread isolation (기본값 true: 배치 시 속도·이벤트 루프 격리)
    
    # Sentiment: HF 분류기. true=asyncio.to_thread(블로킹 격리), false=메인 스레드
    SENTIMENT_CLASSIFIER_USE_THREAD: bool = os.getenv("SENTIMENT_CLASSIFIER_USE_THREAD", "true").lower() == "true"
    # Sentiment: LLM 재판정. true=AsyncOpenAI, false=동기
    SENTIMENT_LLM_ASYNC: bool = os.getenv("SENTIMENT_LLM_ASYNC", "true").lower() == "true"
    # Sentiment 배치: 음식점 간 병렬. true=asyncio.gather(병렬), false=순차
    SENTIMENT_RESTAURANT_ASYNC: bool = os.getenv("SENTIMENT_RESTAURANT_ASYNC", os.getenv("BATCH_RESTAURANT_ASYNC", "true")).lower() == "true"
    # Sentiment 파이프라인을 항상 CPU에서 실행 (meta tensor 오류 회피용, true 시 device=-1)
    SENTIMENT_FORCE_CPU: bool = os.getenv("SENTIMENT_FORCE_CPU", "true").lower() == "true"
    
    # sync vs async (기본값 true: 파이프라인 내·음식점 간 병렬로 최대 속도)
    
    # Summary 배치: LLM 호출. true=AsyncOpenAI/httpx.AsyncClient, false=asyncio.to_thread(동기 래핑)
    SUMMARY_LLM_ASYNC: bool = os.getenv("SUMMARY_LLM_ASYNC", os.getenv("LLM_ASYNC", "true")).lower() == "true"
    # Summary 배치: aspect(service/price/food) 서치 병렬
    SUMMARY_SEARCH_ASYNC: bool = os.getenv("SUMMARY_SEARCH_ASYNC", os.getenv("BATCH_SEARCH_ASYNC", "true")).lower() == "true"
    # Summary 배치: 음식점 간 병렬
    SUMMARY_RESTAURANT_ASYNC: bool = os.getenv("SUMMARY_RESTAURANT_ASYNC", os.getenv("BATCH_RESTAURANT_ASYNC", "true")).lower() == "true"
    
    # Comparison: service/price LLM 호출. true=asyncio.gather(병렬), false=순차
    COMPARISON_ASYNC: bool = os.getenv("COMPARISON_ASYNC", "true").lower() == "true"
    # Comparison 배치: 음식점 간 병렬. true=asyncio.gather(병렬), false=순차
    COMPARISON_BATCH_ASYNC: bool = os.getenv("COMPARISON_BATCH_ASYNC", "true").lower() == "true"


# --- Retrieval (Qdrant, embedding, top_k, rerank 등) ---
class _RetrievalConfig:
    """검색/검열: Qdrant URL, collection, top_k, rerank_k, aspect seed, 벡터 on_disk"""
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", "./qdrant_data")
    QDRANT_VECTORS_ON_DISK: bool = os.getenv("QDRANT_VECTORS_ON_DISK", "false").lower() == "true"
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD)))
    LLM_KEYWORDS: list = DEFAULT_LLM_KEYWORDS

    ENABLE_SENTIMENT_SAMPLING: bool = os.getenv("ENABLE_SENTIMENT_SAMPLING", "false").lower() == "true"
    SENTIMENT_RECENT_TOP_K: int = int(os.getenv("SENTIMENT_RECENT_TOP_K", "100"))
    ASPECT_SEEDS_FILE: Optional[str] = os.getenv("ASPECT_SEEDS_FILE")


# --- Cache (TTL, skip 간격 등) ---
class _CacheConfig:
    """캐시: TTL, skip 최소 간격"""
    SKIP_MIN_INTERVAL_SECONDS: int = int(os.getenv("SKIP_MIN_INTERVAL_SECONDS", "3600"))


# --- Spark (Comparison 전체 평균 데이터 등) ---
class _SparkConfig:
    """Spark/배치: 전체 평균 데이터 경로, 비율"""
    ALL_AVERAGE_ASPECT_DATA_PATH: Optional[str] = os.getenv("ALL_AVERAGE_ASPECT_DATA_PATH", "data/test_data_sample.json")
    ALL_AVERAGE_SERVICE_RATIO: float = float(os.getenv("ALL_AVERAGE_SERVICE_RATIO", "0.60"))
    ALL_AVERAGE_PRICE_RATIO: float = float(os.getenv("ALL_AVERAGE_PRICE_RATIO", "0.55"))


# --- Observability (로깅, 메트릭, 모니터링, watchdog) ---
class _ObservabilityConfig:
    """관측성: 로깅 레벨, 메트릭, CPU/GPU 모니터링, watchdog"""
    METRICS_AND_LOGGING_ENABLE: bool = os.getenv("METRICS_AND_LOGGING_ENABLE", "false").lower() == "true"
    METRICS_ENABLE_LOGGING: bool = os.getenv("METRICS_ENABLE_LOGGING", "true").lower() == "true"
    METRICS_ENABLE_DB: bool = os.getenv("METRICS_ENABLE_DB", "true").lower() == "true"
    METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", "metrics.db")
    METRICS_LOG_DIR: str = os.getenv("METRICS_LOG_DIR", "logs")

    CPU_MONITOR_ENABLE: bool = os.getenv("CPU_MONITOR_ENABLE", "false").lower() == "true"
    CPU_MONITOR_INTERVAL: float = float(os.getenv("CPU_MONITOR_INTERVAL", "1.0"))

    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")
    IDLE_THRESHOLD: int = int(os.getenv("IDLE_THRESHOLD", "5"))
    CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL", "60"))
    IDLE_LIMIT: int = int(os.getenv("IDLE_LIMIT", "5"))
    MIN_RUNTIME: int = int(os.getenv("MIN_RUNTIME", "600"))


# --- Config: 도메인별로 나누되, 기존 Config.XXX 호환 ---
class Config(_ServerConfig, _InferenceConfig, _RetrievalConfig, _CacheConfig, _SparkConfig, _ObservabilityConfig):
    """
    통합 설정. 도메인: Server, Inference, Retrieval, Cache, Spark, Observability.
    기존 Config.USE_GPU, Config.QDRANT_URL 등 그대로 사용 가능.
    """
    # 도메인 네임스페이스 (선택적 접근)
    Server = _ServerConfig
    Inference = _InferenceConfig
    Retrieval = _RetrievalConfig
    Cache = _CacheConfig
    Spark = _SparkConfig
    Observability = _ObservabilityConfig

    @classmethod
    def get_device(cls):
        """GPU 사용 가능 여부"""
        if torch is None:
            return -1
        if cls.USE_GPU and torch.cuda.is_available():
            return cls.GPU_DEVICE
        return -1

    @classmethod
    def get_dtype(cls):
        """양자화 타입"""
        if torch is None:
            return None
        if cls.USE_FP16 and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    @classmethod
    def get_optimal_batch_size(cls, model_type: str = "default"):
        """GPU 메모리에 따른 최적 배치 크기"""
        if torch is None or not cls.USE_GPU or not torch.cuda.is_available():
            return 32
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            return 32
        if model_type == "llm":
            if gpu_memory_gb >= 40:
                return 20
            elif gpu_memory_gb >= 24:
                return 10
            return 5
        elif model_type == "sentiment":
            if gpu_memory_gb >= 40:
                return 128
            elif gpu_memory_gb >= 24:
                return 64
            return 32
        else:
            if gpu_memory_gb >= 40:
                return 128
            elif gpu_memory_gb >= 24:
                return 64
            return 32

    @classmethod
    def calculate_dynamic_batch_size(cls, reviews: List[str], max_tokens_per_batch: Optional[int] = None) -> int:
        """리뷰 리스트 기반 동적 배치 크기"""
        if not reviews:
            return cls.VLLM_DEFAULT_BATCH_SIZE
        if max_tokens_per_batch is None:
            max_tokens_per_batch = cls.VLLM_MAX_TOKENS_PER_BATCH
        sample_size = min(50, len(reviews))
        sample_reviews = reviews[:sample_size]
        total_chars = sum(len(r) for r in sample_reviews)
        avg_chars_per_review = total_chars / sample_size if sample_size > 0 else 100
        avg_tokens_per_review = max(1, int(avg_chars_per_review / 3.5))
        calculated = max(1, int(max_tokens_per_batch / avg_tokens_per_review))
        return max(cls.VLLM_MIN_BATCH_SIZE, min(calculated, cls.VLLM_MAX_BATCH_SIZE))

    @classmethod
    def validate(cls) -> bool:
        """설정 검증"""
        return True
