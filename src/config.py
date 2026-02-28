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
# RunPod vLLM: GET /v1/models 의 id와 일치해야 함 (경로가 id로 노출됨)
DEFAULT_LLM_MODEL = "/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_RETRIES = 3
DEFAULT_COLLECTION_NAME = "reviews_collection"
DEFAULT_LLM_BATCH_SIZE = 10


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
    # RunPod/vLLM 컨텍스트 한도 (max_tokens + input <= 이 값). 4096 = Qwen2.5-7B 기본
    LLM_MAX_CONTEXT_LENGTH: int = int(os.getenv("LLM_MAX_CONTEXT_LENGTH", "4096"))
    # 결과 버전 관리: A/B 비교·재실행 안전용 (restaurant_id + analysis_type + model_version + prompt_version + created_at)
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # device, GPU, quantization
    USE_GPU: bool = (os.getenv("USE_GPU") or os.getenv("USE_GPU#", "true")).lower() == "true"
    GPU_DEVICE: int = int(os.getenv("GPU_DEVICE", "0"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"

    # retry, timeout
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    # batch, concurrency (성능 튜닝)
    LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", str(DEFAULT_LLM_BATCH_SIZE)))
    BATCH_SEARCH_CONCURRENCY: int = int(os.getenv("BATCH_SEARCH_CONCURRENCY", "50"))
    BATCH_LLM_CONCURRENCY: int = int(os.getenv("BATCH_LLM_CONCURRENCY", "8"))
    VLLM_MAX_TOKENS_PER_BATCH: int = int(os.getenv("VLLM_MAX_TOKENS_PER_BATCH", "4000"))
    VLLM_MIN_BATCH_SIZE: int = int(os.getenv("VLLM_MIN_BATCH_SIZE", "10"))
    VLLM_MAX_BATCH_SIZE: int = int(os.getenv("VLLM_MAX_BATCH_SIZE", "100"))
    VLLM_DEFAULT_BATCH_SIZE: int = int(os.getenv("VLLM_DEFAULT_BATCH_SIZE", "50"))

    # credentials (비밀은 .env)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ENABLE_OPENAI_FALLBACK: bool = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() == "true"
    # OpenAI 429 시 vLLM(Qwen2.5-7B 등)으로 폴백. true면 429 발생 시 vLLM 단일 요청 생성 시도(지연 초기화).
    ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT: bool = os.getenv("ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT", "false").lower() == "true"
    # vLLM(폴백·1차) 사용 시 RunPod Serverless GPU 사용. true면 요청 시 GPU 기동·유휴 시 스케일다운(비용 절감). 대상: RunPod.
    VLLM_USE_RUNPOD_GPU: bool = os.getenv("VLLM_USE_RUNPOD_GPU", "false").lower() == "true"
    # RunPod vLLM 전용 엔드포인트 ID. 미설정 시 RUNPOD_ENDPOINT_ID 사용.
    RUNPOD_VLLM_ENDPOINT_ID: Optional[str] = (os.getenv("RUNPOD_VLLM_ENDPOINT_ID", "2mpd5y6lvccfk1") or "").strip() or None
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    RUNPOD_ENDPOINT_ID: str = (os.getenv("RUNPOD_ENDPOINT_ID", "2mpd5y6lvccfk1") or "").strip() or "2mpd5y6lvccfk1"
    USE_RUNPOD: bool = os.getenv("USE_RUNPOD", "true").lower() == "true"
    RUNPOD_POLL_INTERVAL: int = int(os.getenv("RUNPOD_POLL_INTERVAL", "2"))
    RUNPOD_MAX_WAIT_TIME: int = int(os.getenv("RUNPOD_MAX_WAIT_TIME", "300"))
    # RunPod Serverless vLLM 엔드포인트 사용 (앱 내 인프로세스 vLLM 제거됨)
    USE_POD_VLLM: bool = os.getenv("USE_POD_VLLM", "true").lower() == "true"
    # RunPod Pod 직접 URL (vLLM OpenAI 호환 /v1). 설정 시 Serverless 대신 이 URL로 추론. 기본값: 213.173.108.29:16366 (test_all_task 연동)
    VLLM_POD_BASE_URL: Optional[str] = (os.getenv("VLLM_POD_BASE_URL", "http://213.173.108.70:17517/v1") or "").strip() or None

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
    """검색/검열: Qdrant URL, collection, top_k, rerank_k, aspect seed, 벡터 on_disk. 하이브리드 인자는 Summary·Vector API 공통."""
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", "./qdrant_data")
    QDRANT_VECTORS_ON_DISK: bool = os.getenv("QDRANT_VECTORS_ON_DISK", "false").lower() == "true"
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
    # 임베딩/HF 캐시: None이면 라이브러리 기본(/tmp 등). 설정 시 공유 볼륨 권장 (재시작·동시성 안정)
    EMBEDDING_CACHE_DIR: Optional[str] = os.getenv("EMBEDDING_CACHE_DIR", os.getenv("FASTEMBED_CACHE_PATH"))

    # 하이브리드 검색 (Summary·Vector API 공통. Summary는 이 값을 전달, Vector API는 요청 생략 시 이 값을 기본으로 사용)
    DENSE_PREFETCH_LIMIT: int = int(os.getenv("DENSE_PREFETCH_LIMIT", "200"))
    SPARSE_PREFETCH_LIMIT: int = int(os.getenv("SPARSE_PREFETCH_LIMIT", "300"))
    FALLBACK_MIN_SCORE: float = float(os.getenv("FALLBACK_MIN_SCORE", "0.2"))

    ENABLE_SENTIMENT_SAMPLING: bool = os.getenv("ENABLE_SENTIMENT_SAMPLING", "false").lower() == "true"
    SENTIMENT_RECENT_TOP_K: int = int(os.getenv("SENTIMENT_RECENT_TOP_K", "100"))


# --- Cache (TTL, skip 간격 등) ---
class _CacheConfig:
    """캐시: TTL, skip 최소 간격"""
    SKIP_MIN_INTERVAL_SECONDS: int = int(os.getenv("SKIP_MIN_INTERVAL_SECONDS", "3600"))


# --- Batch / Queue (배치 분리, 작업 큐, DLQ) ---
class _BatchConfig:
    """배치: 큐 사용 여부, RQ, 재시도"""
    # 배치 API 호출 시 큐에 넣고 job_id 반환 (true) vs 동기 실행 (false)
    BATCH_USE_QUEUE: bool = os.getenv("BATCH_USE_QUEUE", "false").lower() == "true"
    RQ_QUEUE_NAME: str = os.getenv("RQ_QUEUE_NAME", "batch")
    # Redis URL (미설정 시 REDIS_HOST:REDIS_PORT/REDIS_DB 조합)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    # 배치 작업 단위 재시도 (RQ 기본 + 실패 시 FailedJobRegistry = DLQ)
    BATCH_JOB_MAX_RETRIES: int = int(os.getenv("BATCH_JOB_MAX_RETRIES", "3"))


# --- Spark (Comparison 전체 평균 데이터 등) ---
class _SparkConfig:
    """Spark/배치: 전체 평균 데이터 경로, 비율. DISABLE_SPARK=true 시 JVM 없이 Kiwi만 사용 (Docker 등)."""
    DISABLE_SPARK: bool = os.getenv("DISABLE_SPARK", "false").lower() == "true"
    # Spark 마이크로서비스 URL. 설정 시 메인 앱/워커는 로컬 Spark 미사용, 해당 서비스로 HTTP 호출.
    SPARK_SERVICE_URL: Optional[str] = os.getenv("SPARK_SERVICE_URL", "").strip() or None
    ALL_AVERAGE_ASPECT_DATA_PATH: Optional[str] = os.getenv("ALL_AVERAGE_ASPECT_DATA_PATH", "data/test_data_sample.json")
    ALL_AVERAGE_SERVICE_RATIO: float = float(os.getenv("ALL_AVERAGE_SERVICE_RATIO", "0.60"))
    ALL_AVERAGE_PRICE_RATIO: float = float(os.getenv("ALL_AVERAGE_PRICE_RATIO", "0.55"))
    # 리뷰 수가 이 값 미만이면 recall_seeds·Comparison 비율 계산 시 Spark 대신 Python(Kiwi) 사용. 약 2000건 이하는 단일 프로세스가 유리. Summary와 Comparison 통일.
    RECALL_SEEDS_SPARK_THRESHOLD: int = int(os.getenv("RECALL_SEEDS_SPARK_THRESHOLD", "2000"))


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


# --- Config: 도메인별로 나누되, 기존 Config.XXX 호환 ---
class Config(_ServerConfig, _InferenceConfig, _RetrievalConfig, _CacheConfig, _BatchConfig, _SparkConfig, _ObservabilityConfig):
    """
    통합 설정. 도메인: Server, Inference, Retrieval, Cache, Spark, Observability.
    기존 Config.USE_GPU, Config.QDRANT_URL 등 그대로 사용 가능.
    """
    # 도메인 네임스페이스 (선택적 접근)
    Server = _ServerConfig
    Inference = _InferenceConfig
    Retrieval = _RetrievalConfig
    Cache = _CacheConfig
    Batch = _BatchConfig
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
