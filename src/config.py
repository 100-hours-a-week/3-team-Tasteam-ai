"""
설정 관리 모듈
"""

import os
from typing import Optional, List

try:
    import torch
except ImportError:
    torch = None

# 기본 설정값
DEFAULT_SENTIMENT_MODEL = "Dilwolf/Kakao_app-kr_sentiment"
# final_summary_pipeline과 동일: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768 dim)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_EMBEDDING_DIM = 768  # paraphrase-multilingual-mpnet-base-v2 출력 차원
# Sparse = BM25 등 (vector_search에서 사용)
DEFAULT_SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_MAX_RETRIES = 3
DEFAULT_COLLECTION_NAME = "reviews_collection"
DEFAULT_LLM_BATCH_SIZE = 10  # LLM 분류 배치 크기

# LLM 재분류 키워드
DEFAULT_LLM_KEYWORDS = ["는데", "지만"]


class Config:
    """애플리케이션 설정 클래스"""
    
    # 모델 설정 (환경 변수 지원)
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", DEFAULT_SENTIMENT_MODEL)
    # Sentiment 샘플링 설정
    ENABLE_SENTIMENT_SAMPLING: bool = os.getenv("ENABLE_SENTIMENT_SAMPLING", "false").lower() == "true"  # 샘플링 활성화 여부
    SENTIMENT_RECENT_TOP_K: int = int(os.getenv("SENTIMENT_RECENT_TOP_K", "100"))  # 샘플링 시 사용할 최근 리뷰 수 (기본값: 100)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))
    SPARSE_EMBEDDING_MODEL: str = os.getenv("SPARSE_EMBEDDING_MODEL", DEFAULT_SPARSE_EMBEDDING_MODEL)
    LLM_MODEL: str = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    
    # 분석 설정
    SCORE_THRESHOLD: float = DEFAULT_SCORE_THRESHOLD
    MAX_RETRIES: int = DEFAULT_MAX_RETRIES
    LLM_KEYWORDS: list = DEFAULT_LLM_KEYWORDS
    LLM_BATCH_SIZE: int = DEFAULT_LLM_BATCH_SIZE
    
    # Qdrant 설정
    COLLECTION_NAME: str = DEFAULT_COLLECTION_NAME
    # QDRANT_URL: ":memory:" (메모리), "./qdrant_db" (on-disk), "http://localhost:6333" (원격 서버)
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL", ":memory:")
    
    # GPU 설정
    USE_GPU: bool = os.getenv("USE_GPU#", "true").lower() == "true"
    GPU_DEVICE: int = int(os.getenv("GPU_DEVICE", "0"))
    USE_FP16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    
    # LLM 제공자 선택 (local, runpod, openai)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()  # 기본값: runpod
    
    # OpenAI 설정 (빠른 검증용)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 빠른 검증용 모델
    ENABLE_OPENAI_FALLBACK: bool = os.getenv("ENABLE_OPENAI_FALLBACK", "false").lower() == "true"  # 로컬 큐 오버플로우 시 OpenAI API 폴백 활성화
    
    # RunPod 서버리스 엔드포인트 설정
    RUNPOD_API_KEY: Optional[str] = os.getenv("RUNPOD_API_KEY")
    RUNPOD_ENDPOINT_ID: str = os.getenv("RUNPOD_ENDPOINT_ID", "g09uegksn7h7ed")
    USE_RUNPOD: bool = os.getenv("USE_RUNPOD", "true").lower() == "true"  # RunPod 사용 여부 (하위 호환성)
    RUNPOD_POLL_INTERVAL: int = int(os.getenv("RUNPOD_POLL_INTERVAL", "2"))  # 상태 확인 간격 (초)
    RUNPOD_MAX_WAIT_TIME: int = int(os.getenv("RUNPOD_MAX_WAIT_TIME", "300"))  # 최대 대기 시간 (초)
    
    # vLLM 직접 사용 설정 (RunPod Pod 환경)
    USE_POD_VLLM: bool = os.getenv("USE_POD_VLLM", "false").lower() == "true"  # RunPod Pod에서 vLLM 직접 사용
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))  # 텐서 병렬 크기
    VLLM_MAX_MODEL_LEN: Optional[int] = int(os.getenv("VLLM_MAX_MODEL_LEN")) if os.getenv("VLLM_MAX_MODEL_LEN") else None  # 최대 모델 길이
    
    # Watchdog 설정 (외부 모니터링)
    RUNPOD_POD_ID: Optional[str] = os.getenv("RUNPOD_POD_ID")  # Pod ID (Watchdog에서 사용)
    IDLE_THRESHOLD: int = int(os.getenv("IDLE_THRESHOLD", "5"))  # GPU 사용률 임계값 (%)
    CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL", "60"))  # 체크 간격 (초)
    IDLE_LIMIT: int = int(os.getenv("IDLE_LIMIT", "5"))  # 연속 idle 횟수
    MIN_RUNTIME: int = int(os.getenv("MIN_RUNTIME", "600"))  # 최소 실행 시간 (초)
    
    # 동적 배치 크기 설정 (여러 레스토랑 처리용)
    VLLM_MAX_TOKENS_PER_BATCH: int = int(os.getenv("VLLM_MAX_TOKENS_PER_BATCH", "4000"))  # 배치당 최대 토큰 수
    VLLM_MIN_BATCH_SIZE: int = int(os.getenv("VLLM_MIN_BATCH_SIZE", "10"))  # 최소 배치 크기
    VLLM_MAX_BATCH_SIZE: int = int(os.getenv("VLLM_MAX_BATCH_SIZE", "100"))  # 최대 배치 크기
    VLLM_DEFAULT_BATCH_SIZE: int = int(os.getenv("VLLM_DEFAULT_BATCH_SIZE", "50"))  # 기본 배치 크기
    VLLM_MAX_CONCURRENT_BATCHES: int = int(os.getenv("VLLM_MAX_CONCURRENT_BATCHES", "20"))  # 최대 동시 처리 배치 수 (OOM 방지)
    
    # 우선순위 큐 설정 (Prefill 비용 기반)
    VLLM_USE_PRIORITY_QUEUE: bool = os.getenv("VLLM_USE_PRIORITY_QUEUE", "true").lower() == "true"  # 우선순위 큐 사용 여부
    VLLM_PRIORITY_BY_PREFILL_COST: bool = os.getenv("VLLM_PRIORITY_BY_PREFILL_COST", "true").lower() == "true"  # Prefill 비용 기반 우선순위
    
    # 메트릭/로그 수집 설정 (상시 수집 비활성화, config로 on/off)
    METRICS_AND_LOGGING_ENABLE: bool = os.getenv("METRICS_AND_LOGGING_ENABLE", "false").lower() == "true"  # True일 때만 수집
    METRICS_ENABLE_LOGGING: bool = os.getenv("METRICS_ENABLE_LOGGING", "true").lower() == "true"  # 수집 활성화 시 로그 파일 저장
    METRICS_ENABLE_DB: bool = os.getenv("METRICS_ENABLE_DB", "true").lower() == "true"  # 수집 활성화 시 SQLite 저장
    METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", "metrics.db")
    METRICS_LOG_DIR: str = os.getenv("METRICS_LOG_DIR", "logs")
    
    # CPU 모니터링 설정 (실시간 곡선)
    CPU_MONITOR_ENABLE: bool = os.getenv("CPU_MONITOR_ENABLE", "false").lower() == "true"  # CPU 실시간 추적 on/off
    CPU_MONITOR_INTERVAL: float = float(os.getenv("CPU_MONITOR_INTERVAL", "1.0"))  # 샘플링 간격 (초)
    
    # LLM 개선 타임아웃 설정
    LLM_POLISH_TIMEOUT_SECONDS: float = float(os.getenv("LLM_POLISH_TIMEOUT_SECONDS", "2.0"))  # LLM 개선 최대 대기 시간 (초)
    
    # SKIP 로직 설정 (초기 전략: analysis_metrics 기반)
    SKIP_MIN_INTERVAL_SECONDS: int = int(os.getenv("SKIP_MIN_INTERVAL_SECONDS", "3600"))  # 최소 간격 (초, 기본값: 1시간)
    
    # Comparison - 전체 데이터셋 평균 비율 (배치 작업 결과로 교체 필요)
    ALL_AVERAGE_SERVICE_RATIO: float = float(os.getenv("ALL_AVERAGE_SERVICE_RATIO", "0.60"))  # 전체 평균 서비스 긍정 비율
    ALL_AVERAGE_PRICE_RATIO: float = float(os.getenv("ALL_AVERAGE_PRICE_RATIO", "0.55"))  # 전체 평균 가격 긍정 비율
    # strength_in_aspect와 동일한 '전체' 사용: aspect_data 파일(TSV의 Review 컬럼, JSON의 content)에서 계산
    ALL_AVERAGE_ASPECT_DATA_PATH: Optional[str] = os.getenv("ALL_AVERAGE_ASPECT_DATA_PATH","data/test_data_sample.json")  # 예: data/kr3.tsv
    
    # Aspect Seed 파일 경로 (선택적)
    ASPECT_SEEDS_FILE: Optional[str] = os.getenv("ASPECT_SEEDS_FILE")  # Aspect seed JSON 파일 경로
    
    # 배치 요약: search_async=aspect 3개 병렬, restaurant_async=음식점 간 병렬
    BATCH_SEARCH_ASYNC: bool = os.getenv("BATCH_SEARCH_ASYNC", "false").lower() == "true"  # aspect(service/price/food) 서치 병렬
    BATCH_RESTAURANT_ASYNC: bool = os.getenv("BATCH_RESTAURANT_ASYNC", "false").lower() == "true"  # 음식점 간 병렬
    BATCH_SEARCH_CONCURRENCY: int = int(os.getenv("BATCH_SEARCH_CONCURRENCY", "50"))  # 검색 동시성 상한
    BATCH_LLM_CONCURRENCY: int = int(os.getenv("BATCH_LLM_CONCURRENCY", "8"))  # LLM 동시성 상한
    # LLM 비동기 호출: True면 배치 경로에서 httpx.AsyncClient(진짜 비동기), False면 to_thread(동기 래핑)
    LLM_ASYNC: bool = os.getenv("LLM_ASYNC", "false").lower() == "true"  # llm_async 방식 on/off
    
    @classmethod
    def get_device(cls):
        """GPU 사용 가능 여부 확인"""
        if torch is None:
            return -1
        if cls.USE_GPU and torch.cuda.is_available():
            return cls.GPU_DEVICE
        return -1
    
    @classmethod
    def get_dtype(cls):
        """양자화 타입 반환"""
        if torch is None:
            return None
        if cls.USE_FP16 and torch.cuda.is_available():
            return torch.float16
        return torch.float32
    
    @classmethod
    def get_optimal_batch_size(cls, model_type: str = "default"):
        """GPU 메모리에 따른 최적 배치 크기 계산"""
        if torch is None or not cls.USE_GPU or not torch.cuda.is_available():
            return 32
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            return 32
        
        if model_type == "llm":
            # LLM은 메모리 사용량이 큼 (7B 모델 기준)
            if gpu_memory_gb >= 40:  # A100
                return 20
            elif gpu_memory_gb >= 24:  # RTX 3090
                return 10
            else:
                return 5
        elif model_type == "sentiment":
            # 감성 분석 모델 (작은 모델)
            if gpu_memory_gb >= 40:
                return 128
            elif gpu_memory_gb >= 24:
                return 64
            else:
                return 32
        else:  # embedding
            # 임베딩 모델
            if gpu_memory_gb >= 40:
                return 128
            elif gpu_memory_gb >= 24:
                return 64
            else:
                return 32
    
    @classmethod
    def calculate_dynamic_batch_size(cls, reviews: List[str], max_tokens_per_batch: Optional[int] = None) -> int:
        """
        리뷰 리스트를 기반으로 동적 배치 크기 계산
        
        리뷰당 평균 토큰 수를 추정하여 max_tokens_per_batch 제한 내에서
        최적의 배치 크기를 계산합니다.
        
        Args:
            reviews: 리뷰 문자열 리스트
            max_tokens_per_batch: 배치당 최대 토큰 수 (None이면 Config 값 사용)
            
        Returns:
            계산된 배치 크기 (MIN_BATCH_SIZE ~ MAX_BATCH_SIZE 범위)
        """
        if not reviews:
            return cls.VLLM_DEFAULT_BATCH_SIZE
        
        if max_tokens_per_batch is None:
            max_tokens_per_batch = cls.VLLM_MAX_TOKENS_PER_BATCH
        
        # 샘플링하여 평균 토큰 수 추정 (실제로는 토크나이저 사용 권장, 여기서는 간단한 추정)
        sample_size = min(50, len(reviews))  # 최대 50개 샘플
        sample_reviews = reviews[:sample_size]
        
        # 문자 수 기반 추정 (대략적인 방법)
        # 한국어의 경우: 1 토큰 ≈ 3-4 문자 (Qwen 토크나이저 기준)
        total_chars = sum(len(review) for review in sample_reviews)
        avg_chars_per_review = total_chars / sample_size if sample_size > 0 else 100
        
        # 평균 토큰 수 추정 (한국어 기준 약 3.5 문자/토큰)
        avg_tokens_per_review = max(1, int(avg_chars_per_review / 3.5))
        
        # 배치 크기 계산
        calculated_batch_size = max(1, int(max_tokens_per_batch / avg_tokens_per_review))
        
        # 최소/최대 제한 적용
        batch_size = max(cls.VLLM_MIN_BATCH_SIZE, min(calculated_batch_size, cls.VLLM_MAX_BATCH_SIZE))
        
        return batch_size
    
    @classmethod
    def validate(cls) -> bool:
        """설정값 검증"""
        # OpenAI API 키 검증 제거 (Qwen 모델 사용)
        return True

