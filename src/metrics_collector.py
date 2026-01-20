"""
메트릭 수집 및 저장 통합 모듈
"""

import uuid
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .logger_config import StructuredLogger
from .metrics_db import MetricsDB
from .config import Config
from .goodput_tracker import GoodputTracker

logger = logging.getLogger(__name__)


class MetricsCollector:
    """메트릭 수집 및 저장 클래스"""
    
    def __init__(
        self,
        enable_logging: bool = True,
        enable_db: bool = True,
        db_path: str = "metrics.db",
        log_dir: str = "logs",
    ):
        """
        Args:
            enable_logging: 로그 파일 저장 활성화
            enable_db: SQLite 저장 활성화
            db_path: SQLite 데이터베이스 경로
            log_dir: 로그 디렉토리
        """
        self.enable_logging = enable_logging
        self.enable_db = enable_db
        
        # 로그 설정
        if enable_logging:
            try:
                self.structured_logger = StructuredLogger(log_dir=log_dir)
            except Exception as e:
                logger.warning(f"구조화된 로거 초기화 실패: {e}. 로그 저장이 비활성화됩니다.")
                self.structured_logger = None
                self.enable_logging = False
        else:
            self.structured_logger = None
        
        # SQLite 설정
        if enable_db:
            try:
                self.metrics_db = MetricsDB(db_path=db_path)
            except Exception as e:
                logger.warning(f"메트릭 데이터베이스 초기화 실패: {e}. DB 저장이 비활성화됩니다.")
                self.metrics_db = None
                self.enable_db = False
        else:
            self.metrics_db = None
        
        # Goodput 추적기 초기화
        self.goodput_tracker = GoodputTracker(ttft_sla_ms=2000)  # SLA: TTFT < 2초
    
    def collect_metrics(
        self,
        restaurant_id: Optional[int],
        analysis_type: str,
        start_time: float,
        tokens_used: Optional[int] = None,
        batch_size: Optional[int] = None,
        cache_hit: Optional[bool] = None,
        model_version: Optional[str] = None,
        error_count: int = 0,
        warning_count: int = 0,
        additional_info: Optional[Dict[str, Any]] = None,
        ttft_ms: Optional[float] = None,
        status: Optional[str] = None,
    ) -> str:
        """
        메트릭 수집 및 저장
        
        Args:
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            start_time: 시작 시간 (time.time())
            tokens_used: 사용된 토큰 수
            batch_size: 배치 크기
            cache_hit: 캐시 히트 여부
            model_version: 모델 버전
            error_count: 에러 개수
            warning_count: 경고 개수
            additional_info: 추가 정보 (로그에만 저장)
            
        Returns:
            request_id: 요청 ID
        """
        request_id = str(uuid.uuid4())
        processing_time_ms = (time.time() - start_time) * 1000
        
        # 모델 버전 기본값
        if model_version is None:
            model_version = Config.LLM_MODEL
        
        # status 자동 결정 (명시되지 않은 경우)
        if status is None:
            # additional_info에 status가 있으면 사용 (명시적으로 전달된 경우 우선)
            if additional_info and "status" in additional_info:
                status = additional_info["status"]
            # error_count 기반으로 결정
            elif error_count > 0:
                status = "fail"
            else:
                status = "success"
        
        # 1. 로그 파일에 모든 디버그 정보 저장
        if self.enable_logging and self.structured_logger:
            debug_info = {
                "processing_time_ms": processing_time_ms,
                "tokens_used": tokens_used,
                "batch_size": batch_size,
                "cache_hit": cache_hit,
                "model_version": model_version,
                "error_count": error_count,
                "warning_count": warning_count,
                "status": status,
            }
            
            # additional_info를 업데이트하되, status는 이미 결정된 값 사용
            if additional_info:
                # status를 제외한 additional_info만 업데이트
                additional_info_copy = {k: v for k, v in additional_info.items() if k != "status"}
                debug_info.update(additional_info_copy)
            
            self.structured_logger.log_debug_info(
                request_id=request_id,
                restaurant_id=restaurant_id,
                analysis_type=analysis_type,
                debug_info=debug_info,
            )
        
        # 2. SQLite에 중요한 메트릭만 저장
        if self.enable_db and self.metrics_db:
            try:
                self.metrics_db.insert_metric(
                    restaurant_id=restaurant_id,
                    analysis_type=analysis_type,
                    processing_time_ms=processing_time_ms,
                    tokens_used=tokens_used,
                    batch_size=batch_size,
                    cache_hit=cache_hit,
                    model_version=model_version,
                    error_count=error_count,
                    warning_count=warning_count,
                )
            except Exception as e:
                logger.error(f"메트릭 DB 저장 실패: {e}")
        
        # Goodput 추적 (TTFT가 제공된 경우)
        if ttft_ms is not None and tokens_used is not None:
            processing_time_ms = (time.time() - start_time) * 1000
            self.goodput_tracker.add_request(
                ttft_ms=ttft_ms,
                n_tokens=tokens_used,
                processing_time_ms=processing_time_ms,
            )
        
        return request_id
    
    def get_performance_stats(
        self,
        analysis_type: Optional[str] = None,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        성능 통계 조회 (SQLite에서)
        
        Args:
            analysis_type: 분석 타입 필터
            days: 최근 N일 데이터
            
        Returns:
            성능 통계 리스트
        """
        if self.enable_db and self.metrics_db:
            try:
                return self.metrics_db.get_performance_stats(
                    analysis_type=analysis_type,
                    days=days,
                )
            except Exception as e:
                logger.error(f"성능 통계 조회 실패: {e}")
                return []
        return []
    
    def cleanup_old_data(self, days: int = 90):
        """
        오래된 데이터 삭제 (SQLite)
        
        Args:
            days: N일 이전 데이터 삭제
        """
        if self.enable_db and self.metrics_db:
            try:
                self.metrics_db.cleanup_old_data(days=days)
            except Exception as e:
                logger.error(f"오래된 데이터 삭제 실패: {e}")
    
    def collect_vllm_metrics(
        self,
        request_id: str,
        restaurant_id: Optional[int],
        analysis_type: str,
        vllm_metrics: Dict[str, Any],
    ) -> None:
        """
        vLLM 전용 메트릭 수집
        
        Args:
            request_id: 요청 ID
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            vllm_metrics: vLLM 메트릭 딕셔너리 (avg_prefill_time_ms, avg_decode_time_ms, total_tokens, tps 등)
        """
        # 로그 파일에 모든 디버그 정보 저장
        if self.enable_logging and self.structured_logger:
            debug_info = {
                "vllm_metrics": vllm_metrics,
            }
            
            self.structured_logger.log_debug_info(
                request_id=request_id,
                restaurant_id=restaurant_id,
                analysis_type=analysis_type,
                debug_info=debug_info,
            )
        
        # SQLite에 vLLM 메트릭 저장
        if self.enable_db and self.metrics_db:
            try:
                self.metrics_db.insert_vllm_metric(
                    request_id=request_id,
                    restaurant_id=restaurant_id,
                    analysis_type=analysis_type,
                    prefill_time_ms=vllm_metrics.get("avg_prefill_time_ms"),
                    decode_time_ms=vllm_metrics.get("avg_decode_time_ms"),
                    total_time_ms=vllm_metrics.get("total_time_ms"),
                    n_tokens=vllm_metrics.get("total_tokens"),
                    tpot_ms=vllm_metrics.get("avg_tpot_ms"),
                    tps=vllm_metrics.get("tps"),
                    ttft_ms=vllm_metrics.get("ttft_ms"),
                )
            except Exception as e:
                logger.error(f"vLLM 메트릭 DB 저장 실패: {e}")
    
    def get_goodput_stats(self, recent_n: Optional[int] = None) -> Dict[str, float]:
        """
        Goodput 통계 조회
        
        Args:
            recent_n: 최근 N개 요청만 조회 (None이면 전체)
            
        Returns:
            Goodput 통계 딕셔너리
        """
        if recent_n is not None:
            return self.goodput_tracker.get_recent_stats(n_requests=recent_n)
        else:
            return self.goodput_tracker.calculate_goodput()
    
    def close(self):
        """리소스 정리"""
        if self.metrics_db:
            self.metrics_db.close()

