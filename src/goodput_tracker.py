"""
Goodput 추적 모듈

SLA를 만족하는 요청의 실제 처리량(Goodput)을 측정
"""

import logging
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class GoodputTracker:
    """Goodput 추적 클래스"""
    
    def __init__(self, ttft_sla_ms: float = 2000, max_history: int = 10000):
        """
        Args:
            ttft_sla_ms: TTFT SLA 임계값 (밀리초, 기본값: 2000ms = 2초)
            max_history: 최대 저장 요청 수 (메모리 관리용)
        """
        self.ttft_sla_ms = ttft_sla_ms
        self.max_history = max_history
        self.requests: deque = deque(maxlen=max_history)
    
    def add_request(
        self,
        ttft_ms: float,
        n_tokens: int,
        processing_time_ms: float,
    ) -> None:
        """
        요청 결과 추가
        
        Args:
            ttft_ms: Time To First Token (밀리초)
            n_tokens: 생성된 토큰 수
            processing_time_ms: 전체 처리 시간 (밀리초)
        """
        meets_sla = ttft_ms < self.ttft_sla_ms
        
        self.requests.append({
            "ttft_ms": ttft_ms,
            "n_tokens": n_tokens,
            "processing_time_ms": processing_time_ms,
            "meets_sla": meets_sla,
        })
    
    def calculate_goodput(self) -> Dict[str, float]:
        """
        Goodput 계산
        
        Returns:
            Goodput 메트릭 딕셔너리:
            - throughput_tps: 전체 처리량 (Tokens Per Second)
            - goodput_tps: SLA 만족 처리량 (Tokens Per Second)
            - sla_compliance_rate: SLA 준수율 (%)
            - total_requests: 전체 요청 수
            - sla_met_requests: SLA 만족 요청 수
            - avg_ttft_ms: 평균 TTFT (밀리초)
        """
        if not self.requests:
            return {
                "throughput_tps": 0.0,
                "goodput_tps": 0.0,
                "sla_compliance_rate": 0.0,
                "total_requests": 0,
                "sla_met_requests": 0,
                "avg_ttft_ms": 0.0,
            }
        
        total_time_s = sum(r["processing_time_ms"] for r in self.requests) / 1000
        total_tokens = sum(r["n_tokens"] for r in self.requests)
        sla_met_tokens = sum(r["n_tokens"] for r in self.requests if r["meets_sla"])
        sla_met_requests = sum(1 for r in self.requests if r["meets_sla"])
        avg_ttft = sum(r["ttft_ms"] for r in self.requests) / len(self.requests)
        
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0
        goodput = sla_met_tokens / total_time_s if total_time_s > 0 else 0
        sla_compliance_rate = (sla_met_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return {
            "throughput_tps": throughput,
            "goodput_tps": goodput,
            "sla_compliance_rate": sla_compliance_rate,
            "total_requests": len(self.requests),
            "sla_met_requests": sla_met_requests,
            "avg_ttft_ms": avg_ttft,
        }
    
    def get_recent_stats(self, n_requests: int = 100) -> Dict[str, float]:
        """
        최근 N개 요청에 대한 통계
        
        Args:
            n_requests: 최근 요청 수
            
        Returns:
            최근 요청 통계
        """
        if not self.requests:
            return self.calculate_goodput()
        
        recent_requests = list(self.requests)[-n_requests:]
        
        total_time_s = sum(r["processing_time_ms"] for r in recent_requests) / 1000
        total_tokens = sum(r["n_tokens"] for r in recent_requests)
        sla_met_tokens = sum(r["n_tokens"] for r in recent_requests if r["meets_sla"])
        sla_met_requests = sum(1 for r in recent_requests if r["meets_sla"])
        avg_ttft = sum(r["ttft_ms"] for r in recent_requests) / len(recent_requests)
        
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0
        goodput = sla_met_tokens / total_time_s if total_time_s > 0 else 0
        sla_compliance_rate = (sla_met_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        return {
            "throughput_tps": throughput,
            "goodput_tps": goodput,
            "sla_compliance_rate": sla_compliance_rate,
            "total_requests": len(recent_requests),
            "sla_met_requests": sla_met_requests,
            "avg_ttft_ms": avg_ttft,
        }
    
    def reset(self):
        """요청 히스토리 초기화"""
        self.requests.clear()
        logger.info("Goodput 추적기 초기화 완료")
