#!/usr/bin/env python3
"""
GPU 모니터링 모듈

vLLM 성능 측정을 위한 GPU 메트릭 수집
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml이 설치되지 않았습니다. GPU 메트릭 수집이 비활성화됩니다.")


class GPUMonitor:
    """GPU 메트릭 수집 클래스"""
    
    def __init__(self, device_index: int = 0):
        """
        Args:
            device_index: GPU 디바이스 인덱스 (기본값: 0)
        """
        self.device_index = device_index
        self.device = None
        self.initialized = False
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.initialized = True
                logger.info(f"GPU 모니터 초기화 완료 (디바이스 {device_index})")
            except Exception as e:
                logger.error(f"GPU 모니터 초기화 실패: {e}")
                self.initialized = False
        else:
            logger.warning("pynvml이 없어 GPU 모니터링을 사용할 수 없습니다.")
    
    def get_metrics(self) -> Optional[Dict[str, float]]:
        """
        GPU 메트릭 수집
        
        Returns:
            GPU 메트릭 딕셔너리 또는 None (실패 시)
            - gpu_util_percent: GPU 사용률 (%)
            - memory_util_percent: 메모리 사용률 (%)
            - memory_used_mb: 사용 중인 메모리 (MB)
            - memory_total_mb: 전체 메모리 (MB)
        """
        if not self.initialized or not self.device:
            return None
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
            
            return {
                "gpu_util_percent": util.gpu,
                "memory_util_percent": (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0,
                "memory_used_mb": mem_info.used / (1024 ** 2),
                "memory_total_mb": mem_info.total / (1024 ** 2),
                "memory_free_mb": mem_info.free / (1024 ** 2),
            }
        except Exception as e:
            logger.error(f"GPU 메트릭 수집 실패: {e}")
            return None
    
    def cleanup(self):
        """리소스 정리"""
        if self.initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except Exception as e:
                logger.error(f"GPU 모니터 정리 실패: {e}")


# 전역 인스턴스 (선택적)
_global_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(device_index: int = 0) -> GPUMonitor:
    """
    전역 GPU 모니터 인스턴스 반환
    
    Args:
        device_index: GPU 디바이스 인덱스
        
    Returns:
        GPUMonitor 인스턴스
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMonitor(device_index=device_index)
    return _global_monitor
