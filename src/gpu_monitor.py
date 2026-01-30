"""
서버 측 GPU 사용량 실시간 모니터링 모듈
백그라운드 태스크로 주기적 샘플링 후 logs/gpu_usage.log 에 기록
X-Enable-GPU-Monitor 요청 시 벤치마크용으로 활성화
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

from .config import Config

logger = logging.getLogger(__name__)


class ServerGPUMonitor:
    """서버 측 GPU 사용량 실시간 추적 클래스 (로그 파일 기록)"""

    def __init__(
        self,
        device_index: int = 0,
        interval: float = 1.0,
        log_to_file: bool = True,
        log_dir: str = "logs",
    ):
        """
        Args:
            device_index: GPU 디바이스 인덱스
            interval: 샘플링 간격 (초)
            log_to_file: 파일 로그 활성화 여부
            log_dir: 로그 디렉토리
        """
        if not PYNVML_AVAILABLE:
            raise ImportError("pynvml이 설치되지 않았습니다. pip install pynvml")

        self.device_index = device_index
        self.interval = interval
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._device = None
        self._initialized = False

        try:
            pynvml.nvmlInit()
            self._device = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._initialized = True
        except Exception as e:
            logger.warning(f"GPU 모니터 NVML 초기화 실패: {e}")
            raise

        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.gpu_log_file = log_path / "gpu_usage.log"

            self.gpu_logger = logging.getLogger("server_gpu_monitor")
            self.gpu_logger.setLevel(logging.INFO)
            self.gpu_logger.handlers = []

            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.gpu_log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.gpu_logger.addHandler(handler)
            self.gpu_logger.propagate = False
            logger.info(f"서버 GPU 모니터 로그 파일: {self.gpu_log_file}")

    def _sample(self) -> Optional[dict]:
        """1회 샘플링 (동기)."""
        if not self._initialized or not self._device:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._device)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._device)
            mem_total = mem_info.total
            mem_used = mem_info.used
            mem_util = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
            return {
                "timestamp": datetime.now().isoformat(),
                "device_index": self.device_index,
                "gpu_util_percent": util.gpu,
                "memory_util_percent": round(mem_util, 2),
                "memory_used_mb": round(mem_used / (1024 ** 2), 2),
                "memory_total_mb": round(mem_total / (1024 ** 2), 2),
            }
        except Exception as e:
            logger.debug(f"GPU 샘플링 실패: {e}")
            return None

    async def _monitor_loop(self):
        """백그라운드 샘플링 루프"""
        logger.info(f"서버 GPU 모니터 시작: device={self.device_index}, 간격 {self.interval}초")

        while not self._stop_event.is_set():
            sample = self._sample()
            if sample and self.log_to_file and self.gpu_logger:
                self.gpu_logger.info(json.dumps(sample, ensure_ascii=False))

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                break
            except asyncio.TimeoutError:
                pass

        logger.info("서버 GPU 모니터 종료")

    def start(self):
        """백그라운드 태스크 시작"""
        if self._task is not None and not self._task.done():
            logger.warning("서버 GPU 모니터가 이미 실행 중입니다.")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("서버 GPU 모니터 백그라운드 태스크 생성 완료")

    async def stop(self):
        """백그라운드 태스크 종료 및 NVML 정리"""
        if self._task is not None and not self._task.done():
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        if self._initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
            except Exception as e:
                logger.debug(f"NVML shutdown: {e}")
        logger.info("서버 GPU 모니터 종료 완료")


_benchmark_gpu_monitor_instance: Optional[ServerGPUMonitor] = None


def get_or_create_benchmark_gpu_monitor() -> Optional[ServerGPUMonitor]:
    """
    X-Enable-GPU-Monitor 요청 시 서버 GPU 모니터 반환 및 시작.
    test_all_task.py --benchmark-gpu 시 서버에서 GPU 로그 수집 (logs/gpu_usage.log).
    """
    global _benchmark_gpu_monitor_instance

    if not PYNVML_AVAILABLE:
        logger.warning("pynvml 없음: 서버 GPU 모니터 비활성화")
        return None

    if _benchmark_gpu_monitor_instance is None:
        try:
            _benchmark_gpu_monitor_instance = ServerGPUMonitor(
                device_index=getattr(Config, "GPU_DEVICE", 0),
                interval=getattr(Config, "CPU_MONITOR_INTERVAL", 1.0),
                log_to_file=True,
                log_dir=Config.METRICS_LOG_DIR,
            )
        except Exception as e:
            logger.warning(f"Benchmark 서버 GPU 모니터 초기화 실패: {e}")
            return None

    _benchmark_gpu_monitor_instance.start()
    return _benchmark_gpu_monitor_instance
