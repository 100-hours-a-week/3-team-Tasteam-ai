"""
CPU 사용량 실시간 모니터링 모듈
백그라운드 태스크로 주기적 샘플링 후 로그 기록
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

from .config import Config

logger = logging.getLogger(__name__)


class CPUMonitor:
    """CPU 사용량 실시간 추적 클래스"""
    
    def __init__(
        self,
        interval: float = 1.0,
        log_to_file: bool = True,
        log_dir: str = "logs",
    ):
        """
        Args:
            interval: 샘플링 간격 (초)
            log_to_file: 파일 로그 활성화 여부
            log_dir: 로그 디렉토리
        """
        if psutil is None:
            raise ImportError("psutil이 설치되지 않았습니다. pip install psutil>=7.0.0")
        
        self.interval = interval
        self.log_to_file = log_to_file
        self.log_dir = log_dir
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._process = psutil.Process(os.getpid())
        
        # 파일 로거 설정
        if log_to_file:
            from pathlib import Path
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.cpu_log_file = log_path / "cpu_usage.log"
            
            # CPU 전용 로거 생성
            self.cpu_logger = logging.getLogger("cpu_monitor")
            self.cpu_logger.setLevel(logging.INFO)
            self.cpu_logger.handlers = []  # 기존 핸들러 제거
            
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.cpu_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')  # 타임스탬프는 JSON에 포함
            handler.setFormatter(formatter)
            self.cpu_logger.addHandler(handler)
            self.cpu_logger.propagate = False
            
            logger.info(f"CPU 모니터 로그 파일: {self.cpu_log_file}")
    
    async def _monitor_loop(self):
        """백그라운드 샘플링 루프"""
        logger.info(f"CPU 모니터 시작: 샘플링 간격 {self.interval}초")
        
        while not self._stop_event.is_set():
            try:
                # 시스템 전체 CPU
                system_cpu = psutil.cpu_percent(interval=None)
                
                # 프로세스 CPU (현재 프로세스만)
                process_cpu = self._process.cpu_percent(interval=None)
                
                # 메모리 (선택)
                mem_info = psutil.virtual_memory()
                process_mem_mb = self._process.memory_info().rss / (1024 * 1024)
                
                # 로그 데이터
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "system_cpu_percent": round(system_cpu, 2),
                    "process_cpu_percent": round(process_cpu, 2),
                    "system_mem_percent": round(mem_info.percent, 2),
                    "process_mem_mb": round(process_mem_mb, 2),
                }
                
                # 파일 로그
                if self.log_to_file and self.cpu_logger:
                    import json
                    self.cpu_logger.info(json.dumps(log_data, ensure_ascii=False))
                
                # 콘솔 로그 (간헐적, 10초마다 한 번만)
                if int(time.time()) % 10 == 0:
                    logger.debug(
                        f"CPU: system={system_cpu:.1f}%, process={process_cpu:.1f}%, "
                        f"mem={mem_info.percent:.1f}%"
                    )
                
            except Exception as e:
                logger.error(f"CPU 샘플링 중 오류: {e}")
            
            # interval 대기 (stop_event로 조기 종료 가능)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval)
                break  # stop_event가 set되면 종료
            except asyncio.TimeoutError:
                pass  # 타임아웃은 정상 (다음 샘플링으로)
        
        logger.info("CPU 모니터 종료")
    
    def start(self):
        """백그라운드 태스크 시작"""
        if self._task is not None and not self._task.done():
            logger.warning("CPU 모니터가 이미 실행 중입니다.")
            return
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("CPU 모니터 백그라운드 태스크 생성 완료")
    
    async def stop(self):
        """백그라운드 태스크 종료"""
        if self._task is None or self._task.done():
            return
        
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("CPU 모니터 종료 타임아웃, 태스크 취소")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CPU 모니터 종료 완료")


# 전역 인스턴스 (main.py에서 사용)
_cpu_monitor_instance: Optional[CPUMonitor] = None
# X-Benchmark 요청 시 사용 (Config.CPU_MONITOR_ENABLE 무관)
_benchmark_cpu_monitor_instance: Optional[CPUMonitor] = None


def get_cpu_monitor() -> Optional[CPUMonitor]:
    """CPU 모니터 인스턴스 반환 (싱글톤, Config.CPU_MONITOR_ENABLE=true일 때만)"""
    global _cpu_monitor_instance

    if not Config.CPU_MONITOR_ENABLE:
        return None

    if _cpu_monitor_instance is None:
        try:
            _cpu_monitor_instance = CPUMonitor(
                interval=Config.CPU_MONITOR_INTERVAL,
                log_to_file=True,
                log_dir=Config.METRICS_LOG_DIR,
            )
        except Exception as e:
            logger.warning(f"CPU 모니터 초기화 실패: {e}")
            return None

    return _cpu_monitor_instance


def get_or_create_benchmark_cpu_monitor() -> Optional[CPUMonitor]:
    """
    X-Benchmark 요청 시 CPU 모니터 반환 및 시작 (Config.CPU_MONITOR_ENABLE 무관).
    test_all_task.py --benchmark 시 서버에서 CPU 모니터링 활성화용.
    """
    global _benchmark_cpu_monitor_instance

    if _benchmark_cpu_monitor_instance is None:
        try:
            _benchmark_cpu_monitor_instance = CPUMonitor(
                interval=Config.CPU_MONITOR_INTERVAL,
                log_to_file=True,
                log_dir=Config.METRICS_LOG_DIR,
            )
        except Exception as e:
            logger.warning(f"Benchmark CPU 모니터 초기화 실패: {e}")
            return None

    _benchmark_cpu_monitor_instance.start()  # idempotent
    return _benchmark_cpu_monitor_instance
