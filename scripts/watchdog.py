#!/usr/bin/env python3
"""
RunPod Pod 모니터링 및 자동 종료 Watchdog

외부에서 실행하여 RunPod Pod의 GPU 사용률을 모니터링하고,
일정 시간 idle 상태가 지속되면 Pod를 자동으로 종료합니다.
"""

import os
import sys
import time
import logging
import requests
from typing import Optional

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from scripts.gpu_monitor import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RunPodWatchdog:
    """RunPod Pod 모니터링 및 자동 종료"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        pod_id: Optional[str] = None,
        idle_threshold: Optional[int] = None,
        check_interval: Optional[int] = None,
        idle_limit: Optional[int] = None,
        min_runtime: Optional[int] = None,
    ):
        """
        Args:
            api_key: RunPod API 키 (None이면 Config에서 가져옴)
            pod_id: Pod ID (None이면 Config에서 가져옴)
            idle_threshold: GPU 사용률 임계값 (%) (None이면 Config에서 가져옴)
            check_interval: 체크 간격 (초) (None이면 Config에서 가져옴)
            idle_limit: 연속 idle 횟수 (None이면 Config에서 가져옴)
            min_runtime: 최소 실행 시간 (초) (None이면 Config에서 가져옴)
        """
        self.api_key = api_key or Config.RUNPOD_API_KEY
        self.pod_id = pod_id or Config.RUNPOD_POD_ID
        self.idle_threshold = idle_threshold if idle_threshold is not None else Config.IDLE_THRESHOLD
        self.check_interval = check_interval if check_interval is not None else Config.CHECK_INTERVAL
        self.idle_limit = idle_limit if idle_limit is not None else Config.IDLE_LIMIT
        self.min_runtime = min_runtime if min_runtime is not None else Config.MIN_RUNTIME
        self.idle_count = 0
        self.start_time = time.time()
        
        # GPUMonitor 초기화
        self.gpu_monitor = GPUMonitor(device_index=0)
        
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY가 설정되지 않았습니다.")
        if not self.pod_id:
            raise ValueError("RUNPOD_POD_ID가 설정되지 않았습니다.")
    
    def get_gpu_usage(self) -> Optional[float]:
        """GPU 사용률 조회 (GPUMonitor 사용)"""
        try:
            metrics = self.gpu_monitor.get_metrics()
            if metrics:
                return metrics.get("gpu_util_percent")
        except Exception as e:
            logger.error(f"GPU 사용률 조회 실패: {e}")
        return None
    
    def get_pod_status(self) -> Optional[dict]:
        """
        RunPod Pod 상태 조회
        
        API 엔드포인트: GET https://api.runpod.ai/v1/{pod_id}
        실제 API 형식은 RunPod 공식 문서를 참조하세요.
        
        Returns:
            Pod 상태 정보 딕셔너리 (성공 시), None (실패 시)
        """
        url = f"https://api.runpod.ai/v1/{self.pod_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Pod 상태 조회 실패: HTTP {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Pod 상태 조회 중 네트워크 오류: {e}")
        except Exception as e:
            logger.error(f"Pod 상태 조회 실패: {e}")
        return None
    
    def stop_pod(self) -> bool:
        """
        RunPod Pod 종료
        
        API 엔드포인트: POST https://api.runpod.ai/v1/stop/{pod_id}
        실제 API 형식은 RunPod 공식 문서를 참조하세요.
        
        Returns:
            종료 성공 시 True, 실패 시 False
        """
        url = f"https://api.runpod.ai/v1/stop/{self.pod_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.post(url, headers=headers, timeout=30)
            if response.status_code == 200:
                logger.info(f"Pod {self.pod_id} 종료 성공")
                return True
            else:
                logger.error(f"Pod 종료 실패: HTTP {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Pod 종료 중 네트워크 오류: {e}")
        except Exception as e:
            logger.error(f"Pod 종료 중 오류: {e}")
        return False
    
    def monitor_loop(self):
        """모니터링 루프"""
        logger.info(
            f"Watchdog 시작: Pod {self.pod_id}, "
            f"임계값 {self.idle_threshold}%, "
            f"체크 간격 {self.check_interval}초, "
            f"최소 실행 시간 {self.min_runtime}초"
        )
        
        # Pod 상태 확인 주기 (10회마다 한 번씩 확인)
        pod_status_check_counter = 0
        pod_status_check_interval = 10
        
        while True:
            try:
                # 최소 실행 시간 확인
                runtime = time.time() - self.start_time
                if runtime < self.min_runtime:
                    logger.debug(f"최소 실행 시간 미달 ({int(runtime)}/{self.min_runtime}초), 종료하지 않음")
                    time.sleep(self.check_interval)
                    continue
                
                # Pod 상태 주기적 확인 (이미 종료된 경우 루프 종료)
                pod_status_check_counter += 1
                if pod_status_check_counter >= pod_status_check_interval:
                    pod_status = self.get_pod_status()
                    if pod_status:
                        # Pod 상태 정보에서 종료 상태 확인 (실제 필드명은 API 문서 참조)
                        # 예: pod_status.get("desiredStatus") == "TERMINATED" 등
                        logger.debug(f"Pod 상태 확인: {pod_status}")
                    else:
                        logger.warning("Pod 상태 조회 실패, 계속 모니터링")
                    pod_status_check_counter = 0
                
                # GPU 사용률 확인
                gpu_usage = self.get_gpu_usage()
                
                if gpu_usage is None:
                    logger.warning("GPU 사용률 조회 실패, 다음 체크까지 대기")
                    time.sleep(self.check_interval)
                    continue
                
                logger.info(f"GPU 사용률: {gpu_usage}%")
                
                # Idle 판단 (GPU 사용률만으로 판단)
                if gpu_usage < self.idle_threshold:
                    self.idle_count += 1
                    logger.info(f"Idle 상태 감지 ({self.idle_count}/{self.idle_limit})")
                    
                    if self.idle_count >= self.idle_limit:
                        logger.info("연속 idle 시간 초과, Pod 종료")
                        if self.stop_pod():
                            break  # 종료 성공 시 루프 종료
                        else:
                            # 종료 실패 시 계속 모니터링
                            self.idle_count = 0
                else:
                    # 사용 중이면 idle 카운터 리셋
                    if self.idle_count > 0:
                        logger.info("활성 상태 감지, idle 카운터 리셋")
                    self.idle_count = 0
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Watchdog 중지")
                break
            except Exception as e:
                logger.error(f"모니터링 중 오류: {e}")
                time.sleep(self.check_interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod Pod 모니터링 및 자동 종료 Watchdog")
    parser.add_argument("--api-key", type=str, help="RunPod API 키 (환경변수 우선)")
    parser.add_argument("--pod-id", type=str, help="Pod ID (환경변수 우선)")
    parser.add_argument("--idle-threshold", type=int, help="GPU 사용률 임계값 (%)")
    parser.add_argument("--check-interval", type=int, help="체크 간격 (초)")
    parser.add_argument("--idle-limit", type=int, help="연속 idle 횟수")
    parser.add_argument("--min-runtime", type=int, help="최소 실행 시간 (초)")
    
    args = parser.parse_args()
    
    try:
        watchdog = RunPodWatchdog(
            api_key=args.api_key,
            pod_id=args.pod_id,
            idle_threshold=args.idle_threshold,
            check_interval=args.check_interval,
            idle_limit=args.idle_limit,
            min_runtime=args.min_runtime,
        )
        
        watchdog.monitor_loop()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Watchdog 실행 중 오류: {e}")
        sys.exit(1)

