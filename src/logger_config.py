"""
구조화된 로그 설정 모듈
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StructuredLogger:
    """구조화된 로그 기록기"""
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "debug.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """
        Args:
            log_dir: 로그 디렉토리
            log_file: 로그 파일명
            max_bytes: 로그 파일 최대 크기
            backup_count: 백업 파일 개수
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / log_file
        
        # 로거 설정
        self.logger = logging.getLogger("debug")
        self.logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거 (중복 방지)
        self.logger.handlers = []
        
        # 파일 핸들러 (로테이션)
        try:
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            handler.setLevel(logging.DEBUG)
            
            # JSON 포맷터 (메시지만 저장)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
            self.logger.propagate = False
            
            logger.info(f"구조화된 로거 초기화 완료: {self.log_file}")
        except Exception as e:
            logger.error(f"구조화된 로거 초기화 실패: {e}")
            self.logger = None
    
    def log_debug_info(
        self,
        request_id: str,
        restaurant_id: Optional[int],
        analysis_type: str,
        debug_info: Dict[str, Any],
    ):
        """
        디버그 정보 로그 기록
        
        Args:
            request_id: 요청 ID
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입
            debug_info: 디버그 정보 딕셔너리
        """
        if not self.logger:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "restaurant_id": restaurant_id,
                "analysis_type": analysis_type,
                **debug_info
            }
            
            self.logger.debug(json.dumps(log_entry, ensure_ascii=False))
        except Exception as e:
            logger.error(f"로그 기록 실패: {e}")

