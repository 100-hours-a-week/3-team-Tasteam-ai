"""
중요 메트릭 저장용 SQLite 데이터베이스 모듈
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsDB:
    """중요 메트릭 저장용 SQLite 데이터베이스"""
    
    def __init__(self, db_path: str = "metrics.db"):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화 및 테이블 생성"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            
            # WAL 모드 활성화 (동시성 성능 향상)
            self.conn.execute("PRAGMA journal_mode=WAL")
            
            cursor = self.conn.cursor()
            
            # 분석 메트릭 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    restaurant_id INTEGER,
                    analysis_type TEXT NOT NULL,
                    model_version TEXT,
                    processing_time_ms REAL,
                    tokens_used INTEGER,
                    batch_size INTEGER,
                    cache_hit BOOLEAN,
                    error_count INTEGER DEFAULT 0,
                    warning_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_restaurant_id 
                ON analysis_metrics(restaurant_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_type 
                ON analysis_metrics(analysis_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON analysis_metrics(created_at)
            """)
            
            # vLLM 메트릭 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vllm_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT,
                    restaurant_id INTEGER,
                    analysis_type TEXT,
                    
                    -- Prefill/Decode 분리 지표
                    prefill_time_ms REAL,
                    decode_time_ms REAL,
                    total_time_ms REAL,
                    
                    -- 토큰 관련
                    n_tokens INTEGER,
                    tpot_ms REAL,  -- Time Per Output Token
                    
                    -- 처리량
                    tps REAL,  -- Tokens Per Second
                    ttft_ms REAL,  -- Time To First Token
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # vLLM 메트릭 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_request_id 
                ON vllm_metrics(request_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_restaurant_id 
                ON vllm_metrics(restaurant_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_analysis_type 
                ON vllm_metrics(analysis_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_vllm_created_at 
                ON vllm_metrics(created_at)
            """)
            
            self.conn.commit()
            logger.info(f"메트릭 데이터베이스 초기화 완료: {self.db_path}")
        except Exception as e:
            logger.error(f"메트릭 데이터베이스 초기화 실패: {e}")
            raise
    
    def insert_metric(
        self,
        restaurant_id: Optional[int],
        analysis_type: str,
        processing_time_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        batch_size: Optional[int] = None,
        cache_hit: Optional[bool] = None,
        model_version: Optional[str] = None,
        error_count: int = 0,
        warning_count: int = 0,
    ):
        """
        분석 메트릭 저장
        
        Args:
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            processing_time_ms: 처리 시간 (밀리초)
            tokens_used: 사용된 토큰 수
            batch_size: 배치 크기
            cache_hit: 캐시 히트 여부
            model_version: 모델 버전
            error_count: 에러 개수
            warning_count: 경고 개수
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_metrics (
                    restaurant_id, analysis_type, model_version,
                    processing_time_ms, tokens_used, batch_size,
                    cache_hit, error_count, warning_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                restaurant_id, analysis_type, model_version,
                processing_time_ms, tokens_used, batch_size,
                cache_hit, error_count, warning_count
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"메트릭 저장 실패: {e}")
            self.conn.rollback()
    
    def get_performance_stats(
        self,
        analysis_type: Optional[str] = None,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        성능 통계 조회
        
        Args:
            analysis_type: 분석 타입 필터 (None이면 전체)
            days: 최근 N일 데이터
            
        Returns:
            성능 통계 리스트
        """
        try:
            cursor = self.conn.cursor()
            
            query = """
                SELECT 
                    analysis_type,
                    AVG(processing_time_ms) as avg_processing_time_ms,
                    COUNT(*) as total_requests,
                    SUM(COALESCE(tokens_used, 0)) as total_tokens_used,
                    SUM(error_count) as total_errors,
                    CAST(SUM(error_count) AS REAL) / COUNT(*) as error_rate
                FROM analysis_metrics
                WHERE created_at >= datetime('now', '-' || ? || ' days')
            """
            
            params = [days]
            
            if analysis_type:
                query += " AND analysis_type = ?"
                params.append(analysis_type)
            
            query += " GROUP BY analysis_type"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"성능 통계 조회 실패: {e}")
            return []
    
    def cleanup_old_data(self, days: int = 90):
        """
        오래된 데이터 삭제
        
        Args:
            days: N일 이전 데이터 삭제
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM analysis_metrics
                WHERE created_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            self.conn.commit()
            deleted_count = cursor.rowcount
            logger.info(f"오래된 메트릭 데이터 {deleted_count}개 삭제 완료 (90일 이상)")
        except Exception as e:
            logger.error(f"오래된 데이터 삭제 실패: {e}")
            self.conn.rollback()
    
    def insert_vllm_metric(
        self,
        request_id: str,
        restaurant_id: Optional[int],
        analysis_type: str,
        prefill_time_ms: Optional[float] = None,
        decode_time_ms: Optional[float] = None,
        total_time_ms: Optional[float] = None,
        n_tokens: Optional[int] = None,
        tpot_ms: Optional[float] = None,
        tps: Optional[float] = None,
        ttft_ms: Optional[float] = None,
    ):
        """
        vLLM 메트릭 저장
        
        Args:
            request_id: 요청 ID
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            prefill_time_ms: Prefill 시간 (밀리초)
            decode_time_ms: Decode 시간 (밀리초)
            total_time_ms: 전체 시간 (밀리초)
            n_tokens: 생성된 토큰 수
            tpot_ms: Time Per Output Token (밀리초)
            tps: Tokens Per Second
            ttft_ms: Time To First Token (밀리초)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO vllm_metrics (
                    request_id, restaurant_id, analysis_type,
                    prefill_time_ms, decode_time_ms, total_time_ms,
                    n_tokens, tpot_ms, tps, ttft_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, restaurant_id, analysis_type,
                prefill_time_ms, decode_time_ms, total_time_ms,
                n_tokens, tpot_ms, tps, ttft_ms
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"vLLM 메트릭 저장 실패: {e}")
            self.conn.rollback()
    
    def get_last_success_at(
        self,
        restaurant_id: Optional[int],
        analysis_type: str,
    ) -> Optional[datetime]:
        """
        마지막 성공 실행 시각 조회 (초기 전략: analysis_metrics에서 MAX(created_at))
        
        Args:
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            
        Returns:
            마지막 성공 실행 시각 (datetime), 없으면 None
        """
        if restaurant_id is None:
            return None
        
        try:
            cursor = self.conn.cursor()
            # error_count=0 중 최신 created_at 조회
            cursor.execute("""
                SELECT MAX(created_at) as last_success_at
                FROM analysis_metrics
                WHERE restaurant_id = ? 
                  AND analysis_type = ?
                  AND error_count = 0
            """, (restaurant_id, analysis_type))
            
            row = cursor.fetchone()
            if row and row["last_success_at"]:
                last_success_at_str = row["last_success_at"]
                # ISO 형식 또는 SQLite datetime 형식 파싱
                try:
                    # ISO 형식 시도
                    return datetime.fromisoformat(last_success_at_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # SQLite datetime 형식 시도 (YYYY-MM-DD HH:MM:SS)
                    try:
                        return datetime.strptime(last_success_at_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # 마이크로초 포함 형식
                        return datetime.strptime(last_success_at_str, "%Y-%m-%d %H:%M:%S.%f")
            return None
        except Exception as e:
            logger.error(f"마지막 성공 시각 조회 실패: {e}")
            return None
    
    def should_skip_analysis(
        self,
        restaurant_id: Optional[int],
        analysis_type: str,
        min_interval_seconds: int = 3600,  # 기본값: 1시간
    ) -> bool:
        """
        분석을 건너뛸지 여부 판단 (초기 전략: analysis_metrics 기반)
        
        최근 성공 실행시간(= error_count=0 중 최신 created_at)을 조회해서
        interval 이내면 SKIP
        
        Args:
            restaurant_id: 레스토랑 ID
            analysis_type: 분석 타입 ('sentiment', 'summary', 'strength')
            min_interval_seconds: 최소 간격 (초)
            
        Returns:
            True면 SKIP, False면 실행
        """
        if restaurant_id is None:
            return False
        
        last_success_at = self.get_last_success_at(restaurant_id, analysis_type)
        if last_success_at is None:
            return False  # 이전 기록이 없으면 실행
        
        time_diff = (datetime.now() - last_success_at).total_seconds()
        should_skip = time_diff < min_interval_seconds
        
        if should_skip:
            logger.info(
                f"SKIP: 레스토랑 {restaurant_id}의 {analysis_type} 분석이 "
                f"{int(time_diff)}초 전에 성공적으로 실행됨 (최소 간격: {min_interval_seconds}초)"
            )
        
        return should_skip
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None

