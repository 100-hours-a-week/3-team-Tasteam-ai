"""
캐싱 관리 모듈 (Phase 2: Redis 캐싱 시스템)

통합 캐싱 관리자를 제공하여 LLM 응답, 감성 분석 결과, 임베딩 벡터를 캐싱합니다.
"""
import hashlib
import json
import logging
import uuid
from typing import Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Redis 임포트 (선택적)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis가 설치되지 않았습니다. 캐싱이 비활성화됩니다.")


class CacheManager:
    """통합 캐싱 관리자"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
    ):
        """
        Args:
            redis_host: Redis 호스트
            redis_port: Redis 포트
            redis_db: Redis 데이터베이스 번호
            redis_password: Redis 비밀번호 (선택적)
        """
        self.enabled = False
        self.redis = None
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis가 설치되지 않아 캐싱이 비활성화됩니다.")
            return
        
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,  # 바이너리 모드로 저장
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # 연결 테스트
            self.redis.ping()
            self.enabled = True
            logger.info(f"✅ Redis 연결 성공: {redis_host}:{redis_port}/{redis_db}")
        except Exception as e:
            logger.warning(f"Redis 연결 실패: {e}. 캐싱이 비활성화됩니다.")
            self.redis = None
            self.enabled = False
    
    def _get_key(self, prefix: str, content: str) -> str:
        """
        캐시 키 생성
        
        Args:
            prefix: 키 접두사 (예: "llm", "sentiment", "embedding")
            content: 캐시할 내용 (해시화됨)
            
        Returns:
            캐시 키 문자열
        """
        # 내용을 해시화하여 키 생성
        hash_content = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"{prefix}:{hash_content}"
    
    def get(self, prefix: str, content: str) -> Optional[Any]:
        """
        캐시 조회
        
        Args:
            prefix: 키 접두사
            content: 조회할 내용
            
        Returns:
            캐시된 값 (없으면 None)
        """
        if not self.enabled or self.redis is None:
            return None
        
        try:
            key = self._get_key(prefix, content)
            cached = self.redis.get(key)
            if cached:
                # JSON 디코딩
                return json.loads(cached)
        except Exception as e:
            logger.error(f"캐시 조회 실패: {e}")
        return None
    
    def set(
        self,
        prefix: str,
        content: str,
        value: Any,
        ttl: int = 3600,
    ) -> bool:
        """
        캐시 저장
        
        Args:
            prefix: 키 접두사
            content: 저장할 내용
            value: 저장할 값
            ttl: Time To Live (초 단위, 기본값: 1시간)
            
        Returns:
            저장 성공 여부
        """
        if not self.enabled or self.redis is None:
            return False
        
        try:
            key = self._get_key(prefix, content)
            # JSON 인코딩
            serialized = json.dumps(value, ensure_ascii=False)
            self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False
    
    def delete(self, prefix: str, content: str) -> bool:
        """
        캐시 삭제
        
        Args:
            prefix: 키 접두사
            content: 삭제할 내용
            
        Returns:
            삭제 성공 여부
        """
        if not self.enabled or self.redis is None:
            return False
        
        try:
            key = self._get_key(prefix, content)
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"캐시 삭제 실패: {e}")
            return False
    
    def clear_prefix(self, prefix: str) -> int:
        """
        특정 접두사를 가진 모든 키 삭제
        
        Args:
            prefix: 삭제할 키 접두사
            
        Returns:
            삭제된 키 개수
        """
        if not self.enabled or self.redis is None:
            return 0
        
        try:
            pattern = f"{prefix}:*"
            keys = list(self.redis.scan_iter(match=pattern))
            if keys:
                return self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"접두사 삭제 실패: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """
        캐시 통계 조회
        
        Returns:
            캐시 통계 딕셔너리
        """
        if not self.enabled or self.redis is None:
            return {
                "enabled": False,
                "redis_available": False,
            }
        
        try:
            info = self.redis.info()
            return {
                "enabled": True,
                "redis_available": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {
                "enabled": True,
                "redis_available": True,
                "error": str(e),
            }


# 전역 캐시 매니저 인스턴스 (선택적)
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    전역 캐시 매니저 인스턴스를 반환합니다.
    
    Returns:
        CacheManager 인스턴스
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        import os
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        _global_cache_manager = CacheManager(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            redis_password=redis_password,
        )
    
    return _global_cache_manager


class RedisLock:
    """Redis 기반 분산 락 (중복 실행 방지)"""
    
    def __init__(
        self,
        redis_client: Any,
        key: str,
        ttl: int = 3600,  # 기본값: 1시간
        timeout: int = 10,  # 락 획득 대기 시간 (초)
    ):
        """
        Args:
            redis_client: Redis 클라이언트 인스턴스
            key: 락 키 (예: "lock:1:sentiment")
            ttl: 락 TTL (초 단위)
            timeout: 락 획득 대기 시간 (초 단위, 0이면 즉시 반환)
        """
        self.redis = redis_client
        self.key = key
        self.ttl = ttl
        self.timeout = timeout
        self.lock_value = None  # 락 값 (UUID, 해제 시 검증용)
    
    def acquire(self) -> bool:
        """
        락 획득 시도
        
        Returns:
            True: 락 획득 성공
            False: 락 획득 실패
        """
        if self.redis is None:
            # Redis가 없으면 락 없이 진행 (개발 환경)
            logger.warning(f"Redis가 없어 락을 건너뜁니다: {self.key}")
            return True
        
        self.lock_value = str(uuid.uuid4())
        
        try:
            # SET lock:{key} {value} NX EX {ttl}
            # NX: 키가 없을 때만 설정
            # EX: TTL 설정 (초 단위)
            result = self.redis.set(
                self.key,
                self.lock_value,
                nx=True,  # 키가 없을 때만 설정
                ex=self.ttl,  # TTL 설정
            )
            
            if result:
                logger.debug(f"락 획득 성공: {self.key}")
                return True
            else:
                logger.warning(f"락 획득 실패 (다른 프로세스가 사용 중): {self.key}")
                return False
        except Exception as e:
            logger.error(f"락 획득 중 오류: {e}")
            # 오류 발생 시 락 없이 진행 (안전한 실패)
            return True
    
    def release(self) -> bool:
        """
        락 해제
        
        Returns:
            True: 락 해제 성공
            False: 락 해제 실패
        """
        if self.redis is None or self.lock_value is None:
            return True
        
        try:
            # Lua 스크립트로 원자적 삭제 (값이 일치할 때만 삭제)
            # 이렇게 하면 다른 프로세스가 만료된 락을 삭제하는 것을 방지
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = self.redis.eval(lua_script, 1, self.key, self.lock_value)
            
            if result:
                logger.debug(f"락 해제 성공: {self.key}")
            else:
                logger.warning(f"락 해제 실패 (값 불일치 또는 이미 만료): {self.key}")
            
            return bool(result)
        except Exception as e:
            logger.error(f"락 해제 중 오류: {e}")
            return False
    
    def __enter__(self):
        """Context manager 진입"""
        if not self.acquire():
            raise RuntimeError(f"락 획득 실패: {self.key}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.release()
        return False


@contextmanager
def acquire_lock(
    restaurant_id: Optional[int],
    analysis_type: str,
    ttl: int = 3600,
    timeout: int = 0,
):
    """
    Redis 락 획득 (Context Manager)
    
    사용 예시:
        with acquire_lock(restaurant_id=1, analysis_type="sentiment"):
            # 중복 실행 방지된 코드
            result = analyze_sentiment(...)
    
    Args:
        restaurant_id: 레스토랑 ID
        analysis_type: 분석 타입 ('sentiment', 'summary', 'comparison')
        ttl: 락 TTL (초 단위, 기본값: 1시간)
        timeout: 락 획득 대기 시간 (초 단위, 기본값: 0 = 즉시 반환)
    
    Yields:
        RedisLock 인스턴스
    
    Raises:
        RuntimeError: 락 획득 실패 시
    """
    if restaurant_id is None:
        # restaurant_id가 None이면 락 없이 진행
        yield None
        return
    
    cache_manager = get_cache_manager()
    
    if not cache_manager.enabled or cache_manager.redis is None:
        # Redis가 없으면 락 없이 진행 (개발 환경)
        logger.warning("Redis가 없어 락을 건너뜁니다.")
        yield None
        return
    
    lock_key = f"lock:{restaurant_id}:{analysis_type}"
    lock = RedisLock(
        redis_client=cache_manager.redis,
        key=lock_key,
        ttl=ttl,
        timeout=timeout,
    )
    
    try:
        if not lock.acquire():
            raise RuntimeError(
                f"중복 실행 방지: 레스토랑 {restaurant_id}의 {analysis_type} 분석이 이미 진행 중입니다."
            )
        yield lock
    finally:
        lock.release()

