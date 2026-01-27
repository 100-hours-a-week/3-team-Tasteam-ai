"""
Aspect Seed 관리 모듈
전체 데이터셋 분석 결과를 파일이나 캐시에서 로드
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 기본 Aspect Seed (fallback)
DEFAULT_SERVICE_SEEDS = ['직원 친절', '사장 친절', '친절 기분', '서비스 친절', '사장 직원', '직원 서비스', '아주머니 친절']
DEFAULT_PRICE_SEEDS = ['가격 대비', '무한 리필', '가격 생각', '음식 가격', '합리 가격', '메뉴 가격', '가격 만족', '가격 퀄리티', '리필 가능', '커피 가격', '가격 사악', '런치 가격']
DEFAULT_FOOD_SEEDS = ['가락 국수', '평양 냉면', '수제 버거', '크림 치즈', '치즈 케이크', '크림 파스타', '당근 케이크', '오일 파스타', '카페 커피', '비빔 냉면', '커피 원두', '리코타 치즈', '비빔 막국수', '치즈 돈가스', '커피 산미', '치즈 파스타']

# Aspect Seed 캐시 (메모리 캐싱)
_aspect_seeds_cache = None
_aspect_seeds_file_path = None
_aspect_seeds_file_mtime = None


def load_aspect_seeds(
    file_path: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, List[str]]:
    """
    Aspect Seed 로드 (메모리 캐싱 지원)
    
    우선순위:
    1. 파일에서 로드 (file_path가 제공된 경우)
    2. 환경 변수에서 파일 경로 읽기 (ASPECT_SEEDS_FILE)
    3. 기본값 사용
    
    Args:
        file_path: Aspect seed JSON 파일 경로 (선택적)
        use_cache: 캐시 사용 여부 (기본값: True)
    
    Returns:
        {"service": [...], "price": [...], "food": [...]}
    """
    global _aspect_seeds_cache, _aspect_seeds_file_path, _aspect_seeds_file_mtime
    
    # 1. 파일 경로 결정
    if not file_path:
        file_path = os.getenv("ASPECT_SEEDS_FILE")
    
    # 2. 캐시 확인 (파일이 변경되지 않았는지 확인)
    if use_cache and _aspect_seeds_cache is not None:
        if file_path == _aspect_seeds_file_path:
            # 파일 수정 시간 확인
            if file_path and os.path.exists(file_path):
                current_mtime = os.path.getmtime(file_path)
                if current_mtime == _aspect_seeds_file_mtime:
                    logger.debug("Aspect seed 캐시에서 반환")
                    return _aspect_seeds_cache
            else:
                # 파일이 없으면 캐시된 기본값 반환
                logger.debug("Aspect seed 캐시에서 반환 (기본값)")
                return _aspect_seeds_cache
    
    # 3. 파일에서 로드 시도
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                seeds_data = json.load(f)
            
            # 형식 검증
            if isinstance(seeds_data, dict):
                result = {
                    "service": seeds_data.get("service", DEFAULT_SERVICE_SEEDS),
                    "price": seeds_data.get("price", DEFAULT_PRICE_SEEDS),
                    "food": seeds_data.get("food", DEFAULT_FOOD_SEEDS),
                }
                logger.info(f"Aspect seed 파일에서 로드: {file_path}")
                
                # 캐시 업데이트
                if use_cache:
                    _aspect_seeds_cache = result
                    _aspect_seeds_file_path = file_path
                    _aspect_seeds_file_mtime = os.path.getmtime(file_path)
                
                return result
            else:
                logger.warning(f"Aspect seed 파일 형식이 올바르지 않습니다: {file_path}")
        except Exception as e:
            logger.warning(f"Aspect seed 파일 로드 실패: {e}, 기본값 사용")
    
    # 4. 기본값 사용
    logger.debug("기본 Aspect seed 사용")
    result = {
        "service": DEFAULT_SERVICE_SEEDS,
        "price": DEFAULT_PRICE_SEEDS,
        "food": DEFAULT_FOOD_SEEDS,
    }
    
    # 캐시 업데이트
    if use_cache:
        _aspect_seeds_cache = result
        _aspect_seeds_file_path = file_path
        _aspect_seeds_file_mtime = None
    
    return result


def save_aspect_seeds(
    seeds: Dict[str, List[str]],
    file_path: str,
) -> bool:
    """
    Aspect Seed를 파일에 저장
    
    Args:
        seeds: {"service": [...], "price": [...], "food": [...]}
        file_path: 저장할 파일 경로
    
    Returns:
        성공 여부
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(seeds, f, ensure_ascii=False, indent=2)
        logger.info(f"Aspect seed 저장 완료: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Aspect seed 저장 실패: {e}")
        return False
