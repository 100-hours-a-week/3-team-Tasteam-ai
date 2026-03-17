"""
다른 음식점과의 비교 모듈 (Kiwi + lift 기반)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import Config
from .llm_utils import LLMUtils
from .vector_search import VectorSearch

logger = logging.getLogger(__name__)

# 불용어 캐시 (모듈 레벨)
_stopwords_cache = None
_stopwords_path = None


def _get_stopwords() -> Optional[List[str]]:
    """불용어 리스트 로드 (모듈 레벨 캐싱)"""
    global _stopwords_cache, _stopwords_path

    # 캐시 확인
    if _stopwords_cache is not None:
        return _stopwords_cache

    # 불용어 파일: src/data/ (패키지 기준 경로)
    try:
        _src_dir = Path(__file__).resolve().parent
        stopwords_path = _src_dir / "data" / "stopwords-ko.txt"
        if stopwords_path.exists():
            with open(stopwords_path, encoding="utf-8") as f:
                _stopwords_cache = [w.strip() for w in f if w.strip()]
                _stopwords_path = str(stopwords_path)
                logger.debug(f"불용어 파일 로드 완료: {stopwords_path} ({len(_stopwords_cache)}개)")
                return _stopwords_cache
        logger.debug(f"불용어 파일을 찾을 수 없습니다: {stopwords_path}")
        _stopwords_cache = []
        return _stopwords_cache
    except Exception as e:
        logger.warning(f"불용어 로드 실패: {e}")
        _stopwords_cache = []
        return _stopwords_cache


class ComparisonPipeline:
    """다른 음식점과의 비교 (Kiwi + lift)"""

    def __init__(
        self,
        llm_utils: LLMUtils,
        vector_search: VectorSearch,
    ):
        """
        Args:
            llm_utils: LLMUtils 인스턴스 (API 호환용, 현재 경로에서는 미사용)
            vector_search: VectorSearch 인스턴스
        """
        self.llm_utils = llm_utils
        self.vector_search = vector_search

    # ==================== 전체 파이프라인 실행 ====================

    async def compare(
        self,
        restaurant_id: int,
        restaurant_name: Optional[str] = None,
        all_average_data_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        통계적 비율 기반 다른 음식점과의 비교 (Kiwi + lift).
        레스토랑 리뷰 → Kiwi 명사 bigram → service/price 긍정 비율 → lift.
        comparisons: [{category, lift_percentage}] (lift>0만), comparison_display, category_lift.
        """
        start_time = time.time()
        logger.info("comparison 진행: restaurant_id=%s, restaurant_name=%s", restaurant_id, restaurant_name or "(이름없음)")

        from .comparison_pipeline import (
            calculate_comparison_lift,
            calculate_single_restaurant_ratios,
            format_comparison_display,
            calculate_all_average_ratios_from_reviews,
            calculate_all_average_ratios_from_file,
        )

        stopwords = _get_stopwords() or []
        # 전체 평균: Qdrant(벡터 업로드 리뷰) 우선 → 파일 경로 → Config fallback. 벡터 리뷰 사용 시 Spark는 /all-average-from-reviews만 호출.
        aspect_data_path = all_average_data_path or Config.ALL_AVERAGE_ASPECT_DATA_PATH
        all_average_ratios = None
        try:
            # ① Qdrant(벡터 업로드된 전체 리뷰) 우선. 2000건 이상이면 Spark /all-average-from-reviews 호출 가능.
            logger.info("전체 평균 ① Qdrant 시도: get_all_reviews_for_all_average(5000)")
            all_reviews = self.vector_search.get_all_reviews_for_all_average(limit=5000)
            if all_reviews:
                from .async_workers import run_via_queue
                all_average_ratios = await run_via_queue(
                    "kiwi", calculate_all_average_ratios_from_reviews, all_reviews, stopwords
                )
                logger.info(
                    "전체 평균 ① Qdrant 사용: 리뷰 %d건 → service=%.4f, price=%.4f",
                    len(all_reviews), all_average_ratios.get("service", 0), all_average_ratios.get("price", 0),
                )
            else:
                logger.warning("전체 평균 ① Qdrant: 조회된 리뷰 없음")
            # ② Qdrant 실패 시 파일 경로로 시도 (Spark /all-average-from-file)
            if all_average_ratios is None and aspect_data_path:
                project_root = str(Path(__file__).resolve().parents[1])  # src/ 상위 = 프로젝트 루트
                logger.info(
                    "전체 평균 ② 파일 시도 (Spark 직접 읽기): path=%s",
                    aspect_data_path,
                )
                from .async_workers import run_via_queue
                all_average_ratios = await run_via_queue(
                    "kiwi",
                    calculate_all_average_ratios_from_file,
                    aspect_data_path,
                    stopwords=stopwords,
                    project_root=project_root,
                )
                if all_average_ratios is not None:
                    logger.info(
                        "전체 평균 ② 파일 사용 (Spark 직접 읽기): path=%s → service=%.4f, price=%.4f",
                        aspect_data_path,
                        all_average_ratios.get("service", 0),
                        all_average_ratios.get("price", 0),
                    )
                else:
                    logger.warning(
                        "전체 평균 ② 파일: Spark 직접 읽기 실패 또는 파일 없음 (path=%s)",
                        aspect_data_path,
                    )
        except Exception as e:
            logger.debug(f"전체 평균 계산 건너뜀: {e}")
        if all_average_ratios is None:
            all_average_ratios = {
                "service": Config.ALL_AVERAGE_SERVICE_RATIO,
                "price": Config.ALL_AVERAGE_PRICE_RATIO,
            }
            logger.info(
                "전체 평균 ③ Config fallback: ALL_AVERAGE_SERVICE_RATIO=%.4f, ALL_AVERAGE_PRICE_RATIO=%.4f",
                all_average_ratios["service"], all_average_ratios["price"],
            )

        try:
            restaurant_reviews = self.vector_search.get_restaurant_reviews(str(restaurant_id))
        except Exception as e:
            logger.error(f"레스토랑 리뷰 조회 실패: {e}")
            return {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "comparisons": [],
                "total_candidates": 0,
                "validated_count": 0,
                "category_lift": {"service": 0.0, "price": 0.0},
                "comparison_display": format_comparison_display(0.0, 0.0, 0),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        if not restaurant_reviews:
            logger.warning(f"레스토랑 {restaurant_id}에 리뷰가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "comparisons": [],
                "total_candidates": 0,
                "validated_count": 0,
                "category_lift": {"service": 0.0, "price": 0.0},
                "comparison_display": format_comparison_display(0.0, 0.0, 0),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # 벡터 payload에 없으면 요청에서 전달된 restaurant_name 사용
        _name_from_payload = (restaurant_reviews[0].get("restaurant_name") or None) if restaurant_reviews else None
        restaurant_name = _name_from_payload or restaurant_name
        review_texts = [
            (r.get("content") or r.get("text") or "")
            for r in restaurant_reviews
            if (r.get("content") or r.get("text"))
        ]

        from .async_workers import run_via_queue
        single_restaurant_ratios = await run_via_queue(
            "kiwi",
            calculate_single_restaurant_ratios,
            review_texts,
            stopwords,
        )
        lift_dict = calculate_comparison_lift(single_restaurant_ratios, all_average_ratios)
        n_reviews = len(review_texts)

        logger.info(
            "표본(단일 음식점) restaurant_id=%s service=%.4f, price=%.4f | lift service=%d%%, price=%d%%",
            restaurant_id,
            single_restaurant_ratios.get("service", 0),
            single_restaurant_ratios.get("price", 0),
            lift_dict.get("service", 0),
            lift_dict.get("price", 0),
        )
        # 수치 기반 템플릿으로 비교 문구 생성 (LLM 미사용)
        comparison_display = format_comparison_display(
            lift_dict.get("service", 0.0),
            lift_dict.get("price", 0.0),
            n_reviews,
        )

        comparisons = []
        for category, lift in lift_dict.items():
            if lift <= 0:
                continue
            comparisons.append({
                "category": category,
                "lift_percentage": lift,
            })

        comparisons.sort(key=lambda x: x["lift_percentage"], reverse=True)

        processing_time = (time.time() - start_time) * 1000
        return {
            "restaurant_id": restaurant_id,
            "restaurant_name": restaurant_name,
            "comparisons": comparisons,
            "total_candidates": len(review_texts),
            "validated_count": len(comparisons),
            "category_lift": lift_dict,
            "comparison_display": comparison_display,
            "processing_time_ms": processing_time,
        }

    async def compare_batch(
        self,
        restaurants: List[Dict[str, Any]],
        all_average_data_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        다수 음식점에 대한 비교 배치 처리.
        COMPARISON_BATCH_ASYNC=true면 asyncio.gather(병렬), false면 순차.
        all_average_data_path: 전체 평균·표본 추출용 파일 경로 (예: tasteam_app_all_review_data.json).
        """
        if not restaurants:
            return []

        def _get_restaurant_id(rd: Dict) -> int:
            return int(rd.get("restaurant_id", 0))

        if Config.COMPARISON_BATCH_ASYNC:
            tasks = [
                self.compare(
                    restaurant_id=_get_restaurant_id(rd),
                    restaurant_name=rd.get("restaurant_name") if isinstance(rd, dict) else None,
                    all_average_data_path=all_average_data_path,
                )
                for rd in restaurants
            ]
            return list(await asyncio.gather(*tasks))

        results: List[Dict[str, Any]] = []
        for rd in restaurants:
            rid = _get_restaurant_id(rd)
            name = rd.get("restaurant_name") if isinstance(rd, dict) else None
            result = await self.compare(restaurant_id=rid, restaurant_name=name, all_average_data_path=all_average_data_path)
            results.append(result)
        return results
