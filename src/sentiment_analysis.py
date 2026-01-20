"""
감성 분석 모듈 ()
"""

import json
import logging
from typing import Dict, List, Optional, Any

from .config import Config
from .llm_utils import LLMUtils
from .review_utils import extract_content_list

# 로깅 설정
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    감성 분석 클래스 ()
    
    대표 벡터 기반 TOP-K 리뷰만 사용하여 감성 분석 수행
    1. 레스토랑 대표 벡터 계산
    2. 대표 벡터와 가장 가까운 리뷰 TOP-K 선택
    3. TOP-K 리뷰만 LLM에 넣어 긍/부정 비율 계산
    """
    
    def __init__(
        self,
        llm_utils: Optional[LLMUtils] = None,
        vector_search: Optional[Any] = None,
    ):
        """
        Args:
            llm_utils: LLMUtils 인스턴스 (None이면 자동 생성)
            vector_search: VectorSearch 인스턴스 (대표 벡터 기반 검색용)
        """
        self.llm_utils = llm_utils or LLMUtils()
        self.vector_search = vector_search
    
    def analyze(
        self,
        reviews: Optional[List[Dict]] = None,
        restaurant_id: int = None,
        top_k: int = 20,
        months_back: Optional[int] = None,
        max_retries: int = Config.MAX_RETRIES,
    ) -> Dict:
        """
        대표 벡터 기반 TOP-K 리뷰를 사용하여 감성 분석 수행
        
        프로세스:
        1. 레스토랑 대표 벡터 계산
        2. 대표 벡터와 가장 가까운 리뷰 TOP-K 선택
        3. TOP-K 리뷰만 LLM에 넣어 긍/부정 비율 계산
        
        Args:
            reviews: 리뷰 딕셔너리 리스트 (선택, None이면 vector_search 사용)
            restaurant_id: 레스토랑 ID (필수)
            top_k: 대표 벡터 주위에서 선택할 리뷰 수 (기본값: 20)
            months_back: 최근 N개월 필터 (선택, None이면 필터링 안함)
            max_retries: LLM 호출 실패 시 최대 재시도 횟수
            
        Returns:
            최종 통계 결과 딕셔너리:
            - restaurant_id: 레스토랑 ID
            - positive_count: 긍정 리뷰 개수
            - negative_count: 부정 리뷰 개수
            - total_count: 전체 리뷰 개수
            - positive_ratio: 긍정 비율 (%)
            - negative_ratio: 부정 비율 (%)
        """
        if restaurant_id is None:
            logger.error("restaurant_id는 필수입니다.")
            return {
                "restaurant_id": None,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        # 대표 벡터 기반 TOP-K 리뷰 선택
        if self.vector_search and reviews is None:
            logger.info(f"대표 벡터 기반으로 TOP-{top_k} 리뷰를 선택합니다 (restaurant_id: {restaurant_id}).")
            top_k_results = self.vector_search.query_by_restaurant_vector(
                restaurant_id=restaurant_id,
                top_k=top_k,
                months_back=months_back,
            )
            reviews = [r["payload"] for r in top_k_results]
            logger.info(f"대표 벡터 기반으로 {len(reviews)}개 리뷰를 선택했습니다.")
        elif reviews is None:
            logger.warning("vector_search가 없고 reviews도 없습니다. 빈 결과를 반환합니다.")
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        if not reviews:
            logger.warning("분석할 리뷰가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        # content_list 추출
        content_list = extract_content_list(reviews)
        
        if not content_list:
            logger.warning("content 필드가 있는 리뷰가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": len(reviews),
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        logger.info(f"총 {len(content_list)}개의 리뷰를 LLM으로 분석합니다 (restaurant_id: {restaurant_id}).")
        
        # LLM 입력
        result = self.llm_utils.analyze_all_reviews(
            review_list=content_list,
            restaurant_id=restaurant_id,
            max_retries=max_retries
        )
        
        logger.info("✅ 최종 결과:")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))

        return {
            "restaurant_id": result.get("restaurant_id", restaurant_id),
            "positive_count": result.get("positive_count", 0),
            "negative_count": result.get("negative_count", 0),
            "total_count": result.get("total_count", len(content_list)),
            "positive_ratio": result.get("positive_ratio", 0),
            "negative_ratio": result.get("negative_ratio", 0)
        }
    
    async def analyze_async(
        self,
        reviews: List[Dict],
        restaurant_id: int,
        max_tokens_per_batch: Optional[int] = None,
        max_retries: int = Config.MAX_RETRIES,
    ) -> Dict:
        """
        리뷰 리스트를 비동기로 분석하여 positive_ratio와 negative_ratio를 계산합니다 (vLLM 직접 사용 시).
        
        vLLM 직접 사용 모드에서는 내부적으로 analyze_multiple_restaurants_async()를 재사용하여
        동적 배치 크기와 세마포어 기반 OOM 방지 전략을 적용합니다.
        
        OOM 방지 전략:
        - 동적 배치 크기 계산 (리뷰 길이에 따라)
        - 세마포어를 통한 동시 처리 수 제한
        - 각 배치는 독립적으로 처리 가능 (메모리 사용량 예측 가능)
        - vLLM이 자동으로 여러 배치를 효율적으로 처리 (Continuous Batching)
        
        Args:
            reviews: 리뷰 딕셔너리 리스트 (REVIEW TABLE)
            restaurant_id: 레스토랑 ID (BIGINT FK)
            max_tokens_per_batch: 배치당 최대 토큰 수 (None이면 동적 계산)
            max_retries: LLM 호출 실패 시 최대 재시도 횟수
            
        Returns:
            최종 통계 결과 딕셔너리:
            - restaurant_id: 레스토랑 ID
            - positive_count: 긍정 리뷰 개수
            - negative_count: 부정 리뷰 개수
            - total_count: 전체 리뷰 개수
            - positive_ratio: 긍정 비율 (%)
            - negative_ratio: 부정 비율 (%)
        """
        if not reviews:
            logger.warning("분석할 리뷰가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        # content_list 추출
        content_list = extract_content_list(reviews)
        
        if not content_list:
            logger.warning("content 필드가 있는 리뷰가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "total_count": len(reviews),
                "positive_ratio": 0,
                "negative_ratio": 0
            }
        
        logger.info(f"총 {len(content_list)}개의 리뷰를 vLLM으로 비동기 분석합니다 (restaurant_id: {restaurant_id}).")

        # vLLM 직접 사용 모드인지 확인
        if hasattr(self.llm_utils, 'use_pod_vllm') and self.llm_utils.use_pod_vllm:
            # vLLM 모드: analyze_multiple_restaurants_async() 재사용 (OOM 방지 전략 포함)
            # 단일 레스토랑 요청을 리스트로 감싸서 전달
            restaurants_data = [{
                "restaurant_id": restaurant_id,
                "reviews": reviews
            }]
            
            results = await self.analyze_multiple_restaurants_async(
                restaurants_data=restaurants_data,
                max_tokens_per_batch=max_tokens_per_batch,
                max_retries=max_retries
            )
            
            # 결과가 비어있지 않으면 첫 번째 결과 반환
            if results:
                result = results[0]
            else:
                # 결과가 없는 경우 빈 결과 반환
                result = {
                    "restaurant_id": restaurant_id,
                    "positive_count": 0,
                    "negative_count": 0,
                    "total_count": len(content_list),
                    "positive_ratio": 0,
                    "negative_ratio": 0
                }
        else:
            # 기존 방식 (동기)
            result = self.llm_utils.analyze_all_reviews(
                review_list=content_list,
                restaurant_id=restaurant_id,
                max_retries=max_retries
            )

        logger.info("✅ 최종 결과:")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))

        return {
            "restaurant_id": result.get("restaurant_id", restaurant_id),
            "positive_count": result.get("positive_count", 0),
            "negative_count": result.get("negative_count", 0),
            "total_count": result.get("total_count", len(content_list)),
            "positive_ratio": result.get("positive_ratio", 0),
            "negative_ratio": result.get("negative_ratio", 0)
        }
    
    async def analyze_multiple_restaurants_async(
        self,
        restaurants_data: List[Dict[str, Any]],  # [{"restaurant_id": 1, "reviews": [...]}, ...]
        max_tokens_per_batch: Optional[int] = None,
        max_retries: int = Config.MAX_RETRIES,
    ) -> List[Dict[str, Any]]:
        """
        여러 레스토랑을 비동기 큐 방식으로 감성 분석 (동적 배치 크기)
        
        각 레스토랑의 리뷰를 동적 배치 크기로 나누고, 모든 배치를 비동기 큐에 넣어
        vLLM의 Continuous Batching을 활용하여 처리합니다.
        
        OOM 방지 전략:
        - 각 레스토랑별로 동적 배치 크기 계산 (리뷰 길이에 따라)
        - 세마포어를 통한 동시 처리 수 제한
        - vLLM Continuous Batching으로 GPU 활용률 극대화
    
    Args:
            restaurants_data: 레스토랑 데이터 리스트
                - restaurant_id: 레스토랑 ID
                - reviews: 리뷰 딕셔너리 리스트 (REVIEW TABLE)
            max_tokens_per_batch: 배치당 최대 토큰 수 (None이면 Config 값 사용)
            max_retries: 최대 재시도 횟수
            
        Returns:
            각 레스토랑별 감성 분석 결과 리스트
        """
        if not restaurants_data:
            return []
        
        logger.info(f"총 {len(restaurants_data)}개 레스토랑을 비동기 큐 방식으로 처리합니다.")
        
        # content_list 추출 및 검증
        processed_data = []
        for data in restaurants_data:
            restaurant_id = data["restaurant_id"]
            reviews = data.get("reviews", [])
            
            content_list = extract_content_list(reviews)
            
            if content_list:
                processed_data.append({
                    "restaurant_id": restaurant_id,
                    "content_list": content_list
                })
            else:
                logger.warning(f"레스토랑 {restaurant_id}: content가 있는 리뷰가 없습니다.")
        
        if not processed_data:
            return []
        
        # vLLM 직접 사용 모드인지 확인
        if hasattr(self.llm_utils, 'use_pod_vllm') and self.llm_utils.use_pod_vllm:
            results = await self.llm_utils.analyze_multiple_restaurants_vllm(
                processed_data,
                max_tokens_per_batch=max_tokens_per_batch,
                max_retries=max_retries
            )
        else:
            # 기존 방식: 각 레스토랑을 순차 처리
            results = []
            for data in processed_data:
                result = self.llm_utils.analyze_all_reviews(
                    review_list=data["content_list"],
                    restaurant_id=data["restaurant_id"],
                    max_retries=max_retries
                )
                results.append(result)
        
        return results
