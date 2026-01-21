"""
감성 분석 모듈 ()
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from .config import Config
from .llm_utils import LLMUtils
from .review_utils import extract_content_list

# 로깅 설정
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    감성 분석 클래스 ()
    """
    
    def __init__(
        self,
        llm_utils: Optional[LLMUtils] = None,
        vector_search: Optional[Any] = None,
    ):
        """
        Args:
            llm_utils: LLM sentiment 방식 사용 시 활용 (None이면 필요 시 생성)
            vector_search: VectorSearch 인스턴스 (reviews 미제공 시 전체 리뷰 조회용)
        """
        self.vector_search = vector_search
        self.llm_utils = llm_utils
        self._sentiment_pipeline = None
    
    def _get_sentiment_pipeline(self):
        """HuggingFace sentiment pipeline (lazy init)."""
        if self._sentiment_pipeline is not None:
            return self._sentiment_pipeline
        
        try:
            from transformers import pipeline
        except Exception as e:
            raise ImportError(
                "transformers가 설치되어 있지 않거나 pipeline 로딩에 실패했습니다. "
                "pip install transformers 를 확인하세요."
            ) from e
        
        # 디바이스 선택: GPU가 있으면 사용, 없으면 CPU
        device = -1
        try:
            import torch
            if Config.USE_GPU and torch.cuda.is_available():
                device = 0
        except Exception:
            device = -1
        
        self._sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=Config.SENTIMENT_MODEL,
            device=device,
        )
        logger.info(f"Sentiment 모델 로딩 완료: {Config.SENTIMENT_MODEL} (device={device})")
        return self._sentiment_pipeline
    
    @staticmethod
    def _map_label_to_binary(label: str) -> Optional[str]:
        """
        모델 label을 binary(positive/negative)로 매핑.
        - 긍정: POSITIVE / positive / LABEL_1
        - 부정: NEGATIVE / negative / LABEL_0
        - 그 외(중립 등): None
        """
        if not label:
            return None
        normalized = str(label).strip().lower()
        if normalized in {"positive", "pos", "label_1"}:
            return "positive"
        if normalized in {"negative", "neg", "label_0"}:
            return "negative"
        return None
    
    def _classify_contents(self, content_list: List[str]) -> Tuple[int, int, int]:
        """
        sentiment 모델로 전체 리뷰를 분류하여 (positive_count, negative_count, total_count) 반환
        """
        if not content_list:
            return 0, 0, 0
        
        pipe = self._get_sentiment_pipeline()
        
        positive_count = 0
        negative_count = 0
        total_count = len(content_list)
        
        batch_size = getattr(Config, "LLM_BATCH_SIZE", 10)  # 기존 설정 재사용 (기본 10)
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i : i + batch_size]
            outputs = pipe(batch, truncation=True)
            for out in outputs:
                mapped = self._map_label_to_binary(out.get("label"))
                if mapped == "positive":
                    positive_count += 1
                elif mapped == "negative":
                    negative_count += 1
                # neutral/unknown은 total_count에 포함되지만 count는 증가하지 않음
        
        return positive_count, negative_count, total_count
    
    def analyze(
        self,
        reviews: Optional[List[Dict]] = None,
        restaurant_id: int = None,
    ) -> Dict:
        """
        전체 리뷰를 sentiment 모델로 분류하여 긍/부정 개수를 계산하고,
        코드에서 직접 비율을 산출합니다.
        
        Args:
            reviews: 리뷰 딕셔너리 리스트 (선택, None이면 vector_search에서 전체 리뷰 조회)
            restaurant_id: 레스토랑 ID (필수)
            
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
        
        # reviews 미제공이면: vector_search에서 리뷰 가져오기 (샘플링 활성화 여부에 따라)
        if reviews is None:
            if not self.vector_search:
                logger.warning("vector_search가 없고 reviews도 없습니다. 빈 결과를 반환합니다.")
                return {
                    "restaurant_id": restaurant_id,
                    "positive_count": 0,
                    "negative_count": 0,
                    "total_count": 0,
                    "positive_ratio": 0,
                    "negative_ratio": 0
                }
            
            # 샘플링 활성화 여부에 따라 분기
            if Config.ENABLE_SENTIMENT_SAMPLING:
                # 최근 리뷰부터 100개 샘플링
                limit = Config.SENTIMENT_RECENT_TOP_K
                logger.info(f"최근 리뷰 {limit}개 샘플링 활성화 (restaurant_id: {restaurant_id})")
                reviews = self.vector_search.get_recent_restaurant_reviews(
                    restaurant_id=restaurant_id,
                    limit=limit
                )
                logger.info(f"최근 리뷰 {len(reviews)}개를 샘플링했습니다.")
            else:
                # 전체 리뷰 사용
                reviews = self.vector_search.get_restaurant_reviews(str(restaurant_id))
                logger.info(f"전체 리뷰 사용: {len(reviews)}개 리뷰 (restaurant_id: {restaurant_id})")
        
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
        
        # SENTIMENT_METHOD에 따라 분기 (샘플링 없음: 항상 전체 리뷰)
        method = (Config.SENTIMENT_METHOD or "model").lower()
        if method == "llm":
            logger.info(f"총 {len(content_list)}개의 리뷰를 LLM sentiment 방식으로 집계합니다 (restaurant_id: {restaurant_id}).")
            if self.llm_utils is None:
                self.llm_utils = LLMUtils()
            counts = self.llm_utils.count_sentiments(content_list)
            positive_count = int(counts.get("positive_count", 0))
            negative_count = int(counts.get("negative_count", 0))
            total_count = len(content_list)
        else:
            logger.info(f"총 {len(content_list)}개의 리뷰를 sentiment 모델로 분류합니다 (restaurant_id: {restaurant_id}).")
            positive_count, negative_count, total_count = self._classify_contents(content_list)

        positive_ratio = int(round((positive_count / total_count) * 100)) if total_count else 0
        negative_ratio = int(round((negative_count / total_count) * 100)) if total_count else 0
        
        return {
            "restaurant_id": restaurant_id,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_count": total_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
        }
    
    async def analyze_async(
        self,
        reviews: List[Dict],
        restaurant_id: int,
    ) -> Dict:
        """
        비동기 엔드포인트 호환을 위한 wrapper.
        (현재 sentiment는 LLM/vLLM이 아니라 sentiment 모델 기반으로 동작)
        """
        # 현재 구현은 sync지만, 엔드포인트는 async라 wrapper 유지
        return self.analyze(reviews=reviews, restaurant_id=restaurant_id)
    
    async def analyze_multiple_restaurants_async(
        self,
        restaurants_data: List[Dict[str, Any]],  # [{"restaurant_id": 1, "reviews": [...]}, ...]
    ) -> List[Dict[str, Any]]:
        """
        여러 레스토랑의 리뷰를 sentiment 모델로 분류하여 결과를 반환합니다.
        샘플링이 활성화되어 있으면 대표 벡터 기반 TOP-K를 사용하고,
        비활성화되어 있으면 전체 리뷰를 사용합니다.
        """
        if not restaurants_data:
            return []

        results: List[Dict[str, Any]] = []
        for data in restaurants_data:
            restaurant_id = data.get("restaurant_id")
            reviews = data.get("reviews", [])
            
            # 샘플링이 활성화되어 있으면 reviews를 None으로 전달하여 샘플링 로직 적용
            # 샘플링이 비활성화되어 있으면 제공된 reviews를 사용 (없으면 전체 리뷰 조회)
            if Config.ENABLE_SENTIMENT_SAMPLING:
                results.append(self.analyze(reviews=None, restaurant_id=restaurant_id))
            else:
                results.append(self.analyze(reviews=reviews, restaurant_id=restaurant_id))
        return results
