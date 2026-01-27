"""
감성 분석 모듈 ()
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union

from .config import Config
from .review_utils import extract_content_list

# 로깅 설정
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    감성 분석 클래스 ()
    """
    
    def __init__(
        self,
        vector_search: Optional[Any] = None,
    ):
        """
        Args:
            vector_search: VectorSearch 인스턴스 (reviews 미제공 시 전체 리뷰 조회용)
        """
        self.vector_search = vector_search
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
        
        # 새로운 파이프라인: Dilwolf/Kakao_app-kr_sentiment 모델 사용
        self._sentiment_pipeline = pipeline(
            task="text-classification",
            model="Dilwolf/Kakao_app-kr_sentiment",
            device=device,
        )
        logger.info(f"Sentiment 모델 로딩 완료: Dilwolf/Kakao_app-kr_sentiment (device={device})")
        return self._sentiment_pipeline
    
    @staticmethod
    def _map_label_to_binary(label: str) -> Optional[str]:
        """
        모델 label을 binary(positive/negative)로 매핑.
        - 긍정: "1" / POSITIVE / POS / LABEL_1 
        - 부정: "0" / NEGATIVE / NEG / LABEL_0  
        - 그 외: None (중립 라벨은 모델에 없으므로 score 기반으로 판정)
        """
        if not label:
            return None
        normalized = str(label).strip().lower()
        # 숫자 ID 처리 (0: negative, 1: positive)
        if normalized in {"1", "positive", "pos", "label_1"}:
            return "positive"
        if normalized in {"0", "negative", "neg", "label_0"}:
            return "negative"
        return None  # 알 수 없는 경우 None 반환 (score 기반 판정에서 처리)
    
    def _classify_contents(
        self, 
        content_list: List[str],
        reviews: Optional[List[Union[Dict, Any]]] = None,
    ) -> Tuple[int, int, int, int, List[Optional[str]]]:
        """
        새로운 파이프라인: HuggingFace 모델로 1차 분류 후, negative만 LLM으로 재판정
        
        Args:
            content_list: 리뷰 텍스트 리스트
            reviews: 리뷰 딕셔너리 리스트 (sentiment 라벨 부여용, 선택)
        
        Returns:
            (positive_count, negative_count, neutral_count, total_count, sentiment_labels)
            sentiment_labels: 각 리뷰의 sentiment 라벨 리스트 ("positive", "negative", "neutral", None)
        """
        if not content_list:
            return 0, 0, 0, 0, []
        
        pipe = self._get_sentiment_pipeline()
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_count = len(content_list)
        sentiment_labels: List[Optional[str]] = []
        
        # 1차 분류: HuggingFace 모델로 전체 리뷰 분류
        batch_size = getattr(Config, "LLM_BATCH_SIZE", 10)
        negative_reviews_for_llm = []  # LLM 재판정 대상
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i : i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(content_list))))
            
            # return_all_scores=True로 호출하여 모든 점수 확인
            outputs = pipe(batch, return_all_scores=True)
            
            for idx, out in zip(batch_indices, outputs):
                # outputs는 리스트의 리스트: [[{"label": "...", "score": ...}, ...], ...]
                if isinstance(out, list) and len(out) >= 2:
                    # positive score (인덱스 1) 확인
                    positive_score = out[1].get("score", 0.0) if len(out) > 1 else 0.0
                    is_positive = positive_score > 0.8
                    
                    if is_positive:
                        sentiment_labels.append("positive")
                        positive_count += 1
                    else:
                        # negative로 1차 분류, LLM 재판정 대상에 추가
                        sentiment_labels.append("negative")  # 임시로 negative
                        # ReviewModel (Pydantic)인 경우 딕셔너리로 변환
                        review_obj = None
                        if reviews and idx < len(reviews):
                            review = reviews[idx]
                            if hasattr(review, 'model_dump'):
                                review_obj = review.model_dump()
                            elif hasattr(review, 'dict'):
                                review_obj = review.dict()
                            elif isinstance(review, dict):
                                review_obj = review
                        
                        negative_reviews_for_llm.append({
                            "index": idx,
                            "content": content_list[idx],
                            "review": review_obj
                        })
                        negative_count += 1
        
        # 2차 분류: Negative로 분류된 리뷰만 LLM으로 재판정
        if negative_reviews_for_llm:
            try:
                import os
                import json
                from openai import OpenAI
                
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # LLM 입력 문자열 생성
                llm_input = "\n".join([
                    f'{item["review"].get("id") if item["review"] else item["index"]}\t{item["content"]}'
                    for item in negative_reviews_for_llm
                ])
                
                # LLM 호출
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a sentiment classification engine.\n"
                                "Each input line is formatted as: id<TAB>review.\n"
                                "Classify sentiment as one of: positive, negative, neutral.\n"
                                "Return ONLY a valid JSON array like:\n"
                                '[{"id":105,"sentiment":"positive"}]'
                            )
                        },
                        {"role": "user", "content": llm_input}
                    ],
                    temperature=0
                )
                
                raw = response.choices[0].message.content.strip()
                raw = raw[raw.find("["): raw.rfind("]")+1]  # JSON 방어
                results = json.loads(raw)
                
                # id -> sentiment 맵 생성
                sentiment_map = {x["id"]: x["sentiment"] for x in results}
                
                # Negative로 분류된 리뷰들의 sentiment 업데이트
                for item in negative_reviews_for_llm:
                    idx = item["index"]
                    review_id = item["review"].get("id") if item["review"] else idx
                    
                    if review_id in sentiment_map:
                        new_sentiment = sentiment_map[review_id]
                        sentiment_labels[idx] = new_sentiment
                        
                        # 카운트 조정
                        if new_sentiment == "positive":
                            positive_count += 1
                            negative_count -= 1
                        elif new_sentiment == "neutral":
                            neutral_count += 1
                            negative_count -= 1
                        # negative는 그대로 유지
                
                logger.info(f"LLM 재판정 완료: {len(negative_reviews_for_llm)}개 리뷰 중 {len(results)}개 재분류")
            except Exception as e:
                logger.warning(f"LLM 재판정 실패: {e}. 1차 분류 결과를 사용합니다.")
        
        # 리뷰 객체에 sentiment 라벨 부여
        # ReviewModel (Pydantic)인 경우 딕셔너리로 변환 후 수정
        if reviews is not None and len(reviews) == len(sentiment_labels):
            for i, (review, label) in enumerate(zip(reviews, sentiment_labels)):
                if label:  # positive, negative, neutral 모두 부여
                    # Pydantic 모델인 경우 딕셔너리로 변환
                    if hasattr(review, 'model_dump'):
                        review_dict = review.model_dump()
                        review_dict["sentiment"] = label
                        reviews[i] = review_dict
                    elif hasattr(review, 'dict'):
                        review_dict = review.dict()
                        review_dict["sentiment"] = label
                        reviews[i] = review_dict
                    elif isinstance(review, dict):
                        review["sentiment"] = label
                    # Pydantic 모델이고 수정이 필요한 경우는 이미 처리됨
        
        return positive_count, negative_count, neutral_count, total_count, sentiment_labels
    
    def analyze(
        self,
        reviews: Optional[List[Union[Dict, Any]]] = None,
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
        
        # ReviewModel (Pydantic)을 딕셔너리로 변환 (필요한 경우)
        reviews_dict = []
        for review in reviews:
            if hasattr(review, 'model_dump'):
                reviews_dict.append(review.model_dump())
            elif hasattr(review, 'dict'):
                reviews_dict.append(review.dict())
            elif isinstance(review, dict):
                reviews_dict.append(review)
            else:
                # 알 수 없는 형식은 스킵
                logger.warning(f"알 수 없는 리뷰 형식: {type(review)}")
                continue
        
        # content_list 추출
        content_list = extract_content_list(reviews_dict)
        
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
        
        # sentiment 모델로 분류 (새로운 파이프라인: 1차 분류 + LLM 재판정)
        logger.info(f"총 {len(content_list)}개의 리뷰를 sentiment 모델로 분류합니다 (restaurant_id: {restaurant_id}).")
        positive_count, negative_count, neutral_count, total_count, _ = self._classify_contents(
            content_list, 
            reviews=reviews_dict  # 딕셔너리 형식으로 변환된 리뷰 사용
        )
        # sentiment 라벨은 비율 계산에만 사용하며, Qdrant에는 저장하지 않음.

        # 비율 계산 (새로운 파이프라인 방식)
        # positive_rate = positive_count / (positive_count + negative_count)
        # neutral_rate = neutral_count / total_count
        # negative_rate = 1 - positive_rate
        total_with_sentiment = positive_count + negative_count
        positive_ratio = int(round((positive_count / total_with_sentiment) * 100)) if total_with_sentiment > 0 else 0
        negative_ratio = int(round((negative_count / total_with_sentiment) * 100)) if total_with_sentiment > 0 else 0
        neutral_ratio = int(round((neutral_count / total_count) * 100)) if total_count > 0 else 0
        
        return {
            "restaurant_id": restaurant_id,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_count": total_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
        }
    
    async def analyze_async(
        self,
        reviews: List[Union[Dict, Any]],
        restaurant_id: int,
    ) -> Dict:
        """
        비동기 엔드포인트 호환을 위한 wrapper.
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
