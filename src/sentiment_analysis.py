"""
감성 분석 모듈 ()
"""

import asyncio
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Any, Tuple, Union

from .config import Config
from .review_utils import extract_content_list

# 로깅 설정
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    감성 분석 클래스 ()
    """
    # 클래스 레벨에서 공유 (인스턴스/스레드 간 파이프라인 싱글톤)
    _shared_pipeline = None
    _shared_lock = threading.Lock()

    def __init__(
        self,
        vector_search: Optional[Any] = None,
    ):
        """
        Args:
            vector_search: VectorSearch 인스턴스 (reviews 미제공 시 전체 리뷰 조회용)
        """
        self.vector_search = vector_search

    @classmethod
    def _get_sentiment_pipeline(cls):
        """전역 싱글톤 파이프라인 (클래스 레벨, 쓰레드 안전)."""
        if cls._shared_pipeline is not None:
            return cls._shared_pipeline

        with cls._shared_lock:
            if cls._shared_pipeline is not None:
                return cls._shared_pipeline

            try:
                from transformers import pipeline
                import torch
            except Exception as e:
                raise ImportError(
                    "transformers가 설치되어 있지 않거나 pipeline 로딩에 실패했습니다. "
                    "pip install transformers 를 확인하세요."
                ) from e

            model_name = getattr(Config, "SENTIMENT_MODEL", "Dilwolf/Kakao_app-kr_sentiment")
            use_cpu = getattr(Config, "SENTIMENT_FORCE_CPU", True)
            device = 0 if (not use_cpu and torch.cuda.is_available()) else -1

            # low_cpu_mem_usage=False: meta 텐서 로딩 방지
            # attn_implementation="eager": SDPA meta tensor 오류 우회
            model_kwargs = {"low_cpu_mem_usage": False, "attn_implementation": "eager"}
            cls._shared_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
                model_kwargs=model_kwargs,
            )
            logger.info(f"Sentiment 모델 로딩 완료: {model_name} (device={device})")
            return cls._shared_pipeline
    
    @staticmethod
    def _score_to_float(score: Any) -> float:
        """
        파이프라인 출력의 score를 float으로 변환.
        meta tensor 등 .item() 호출 시 RuntimeError가 나는 경우 0.0 반환.
        """
        if score is None:
            return 0.0
        if isinstance(score, (int, float)):
            return float(score)
        if hasattr(score, "item"):
            try:
                return float(score.item())
            except (RuntimeError, ValueError):
                pass
        if hasattr(score, "cpu") and hasattr(score, "numpy"):
            try:
                return float(score.cpu().numpy().item())
            except (RuntimeError, ValueError, AttributeError):
                pass
        return 0.0

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
    
    def _classify_with_hf_only(
        self,
        content_list: List[str],
        reviews: Optional[List[Union[Dict, Any]]] = None,
    ) -> Tuple[int, int, int, int, List[Optional[str]], List[Dict[str, Any]]]:
        """HuggingFace 1차 분류만 (LLM 없음). Returns: (pos, neg, neu, total, labels, neg_for_llm)."""
        if not content_list:
            return 0, 0, 0, 0, [], []

        pipe = self._get_sentiment_pipeline()
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        total_count = len(content_list)
        sentiment_labels: List[Optional[str]] = []
        negative_reviews_for_llm: List[Dict[str, Any]] = []
        batch_size = getattr(Config, "LLM_BATCH_SIZE", 10)

        for i in range(0, len(content_list), batch_size):
            batch = content_list[i : i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(content_list))))
            
            # top_k=None으로 모든 레이블 점수 반환 (return_all_scores 대체)
            outputs = pipe(batch, top_k=None)
            
            for idx, out in zip(batch_indices, outputs):
                # outputs는 리스트의 리스트: [[{"label": "...", "score": ...}, ...], ...]
                if isinstance(out, list) and len(out) >= 2:
                    # positive score (인덱스 1) 확인 (meta tensor 등은 _score_to_float로 안전 변환)
                    raw_score = out[1].get("score", 0.0) if len(out) > 1 else 0.0
                    positive_score = self._score_to_float(raw_score)
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

        return positive_count, negative_count, neutral_count, total_count, sentiment_labels, negative_reviews_for_llm

    def _apply_llm_reclassify_sync(
        self,
        negative_reviews_for_llm: List[Dict[str, Any]],
        sentiment_labels: List[Optional[str]],
        positive_count: int,
        negative_count: int,
        neutral_count: int,
    ) -> Tuple[int, int, int, List[Optional[str]]]:
        """LLM 재판정 (동기)."""
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        llm_input = "\n".join([
            f'{item["review"].get("id") if item["review"] else item["index"]}\t{item["content"]}'
            for item in negative_reviews_for_llm
        ])
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
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
        raw = raw[raw.find("["): raw.rfind("]")+1]
        results = json.loads(raw)
        sentiment_map = {x["id"]: x["sentiment"] for x in results}
        for item in negative_reviews_for_llm:
            idx = item["index"]
            review_id = item["review"].get("id") if item["review"] else idx
            if review_id in sentiment_map:
                new_sentiment = sentiment_map[review_id]
                sentiment_labels[idx] = new_sentiment
                if new_sentiment == "positive":
                    positive_count += 1
                    negative_count -= 1
                elif new_sentiment == "neutral":
                    neutral_count += 1
                    negative_count -= 1
        logger.info(f"LLM 재판정 완료: {len(negative_reviews_for_llm)}개 리뷰 중 {len(results)}개 재분류")
        return positive_count, negative_count, neutral_count, sentiment_labels

    async def _apply_llm_reclassify_async(
        self,
        negative_reviews_for_llm: List[Dict[str, Any]],
        sentiment_labels: List[Optional[str]],
        positive_count: int,
        negative_count: int,
        neutral_count: int,
    ) -> Tuple[int, int, int, List[Optional[str]]]:
        """LLM 재판정 (비동기)."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        llm_input = "\n".join([
            f'{item["review"].get("id") if item["review"] else item["index"]}\t{item["content"]}'
            for item in negative_reviews_for_llm
        ])
        response = await client.chat.completions.create(
            model=Config.OPENAI_MODEL,
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
        raw = raw[raw.find("["): raw.rfind("]")+1]
        results = json.loads(raw)
        sentiment_map = {x["id"]: x["sentiment"] for x in results}
        for item in negative_reviews_for_llm:
            idx = item["index"]
            review_id = item["review"].get("id") if item["review"] else idx
            if review_id in sentiment_map:
                new_sentiment = sentiment_map[review_id]
                sentiment_labels[idx] = new_sentiment
                if new_sentiment == "positive":
                    positive_count += 1
                    negative_count -= 1
                elif new_sentiment == "neutral":
                    neutral_count += 1
                    negative_count -= 1
        logger.info(f"LLM 재판정 완료: {len(negative_reviews_for_llm)}개 리뷰 중 {len(results)}개 재분류")
        return positive_count, negative_count, neutral_count, sentiment_labels

    @staticmethod
    def _assign_labels_to_reviews(
        reviews: Optional[List[Union[Dict, Any]]],
        sentiment_labels: List[Optional[str]],
    ) -> None:
        """리뷰 객체에 sentiment 라벨 부여 (in-place)."""
        if reviews is None or len(reviews) != len(sentiment_labels):
            return
        for i, (review, label) in enumerate(zip(reviews, sentiment_labels)):
            if not label:
                continue
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

    def _classify_contents(
        self,
        content_list: List[str],
        reviews: Optional[List[Union[Dict, Any]]] = None,
    ) -> Tuple[int, int, int, int, List[Optional[str]]]:
        """HuggingFace 1차 분류 + LLM 재판정 (동기, analyze/배치용)."""
        pos, neg, neu, total, labels, neg_for_llm = self._classify_with_hf_only(content_list, reviews)
        if neg_for_llm:
            try:
                pos, neg, neu, labels = self._apply_llm_reclassify_sync(
                    neg_for_llm, labels, pos, neg, neu
                )
            except Exception as e:
                logger.warning(f"LLM 재판정 실패: {e}. 1차 분류 결과를 사용합니다.")
        self._assign_labels_to_reviews(reviews, labels)
        return pos, neg, neu, total, labels

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
        reviews: Optional[List[Union[Dict, Any]]] = None,
        restaurant_id: Optional[int] = None,
    ) -> Dict:
        """
        비동기 엔드포인트용. SENTIMENT_CLASSIFIER_USE_THREAD/SENTIMENT_LLM_ASYNC에 따라 분기.
        기본값(둘 다 false)이면 analyze() 호출.
        """
        if not Config.SENTIMENT_CLASSIFIER_USE_THREAD and not Config.SENTIMENT_LLM_ASYNC:
            return self.analyze(reviews=reviews, restaurant_id=restaurant_id)

        # 토글이 켜진 경우: 비동기/스레드 격리 경로
        if restaurant_id is None:
            return {
                "restaurant_id": None,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 0,
            }
        if reviews is None:
            if not self.vector_search:
                return {
                    "restaurant_id": restaurant_id,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "total_count": 0,
                    "positive_ratio": 0,
                    "negative_ratio": 0,
                    "neutral_ratio": 0,
                }
            if Config.ENABLE_SENTIMENT_SAMPLING:
                limit = Config.SENTIMENT_RECENT_TOP_K
                reviews = self.vector_search.get_recent_restaurant_reviews(restaurant_id=restaurant_id, limit=limit)
            else:
                reviews = self.vector_search.get_restaurant_reviews(str(restaurant_id))
        if not reviews:
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 0,
            }
        reviews_dict = []
        for r in reviews:
            if hasattr(r, 'model_dump'):
                reviews_dict.append(r.model_dump())
            elif hasattr(r, 'dict'):
                reviews_dict.append(r.dict())
            elif isinstance(r, dict):
                reviews_dict.append(r)
        content_list = extract_content_list(reviews_dict)
        if not content_list:
            return {
                "restaurant_id": restaurant_id,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_count": len(reviews),
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 0,
            }
        logger.info(f"총 {len(content_list)}개의 리뷰를 sentiment 모델로 분류합니다 (restaurant_id: {restaurant_id}).")
        if Config.SENTIMENT_CLASSIFIER_USE_THREAD:
            hf_result = await asyncio.to_thread(
                self._classify_with_hf_only, content_list, reviews_dict
            )
        else:
            hf_result = self._classify_with_hf_only(content_list, reviews_dict)
        pos, neg, neu, total, labels, neg_for_llm = hf_result
        if neg_for_llm:
            try:
                if Config.SENTIMENT_LLM_ASYNC:
                    pos, neg, neu, labels = await self._apply_llm_reclassify_async(
                        neg_for_llm, labels, pos, neg, neu
                    )
                else:
                    pos, neg, neu, labels = self._apply_llm_reclassify_sync(
                        neg_for_llm, labels, pos, neg, neu
                    )
            except Exception as e:
                logger.warning(f"LLM 재판정 실패: {e}. 1차 분류 결과를 사용합니다.")
        self._assign_labels_to_reviews(reviews_dict, labels)
        total_with_sentiment = pos + neg
        positive_ratio = int(round((pos / total_with_sentiment) * 100)) if total_with_sentiment > 0 else 0
        negative_ratio = int(round((neg / total_with_sentiment) * 100)) if total_with_sentiment > 0 else 0
        neutral_ratio = int(round((neu / total) * 100)) if total > 0 else 0
        return {
            "restaurant_id": restaurant_id,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "total_count": total,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
        }
    
    async def analyze_multiple_restaurants_async(
        self,
        restaurants_data: List[Any],  # List[SentimentRestaurantBatchInput] 또는 [{"restaurant_id", "reviews"}]
    ) -> List[Dict[str, Any]]:
        """
        여러 레스토랑의 리뷰를 sentiment 모델로 분류하여 결과를 반환합니다.
        SENTIMENT_RESTAURANT_ASYNC=true면 음식점 간 asyncio.gather(병렬), false면 순차.
        샘플링이 활성화되어 있으면 대표 벡터 기반 TOP-K를 사용하고,
        비활성화되어 있으면 제공된 reviews를 사용합니다.
        """
        if not restaurants_data:
            return []

        async def _analyze_one(data: Any) -> Dict[str, Any]:
            restaurant_id = data.restaurant_id if hasattr(data, "restaurant_id") else data.get("restaurant_id")
            reviews = data.reviews if hasattr(data, "reviews") else data.get("reviews", [])
            if Config.ENABLE_SENTIMENT_SAMPLING:
                return await self.analyze_async(reviews=None, restaurant_id=restaurant_id)
            return await self.analyze_async(reviews=reviews, restaurant_id=restaurant_id)

        if Config.SENTIMENT_RESTAURANT_ASYNC:
            tasks = [_analyze_one(d) for d in restaurants_data]
            return list(await asyncio.gather(*tasks))

        results: List[Dict[str, Any]] = []
        for data in restaurants_data:
            restaurant_id = data.restaurant_id if hasattr(data, "restaurant_id") else data.get("restaurant_id")
            reviews = data.reviews if hasattr(data, "reviews") else data.get("reviews", [])
            if Config.ENABLE_SENTIMENT_SAMPLING:
                results.append(self.analyze(reviews=None, restaurant_id=restaurant_id))
            else:
                results.append(self.analyze(reviews=reviews, restaurant_id=restaurant_id))
        return results
