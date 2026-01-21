"""
강점 추출 파이프라인 모듈 (새로운 구조화된 방식)
"""

import json
import logging
import math
import re
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .config import Config
from .review_utils import preprocess_reviews, estimate_tokens
from .llm_utils import LLMUtils
from .vector_search import VectorSearch
from qdrant_client import models

logger = logging.getLogger(__name__)

# 일반적인 aspect 필터링 (너무 일반적인 표현은 제거)
GENERIC_ASPECTS = {
    "맛", "맛있다", "좋다", "괜찮다", "추천", "만족", "별로"
}


class StrengthExtractionPipeline:
    """강점 추출 파이프라인 클래스"""
    
    def __init__(
        self,
        llm_utils: LLMUtils,
        vector_search: VectorSearch,
    ):
        """
        Args:
            llm_utils: LLMUtils 인스턴스
            vector_search: VectorSearch 인스턴스
        """
        self.llm_utils = llm_utils
        self.vector_search = vector_search
        # 임베딩 캐시 (메모리 기반)
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    # ==================== Step A: 타겟 긍정 근거 후보 수집 ====================
    
    def collect_positive_evidence_candidates(
        self,
        restaurant_id: int,
        max_candidates: int = 300,
        months_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        대표 벡터 기반 + 다양성 샘플링으로 근거 후보 수집
        
        프로세스:
        1. 레스토랑 대표 벡터 기반 TOP-K 선택 (대표성)
        2. 최근 리뷰 일부 추가 (최신성)
        3. 랜덤 샘플링 일부 추가 (다양성)
        4. 날짜 필터링 (선택)
        
        Args:
            restaurant_id: 레스토랑 ID
            max_candidates: 최대 후보 개수
            months_back: 최근 N개월 필터 (선택, None이면 필터링 안함)
        
        Returns:
            근거 후보 리스트 (중복 제거됨)
        """
        try:
            import random
            
            # 1. 대표 벡터 기반 TOP-K (대표성)
            representative_count = max(20, max_candidates // 3)
            top_k_results = self.vector_search.query_by_restaurant_vector(
                restaurant_id=restaurant_id,
                top_k=representative_count,
                months_back=months_back,
            )
            representative_candidates = [r["payload"] for r in top_k_results]
            
            # 2. 최근 리뷰 일부 추가 (최신성)
            recent_count = max(20, max_candidates // 3)
            # 최근 리뷰는 날짜 기준으로 정렬하여 가져오기
            # query_by_restaurant_vector는 이미 유사도 기준이므로, 별도로 최근 리뷰 조회 필요
            # 간단히 대표 벡터 결과에서 날짜 필터링하여 최근 것 선택
            all_reviews = self.vector_search.get_restaurant_reviews(str(restaurant_id))
            if all_reviews:
                # 날짜 기준 정렬 (최신순)
                sorted_reviews = sorted(
                    all_reviews,
                    key=lambda r: r.get("created_at", ""),
                    reverse=True
                )
                recent_candidates = sorted_reviews[:recent_count]
            else:
                recent_candidates = []
            
            # 3. 랜덤 샘플링 (다양성)
            random_count = max(20, max_candidates // 3)
            if all_reviews and len(all_reviews) > random_count:
                random_candidates = random.sample(all_reviews, random_count)
            else:
                random_candidates = all_reviews if all_reviews else []
            
            # 4. 중복 제거 (review_id 기준)
            seen_ids = set()
            candidates = []
            
            for candidate in representative_candidates + recent_candidates + random_candidates:
                review_id = str(candidate.get("review_id") or candidate.get("id", ""))
                if review_id and review_id not in seen_ids:
                    seen_ids.add(review_id)
                    candidates.append(candidate)
                    if len(candidates) >= max_candidates:
                        break
            
            logger.info(
                f"레스토랑 {restaurant_id}: 대표 {len(representative_candidates)}개 + "
                f"최근 {len(recent_candidates)}개 + 랜덤 {len(random_candidates)}개 → "
                f"총 {len(candidates)}개 후보 수집 (중복 제거 후, max: {max_candidates})"
            )
            
            return candidates
            
        except Exception as e:
            logger.error(f"근거 후보 수집 중 오류: {e}", exc_info=True)
            return []
    
    # ==================== Step B: Recall 단계 (강점 후보 넓게 수집) ====================
    
    def extract_strength_candidates(
        self,
        evidence_candidates: List[Dict[str, Any]],
        max_tokens: int = 4000,
        min_output: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        LLM으로 강점 후보 생성 (Recall 단계: 최소 N개 강제)
        
        Args:
            evidence_candidates: 근거 후보 리스트
            max_tokens: 최대 토큰 수
            min_output: 최소 출력 개수 (기본값: 5)
        
        Returns:
            강점 후보 리스트 (최소 min_output개 보장)
        """
        if not evidence_candidates:
            logger.warning("근거 후보가 없어 강점 후보를 생성할 수 없습니다.")
            return []
        
        # 1. 토큰 제한 고려해 샘플링
        sampled_reviews = self._sample_reviews_by_tokens(
            evidence_candidates,
            max_tokens=max_tokens
        )
        
        # 2. LLM 프롬프트 (Recall 단계: 넓게 수집)
        reviews_text = "\n".join([
            f"[{r.get('review_id', r.get('id', ''))}] {r.get('content', r.get('text', ''))}"
            for r in sampled_reviews
        ])
        
        prompt = f"""다음 리뷰들을 읽고 이 레스토랑의 강점을 aspect 단위로 추출하세요.

리뷰들:
{reviews_text}

**중요 지시사항**:
1. **최소 {min_output}개는 반드시 출력하세요.** 확신이 낮아도 후보로 제시하세요.
2. **확신이 낮으면 confidence를 낮게 표시하세요** (예: "confidence": 0.3)
3. **generic 표현(맛있다/좋다/괜찮다)이라도 일단 후보로 내되, type을 'generic'으로 태깅하세요**
4. Step C에서 support_count로 걸러낼 예정이므로, 가능한 한 넓게 후보를 제시하세요.

각 강점에 대해:
- aspect: 강점의 카테고리 (예: "불맛", "서비스", "주차", "맛", "분위기", "가격")
- claim: 실제 리뷰에서 자주 사용되는 표현 (예: "불맛이 좋다", "맛있다", "서비스가 친절하다")
- type: "specific" (구체적) 또는 "generic" (일반적, 예: 맛있다/좋다/괜찮다)
- confidence: 확신도 (0.0~1.0, 낮으면 낮게 표시)
- evidence_quotes: 해당 강점을 언급한 리뷰 인용문 (최대 3개)
- evidence_review_ids: 해당 리뷰 ID 리스트

JSON 형식 (반드시 최소 {min_output}개 이상 출력):
{{
  "strengths": [
    {{
      "aspect": "불맛",
      "claim": "불맛이 좋다",
      "type": "specific",
      "confidence": 0.9,
      "evidence_quotes": ["숫불향이 진해서 맛있어요", "불맛이 강해서 고기 맛이 살아있어요"],
      "evidence_review_ids": ["rev_1", "rev_5"]
    }},
    {{
      "aspect": "맛",
      "claim": "맛있다",
      "type": "generic",
      "confidence": 0.6,
      "evidence_quotes": ["맛있어요", "맛이 좋아요"],
      "evidence_review_ids": ["rev_2", "rev_3"]
    }},
    ...
  ]
}}
"""
        
        raw_response = None
        try:
            response = self.llm_utils._generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_new_tokens=1000,  # 더 많은 후보를 위해 토큰 증가
            )
            
            # 로그: LLM raw output 앞 300자
            raw_response = response
            logger.info(f"Step B LLM raw output (앞 300자): {raw_response[:300]}")
            
            # JSON 파싱
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:].strip()
            elif response.startswith("```"):
                response = response[3:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()
            
            # JSON 부분만 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response)
            strengths = result.get("strengths", [])
            
            # 로그: 파싱 성공/실패 + 예외 메시지, 파싱 후 item 개수 (필터링 전)
            logger.info(f"Step B 파싱 성공: {len(strengths)}개 강점 후보 (필터링 전)")
            
            # 최소 개수 검증
            if len(strengths) < min_output:
                logger.warning(
                    f"Step B: 최소 {min_output}개를 요청했지만 {len(strengths)}개만 반환됨. "
                    f"재시도하거나 generic 후보를 추가 생성해야 합니다."
                )
                # generic 후보 자동 생성 (최소 개수 보장)
                missing_count = min_output - len(strengths)
                for i in range(missing_count):
                    strengths.append({
                        "aspect": "맛",
                        "claim": "맛있다",
                        "type": "generic",
                        "confidence": 0.3,
                        "evidence_quotes": [],
                        "evidence_review_ids": [],
                    })
                logger.info(f"Step B: generic 후보 {missing_count}개 자동 추가 (최소 개수 보장)")
            
            # 일반적인 aspect는 필터링하지 않음 (Step C에서 support_count로 걸러냄)
            # type이 'generic'인 것도 유지 (Step C에서 검증)
            
            logger.info(f"Step B 완료: {len(strengths)}개 강점 후보 추출 (최소 {min_output}개 보장)")
            return strengths
            
        except json.JSONDecodeError as e:
            logger.error(f"Step B JSON 파싱 실패: {e}")
            logger.error(f"Step B raw output (앞 500자): {raw_response[:500] if raw_response else 'N/A'}")
            # 파싱 실패 시 빈 리스트 반환하지 않고 최소 개수만큼 generic 후보 생성
            logger.warning(f"Step B: 파싱 실패로 인해 generic 후보 {min_output}개 자동 생성")
            return [
                {
                    "aspect": "맛",
                    "claim": "맛있다",
                    "type": "generic",
                    "confidence": 0.1,
                    "evidence_quotes": [],
                    "evidence_review_ids": [],
                }
                for _ in range(min_output)
            ]
        except Exception as e:
            logger.error(f"Step B 강점 후보 생성 중 오류: {e}", exc_info=True)
            logger.error(f"Step B raw output (앞 500자): {raw_response[:500] if raw_response else 'N/A'}")
            # 오류 발생 시에도 최소 개수만큼 generic 후보 생성
            logger.warning(f"Step B: 오류 발생으로 인해 generic 후보 {min_output}개 자동 생성")
            return [
                {
                    "aspect": "맛",
                    "claim": "맛있다",
                    "type": "generic",
                    "confidence": 0.1,
                    "evidence_quotes": [],
                    "evidence_review_ids": [],
                }
                for _ in range(min_output)
            ]
    
    def _sample_reviews_by_tokens(
        self,
        reviews: List[Dict[str, Any]],
        max_tokens: int = 4000,
    ) -> List[Dict[str, Any]]:
        """토큰 제한 고려해 리뷰 샘플링"""
        sampled = []
        total_tokens = 0
        
        for review in reviews:
            text = review.get("content", "") or review.get("text", "")
            tokens = estimate_tokens(text)
            
            if total_tokens + tokens > max_tokens:
                break
            
            sampled.append(review)
            total_tokens += tokens
        
        return sampled
    
    # ==================== Step C: 강점별 근거 확장/검증 ====================
    
    def _calculate_dynamic_min_support(self, total_reviews: int) -> int:
        """
        총 리뷰 수에 따라 min_support 동적 조정
        
        Args:
            total_reviews: 레스토랑의 총 리뷰 수
        
        Returns:
            동적 조정된 min_support 값
        """
        if total_reviews < 20:
            return 2  # 작은 레스토랑
        elif total_reviews < 50:
            return 3  # 중간 레스토랑
        elif total_reviews < 100:
            return 4  # 큰 레스토랑
        else:
            return 5  # 매우 큰 레스토랑 (기본값)
    
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """
        임베딩 캐시에서 조회하거나 새로 생성
        
        Args:
            text: 임베딩할 텍스트
        
        Returns:
            임베딩 벡터
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # 캐시 미스 시 새로 생성
        embedding = self.vector_search.encoder.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = embedding
        return embedding
    
    async def _validate_single_strength(
        self,
        strength: Dict[str, Any],
        restaurant_id: int,
        min_support: int,
        index: int,
        total: int,
    ) -> Optional[Dict[str, Any]]:
        """
        단일 강점 검증 (비동기)
        
        Args:
            strength: 검증할 강점 후보
            restaurant_id: 레스토랑 ID
            min_support: 최소 support_count
            index: 현재 인덱스 (로깅용)
            total: 전체 개수 (로깅용)
        
        Returns:
            검증된 강점 또는 None
        """
        aspect = strength.get("aspect", "")
        claim = strength.get("claim", "")
        strength_type = strength.get("type", "specific")
        
        logger.info(f"강점 후보 {index}/{total}: aspect='{aspect}', claim='{claim[:50] if claim else 'N/A'}...', type='{strength_type}'")
        
        if not aspect or not claim:
            logger.warning(f"강점 후보 {index}: aspect 또는 claim이 비어있어 스킵")
            return None
        
        # 1. 쿼리 문장 생성
        claim_words = claim.split()
        is_simple_claim = (
            len(claim_words) <= 3 and 
            any(word in claim for word in ["좋다", "맛있다", "친절하다", "깨끗하다", "편하다", "넓다", "괜찮다", "만족"])
        )
        
        if is_simple_claim:
            query_text = f"{aspect} {claim}"
        else:
            query_text = f"{aspect} 좋다"
        
        try:
            # 2. Qdrant 검색
            logger.info(f"강점 '{aspect}' 검증: Qdrant 검색 시작 (restaurant_id: {restaurant_id}, query: {query_text[:50]}...)")
            
            search_results = self.vector_search.query_similar_reviews(
                query_text=query_text,
                restaurant_id=restaurant_id,
                limit=50,
                min_score=0.0,
            )
            
            # 3. Support 계산
            support_count_raw = len(search_results)
            score_threshold = 0.3
            valid_results = [
                r for r in search_results 
                if r.get("score", 0) >= score_threshold
            ]
            support_count_valid = len(valid_results)
            
            # 4. 긍정 리뷰 필터링
            negative_keywords = ["별로", "불친절", "최악", "비추", "안가", "싫다", "나쁘다", "더럽다", "불만", "별점"]
            
            filtered_results = []
            for r in valid_results:
                review = r["payload"]
                content = review.get("content", review.get("text", "")).lower()
                
                sentiment = review.get("sentiment") or review.get("is_recommended")
                if sentiment is not None:
                    if sentiment == "positive" or sentiment is True or (isinstance(sentiment, str) and "positive" in sentiment.lower()):
                        filtered_results.append(r)
                    continue
                
                has_negative = any(keyword in content for keyword in negative_keywords)
                if not has_negative:
                    filtered_results.append(r)
            
            evidence_reviews = [r["payload"] for r in filtered_results]
            support_count = len(evidence_reviews)
            
            logger.info(
                f"강점 '{aspect}' 검증: Qdrant 검색 결과 "
                f"raw={support_count_raw}개, valid(score>={score_threshold})={support_count_valid}개, "
                f"positive_filtered={support_count}개 (min_support: {min_support})"
            )
            
            if support_count < min_support:
                logger.warning(
                    f"강점 '{aspect}': support_count {support_count} < {min_support}, 버림 "
                    f"(raw: {support_count_raw}, valid: {support_count_valid})"
                )
                return None
            
            # 5. 일관성 체크 (임베딩 캐싱 사용)
            if len(evidence_reviews) > 0:
                embeddings = []
                for r in evidence_reviews:
                    text = r.get("content", r.get("text", ""))
                    if text:
                        embedding = self._get_cached_embedding(text)
                        embeddings.append(embedding.tolist())
                
                consistency = self._calculate_consistency(embeddings)
            else:
                consistency = 1.0
            
            consistency_value = float(consistency) if isinstance(consistency, (int, float)) else 0.0
            
            if len(evidence_reviews) > 0:
                logger.info(
                    f"강점 '{aspect}': consistency {consistency_value:.2f} (임계값: 0.3)"
                )
                
                if consistency_value < 0.25:
                    logger.warning(
                        f"강점 '{aspect}': consistency {consistency_value:.2f} < 0.25, 버림"
                    )
                    return None
                else:
                    logger.info(f"강점 '{aspect}': consistency 통과 ({consistency_value:.2f})")
            
            # 6. Recency 가중치
            recency = self._calculate_recency_weight(evidence_reviews)
            
            # 7. Support ratio 계산
            total_reviews = len(evidence_reviews)
            support_ratio = support_count / total_reviews if total_reviews > 0 else 0
            
            # 8. Evidence review IDs 추출
            evidence_review_ids = [
                str(r.get("review_id") or r.get("id", ""))
                for r in evidence_reviews
            ]
            
            recency_value = float(recency) if isinstance(recency, (int, float)) else 0.0
            logger.info(
                f"강점 '{aspect}': 검증 통과! "
                f"(support: {support_count}, raw: {support_count_raw}, valid: {support_count_valid}, "
                f"consistency: {consistency_value:.2f}, recency: {recency_value:.2f})"
            )
            
            return {
                **strength,
                "support_count": support_count,
                "support_count_raw": support_count_raw,
                "support_count_valid": support_count_valid,
                "support_ratio": support_ratio,
                "consistency": consistency_value,
                "recency": recency_value,
                "evidence_reviews": evidence_reviews[:10],
                "evidence_review_ids": evidence_review_ids,
            }
            
        except Exception as e:
            logger.error(f"강점 '{aspect}' 검증 중 오류: {e}")
            return None
    
    async def expand_and_validate_evidence(
        self,
        strength_candidates: List[Dict[str, Any]],
        restaurant_id: int,
        min_support: int = 5,
        total_reviews: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        각 aspect에 대해 Qdrant 벡터 검색으로 근거 확장 및 검증 (비동기 병렬 처리)
        
        Args:
            strength_candidates: 강점 후보 리스트
            restaurant_id: 레스토랑 ID
            min_support: 최소 support_count (동적 조정됨)
            total_reviews: 레스토랑의 총 리뷰 수 (동적 min_support 계산용)
        
        Returns:
            검증된 강점 리스트
        """
        # 동적 min_support 조정
        if total_reviews is not None:
            adjusted_min_support = self._calculate_dynamic_min_support(total_reviews)
            if adjusted_min_support != min_support:
                logger.info(
                    f"min_support 동적 조정: {min_support} → {adjusted_min_support} "
                    f"(총 리뷰 수: {total_reviews})"
                )
                min_support = adjusted_min_support
        
        logger.info(f"Step C: {len(strength_candidates)}개 강점 후보 검증 시작 (min_support: {min_support}, 병렬 처리)")
        
        # 병렬 처리: 모든 강점을 동시에 검증
        tasks = [
            self._validate_single_strength(
                strength=strength,
                restaurant_id=restaurant_id,
                min_support=min_support,
                index=i + 1,
                total=len(strength_candidates),
            )
            for i, strength in enumerate(strength_candidates)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # None 제거
        validated_strengths = [r for r in results if r is not None]
        
        logger.info(f"{len(validated_strengths)}개 강점 검증 통과 (후보: {len(strength_candidates)}개)")
        
        # 임베딩 캐시 정리 (메모리 절약)
        if len(self._embedding_cache) > 1000:
            logger.info(f"임베딩 캐시 정리: {len(self._embedding_cache)}개 → 500개")
            # 최근 사용한 것만 유지 (간단한 전략: 절반만 유지)
            keys_to_keep = list(self._embedding_cache.keys())[:500]
            self._embedding_cache = {k: self._embedding_cache[k] for k in keys_to_keep}
        
        return validated_strengths
    
    def _calculate_consistency(self, embeddings: List[List[float]]) -> float:
        """임베딩 분산 계산 (일관성)"""
        if len(embeddings) < 2:
            return 1.0
        
        embeddings_array = np.array(embeddings)
        cluster_center = np.mean(embeddings_array, axis=0)
        
        distances = [
            np.linalg.norm(emb - cluster_center)
            for emb in embeddings_array
        ]
        
        # 표준편차가 낮을수록 일관성 높음
        std_dev = np.std(distances)
        consistency = 1.0 / (1.0 + std_dev)  # 0~1 범위로 정규화
        
        return float(consistency)
    
    def _calculate_recency_weight(self, reviews: List[Dict[str, Any]]) -> float:
        """최근 가중치 계산"""
        now = datetime.now()
        recency_scores = []
        
        for review in reviews:
            created_at = review.get("created_at")
            if not created_at:
                recency_scores.append(0.5)  # 기본값
                continue
            
            try:
                if isinstance(created_at, str):
                    review_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    review_date = created_at
                
                days_ago = (now - review_date).days
                
                # 최근 30일 내면 1.0, 90일이면 0.5, 그 이상이면 0.1
                if days_ago <= 30:
                    recency_scores.append(1.0)
                elif days_ago <= 90:
                    recency_scores.append(0.5)
                else:
                    recency_scores.append(0.1)
            except Exception:
                recency_scores.append(0.5)
        
        return float(np.mean(recency_scores)) if recency_scores else 0.5
    
    # ==================== Step D: 의미 중복 제거 (Connected Components) ====================
    
    def _union_find(self, n: int, edges: List[Tuple[int, int]]) -> List[int]:
        """
        Union-Find 알고리즘으로 Connected Components 찾기
        
        Args:
            n: 노드 개수
            edges: 간선 리스트 [(i, j), ...]
        
        Returns:
            각 노드의 루트 노드 ID 리스트
        """
        parent = list(range(n))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x: int, y: int):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        
        for i, j in edges:
            union(i, j)
        
        # 모든 노드의 최종 루트 찾기
        for i in range(n):
            find(i)
        
        return parent
    
    def _calculate_evidence_overlap(
        self,
        evidence_ids_1: List[str],
        evidence_ids_2: List[str],
    ) -> float:
        """
        두 aspect의 evidence 리뷰 ID 간 overlap 비율 계산
        
        Args:
            evidence_ids_1: 첫 번째 aspect의 evidence 리뷰 ID 리스트
            evidence_ids_2: 두 번째 aspect의 evidence 리뷰 ID 리스트
        
        Returns:
            Overlap 비율 (0.0 ~ 1.0)
        """
        if not evidence_ids_1 or not evidence_ids_2:
            return 0.0
        
        set1 = set(str(id) for id in evidence_ids_1)
        set2 = set(str(id) for id in evidence_ids_2)
        
        intersection = len(set1 & set2)
        union_size = len(set1 | set2)
        
        if union_size == 0:
            return 0.0
        
        return intersection / union_size
    
    def _compute_evidence_centroid(
        self,
        evidence_reviews: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Evidence 리뷰들의 임베딩 centroid 계산 (임베딩 캐싱 사용)
        
        Args:
            evidence_reviews: Evidence 리뷰 리스트
        
        Returns:
            Centroid 벡터 또는 None
        """
        if not evidence_reviews:
            return None
        
        embeddings = []
        for review in evidence_reviews:
            text = review.get("content", "") or review.get("text", "")
            if text:
                emb = self._get_cached_embedding(text)
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def merge_similar_strengths(
        self,
        validated_strengths: List[Dict[str, Any]],
        threshold_high: float = 0.88,
        threshold_low: float = 0.82,
        evidence_overlap_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Connected Components (Union-Find) 방식으로 유사한 aspect 병합
        
        가드레일:
        - 가드레일 A) 이중 임계값: T_high (즉시 union), T_low (evidence overlap 체크 후 union)
        - 가드레일 B) evidence overlap: 일정 비율 이상 겹치면 merge 허용
        
        Args:
            validated_strengths: 검증된 강점 리스트
            threshold_high: 높은 임계값 (즉시 union)
            threshold_low: 낮은 임계값 (evidence overlap 체크 후 union)
            evidence_overlap_threshold: Evidence overlap 최소 비율
        
        Returns:
            병합된 강점 리스트
        """
        if len(validated_strengths) < 2:
            return validated_strengths
        
        n = len(validated_strengths)
        
        # 1. 각 강점의 대표 벡터 생성 (근거 리뷰 벡터들의 centroid)
        # 제시된 로직: "그 강점의 근거 리뷰 벡터들의 centroid"
        representative_vectors = []
        for strength in validated_strengths:
            evidence_reviews = strength.get("evidence_reviews", [])
            if evidence_reviews:
                # Evidence centroid 계산
                centroid = self._compute_evidence_centroid(evidence_reviews)
                if centroid is not None:
                    representative_vectors.append(centroid)
                else:
                    # Fallback: aspect 텍스트 임베딩
                    aspect_text = strength.get("aspect", "")
                    representative_vectors.append(
                        self._get_cached_embedding(aspect_text) if aspect_text else self._get_cached_embedding("")
                    )
            else:
                # Fallback: aspect 텍스트 임베딩
                aspect_text = strength.get("aspect", "")
                representative_vectors.append(
                    self._get_cached_embedding(aspect_text) if aspect_text else self._get_cached_embedding("")
                )
        
        # 2. 유사도 그래프 만들기 (배치 벡터 연산 + aspect별 그룹화)
        # - 개선 1) aspect가 같을 때만 비교 (기존 로직 유지)
        # - 개선 2) threshold_high 이상이면 즉시 union
        # - 개선 3) 같은 aspect 내 유사도는 행렬곱으로 배치 계산
        edges: List[Tuple[int, int]] = []

        # aspect별 인덱스 그룹화
        aspect_to_indices: Dict[str, List[int]] = {}
        for idx, s in enumerate(validated_strengths):
            aspect = (s.get("aspect", "") or "").strip()
            aspect_to_indices.setdefault(aspect, []).append(idx)

        for aspect, indices in aspect_to_indices.items():
            if len(indices) < 2:
                continue

            # (m, d) 형태로 스택 후 row-wise 정규화
            vecs = np.stack([representative_vectors[i] for i in indices], axis=0).astype(np.float32, copy=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            # zero-norm 방지
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms

            # cosine similarity matrix (m, m)
            sim_matrix = vecs @ vecs.T

            # upper triangle만 사용 (i < j)
            tri_r, tri_c = np.triu_indices(len(indices), k=1)
            sims = sim_matrix[tri_r, tri_c]

            # threshold_high: 즉시 union
            high_mask = sims >= threshold_high
            if np.any(high_mask):
                for r, c in zip(tri_r[high_mask], tri_c[high_mask]):
                    edges.append((indices[int(r)], indices[int(c)]))

            # threshold_low: evidence overlap 체크 후 union
            low_mask = (sims >= threshold_low) & (~high_mask)
            if np.any(low_mask):
                for r, c in zip(tri_r[low_mask], tri_c[low_mask]):
                    i = indices[int(r)]
                    j = indices[int(c)]
                    evidence_ids_1 = validated_strengths[i].get("evidence_review_ids", [])
                    evidence_ids_2 = validated_strengths[j].get("evidence_review_ids", [])
                    overlap = self._calculate_evidence_overlap(evidence_ids_1, evidence_ids_2)
                    if overlap >= evidence_overlap_threshold:
                        edges.append((i, j))
        
        # 3. Connected Components로 그룹 생성 (Union-Find)
        parent = self._union_find(n, edges)
        
        # 4. 클러스터별로 그룹화
        clusters = {}
        for i, root in enumerate(parent):
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        
        # 5. 클러스터별 병합
        merged_strengths = []
        merged_count = 0  # 병합된 강점 수 추적
        
        for cluster_id, indices in clusters.items():
            cluster_strengths = [validated_strengths[i] for i in indices]
            
            if len(cluster_strengths) < 2:
                # 병합되지 않은 단일 강점
                strength = cluster_strengths[0]
                aspect = strength.get("aspect", "")
                claim = strength.get("claim", "")[:50] + "..." if len(strength.get("claim", "")) > 50 else strength.get("claim", "")
                logger.debug(f"  병합 안됨 (단일): aspect='{aspect}', claim='{claim}'")
                merged_strengths.extend(cluster_strengths)
                continue
            
            # 대표 aspect 선정 (support_count가 가장 큰 것)
            representative_strength = max(
                cluster_strengths,
                key=lambda s: s.get("support_count", 0)
            )
            representative_aspect = representative_strength.get("aspect", "")
            
            # 병합된 강점들 로깅
            merged_aspects = [s.get("aspect", "") for s in cluster_strengths]
            merged_claims = [s.get("claim", "")[:30] + "..." if len(s.get("claim", "")) > 30 else s.get("claim", "") for s in cluster_strengths]
            logger.info(
                f"  병합됨 ({len(cluster_strengths)}개 → 1개): "
                f"aspects={merged_aspects}, "
                f"대표 aspect='{representative_aspect}'"
            )
            merged_count += len(cluster_strengths) - 1  # 병합으로 제거된 강점 수
            
            # Evidence 합치기 (중복 제거)
            all_evidence_ids = set()
            all_evidence_reviews = []
            for s in cluster_strengths:
                evidence_ids = s.get("evidence_review_ids", [])
                if isinstance(evidence_ids, list):
                    all_evidence_ids.update(str(id) for id in evidence_ids)
                all_evidence_reviews.extend(s.get("evidence_reviews", []))
            
            # 중복 제거 (review_id 기준)
            unique_evidence_reviews = {}
            for r in all_evidence_reviews:
                review_id = str(r.get("review_id") or r.get("id", ""))
                if review_id and review_id not in unique_evidence_reviews:
                    unique_evidence_reviews[review_id] = r
            
            # Evidence centroid 재계산 (병합된 evidence 기반)
            evidence_centroid = self._compute_evidence_centroid(
                list(unique_evidence_reviews.values())
            )
            
            # Support 통계 합치기
            total_support = sum(s.get("support_count", 0) for s in cluster_strengths)
            avg_support_ratio = np.mean([s.get("support_ratio", 0) for s in cluster_strengths])
            avg_consistency = np.mean([s.get("consistency", 0) for s in cluster_strengths])
            avg_recency = np.mean([s.get("recency", 0) for s in cluster_strengths])
            
            merged_strengths.append({
                "aspect": representative_aspect,
                "claim": representative_strength.get("claim", ""),
                "support_count": total_support,
                "support_ratio": float(avg_support_ratio),
                "consistency": float(avg_consistency),
                "recency": float(avg_recency),
                "evidence_review_ids": list(all_evidence_ids),
                "evidence_reviews": list(unique_evidence_reviews.values())[:10],
                "evidence_centroid": evidence_centroid.tolist() if evidence_centroid is not None else None,
            })
        
        logger.info(
            f"Step D 완료: {len(validated_strengths)}개 강점 → {len(merged_strengths)}개로 병합 "
            f"(병합으로 제거된 강점: {merged_count}개, "
            f"threshold_high={threshold_high}, threshold_low={threshold_low})"
        )
        
        # 병합 후 각 강점 상세 로깅
        for idx, strength in enumerate(merged_strengths, 1):
            aspect = strength.get("aspect", "")
            claim = strength.get("claim", "")[:50] + "..." if len(strength.get("claim", "")) > 50 else strength.get("claim", "")
            support_count = strength.get("support_count", 0)
            logger.info(
                f"  병합 후 강점 {idx}: aspect='{aspect}', claim='{claim}', support_count={support_count}"
            )
        return merged_strengths
    
    def regenerate_claims(
        self,
        strengths: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Step D 이후 Claim 후처리 재생성 (LLM 1회)
        
        제약 조건:
        - 20자~35자
        - 이모지/감탄사 금지
        - "맛있다/좋다/추천" 단독 금지
        
        Args:
            strengths: 병합된 강점 리스트
        
        Returns:
            claim이 재생성된 강점 리스트
        """
        if not strengths:
            return strengths
        
        # LLMUtils 인스턴스 생성 (필요시)
        if not hasattr(self, 'llm_utils'):
            self.llm_utils = LLMUtils()
        
        # 각 강점에 대해 claim 재생성
        regenerated_strengths = []
        for strength in strengths:
            aspect = strength.get("aspect", "")
            original_claim = strength.get("claim", "")
            
            # (1) 템플릿 기반 보정 먼저 시도 (LLM 없이, evidence_reviews 없어도 가능)
            if original_claim:
                corrected_claim = self._apply_claim_template(original_claim, aspect)
                logger.debug(f"강점 '{aspect}': 템플릿 보정 시도 - '{original_claim}' → '{corrected_claim}'")
                if corrected_claim and corrected_claim != original_claim:
                    # 템플릿 보정 성공
                    if self._validate_claim(corrected_claim):
                        strength["claim"] = corrected_claim
                        logger.info(f"강점 '{aspect}': 템플릿 기반 claim 보정 성공 - '{original_claim}' → '{corrected_claim}'")
                        regenerated_strengths.append(strength)
                        continue
                    else:
                        logger.warning(f"강점 '{aspect}': 템플릿 보정 결과가 제약 조건 위반 (길이: {len(corrected_claim)}, 범위: 15-28자) - '{corrected_claim}'")
                elif corrected_claim == original_claim:
                    logger.debug(f"강점 '{aspect}': 템플릿 매칭 실패 - '{original_claim}'")
            
            # (2) LLM 기반 생성 (템플릿 보정 실패 시 또는 evidence_reviews가 있는 경우)
            evidence_reviews = strength.get("evidence_reviews", [])[:5]  # 대표 근거 3~5개
            
            if not evidence_reviews:
                # 근거가 없으면 템플릿 보정 결과 또는 기존 claim 유지
                regenerated_strengths.append(strength)
                continue
            
            # 근거 리뷰 텍스트 추출
            evidence_texts = [
                review.get("content", review.get("text", ""))
                for review in evidence_reviews
            ]
            evidence_texts = [t for t in evidence_texts if t]  # 빈 텍스트 제거
            
            if not evidence_texts:
                regenerated_strengths.append(strength)
                continue
            
            # LLM으로 claim 재생성
            try:
                new_claim = self._generate_claim_from_evidence(
                    aspect=aspect,
                    evidence_texts=evidence_texts,
                    original_claim=original_claim,
                )
                
                # 제약 조건 검증
                if self._validate_claim(new_claim):
                    strength["claim"] = new_claim
                    logger.info(f"강점 '{aspect}': claim 재생성 성공 - '{new_claim}'")
                else:
                    logger.warning(f"강점 '{aspect}': claim 재생성 실패 (제약 조건 위반) - '{new_claim}', 기존 claim 유지")
                    # 기존 claim 유지
                
            except Exception as e:
                logger.error(f"강점 '{aspect}': claim 재생성 중 오류 - {e}, 기존 claim 유지")
                # 기존 claim 유지
            
            regenerated_strengths.append(strength)
        
        return regenerated_strengths
    
    def _generate_claim_from_evidence(
        self,
        aspect: str,
        evidence_texts: List[str],
        original_claim: Optional[str] = None,
    ) -> str:
        """
        근거 리뷰로부터 자연스러운 claim 생성 (템플릿 기반 보정 + LLM)
        
        Args:
            aspect: 강점 카테고리
            evidence_texts: 근거 리뷰 텍스트 리스트
            original_claim: 원본 claim (템플릿 보정용)
        
        Returns:
            재생성된 claim (15-28자)
        """
        # (1) 템플릿 기반 보정 (LLM 없이도 가능)
        if original_claim:
            corrected_claim = self._apply_claim_template(original_claim, aspect)
            if corrected_claim and self._validate_claim(corrected_claim):
                logger.info(f"강점 '{aspect}': 템플릿 기반 claim 보정 성공 - '{original_claim}' → '{corrected_claim}'")
                return corrected_claim
        
        # (2) LLM 기반 생성 (템플릿 보정 실패 시)
        evidence_snippets = "\n".join([f"- {text[:100]}" for text in evidence_texts[:5]])
        
        prompt = f"""다음 근거 리뷰들을 바탕으로 "{aspect}"에 대한 자연스러운 1문장 claim을 생성하세요.

근거 리뷰:
{evidence_snippets}

요구사항:
1. 15자~28자 사이 (모바일 카드 1줄 기준)
2. 이모지, 감탄사(와!, 대박!, 최고!) 금지
3. "맛있다", "좋다", "추천" 단독 사용 금지 (예: "맛있다" ❌, "맛이 좋다" ✅)
4. 자연스러운 서술형 문장, 메타 표현 통일: "언급이 많음" 사용 (예: "가격 대비 양이 넉넉하다는 언급이 많음", "응대가 빠르고 친절하다는 언급이 많음")
5. 맛 관련 claim은 구체명사 1개 포함 필수 (예: "국물", "면", "유자라멘", "디저트", "커피", "육즙", "불맛" 등)
6. JSON 형식으로 반환: {{"claim": "생성된 claim"}}

예시:
- 가격: "가격 대비 양이 넉넉하다는 언급이 많음"
- 서비스: "응대가 빠르고 친절하다는 언급이 많음"
- 메뉴 다양성: "메뉴 선택지가 넓어 취향대로 고르기 좋음"
- 맛: "유자라멘 국물이 진하다는 언급이 많음" (구체명사 "유자라멘", "국물" 포함)

claim만 반환하세요 (JSON 형식):"""

        try:
            # _generate_response는 messages 리스트를 받음
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.llm_utils._generate_response(
                messages=messages,
                temperature=0.3,
                max_new_tokens=100,
            )
            
            # JSON 파싱
            import json
            # JSON 부분만 추출
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                claim = result.get("claim", "").strip()
            else:
                # JSON이 없으면 직접 추출
                claim = response.strip()
                # 따옴표 제거
                claim = claim.strip('"').strip("'")
            
            # LLM 생성 결과도 템플릿 보정 적용
            if claim:
                corrected_claim = self._apply_claim_template(claim, aspect)
                if corrected_claim and self._validate_claim(corrected_claim):
                    return corrected_claim
                elif self._validate_claim(claim):
                    return claim
            
            # 모두 실패 시 기본 claim 반환
            return f"{aspect}에 대한 긍정적 평가가 많음"
            
        except Exception as e:
            logger.error(f"Claim 생성 중 오류: {e}")
            # 기본 claim 반환
            return f"{aspect}에 대한 긍정적 평가가 많음"
    
    def _apply_claim_template(self, claim: str, aspect: str) -> Optional[str]:
        """
        템플릿 기반 claim 보정 (너무 짧거나 추상적인 claim 보정)
        
        Args:
            claim: 원본 claim
            aspect: 강점 카테고리
        
        Returns:
            보정된 claim 또는 None
        """
        if not claim:
            return None
        
        claim_lower = claim.lower()
        
        # 템플릿 매핑 (15-28자 범위 준수, 메타 표현 통일: "언급이 많음"으로 통일)
        template_map = {
            # 너무 짧은 claim (맛 관련은 구체명사 포함하도록 나중에 LLM으로 처리)
            "맛있다": "맛에 대한 만족도가 높다는 언급이 많음",  # 구체명사는 LLM에서 추가
            "맛이 좋다": f"{aspect}에 대한 만족도가 높다는 언급이 많음",
            "좋다": f"{aspect}에 대한 평가가 좋다는 언급이 많음",
            "괜찮다": f"{aspect}에 대한 평가가 괜찮다는 언급이 많음",
            "추천": f"{aspect}에 대한 추천이 많다는 언급이 많음",
            "최고": f"{aspect}에 대한 평가가 최고라는 언급이 많음",
            
            # 가격 관련
            "좋은 가격": "가격이 합리적이라는 언급이 많음",
            "가격이 좋다": "가격이 합리적이라는 언급이 많음",
            "저렴하다": "가격이 저렴하다는 언급이 많음",
            "가성비": "가성비가 좋다는 언급이 많음",
            
            # 메뉴 관련
            "메뉴가 다양하다": "메뉴 선택지가 다양해 고르기 좋다는 언급이 많음",
            "메뉴 다양": "메뉴 선택지가 다양해 고르기 좋다는 언급이 많음",
            "다양한 메뉴": "메뉴 선택지가 다양해 고르기 좋다는 언급이 많음",
            
            # 서비스 관련
            "서비스가 좋다": "서비스에 대한 만족도가 높다는 언급이 많음",
            "친절하다": "서비스가 친절하다는 언급이 많음",
            "응대가 빠르다": "응대가 빠르고 친절하다는 언급이 많음",
            
            # 분위기 관련
            "분위기가 좋다": "분위기가 좋다는 언급이 많음",
            "깨끗하다": "환경이 깨끗하다는 언급이 많음",
            "편하다": "이용하기 편하다는 언급이 많음",
        }
        
        # 정확히 일치하는 경우
        if claim in template_map:
            return template_map[claim]
        
        # 부분 일치하는 경우 (대소문자 무시)
        # 정확히 일치하거나 공백으로 구분된 단어로 포함되어 있는지 확인
        for key, template in template_map.items():
            key_lower = key.lower()
            # 정확히 일치하거나 공백으로 구분된 단어로 포함되어 있는지 확인
            if (key_lower == claim_lower or 
                f" {key_lower} " in f" {claim_lower} " or 
                claim_lower.startswith(key_lower + " ") or 
                claim_lower.endswith(" " + key_lower)):
                # aspect가 있으면 aspect를 포함한 템플릿 사용
                if "{aspect}" in template:
                    return template.format(aspect=aspect)
                else:
                    return template
        
        # 템플릿 매칭 실패 시 원본 반환
        return claim
    
    def _validate_claim(self, claim: str) -> bool:
        """
        Claim 제약 조건 검증
        
        Args:
            claim: 검증할 claim
        
        Returns:
            True if valid, False otherwise
        """
        if not claim:
            return False
        
        # 1. 길이 체크 (15-28자, 모바일 카드 1줄 기준)
        if len(claim) < 15 or len(claim) > 28:
            return False
        
        # 2. 이모지/감탄사 체크
        # 이모지 패턴 (더 정확한 범위)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002600-\U000027BF"  # miscellaneous symbols
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "]+",
            flags=re.UNICODE
        )
        # 이모지가 실제로 포함되어 있는지 확인 (한글 문자는 제외)
        emoji_match = emoji_pattern.search(claim)
        if emoji_match:
            # 한글 범위 확인 (가-힣)
            korean_only = re.sub(r'[가-힣\s.,!?]', '', claim)
            if korean_only:  # 한글이 아닌 문자가 남아있으면 이모지로 간주
                return False
        
        exclamation_words = ["와!", "대박!", "최고!", "완전!", "진짜!"]
        if any(word in claim for word in exclamation_words):
            return False
        
        # 3. "맛있다/좋다/추천" 단독 사용 금지
        forbidden_standalone = ["맛있다", "좋다", "추천"]
        # 단독으로 사용된 경우 체크 (앞뒤로 공백이나 문장 끝)
        for word in forbidden_standalone:
            pattern = rf'^\s*{word}\s*[.!?]?\s*$'
            if re.match(pattern, claim):
                return False
        
        return True
    
    # ==================== Step E~H: 비교군 기반 차별 강점 계산 ====================
    
    def calculate_distinct_strengths(
        self,
        target_strengths: List[Dict[str, Any]],
        restaurant_id: int,
        category_filter: Optional[int] = None,
        region_filter: Optional[str] = None,
        price_band_filter: Optional[str] = None,
        comparison_count: int = 20,
        alpha: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        비교군 기반 차별 강점 계산
        
        Args:
            target_strengths: 타겟 강점 리스트
            restaurant_id: 타겟 레스토랑 ID
            category_filter: 카테고리 필터
            region_filter: 지역 필터
            price_band_filter: 가격대 필터
            comparison_count: 비교군 개수
            alpha: 차별성 가중치
        
        Returns:
            차별 강점 리스트
        """
        # Step E: 비교군 구성
        comparison_restaurants = self._find_comparison_restaurants(
            restaurant_id,
            category_filter,
            region_filter,
            price_band_filter,
            comparison_count,
        )
        
        if not comparison_restaurants:
            logger.warning(
                f"비교군을 찾을 수 없습니다 (restaurant_id: {restaurant_id}, "
                f"category_filter: {category_filter}, region_filter: {region_filter}, "
                f"price_band_filter: {price_band_filter}). distinct 강점을 계산할 수 없습니다."
            )
            return []  # 빈 리스트 반환 (distinct 계산 불가)
        
        # Step F: 비교군 강점 인덱스
        # 제시된 로직: "비교군도 음식점별 강점 프로필이 있어야 함(배치로 미리 만들어두는 게 좋음)"
        # TODO: 배치 작업으로 비교군 강점 프로필을 미리 계산해 캐시에 저장
        # 현재는 실시간 계산 (성능 최적화 필요)
        comparison_strength_profiles = []
        for comp_restaurant_id in comparison_restaurants[:5]:  # 성능을 위해 5개만
            try:
                # 간단한 대표 벡터 기반 비교
                # TODO: 실제로는 각 비교군의 aspect-level 강점 프로필을 사용해야 함
                comp_vector = self.vector_search.compute_restaurant_vector(comp_restaurant_id)
                if comp_vector is not None:
                    comparison_strength_profiles.append({
                        "restaurant_id": comp_restaurant_id,
                        "vector": comp_vector,
                    })
            except Exception as e:
                logger.warning(f"비교군 {comp_restaurant_id} 벡터 계산 실패: {e}")
                continue
        
        # Step G: 타겟 aspect vs 비교군 aspect 유사도
        # 제시된 로직: "타겟 강점(클러스터)마다: 비교군 강점 벡터들과 max_sim 계산"
        distinct_strengths = []
        for target_strength in target_strengths:
            aspect = target_strength.get("aspect", "")
            if not aspect:
                continue
            
            # 타겟 강점의 대표 벡터 사용 (evidence centroid, Step 4에서 계산됨)
            # Fallback: evidence_centroid가 없으면 aspect 텍스트 임베딩 사용
            evidence_centroid = target_strength.get("evidence_centroid")
            if evidence_centroid is not None:
                target_vector = np.array(evidence_centroid)
            else:
                # Fallback: aspect 텍스트 임베딩
                target_vector = np.array(
                    self.vector_search.encoder.encode(aspect, convert_to_numpy=True)
                )
            
            # 비교군 전체 aspect 벡터 풀에서 max_sim 구하기
            # TODO: 실제로는 비교군의 aspect-level 강점 벡터들과 비교해야 함
            # 현재는 레스토랑 대표 벡터와 비교 (간소화된 구현)
            max_sim = 0.0
            closest_competitor_id = None
            
            for comp_profile in comparison_strength_profiles:
                comp_vector = np.array(comp_profile["vector"])
                similarity = np.dot(target_vector, comp_vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(comp_vector)
                )
                
                if similarity > max_sim:
                    max_sim = similarity
                    closest_competitor_id = comp_profile["restaurant_id"]
            
            # Distinct 계산
            distinct = 1.0 - max_sim
            
            # Step H: 최종 점수
            # 제시된 로직: rep_score = log(1 + support_count) * consistency * recency_weight
            rep_score = (
                math.log(1 + target_strength.get("support_count", 0)) *
                target_strength.get("consistency", 1.0) *
                target_strength.get("recency", 1.0)
            )
            final_score = rep_score * (1 + alpha * distinct)
            
            distinct_strengths.append({
                **target_strength,
                "distinct_score": round(distinct, 3),
                "closest_competitor_sim": round(max_sim, 3),
                "closest_competitor_id": closest_competitor_id,
                "final_score": round(final_score, 3),
            })
        
        return distinct_strengths
    
    def _find_comparison_restaurants(
        self,
        restaurant_id: int,
        category_filter: Optional[int],
        region_filter: Optional[str],
        price_band_filter: Optional[str],
        comparison_count: int,
    ) -> List[int]:
        """비교군 레스토랑 찾기"""
        try:
            logger.info(
                f"비교군 검색 시작: restaurant_id={restaurant_id}, "
                f"category_filter={category_filter}, comparison_count={comparison_count}"
            )
            
            # 대표 벡터 기반 유사 레스토랑 검색
            similar_restaurants = self.vector_search.find_similar_restaurants(
                target_restaurant_id=restaurant_id,
                top_n=comparison_count,
                food_category_id=category_filter,
                exclude_self=True,
            )
            
            logger.info(f"비교군 검색 결과: {len(similar_restaurants)}개 레스토랑 발견")
            
            if not similar_restaurants:
                logger.warning(
                    f"비교군을 찾을 수 없습니다. "
                    f"restaurant_vectors 컬렉션이 없거나, "
                    f"유사한 레스토랑이 없거나, "
                    f"필터 조건(category_filter={category_filter})이 너무 엄격할 수 있습니다."
                )
            
            return [r["restaurant_id"] for r in similar_restaurants]
            
        except Exception as e:
            logger.error(f"비교군 찾기 중 오류: {e}", exc_info=True)
            return []
    
    # ==================== 전체 파이프라인 실행 ====================
    
    async def extract_strengths(
        self,
        restaurant_id: int,
        strength_type: str = "both",
        category_filter: Optional[int] = None,
        region_filter: Optional[str] = None,
        price_band_filter: Optional[str] = None,
        top_k: int = 10,
        max_candidates: int = 300,
        months_back: int = 6,
        min_support: int = 5,
    ) -> Dict[str, Any]:
        """
        전체 강점 추출 파이프라인 실행
        
        Args:
            restaurant_id: 레스토랑 ID
            strength_type: 강점 타입 ('representative', 'distinct', 'both')
            category_filter: 카테고리 필터
            region_filter: 지역 필터
            price_band_filter: 가격대 필터
            top_k: 반환할 최대 강점 개수
            max_candidates: 근거 후보 최대 개수
            months_back: 최근 N개월 리뷰만 사용
            min_support: 최소 support_count
        
        Returns:
            강점 추출 결과
        """
        start_time = time.time()
        
        # Step A: 타겟 긍정 근거 후보 수집
        logger.info(f"강점 추출 시작: 레스토랑 {restaurant_id}, 최대 후보 {max_candidates}개, 최근 {months_back}개월")
        evidence_candidates = self.collect_positive_evidence_candidates(
            restaurant_id=restaurant_id,
            max_candidates=max_candidates,
            months_back=months_back,
        )
        
        logger.info(f"Step A 완료: {len(evidence_candidates)}개 근거 후보 수집")
        
        if not evidence_candidates:
            logger.warning(f"레스토랑 {restaurant_id}에 대한 근거 후보가 없습니다.")
            return {
                "restaurant_id": restaurant_id,
                "strength_type": strength_type,
                "strengths": [],
                "total_candidates": 0,
                "validated_count": 0,
            }
        
        # Step B: 강점 후보 생성
        logger.info(f"Step B 시작: {len(evidence_candidates)}개 후보에서 강점 추출")
        strength_candidates = self.extract_strength_candidates(
            evidence_candidates=evidence_candidates,
            max_tokens=4000,
        )
        
        logger.info(f"Step B 완료: {len(strength_candidates)}개 강점 후보 생성")
        
        if not strength_candidates:
            return {
                "restaurant_id": restaurant_id,
                "strength_type": strength_type,
                "strengths": [],
                "total_candidates": len(evidence_candidates),
                "validated_count": 0,
            }
        
        # Step C: 강점별 근거 확장/검증 (비동기 병렬 처리)
        # 총 리뷰 수 조회 (동적 min_support 계산용)
        all_reviews = self.vector_search.get_restaurant_reviews(str(restaurant_id))
        total_reviews = len(all_reviews) if all_reviews else None
        
        logger.info(f"Step C 시작: {len(strength_candidates)}개 강점 후보 검증 (min_support: {min_support}, 총 리뷰: {total_reviews})")
        validated_strengths = await self.expand_and_validate_evidence(
            strength_candidates=strength_candidates,
            restaurant_id=restaurant_id,
            min_support=min_support,
            total_reviews=total_reviews,
        )
        
        logger.info(f"Step C 완료: {len(validated_strengths)}개 강점 검증 통과")
        
        # 검증 통과한 각 강점 상세 로깅
        for idx, strength in enumerate(validated_strengths, 1):
            aspect = strength.get("aspect", "")
            claim = strength.get("claim", "")[:50] + "..." if len(strength.get("claim", "")) > 50 else strength.get("claim", "")
            support_count = strength.get("support_count", 0)
            consistency = strength.get("consistency", 0)
            # consistency가 딕셔너리일 경우를 대비해 숫자로 변환
            consistency_value = float(consistency) if isinstance(consistency, (int, float)) else 0.0
            logger.info(
                f"  검증 통과 강점 {idx}: aspect='{aspect}', claim='{claim}', "
                f"support_count={support_count}, consistency={consistency_value:.3f}"
            )
        
        if not validated_strengths:
            return {
                "restaurant_id": restaurant_id,
                "strength_type": strength_type,
                "strengths": [],
                "total_candidates": len(evidence_candidates),
                "validated_count": 0,
            }
        
        # Step D: 의미 중복 제거 (Connected Components)
        merged_strengths = self.merge_similar_strengths(
            validated_strengths=validated_strengths,
            threshold_high=0.88,
            threshold_low=0.82,
            evidence_overlap_threshold=0.3,
        )
        
        logger.info(f"Step D 완료: {len(validated_strengths)}개 강점 → {len(merged_strengths)}개로 병합")
        
        # Step D-1: Claim 후처리 재생성 (LLM 1회)
        logger.info(f"Step D-1 시작: {len(merged_strengths)}개 강점의 claim 재생성")
        merged_strengths = self.regenerate_claims(merged_strengths)
        logger.info(f"Step D-1 완료: {len(merged_strengths)}개 강점의 claim 재생성 완료")
        
        # Step E~H: 비교군 기반 차별 강점 계산 (distinct 또는 both일 때만)
        if strength_type in ["distinct", "both"]:
            distinct_strengths = self.calculate_distinct_strengths(
                target_strengths=merged_strengths,
                restaurant_id=restaurant_id,
                category_filter=category_filter,
                region_filter=region_filter,
                price_band_filter=price_band_filter,
                comparison_count=20,
                alpha=1.0,
            )
            
            # final_score 기준으로 정렬
            distinct_strengths.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            if strength_type == "distinct":
                merged_strengths = distinct_strengths
            else:  # both
                # distinct_score 기준으로 strength_type 구분
                # distinct_score >= 0.3이면 distinct, 그 외는 representative
                for s in distinct_strengths:
                    distinct_score = s.get("distinct_score", 0)
                    if distinct_score is not None and distinct_score >= 0.3:
                        s["strength_type"] = "distinct"
                    else:
                        s["strength_type"] = "representative"
                        # distinct_score가 낮으면 None으로 표시 (representative로 간주)
                        if distinct_score is not None and distinct_score < 0.3:
                            s["distinct_score"] = None
                            s["closest_competitor_sim"] = None
                            s["closest_competitor_id"] = None
                
                # both일 때는 distinct_strengths만 사용 (중복 제거)
                merged_strengths = distinct_strengths
                logger.info(
                    f"both 모드: {len([s for s in merged_strengths if s.get('strength_type') == 'distinct'])}개 distinct, "
                    f"{len([s for s in merged_strengths if s.get('strength_type') == 'representative'])}개 representative"
                )
        else:
            # representative만
            for s in merged_strengths:
                s["strength_type"] = "representative"
                s["distinct_score"] = None
                s["closest_competitor_sim"] = None
                s["closest_competitor_id"] = None
                # 제시된 로직: rep_score = log(1 + support_count) * consistency * recency_weight
                s["final_score"] = (
                    math.log(1 + s.get("support_count", 0)) *
                    s.get("consistency", 1.0) *
                    s.get("recency", 1.0)
                )
        
        # Top-K 선택 및 최종 형식 변환 (쿼터 적용: both 모드일 때만)
        logger.info(f"Top-K 선택 전: {len(merged_strengths)}개 강점 (top_k={top_k})")
        
        if strength_type == "both":
            # 쿼터 적용: 대표 최소 2개 확보, distinct 최대 3개
            representative_strengths = [s for s in merged_strengths if s.get("strength_type") == "representative"]
            distinct_strengths = [s for s in merged_strengths if s.get("strength_type") == "distinct"]
            
            # final_score 기준으로 정렬
            representative_strengths.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            distinct_strengths.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            # (3) 대표 후보가 적을 때의 fallback
            # 대표가 1개뿐이면: "대표 장점" 1개, "차별 포인트" 3개, "추가 장점" 1개
            if len(representative_strengths) == 1:
                min_representative = 1
                max_distinct = min(3, len(distinct_strengths))
                max_additional = 1
            else:
                min_representative = min(2, len(representative_strengths))  # 최소 2개 (가능한 범위 내)
                max_distinct = min(3, len(distinct_strengths))  # 최대 3개
                max_additional = top_k - min_representative - max_distinct  # 나머지
            
            selected_representative = representative_strengths[:min_representative]
            selected_distinct = distinct_strengths[:max_distinct]
            
            # (2) 같은 타입 중복 방지: 대표 섹션에 aspect가 들어갔으면 차별 섹션의 같은 aspect는 1개만 허용
            representative_aspects = {s.get("aspect", "").lower() for s in selected_representative}
            
            # 차별 강점에서 대표와 같은 aspect 필터링 (최대 1개만 허용)
            filtered_distinct = []
            aspect_count_map = {}  # aspect별 카운트
            
            for s in selected_distinct:
                aspect = s.get("aspect", "").lower()
                if aspect in representative_aspects:
                    # 대표와 같은 aspect면 카운트 체크
                    count = aspect_count_map.get(aspect, 0)
                    if count < 1:  # 최대 1개만 허용
                        # "맛" 같은 경우 "시그니처 메뉴"로 라벨 변경
                        if aspect in ["맛", "taste", "맛있다"]:
                            s["aspect"] = "시그니처 메뉴"
                            s["original_aspect"] = aspect  # 원본 보존
                        filtered_distinct.append(s)
                        aspect_count_map[aspect] = count + 1
                    # 이미 1개가 있으면 스킵
                else:
                    # 대표와 다른 aspect면 그대로 추가
                    filtered_distinct.append(s)
            
            selected_distinct = filtered_distinct
            
            # 나머지 슬롯 채우기 (final_score 기준)
            remaining_slots = top_k - len(selected_representative) - len(selected_distinct)
            if remaining_slots > 0:
                # 아직 선택되지 않은 강점들
                remaining = [
                    s for s in merged_strengths
                    if s not in selected_representative and s not in selected_distinct
                ]
                remaining.sort(key=lambda x: x.get("final_score", 0), reverse=True)
                selected_remaining = remaining[:min(remaining_slots, max_additional if len(representative_strengths) == 1 else remaining_slots)]
            else:
                selected_remaining = []
            
            final_strengths = selected_representative + selected_distinct + selected_remaining
            # final_score 기준으로 최종 정렬
            final_strengths.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            logger.info(
                f"Top-K 선택 (쿼터 적용): "
                f"representative {len(selected_representative)}개, "
                f"distinct {len(selected_distinct)}개 (중복 방지 적용), "
                f"추가 {len(selected_remaining)}개 → 총 {len(final_strengths)}개"
            )
        else:
            # both가 아니면 기존 방식 (final_score 기준 상위 K개)
            final_strengths = merged_strengths[:top_k]
            
            if len(merged_strengths) > top_k:
                logger.info(
                    f"Top-K 선택으로 {len(merged_strengths) - top_k}개 강점 제외됨 "
                    f"(상위 {top_k}개만 선택)"
                )
                # 제외된 강점 로깅
                for idx, strength in enumerate(merged_strengths[top_k:], top_k + 1):
                    aspect = strength.get("aspect", "")
                    claim = strength.get("claim", "")[:50] + "..." if len(strength.get("claim", "")) > 50 else strength.get("claim", "")
                    final_score = strength.get("final_score", 0)
                    # final_score가 딕셔너리일 경우를 대비해 숫자로 변환
                    if isinstance(final_score, (int, float)):
                        final_score_value = float(final_score)
                    else:
                        final_score_value = 0.0
                    logger.info(
                        f"  제외된 강점 {idx}: aspect='{aspect}', claim='{claim}', final_score={final_score_value:.3f}"
                    )
        
        # 강점별 상세 로그
        logger.info(f"강점 추출 완료: 총 {len(final_strengths)}개 강점 추출됨")
        for idx, strength in enumerate(final_strengths, 1):
            aspect = strength.get("aspect", "")
            claim = strength.get("claim", "")[:50] + "..." if len(strength.get("claim", "")) > 50 else strength.get("claim", "")
            support_count = strength.get("support_count", 0)
            strength_type_value = strength.get("strength_type", "representative")  # 함수 파라미터와 충돌 방지
            distinct_score = strength.get("distinct_score")
            final_score = strength.get("final_score", 0)
            # final_score와 distinct_score가 딕셔너리일 경우를 대비해 숫자로 변환
            if isinstance(final_score, (int, float)):
                final_score_value = float(final_score)
            else:
                final_score_value = 0.0
            if distinct_score is not None and isinstance(distinct_score, (int, float)):
                distinct_score_value = float(distinct_score)
            else:
                distinct_score_value = None
            
            log_msg = (
                f"강점 {idx}: aspect='{aspect}', claim='{claim}', "
                f"support_count={support_count}, strength_type={strength_type_value}, "
                f"final_score={final_score_value:.3f}"
            )
            if distinct_score_value is not None:
                log_msg += f", distinct_score={distinct_score_value:.3f}"
            logger.info(log_msg)
        
        # Evidence 스니펫 생성 및 대표 근거 1줄 추출
        formatted_strengths = []
        for strength in final_strengths:
            evidence_reviews = strength.get("evidence_reviews", [])
            
            # 대표 근거 1줄 추출 (가장 대표적인 리뷰 1개)
            representative_evidence = None
            if evidence_reviews:
                # 첫 번째 리뷰를 대표 근거로 사용 (이미 정렬되어 있음)
                first_review = evidence_reviews[0]
                text = first_review.get("content", first_review.get("text", ""))
                if text:
                    # 50자 이내로 제한
                    representative_evidence = text[:50] + "..." if len(text) > 50 else text
            
            evidence_snippets = []
            for review in evidence_reviews[:5]:
                text = review.get("content", review.get("text", ""))
                snippet = text[:100] + "..." if len(text) > 100 else text
                
                evidence_snippets.append({
                    "review_id": str(review.get("review_id") or review.get("id", "")),
                    "snippet": snippet,
                    "rating": review.get("rating") or review.get("is_recommended"),
                    "created_at": str(review.get("created_at", "")),
                })
            
            formatted_strengths.append({
                "aspect": strength.get("aspect", ""),
                "claim": strength.get("claim", ""),
                "strength_type": strength.get("strength_type", "representative"),
                "support_count": strength.get("support_count", 0),
                "support_count_raw": strength.get("support_count_raw"),  # 디버깅용
                "support_count_valid": strength.get("support_count_valid"),  # 디버깅용
                "support_ratio": round(strength.get("support_ratio", 0), 3),
                "distinct_score": strength.get("distinct_score"),
                "closest_competitor_sim": strength.get("closest_competitor_sim"),
                "closest_competitor_id": strength.get("closest_competitor_id"),
                "evidence": evidence_snippets,
                "representative_evidence": representative_evidence,  # 대표 근거 1줄
                "final_score": round(strength.get("final_score", 0), 3),
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "restaurant_id": restaurant_id,
            "strength_type": strength_type,
            "strengths": formatted_strengths,
            "total_candidates": len(evidence_candidates),
            "validated_count": len(validated_strengths),
            "processing_time_ms": processing_time,
        }
