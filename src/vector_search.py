"""
벡터 검색 모듈
"""

import uuid
import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from .config import Config
from .review_utils import extract_image_urls, validate_review_data, validate_restaurant_data

logger = logging.getLogger(__name__)


class VectorSearch:
    """벡터 검색 클래스"""
    
    def __init__(
        self,
        encoder: SentenceTransformer,
        qdrant_client: QdrantClient,
        collection_name: str = Config.COLLECTION_NAME,
    ):
        """
        Args:
            encoder: SentenceTransformer 인코더
            qdrant_client: Qdrant 클라이언트
            collection_name: 컬렉션 이름
        """
        self.encoder = encoder
        
        # GPU 및 FP16 최적화 적용
        if Config.USE_GPU and torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            if Config.USE_FP16:
                self.encoder = self.encoder.half()  # FP16 양자화
            self.batch_size = Config.get_optimal_batch_size("embedding")
        else:
            self.batch_size = 32
        
        self.client = qdrant_client
        self.collection_name = collection_name
        
        # 컬렉션이 없으면 생성
        try:
            self.client.get_collection(collection_name)
        except Exception:
            # 컬렉션이 없으면 생성
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=encoder.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
    
    def _get_point_id(self, restaurant_id: Union[int, str], review_id: Union[int, str]) -> str:
        """
        리뷰 ID 기반 Point ID 생성 (일관성 보장)
: id (BIGINT PK)를 사용
        
        Args:
            restaurant_id: 레스토랑 ID (int 또는 str)
            review_id: 리뷰 ID (int 또는 str, 의 id 필드)
            
        Returns:
            Point ID (MD5 해시)
        """
        content = f"{restaurant_id}:{review_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def prepare_points(self, data: Dict, batch_size: Optional[int] = None) -> List[PointStruct]:
        """
        레스토랑 데이터를 Qdrant 포인트로 변환합니다. (대용량 처리 최적화)
        
        Args:
            data: 레스토랑 데이터 딕셔너리
            batch_size: 배치 인코딩 크기 (None이면 자동으로 최적 크기 사용)
            
        Returns:
            Qdrant PointStruct 리스트
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        points = []
        review_texts = []
        review_metadata = []
        
        # 1단계: 모든 리뷰 텍스트와 메타데이터 수집 ( 기반)
        # data는 reviews 리스트를 직접 받거나, restaurants 구조를 받을 수 있음
        reviews_list = data.get("reviews", [])
        if not reviews_list and "restaurants" in data:
            # 기존 형식 지원 (하위 호환성)
            for restaurant in data.get("restaurants", []):
                if not validate_restaurant_data(restaurant):
                    logger.warning(f"레스토랑 정보가 불완전합니다: {restaurant}")
                    continue
                reviews_list.extend(restaurant.get("reviews", []))
        
        for review in reviews_list:
            # Pydantic 모델을 딕셔너리로 변환
            if hasattr(review, 'dict'):
                review = review.dict()
            elif hasattr(review, 'model_dump'):
                review = review.model_dump()
            elif not isinstance(review, dict):
                logger.warning(f"리뷰 형식이 올바르지 않습니다: {type(review)}")
                continue
            
            if not validate_review_data(review):
                logger.warning(f"리뷰 정보가 불완전합니다: {review}")
                continue
                    
            # : content 필드 사용
            review_text = review.get("content") or review.get("review", "")
            if not review_text:
                continue
            
            review_texts.append(review_text)
            
            #  컬럼명 사용
            review_id = review.get("id") or review.get("review_id")
            restaurant_id = review.get("restaurant_id")
            
            # datetime 객체를 문자열로 변환
            def to_iso_string(dt):
                if dt is None:
                    return None
                if isinstance(dt, datetime):
                    return dt.isoformat()
                if isinstance(dt, str):
                    return dt
                return str(dt)
            
            created_at_value = review.get("created_at") or review.get("datetime")
            updated_at_value = review.get("updated_at")
            deleted_at_value = review.get("deleted_at")
            
            review_metadata.append({
                #  컬럼명
                "id": int(review_id) if review_id and str(review_id).isdigit() else review_id,
                "restaurant_id": str(restaurant_id) if restaurant_id else None,  # Qdrant 검색을 위해 문자열로 저장
                "member_id": review.get("member_id") or (int(review.get("user_id")) if review.get("user_id") and str(review.get("user_id")).isdigit() else review.get("user_id")),
                "group_id": review.get("group_id") or (int(review.get("group")) if review.get("group") and str(review.get("group")).isdigit() else review.get("group")),
                "subgroup_id": review.get("subgroup_id"),
                "content": review_text,
                "is_recommended": review.get("is_recommended"),
                "created_at": to_iso_string(created_at_value) or datetime.now().isoformat(),
                "updated_at": to_iso_string(updated_at_value) or datetime.now().isoformat(),
                "deleted_at": to_iso_string(deleted_at_value),
                # 하위 호환성을 위한 필드도 유지
                "review_id": review_id,
                "restaurant_name": review.get("restaurant_name") or data.get("restaurant_name"),
                "user_id": review.get("user_id") or review.get("member_id"),
                "group": review.get("group") or review.get("group_id"),
                "review": review_text,
                "datetime": review.get("created_at") or review.get("datetime"),
                "image_urls": extract_image_urls(review.get("images")),
                "version": review.get("version", 1),
            })
        
        # 2단계: 배치로 벡터 인코딩 (대용량 처리 최적화)
        logger.info(f"총 {len(review_texts)}개의 리뷰를 배치로 인코딩합니다 (배치 크기: {batch_size})")
        
        for i in range(0, len(review_texts), batch_size):
            batch_texts = review_texts[i:i + batch_size]
            batch_metadata = review_metadata[i:i + batch_size]
            
            try:
                # 배치 인코딩 (한 번에 처리하여 성능 향상)
                batch_vectors = self.encoder.encode(batch_texts)
                
                for text, vector, metadata in zip(batch_texts, batch_vectors, batch_metadata):
                    try:
                        # : id 기반 Point ID 생성
                        review_id = metadata.get("id") or metadata.get("review_id")
                        restaurant_id = metadata.get("restaurant_id")
                        point_id = self._get_point_id(
                            restaurant_id or "",
                            review_id or ""
                        )
                        point = PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload=metadata
                        )
                        points.append(point)
                    except Exception as e:
                        logger.error(f"포인트 생성 중 오류: {metadata.get('id') or metadata.get('review_id')} | {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"배치 인코딩 중 오류 발생 (배치 {i//batch_size + 1}): {str(e)}")
                # 배치 실패 시 개별 처리
                for text, metadata in zip(batch_texts, batch_metadata):
                    try:
                        vector = self.encoder.encode(text)
                        # : id 기반 Point ID 생성
                        review_id = metadata.get("id") or metadata.get("review_id")
                        restaurant_id = metadata.get("restaurant_id")
                        point_id = self._get_point_id(
                            restaurant_id or "",
                            review_id or ""
                        )
                        point = PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload=metadata
                        )
                        points.append(point)
                    except Exception as e2:
                        logger.error(f"개별 인코딩 중 오류: {metadata.get('id') or metadata.get('review_id')} | {str(e2)}")
                        continue
        
        logger.info(f"총 {len(points)}개의 포인트를 생성했습니다.")
        return points
    
    def upload_points(self, points: List[PointStruct]) -> None:
        """
        포인트를 Qdrant에 업로드합니다 (기존 방식).
        
        Args:
            points: 업로드할 포인트 리스트
        """
        try:
            if not points:
                logger.warning("업로드할 포인트가 없습니다.")
                return
            
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"{len(points)}개의 포인트를 업로드했습니다.")
        except Exception as e:
            logger.error(f"포인트 업로드 중 오류: {str(e)}")
            raise
    
    def upload_collection(self, points: List[PointStruct]) -> None:
        """
        포인트를 Qdrant에 업로드합니다 (upload_collection 방식, 대용량 데이터에 효율적).
        
        Args:
            points: 업로드할 포인트 리스트
        """
        try:
            if not points:
                logger.warning("업로드할 포인트가 없습니다.")
                return
            
            # PointStruct 리스트를 upload_collection 형식으로 변환
            # upload_collection은 벡터, 페이로드, ID를 리스트 형태로 받습니다
            vectors = []
            payloads = []
            ids = []
            
            for point in points:
                ids.append(point.id)
                vectors.append(point.vector)
                payloads.append(point.payload)
            
            logger.info(f"{len(points)}개의 포인트를 upload_collection으로 업로드 시작...")
            
            # upload_collection 사용 (대용량 데이터에 효율적)
            # wait=True로 설정하여 업로드 완료를 기다림
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=vectors,
                payload=payloads,
                ids=ids,
                batch_size=100,  # 배치 크기 설정
                parallel=1,  # 병렬 처리 수
                wait=True,  # 업로드 완료 대기
            )
            logger.info(f"{len(points)}개의 포인트를 upload_collection으로 업로드 완료했습니다.")
        except Exception as e:
            logger.error(f"포인트 업로드 중 오류: {str(e)}", exc_info=True)
            raise
    
    def get_restaurant_reviews(self, restaurant_id: Union[int, str]) -> List[Dict]:
        """
        레스토랑 ID로 리뷰를 조회합니다 ( 기반).
        
        Args:
            restaurant_id: 레스토랑 ID (int 또는 str)
            
        Returns:
            리뷰 payload 리스트 ( 컬럼명 포함)
        """
        try:
            # int를 str로 변환 (Qdrant는 문자열 키를 사용할 수 있음)
            restaurant_id_str = str(restaurant_id) if isinstance(restaurant_id, int) else restaurant_id
            
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="restaurant_id",
                            match=models.MatchValue(value=restaurant_id_str)
                        )
                    ]
                )
            )
            return [r.payload for r in records]
        except Exception as e:
            logger.error(f"리뷰 조회 중 오류: {str(e)}")
            return []
    
    def get_all_restaurant_ids(self) -> List[str]:
        """
        컬렉션에 있는 모든 고유한 레스토랑 ID를 반환합니다.
        
        Returns:
            레스토랑 ID 리스트 (정렬됨)
        """
        try:
            restaurant_ids = set()
            
            # Scroll을 사용하여 모든 포인트 조회
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=100,  # 한 번에 100개씩 조회
                    with_payload=True,
                    with_vectors=False
                )
                
                for record in records:
                    restaurant_id = record.payload.get("restaurant_id")
                    if restaurant_id:
                        restaurant_ids.add(restaurant_id)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            result = sorted(list(restaurant_ids))
            logger.info(f"총 {len(result)}개의 고유한 레스토랑 ID를 찾았습니다.")
            return result
        except Exception as e:
            logger.error(f"레스토랑 ID 조회 중 오류: {str(e)}")
            return []
    
    # ==================== Phase 1: 대표 벡터 기반 비교군 선정 ====================
    
    def compute_restaurant_vector(
        self,
        restaurant_id: Union[int, str],
        weight_by_date: bool = True,
        weight_by_rating: bool = True,
    ) -> Optional[np.ndarray]:
        """
        레스토랑의 모든 리뷰 임베딩을 평균/가중 평균하여 대표 벡터 생성
        
        Args:
            restaurant_id: 레스토랑 ID
            weight_by_date: 최근 리뷰에 가중치 부여 (기본값: True)
            weight_by_rating: 높은 별점에 가중치 부여 (기본값: True)
            
        Returns:
            대표 벡터 (numpy array) 또는 None
        """
        try:
            # 1. 해당 레스토랑의 모든 리뷰 검색
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="restaurant_id",
                            match=models.MatchValue(value=str(restaurant_id))
                        )
                    ]
                ),
                limit=10000,  # 충분히 큰 값
                with_payload=True,
                with_vectors=True,
            )
            
            if not records:
                logger.warning(f"레스토랑 {restaurant_id}의 리뷰를 찾을 수 없습니다.")
                return None
            
            # 2. 각 리뷰의 벡터와 메타데이터 추출
            vectors = []
            weights = []
            
            for record in records:
                if not hasattr(record, 'vector') or record.vector is None:
                    # 벡터가 없으면 임베딩 생성
                    review_text = record.payload.get("content", "") or record.payload.get("review", "")
                    if not review_text:
                        continue
                    vector = self.encoder.encode(review_text, convert_to_numpy=True)
                else:
                    vector = np.array(record.vector)
                
                # 가중치 계산
                weight = 1.0
                
                if weight_by_date and "created_at" in record.payload:
                    # 최근 리뷰일수록 높은 가중치
                    try:
                        from datetime import datetime, timezone
                        created_at = record.payload["created_at"]
                        if isinstance(created_at, str):
                            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        else:
                            created_dt = created_at
                        
                        now = datetime.now(timezone.utc)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                        
                        days_ago = (now - created_dt).days
                        # 최근 1년 = 1.0, 2년 전 = 0.8, 3년 전 = 0.6
                        if days_ago <= 365:
                            weight *= 1.0
                        elif days_ago <= 730:
                            weight *= 0.8
                        else:
                            weight *= 0.6
                    except Exception:
                        pass  # 날짜 파싱 실패 시 가중치 유지
                
                if weight_by_rating and "is_recommended" in record.payload:
                    # 추천 리뷰에 높은 가중치
                    if record.payload["is_recommended"]:
                        weight *= 1.5
                
                vectors.append(vector)
                weights.append(weight)
            
            if not vectors:
                return None
            
            # 3. 가중 평균 계산
            vectors = np.array(vectors)
            weights = np.array(weights)
            weights = weights / weights.sum()  # 정규화
            
            restaurant_vector = np.average(vectors, axis=0, weights=weights)
            
            return restaurant_vector
            
        except Exception as e:
            logger.error(f"레스토랑 {restaurant_id} 대표 벡터 계산 중 오류: {str(e)}")
            return None
    
    def upsert_restaurant_vector(
        self,
        restaurant_id: Union[int, str],
        restaurant_name: str,
        food_category_id: Optional[int] = None,
    ) -> bool:
        """
        레스토랑 대표 벡터를 Qdrant에 저장/업데이트
        
        Args:
            restaurant_id: 레스토랑 ID
            restaurant_name: 레스토랑 이름
            food_category_id: 음식 카테고리 ID (선택)
            
        Returns:
            성공 여부
        """
        try:
            # 1. 대표 벡터 계산
            restaurant_vector = self.compute_restaurant_vector(restaurant_id)
            
            if restaurant_vector is None:
                logger.warning(f"레스토랑 {restaurant_id}의 대표 벡터를 생성할 수 없습니다.")
                return False
            
            # 2. restaurant_vectors 컬렉션이 없으면 생성
            try:
                self.client.get_collection(RESTAURANT_VECTORS_COLLECTION)
            except Exception:
                self.client.create_collection(
                    collection_name=RESTAURANT_VECTORS_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=self.encoder.get_sentence_embedding_dimension(),
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(f"컬렉션 {RESTAURANT_VECTORS_COLLECTION} 생성 완료")
            
            # 3. 포인트 생성 및 업로드
            # Qdrant의 id는 UUID 형식이어야 하므로, restaurant_id를 기반으로 일관된 UUID 생성
            # 같은 restaurant_id는 항상 같은 UUID를 생성하기 위해 UUID5 사용
            restaurant_id_str = str(restaurant_id)
            # UUID5를 사용하여 restaurant_id를 기반으로 일관된 UUID 생성
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"restaurant_{restaurant_id_str}")
            
            point = PointStruct(
                id=str(point_id),  # UUID를 문자열로 변환
                vector=restaurant_vector.tolist(),
                payload={
                    "restaurant_id": str(restaurant_id),
                    "restaurant_name": restaurant_name,
                    "food_category_id": food_category_id,
                }
            )
            
            self.client.upsert(
                collection_name=RESTAURANT_VECTORS_COLLECTION,
                points=[point]
            )
            
            logger.info(f"레스토랑 {restaurant_id}의 대표 벡터를 업데이트했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"레스토랑 {restaurant_id} 대표 벡터 업데이트 중 오류: {str(e)}")
            return False
    
    def find_similar_restaurants(
        self,
        target_restaurant_id: Union[int, str],
        top_n: int = 20,
        food_category_id: Optional[int] = None,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """
        타겟 레스토랑과 유사한 레스토랑을 대표 벡터로 검색
        
        Args:
            target_restaurant_id: 타겟 레스토랑 ID
            top_n: 반환할 상위 N개 (기본값: 20)
            food_category_id: 음식 카테고리 필터 (선택)
            exclude_self: 타겟 레스토랑 제외 여부 (기본값: True)
            
        Returns:
            유사 레스토랑 리스트 [{"restaurant_id": ..., "score": ..., ...}, ...]
        """
        try:
            # 1. 타겟 레스토랑의 대표 벡터 가져오기 (또는 계산)
            target_vector = self.compute_restaurant_vector(target_restaurant_id)
            
            if target_vector is None:
                logger.warning(f"타겟 레스토랑 {target_restaurant_id}의 대표 벡터를 찾을 수 없습니다.")
                return []
            
            # 2. restaurant_vectors 컬렉션이 없으면 fallback: 리뷰 컬렉션에서 직접 비교군 찾기
            try:
                collection_info = self.client.get_collection(RESTAURANT_VECTORS_COLLECTION)
                logger.info(f"restaurant_vectors 컬렉션 발견: {collection_info.points_count}개 포인트")
                use_fallback = False
            except Exception as e:
                logger.info(
                    f"컬렉션 {RESTAURANT_VECTORS_COLLECTION}가 없습니다 (오류: {e}). "
                    f"리뷰 컬렉션에서 직접 비교군을 찾습니다."
                )
                use_fallback = True
            
            if use_fallback:
                # Fallback: 리뷰 컬렉션에서 모든 레스토랑의 대표 벡터를 계산하여 비교
                return self._find_similar_restaurants_fallback(
                    target_restaurant_id=target_restaurant_id,
                    target_vector=target_vector,
                    top_n=top_n,
                    food_category_id=food_category_id,
                    exclude_self=exclude_self,
                )
            
            # 3. 필터 구성
            filter_conditions = []
            
            if food_category_id is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="food_category_id",
                        match=models.MatchValue(value=food_category_id)
                    )
                )
            
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(must=filter_conditions)
            
            # 4. 유사도 검색 (1번만!)
            results = self.client.query_points(
                collection_name=RESTAURANT_VECTORS_COLLECTION,
                query=target_vector.tolist(),
                limit=top_n + (1 if exclude_self else 0),  # 자기 자신 제외 고려
                query_filter=query_filter,
            ).points
            
            # 5. 결과 변환
            similar_restaurants = []
            for result in results:
                result_restaurant_id = result.payload.get("restaurant_id")
                
                if exclude_self and str(result_restaurant_id) == str(target_restaurant_id):
                    continue  # 자기 자신 제외
                
                similar_restaurants.append({
                    "restaurant_id": result_restaurant_id,
                    "restaurant_name": result.payload.get("restaurant_name", ""),
                    "score": float(result.score),
                    "food_category_id": result.payload.get("food_category_id"),
                })
            
            logger.info(f"restaurant_vectors 컬렉션에서 {len(similar_restaurants)}개 비교군 레스토랑 찾음")
            return similar_restaurants[:top_n]
            
        except Exception as e:
            logger.error(f"유사 레스토랑 검색 중 오류: {str(e)}")
            return []
    
    def _find_similar_restaurants_fallback(
        self,
        target_restaurant_id: Union[int, str],
        target_vector: np.ndarray,
        top_n: int = 20,
        food_category_id: Optional[int] = None,
        exclude_self: bool = True,
    ) -> List[Dict]:
        """
        Fallback: restaurant_vectors 컬렉션이 없을 때 리뷰 컬렉션에서 직접 비교군 찾기
        
        Args:
            target_restaurant_id: 타겟 레스토랑 ID
            target_vector: 타겟 레스토랑의 대표 벡터
            top_n: 반환할 상위 N개
            food_category_id: 음식 카테고리 필터
            exclude_self: 타겟 레스토랑 제외 여부
            
        Returns:
            유사 레스토랑 리스트
        """
        try:
            # 1. 리뷰 컬렉션에서 모든 레스토랑 ID 수집
            restaurant_ids = set()
            filter_conditions = []
            
            if food_category_id is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key="food_category_id",
                        match=models.MatchValue(value=food_category_id)
                    )
                )
            
            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            # Scroll을 사용하여 모든 레스토랑 ID 수집
            try:
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=10000,  # 충분히 큰 값
                )
                logger.info(f"Fallback: scroll 결과 {len(records)}개 레코드 조회")
            except Exception as e:
                logger.error(f"Fallback: scroll 실패: {e}")
                return []
            
            for record in records:
                rid = record.payload.get("restaurant_id")
                if rid:
                    restaurant_ids.add(str(rid))
            
            logger.info(f"Fallback: 리뷰 컬렉션에서 {len(restaurant_ids)}개 레스토랑 ID 수집")
            
            if exclude_self:
                restaurant_ids.discard(str(target_restaurant_id))
                logger.info(f"Fallback: 타겟 레스토랑 {target_restaurant_id} 제외 후 {len(restaurant_ids)}개 레스토랑")
            
            if not restaurant_ids:
                logger.warning(
                    f"비교군 레스토랑을 찾을 수 없습니다. "
                    f"리뷰 컬렉션에 다른 레스토랑의 리뷰가 없거나, "
                    f"타겟 레스토랑({target_restaurant_id})만 존재합니다."
                )
                return []
            
            # 2. 각 레스토랑의 대표 벡터 계산 및 유사도 계산
            similar_restaurants = []
            processed_count = 0
            failed_count = 0
            
            for rid in list(restaurant_ids)[:100]:  # 성능을 위해 최대 100개만
                try:
                    comp_vector = self.compute_restaurant_vector(rid)
                    if comp_vector is None:
                        logger.debug(f"레스토랑 {rid}의 대표 벡터가 None입니다.")
                        failed_count += 1
                        continue
                    
                    # 코사인 유사도 계산
                    similarity = np.dot(target_vector, comp_vector) / (
                        np.linalg.norm(target_vector) * np.linalg.norm(comp_vector)
                    )
                    
                    similar_restaurants.append({
                        "restaurant_id": int(rid) if str(rid).isdigit() else rid,
                        "restaurant_name": "",  # 이름은 리뷰에서 가져올 수 없음
                        "score": float(similarity),
                        "food_category_id": food_category_id,
                    })
                    processed_count += 1
                except Exception as e:
                    logger.warning(f"레스토랑 {rid} 벡터 계산 실패: {e}")
                    failed_count += 1
                    continue
            
            logger.info(
                f"Fallback: {processed_count}개 레스토랑 벡터 계산 성공, "
                f"{failed_count}개 실패, {len(similar_restaurants)}개 유사도 계산 완료"
            )
            
            # 3. 유사도 기준으로 정렬
            similar_restaurants.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Fallback 방식으로 {len(similar_restaurants)}개 비교군 레스토랑 찾음")
            return similar_restaurants[:top_n]
            
        except Exception as e:
            logger.error(f"Fallback 비교군 검색 중 오류: {str(e)}")
            return []
    
    # ==================== Phase 2: 강점 임베딩 기반 차별점 계산 ====================
    
    def compute_strength_embeddings(
        self,
        strengths: List[str],
    ) -> np.ndarray:
        """
        강점 리스트를 임베딩으로 변환
        
        Args:
            strengths: 강점 리스트 ["맛이 좋다", "서비스가 친절하다", ...]
            
        Returns:
            강점 임베딩 배열 (n_strengths, embedding_dim)
        """
        if not strengths:
            return np.array([])
        
        # 배치 인코딩
        embeddings = self.encoder.encode(
            strengths,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def find_unique_strengths(
        self,
        target_strengths: List[str],
        comparison_strengths_list: List[List[str]],  # 각 비교군의 강점 리스트
        similarity_threshold: float = 0.7,
    ) -> List[str]:
        """
        타겟 레스토랑에만 있는 (또는 더 강한) 강점을 찾기
        
        Args:
            target_strengths: 타겟 레스토랑의 강점 리스트
            comparison_strengths_list: 비교군들의 강점 리스트 리스트
            similarity_threshold: 유사도 임계점 (이상이면 같은 강점으로 간주)
            
        Returns:
            차별화된 강점 리스트
        """
        if not target_strengths:
            return []
        
        # 1. 타겟 강점 임베딩
        target_embeddings = self.compute_strength_embeddings(target_strengths)
        
        if len(target_embeddings) == 0:
            return []
        
        # 2. 비교군 강점 임베딩 (모든 비교군 합치기)
        all_comparison_strengths = []
        for comp_strengths in comparison_strengths_list:
            all_comparison_strengths.extend(comp_strengths)
        
        if not all_comparison_strengths:
            # 비교군에 강점이 없으면 타겟의 모든 강점이 차별점
            return target_strengths
        
        comparison_embeddings = self.compute_strength_embeddings(all_comparison_strengths)
        
        if len(comparison_embeddings) == 0:
            return target_strengths
        
        # 3. 각 타겟 강점이 비교군 강점과 유사한지 확인
        unique_strengths = []
        
        for i, target_strength in enumerate(target_strengths):
            target_emb = target_embeddings[i]
            
            # 비교군 강점들과의 최대 유사도 계산 (코사인 유사도)
            target_norm = np.linalg.norm(target_emb)
            if target_norm == 0:
                continue
            
            similarities = np.dot(comparison_embeddings, target_emb) / (
                np.linalg.norm(comparison_embeddings, axis=1) * target_norm
            )
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            
            # 임계점 이하면 차별점으로 간주
            if max_similarity < similarity_threshold:
                unique_strengths.append(target_strength)
        
        return unique_strengths
    
    def query_by_restaurant_vector(
        self,
        restaurant_id: Union[int, str],
        top_k: int = 20,
        months_back: Optional[int] = None,
    ) -> List[Dict]:
        """
        레스토랑 대표 벡터를 쿼리로 사용하여 TOP-K 리뷰 검색
        
        Args:
            restaurant_id: 레스토랑 ID
            top_k: 반환할 최대 리뷰 수
            months_back: 최근 N개월 필터 (선택, None이면 필터링 안함)
        
        Returns:
            검색 결과 리스트 [{"payload": {...}, "score": 0.95}, ...]
        """
        try:
            # 1. 대표 벡터 계산
            restaurant_vector = self.compute_restaurant_vector(restaurant_id)
            if restaurant_vector is None:
                logger.warning(f"레스토랑 {restaurant_id}의 대표 벡터를 계산할 수 없습니다.")
                return []
            
            # 2. 필터 구성
            filter_conditions = [
                models.FieldCondition(
                    key="restaurant_id",
                    match=models.MatchValue(value=str(restaurant_id))
                )
            ]
            
            query_filter = models.Filter(must=filter_conditions)
            
            # 3. 대표 벡터로 검색
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=restaurant_vector.tolist(),
                query_filter=query_filter,
                limit=top_k * 2,  # 날짜 필터링 전에 더 많이 가져옴
            ).points
            
            # 4. 결과 변환
            candidates = []
            for r in results:
                candidates.append({
                    "payload": r.payload,
                    "score": float(r.score)
                })
            
            # 5. 날짜 필터링 (클라이언트 측에서 수행)
            if months_back is not None:
                from datetime import datetime, timedelta
                cutoff_date = datetime.now() - timedelta(days=months_back * 30)
                
                filtered_candidates = []
                for candidate in candidates:
                    created_at_str = candidate["payload"].get("created_at", "")
                    if created_at_str:
                        try:
                            if isinstance(created_at_str, str):
                                try:
                                    candidate_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                                except ValueError:
                                    candidate_date = datetime.now()
                            else:
                                candidate_date = created_at_str
                            
                            if candidate_date >= cutoff_date:
                                filtered_candidates.append(candidate)
                        except (ValueError, TypeError):
                            # 파싱 실패 시 포함
                            filtered_candidates.append(candidate)
                    else:
                        # created_at이 없으면 포함
                        filtered_candidates.append(candidate)
                
                candidates = filtered_candidates
            
            # 6. 날짜 기준 정렬 (최신순)
            candidates.sort(
                key=lambda x: x["payload"].get("created_at", ""),
                reverse=True
            )
            
            # 7. TOP-K만 반환
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"대표 벡터 기반 검색 중 오류: {str(e)}")
            return []
    
    def query_similar_reviews(
        self,
        query_text: str,
        restaurant_id: Optional[Union[int, str]] = None,
        limit: int = 3,
        min_score: float = 0.0,
        food_category_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        의미 기반으로 유사한 리뷰를 검색합니다 ( 기반).
        
        Args:
            query_text: 검색 쿼리 텍스트
            restaurant_id: 필터링할 레스토랑 ID (None이면 전체)
            limit: 반환할 최대 개수
            min_score: 최소 유사도 점수
            food_category_id: 음식 카테고리 ID 필터 (선택사항, strength 기능용)
            
        Returns:
            검색 결과 리스트 (payload와 score 포함,  컬럼명)
        """
        try:
            query_vector = self.encoder.encode(query_text).tolist()
            
            filter_conditions = []
            if restaurant_id:
                restaurant_id_str = str(restaurant_id) if isinstance(restaurant_id, int) else restaurant_id
                filter_conditions.append(
                    models.FieldCondition(
                        key="restaurant_id",
                        match=models.MatchValue(value=restaurant_id_str)
                    )
                )
            
            # food_category_id 필터 (strength 기능용)
            if food_category_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="food_category_id",
                        match=models.MatchValue(value=food_category_id)
                    )
                )
            
            query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
            
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points
            
            results = []
            for hit in hits:
                if hit.score and hit.score >= min_score:
                    results.append({
                        "payload": hit.payload,
                        "score": hit.score
                    })
            
            return results
        except Exception as e:
            logger.error(f"리뷰 검색 중 오류: {str(e)}")
            return []
    
    @staticmethod
    def _should_expand_query(query: str) -> bool:
        """
        쿼리 확장이 필요한지 판단하는 하이브리드 로직
        
        단순 키워드: 확장 불필요 (빠른 응답)
        복잡한 의도: 확장 필요 (검색 품질 향상)
        
        Args:
            query: 사용자 검색 쿼리
            
        Returns:
            확장 필요 여부 (True: 확장 필요, False: 확장 불필요)
        """
        # 복잡한 의도 표현 패턴 (확장 필요)
        complex_patterns = [
            r"데이트|가족|친구|혼자|모임",  # 상황 표현
            r"좋은|나쁜|추천|만족|적합",  # 평가 표현
            r"적합|맞는|어울리는|좋을|나을",  # 적합성 표현
            r"원하는|원하는|찾는|원하는",  # 욕구 표현
        ]
        
        # 단순 키워드 패턴 (확장 불필요)
        simple_keywords = [
            "분위기", "맛", "서비스", "가격", "주차",
            "로맨틱", "조용한", "시끄러운", "넓은", "좁은",
        ]
        
        # 단순 키워드만 포함된 경우 확장 불필요
        query_lower = query.lower()
        is_simple = any(keyword in query_lower for keyword in simple_keywords)
        if is_simple and len(query.split()) <= 3:
            return False
        
        # 복잡한 패턴이 있으면 확장 필요
        for pattern in complex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # 기본값: 짧은 쿼리(3단어 이하)이고 패턴이 없으면 확장 불필요
        if len(query.split()) <= 2:
            return False
        
        # 긴 쿼리는 확장 필요
        return True
    
    async def query_similar_reviews_with_expansion(
        self,
        query_text: str,
        restaurant_id: Optional[Union[int, str]] = None,
        limit: int = 3,
        min_score: float = 0.0,
        food_category_id: Optional[int] = None,
        expand_query: Optional[bool] = None,  # None: 자동 판단, True: 강제 확장, False: 확장 안함
        llm_utils: Optional[Any] = None,  # LLMUtils 인스턴스
    ) -> List[Dict]:
        """
        Query Expansion을 지원하는 의미 기반 검색 (하이브리드 접근)
        
        Args:
            query_text: 검색 쿼리 텍스트
            restaurant_id: 필터링할 레스토랑 ID (None이면 전체)
            limit: 반환할 최대 개수
            min_score: 최소 유사도 점수
            food_category_id: 음식 카테고리 ID 필터 (선택사항)
            expand_query: 쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)
            llm_utils: LLMUtils 인스턴스 (확장 시 필요)
            
        Returns:
            검색 결과 리스트 (payload와 score 포함)
        """
        original_query = query_text
        
        # 쿼리 확장 여부 결정 (하이브리드 로직)
        should_expand = False
        if expand_query is None:
            # 자동 판단
            should_expand = self._should_expand_query(query_text)
        elif expand_query:
            # 강제 확장
            should_expand = True
        
        # 쿼리 확장 실행
        if should_expand and llm_utils:
            try:
                expanded_query = await llm_utils.expand_query_for_dense_search(query_text)
                if expanded_query and expanded_query != query_text:
                    logger.info(f"쿼리 확장: '{query_text}' → '{expanded_query}'")
                    query_text = expanded_query
                else:
                    logger.debug(f"쿼리 확장 결과가 원본과 동일, 원본 사용")
            except Exception as e:
                logger.warning(f"쿼리 확장 실패, 원본 사용: {e}")
                # 실패 시 원본 쿼리 사용
        
        # 기존 검색 로직 호출
        return self.query_similar_reviews(
            query_text=query_text,
            restaurant_id=restaurant_id,
            limit=limit,
            min_score=min_score,
            food_category_id=food_category_id,
        )
    
    async def get_reviews_with_images(
        self,
        query_text: str,
        limit: int = 10,
        min_score: float = 0.0,
        expand_query: Optional[bool] = None,  # None: 자동 판단, True: 강제 확장, False: 확장 안함
        llm_utils: Optional[Any] = None,  # LLMUtils 인스턴스
    ) -> List[Dict]:
        """
        이미지가 있는 리뷰를 검색합니다 (Query Expansion 지원).
        
        Args:
            query_text: 검색 쿼리 텍스트
            limit: 반환할 최대 개수
            min_score: 최소 유사도 점수
            expand_query: 쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)
            llm_utils: LLMUtils 인스턴스 (확장 시 필요)
            
        Returns:
            이미지 URL이 있는 리뷰 리스트
        """
        results = await self.query_similar_reviews_with_expansion(
            query_text=query_text,
            restaurant_id=None,
            limit=limit,
            min_score=min_score,
            expand_query=expand_query,
            llm_utils=llm_utils,
        )
        
        reviews_with_images = []
        for result in results:
            image_urls = result["payload"].get("image_urls", [])
            if image_urls:  # 이미지가 있는 경우만
                reviews_with_images.append({
                    "payload": result["payload"],
                    "score": result["score"],
                    "image_urls": image_urls
                })
        
        return reviews_with_images
    
    def upsert_review(
        self,
        restaurant_id: str,
        restaurant_name: str,
        review: Dict[str, Any],
        update_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        리뷰를 upsert합니다 (있으면 업데이트, 없으면 삽입).
        update_filter를 사용하여 낙관적 잠금(Optimistic Locking)을 지원합니다.
        
        Args:
            restaurant_id: 레스토랑 ID
            restaurant_name: 레스토랑 이름
            review: 리뷰 딕셔너리 (review_id, review, user_id, datetime, group, images, version 등)
            update_version: 업데이트할 버전 (None이면 항상 업데이트, 지정하면 해당 버전일 때만 업데이트)
            
        Returns:
            {
                "action": "inserted" | "updated" | "skipped",
                "review_id": str,
                "version": int,
                "point_id": str,
                "reason": str (skipped인 경우)
            }
        """
        try:
            # 1. 리뷰 검증
            if not validate_review_data(review):
                raise ValueError("리뷰 데이터가 유효하지 않습니다.")
            
            # : id 필드 사용
            review_id = review.get("id") or review.get("review_id")
            if not review_id:
                raise ValueError("id (review_id)가 필요합니다.")
            
            # 2. Point ID 생성 (id 기반)
            point_id = self._get_point_id(restaurant_id, review_id)
            
            # 3. 벡터 인코딩 (: content 필드)
            review_text = review.get("content") or review.get("review", "")
            if not review_text:
                raise ValueError("content가 필요합니다.")
            vector = self.encoder.encode(review_text).tolist()
            
            # 4. 현재 버전 확인
            current_version = review.get("version", 1)
            new_version = current_version + 1 if update_version is not None else current_version + 1
            
            # 5. Payload 구성 ( 컬럼명)
            payload = {
                #  컬럼명
                "id": int(review_id) if review_id and str(review_id).isdigit() else review_id,
                "restaurant_id": int(restaurant_id) if restaurant_id and str(restaurant_id).isdigit() else restaurant_id,
                "member_id": review.get("member_id") or (int(review.get("user_id")) if review.get("user_id") and str(review.get("user_id")).isdigit() else review.get("user_id")),
                "group_id": review.get("group_id") or (int(review.get("group")) if review.get("group") and str(review.get("group")).isdigit() else review.get("group")),
                "subgroup_id": review.get("subgroup_id"),
                "content": review_text,
                "is_recommended": review.get("is_recommended"),
                "created_at": review.get("created_at") or review.get("datetime") or datetime.now().isoformat(),
                "updated_at": review.get("updated_at") or datetime.now().isoformat(),
                "deleted_at": review.get("deleted_at"),
                # 하위 호환성 필드
                "review_id": review_id,
                "restaurant_name": restaurant_name,
                "user_id": review.get("user_id") or review.get("member_id"),
                "group": review.get("group") or review.get("group_id"),
                "review": review_text,
                "datetime": review.get("created_at") or review.get("datetime"),
                "image_urls": extract_image_urls(review.get("images")),
                "version": new_version,
            }
            
            # 6. Upsert 실행
            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
            
            # update_filter 설정 (version 기반 낙관적 잠금)
            update_filter = None
            if update_version is not None:
                # 특정 버전일 때만 업데이트 (낙관적 잠금)
                update_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="version",
                            match=models.MatchValue(value=update_version)
                        ),
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=review_id)
                        ),
                    ]
                )
            
            # Upsert 실행
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                update_filter=update_filter,
            )
            
            # 7. 결과 확인
            # Qdrant는 업데이트/삽입 여부를 직접 반환하지 않으므로
            # 기존 포인트 존재 여부로 판단
            existing = self._check_point_exists(point_id)
            
            if existing and update_version is not None:
                # 기존 포인트의 버전 확인
                existing_point = self._get_point_by_id(point_id)
                if existing_point and existing_point.get("version") != update_version:
                    # 버전이 맞지 않으면 스킵
                    logger.warning(
                        f"리뷰 {review_id}: 버전 불일치 "
                        f"(요청: {update_version}, 실제: {existing_point.get('version')})"
                    )
                    return {
                        "action": "skipped",
                        "reason": "version_mismatch",
                        "review_id": review_id,
                        "requested_version": update_version,
                        "current_version": existing_point.get("version") if existing_point else None,
                        "point_id": point_id
                    }
            
            action = "updated" if existing else "inserted"
            logger.info(f"리뷰 {review_id}: {action} (version {new_version})")
            
            return {
                "action": action,
                "review_id": review_id,
                "version": new_version,
                "point_id": point_id
            }
            
        except Exception as e:
            logger.error(f"리뷰 upsert 중 오류: {str(e)}")
            raise
    
    def _check_point_exists(self, point_id: str) -> bool:
        """
        포인트 존재 여부 확인
        
        Args:
            point_id: Point ID
            
        Returns:
            존재 여부
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            return len(result) > 0
        except Exception:
            return False
    
    def _get_point_by_id(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Point ID로 포인트 조회
        
        Args:
            point_id: Point ID
            
        Returns:
            Payload 딕셔너리 (없으면 None)
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            if result:
                return result[0].payload
            return None
        except Exception:
            return None
    
    def delete_review(
        self,
        restaurant_id: Union[int, str],
        review_id: Union[int, str],
    ) -> Dict[str, Any]:
        """
        리뷰를 삭제합니다.
        
        Args:
            restaurant_id: 레스토랑 ID
            review_id: 리뷰 ID
            
        Returns:
            {
                "action": "deleted" | "not_found",
                "review_id": str,
                "point_id": str
            }
        """
        try:
            # Point ID 생성
            point_id = self._get_point_id(restaurant_id, review_id)
            
            # 포인트 존재 여부 확인
            if not self._check_point_exists(point_id):
                logger.warning(f"리뷰 {review_id}를 찾을 수 없습니다 (point_id: {point_id})")
                return {
                    "action": "not_found",
                    "review_id": review_id,
                    "point_id": point_id
                }
            
            # 삭제 실행
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            
            logger.info(f"리뷰 {review_id} 삭제 완료 (point_id: {point_id})")
            
            return {
                "action": "deleted",
                "review_id": review_id,
                "point_id": point_id
            }
            
        except Exception as e:
            logger.error(f"리뷰 삭제 중 오류: {str(e)}")
            raise
    
    def delete_reviews_batch(
        self,
        restaurant_id: Union[int, str],
        review_ids: List[Union[int, str]],
    ) -> Dict[str, Any]:
        """
        여러 리뷰를 배치로 삭제합니다.
        
        Args:
            restaurant_id: 레스토랑 ID
            review_ids: 리뷰 ID 리스트
            
        Returns:
            {
                "results": [
                    {
                        "action": "deleted" | "not_found",
                        "review_id": str,
                        "point_id": str
                    },
                    ...
                ],
                "total": int,
                "deleted_count": int,
                "not_found_count": int
            }
        """
        if not review_ids:
            return {
                "results": [],
                "total": 0,
                "deleted_count": 0,
                "not_found_count": 0
            }
        
        try:
            # Point ID 리스트 생성
            point_ids = []
            review_id_to_point_id = {}
            
            for review_id in review_ids:
                point_id = self._get_point_id(restaurant_id, review_id)
                point_ids.append(point_id)
                review_id_to_point_id[review_id] = point_id
            
            # 존재 여부 확인
            existing_points = {}
            try:
                retrieved = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=point_ids
                )
                for point in retrieved:
                    existing_points[point.id] = point.payload
            except Exception as e:
                logger.warning(f"포인트 존재 여부 확인 중 오류: {str(e)}")
            
            # 존재하는 포인트만 삭제
            existing_point_ids = [pid for pid in point_ids if pid in existing_points]
            
            if existing_point_ids:
                # 배치 삭제 실행
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=existing_point_ids
                    )
                )
                logger.info(f"총 {len(existing_point_ids)}개의 리뷰를 삭제했습니다.")
            
            # 결과 구성
            results = []
            for review_id in review_ids:
                point_id = review_id_to_point_id[review_id]
                if point_id in existing_points:
                    results.append({
                        "action": "deleted",
                        "review_id": review_id,
                        "point_id": point_id
                    })
                else:
                    results.append({
                        "action": "not_found",
                        "review_id": review_id,
                        "point_id": point_id
                    })
            
            deleted_count = sum(1 for r in results if r["action"] == "deleted")
            not_found_count = sum(1 for r in results if r["action"] == "not_found")
            
            logger.info(f"✅ 배치 삭제 완료: {deleted_count}개 삭제, {not_found_count}개 미발견")
            
            return {
                "results": results,
                "total": len(results),
                "deleted_count": deleted_count,
                "not_found_count": not_found_count
            }
            
        except Exception as e:
            logger.error(f"배치 삭제 중 오류: {str(e)}")
            raise
    
    def upsert_reviews_batch(
        self,
        restaurant_id: str,
        restaurant_name: str,
        reviews: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        여러 리뷰를 배치로 upsert합니다. (성능 최적화)
        
        배치 처리로 벡터 인코딩과 Qdrant upsert를 효율적으로 수행합니다.
        단, update_filter는 지원하지 않습니다 (중복 방지만 가능).
        
        Args:
            restaurant_id: 레스토랑 ID
            restaurant_name: 레스토랑 이름
            reviews: 리뷰 딕셔너리 리스트 (review_id, review, user_id, datetime, group, images, version 등)
            batch_size: 벡터 인코딩 배치 크기 (None이면 자동으로 최적 크기 사용)
            
        Returns:
            각 리뷰의 upsert 결과 리스트:
            [
                {
                    "action": "inserted" | "updated",
                    "review_id": str,
                    "version": int,
                    "point_id": str
                },
                ...
            ]
        """
        if not reviews:
            return []
        
        if batch_size is None:
            batch_size = self.batch_size
        
        results = []
        all_points = []
        point_id_to_review = {}  # point_id -> review 매핑
        
        try:
            # 1. 리뷰 검증 및 텍스트 추출
            review_texts = []
            valid_reviews = []
            
            for review in reviews:
                if not validate_review_data(review):
                    logger.warning(f"리뷰 데이터가 유효하지 않습니다: {review.get('review_id')}")
                    results.append({
                        "action": "error",
                        "review_id": review.get("review_id", "unknown"),
                        "error": "리뷰 데이터가 유효하지 않습니다."
                    })
                    continue
                
                # : id 필드 사용
                review_id = review.get("id") or review.get("review_id")
                if not review_id:
                    logger.warning("id (review_id)가 없는 리뷰를 건너뜁니다.")
                    results.append({
                        "action": "error",
                        "review_id": "unknown",
                        "error": "id (review_id)가 필요합니다."
                    })
                    continue
                
                # : content 필드
                review_text = review.get("content") or review.get("review", "")
                if not review_text:
                    logger.warning(f"content가 없는 리뷰를 건너뜁니다: {review_id}")
                    results.append({
                        "action": "error",
                        "review_id": review_id,
                        "error": "content가 필요합니다."
                    })
                    continue
                
                review_texts.append(review_text)
                valid_reviews.append(review)
            
            if not valid_reviews:
                return results
            
            # 2. 배치로 벡터 인코딩 (성능 최적화)
            logger.info(f"총 {len(valid_reviews)}개의 리뷰를 배치로 인코딩합니다 (배치 크기: {batch_size})")
            
            all_vectors = []
            for i in range(0, len(review_texts), batch_size):
                batch_texts = review_texts[i:i + batch_size]
                try:
                    batch_vectors = self.encoder.encode(batch_texts)
                    all_vectors.extend(batch_vectors)
                except Exception as e:
                    logger.error(f"배치 인코딩 중 오류 발생 (배치 {i//batch_size + 1}): {str(e)}")
                    # 배치 실패 시 개별 처리
                    for text in batch_texts:
                        try:
                            vector = self.encoder.encode(text)
                            all_vectors.append(vector)
                        except Exception as e2:
                            logger.error(f"개별 인코딩 중 오류: {str(e2)}")
                            all_vectors.append(None)
            
            # 3. 포인트 생성
            for review, vector in zip(valid_reviews, all_vectors):
                if vector is None:
                    results.append({
                        "action": "error",
                        "review_id": review.get("review_id"),
                        "error": "벡터 인코딩 실패"
                    })
                    continue
                
                # : id 필드 우선 사용
                review_id = review.get("id") or review.get("review_id")
                # restaurant_id와 review_id를 문자열로 변환
                point_id = self._get_point_id(str(restaurant_id), str(review_id))
                
                # 버전 관리
                current_version = review.get("version", 1)
                new_version = current_version + 1
                
                # Payload 구성 ( 컬럼명)
                review_text = review.get("content") or review.get("review", "")
                payload = {
                    #  컬럼명
                    "id": int(review_id) if review_id and str(review_id).isdigit() else review_id,
                    "restaurant_id": int(restaurant_id) if restaurant_id and str(restaurant_id).isdigit() else restaurant_id,
                    "member_id": review.get("member_id") or (int(review.get("user_id")) if review.get("user_id") and str(review.get("user_id")).isdigit() else review.get("user_id")),
                    "group_id": review.get("group_id") or (int(review.get("group")) if review.get("group") and str(review.get("group")).isdigit() else review.get("group")),
                    "subgroup_id": review.get("subgroup_id"),
                    "content": review_text,
                    "is_recommended": review.get("is_recommended"),
                    "created_at": review.get("created_at") or review.get("datetime") or datetime.now().isoformat(),
                    "updated_at": review.get("updated_at") or datetime.now().isoformat(),
                    "deleted_at": review.get("deleted_at"),
                    # 하위 호환성 필드
                    "review_id": review_id,
                    "restaurant_name": restaurant_name,
                    "user_id": review.get("user_id") or review.get("member_id"),
                    "group": review.get("group") or review.get("group_id"),
                    "review": review_text,
                    "datetime": review.get("created_at") or review.get("datetime"),
                    "image_urls": extract_image_urls(review.get("images")),
                    "version": new_version,
                }
                
                point = models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
                
                all_points.append(point)
                point_id_to_review[point_id] = {
                    "review": review,
                    "new_version": new_version
                }
            
            if not all_points:
                return results
            
            # 4. 배치로 upsert (update_filter 없이 - 중복 방지만)
            logger.info(f"총 {len(all_points)}개의 포인트를 배치로 upsert합니다.")
            self.client.upsert(
                collection_name=self.collection_name,
                points=all_points
            )
            
            # 5. 결과 확인 (기존 포인트 존재 여부 확인)
            point_ids = [p.id for p in all_points]
            existing_points = {}
            
            try:
                retrieved = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=point_ids
                )
                for point in retrieved:
                    existing_points[point.id] = point.payload
            except Exception as e:
                logger.warning(f"기존 포인트 조회 중 오류: {str(e)}")
            
            # 6. 결과 구성
            for point_id in point_ids:
                review_info = point_id_to_review[point_id]
                review_id = review_info["review"]["review_id"]
                new_version = review_info["new_version"]
                
                existing = point_id in existing_points
                action = "updated" if existing else "inserted"
                
                results.append({
                    "action": action,
                    "review_id": review_id,
                    "version": new_version,
                    "point_id": point_id
                })
                
                logger.debug(f"리뷰 {review_id}: {action} (version {new_version})")
            
            logger.info(f"✅ 배치 upsert 완료: {len(results)}개 리뷰 처리")
            return results
            
        except Exception as e:
            logger.error(f"배치 upsert 중 오류: {str(e)}")
            raise


# 편의 함수들
def prepare_qdrant_points(
    data: Dict,
    encoder: SentenceTransformer,
    collection_name: str = Config.COLLECTION_NAME,
) -> List[PointStruct]:
    """
    레스토랑 데이터를 Qdrant 포인트로 변환하는 편의 함수.
    
    Args:
        data: 레스토랑 데이터 딕셔너리
        encoder: SentenceTransformer 인코더
        collection_name: 컬렉션 이름
        
    Returns:
        Qdrant PointStruct 리스트
    """
    # 임시 클라이언트 생성 (실제로는 전달받아야 함)
    from qdrant_client import QdrantClient
    temp_client = QdrantClient(":memory:")
    
    vector_search = VectorSearch(encoder, temp_client, collection_name)
    return vector_search.prepare_points(data)


def get_restaurant_reviews(
    qdrant_client: QdrantClient,
    restaurant_id: str,
    collection_name: str = Config.COLLECTION_NAME,
) -> List[Dict]:
    """
    레스토랑 ID로 리뷰를 조회하는 편의 함수.
    
    Args:
        qdrant_client: Qdrant 클라이언트
        restaurant_id: 레스토랑 ID
        collection_name: 컬렉션 이름
        
    Returns:
        리뷰 payload 리스트
    """
    # encoder는 실제로 필요 없지만 클래스 구조상 필요
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    vector_search = VectorSearch(encoder, qdrant_client, collection_name)
    return vector_search.get_restaurant_reviews(restaurant_id)


def query_similar_reviews(
    qdrant_client: QdrantClient,
    encoder: SentenceTransformer,
    query_text: str,
    restaurant_id: Optional[str] = None,
    collection_name: str = Config.COLLECTION_NAME,
    limit: int = 3,
    min_score: float = 0.0,
) -> List[Dict]:
    """
    의미 기반으로 유사한 리뷰를 검색하는 편의 함수.
    
    Args:
        qdrant_client: Qdrant 클라이언트
        encoder: SentenceTransformer 인코더
        query_text: 검색 쿼리 텍스트
        restaurant_id: 필터링할 레스토랑 ID (None이면 전체)
        collection_name: 컬렉션 이름
        limit: 반환할 최대 개수
        min_score: 최소 유사도 점수
        
    Returns:
        검색 결과 리스트 (payload와 score 포함)
    """
    vector_search = VectorSearch(encoder, qdrant_client, collection_name)
    return vector_search.query_similar_reviews(query_text, restaurant_id, limit, min_score)


def get_reviews_with_images(
    qdrant_client: QdrantClient,
    encoder: SentenceTransformer,
    query_text: str,
    collection_name: str = Config.COLLECTION_NAME,
    limit: int = 10,
    min_score: float = 0.0,
) -> List[Dict]:
    """
    이미지가 있는 리뷰를 검색하는 편의 함수.
    
    Args:
        qdrant_client: Qdrant 클라이언트
        encoder: SentenceTransformer 인코더
        query_text: 검색 쿼리 텍스트
        collection_name: 컬렉션 이름
        limit: 반환할 최대 개수
        min_score: 최소 유사도 점수
        
    Returns:
        이미지 URL이 있는 리뷰 리스트
    """
    vector_search = VectorSearch(encoder, qdrant_client, collection_name)
    return vector_search.get_reviews_with_images(query_text, limit, min_score)


# ==================== Phase 1: 대표 벡터 기반 비교군 선정 ====================

RESTAURANT_VECTORS_COLLECTION = "restaurant_vectors"

