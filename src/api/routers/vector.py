"""
벡터 검색 라우터
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ...vector_search import VectorSearch

logger = logging.getLogger(__name__)
from ...models import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
    VectorUploadRequest,
    VectorUploadResponse,
)
from ..dependencies import get_vector_search

router = APIRouter()


@router.post("/search/similar", response_model=VectorSearchResponse)
def search_similar_reviews(
    request: VectorSearchRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    의미 기반 검색(벡터검색)을 통해 유사한 리뷰를 검색합니다 ( 기반).
    
    반환: id, restaurant_id, content, score (유사도 점수)
    
    - **query_text**: 검색 쿼리 텍스트
    - **restaurant_id**: 레스토랑 ID 필터 (선택사항, None이면 전체 검색)
    - **limit**: 반환할 최대 개수 (기본값: 3, 최대: 100)
    - **min_score**: 최소 유사도 점수 (기본값: 0.0)
    """
    try:
        results = vector_search.query_similar_reviews(
            query_text=request.query_text,
            restaurant_id=request.restaurant_id,
            limit=request.limit,
            min_score=request.min_score,
        )
        # VectorSearchResult 형식으로 변환 (review와 score 포함)
        # payload의 id/review_id가 없을 수 있어 ReviewModel 호환 dict로 변환
        def _to_review_model(payload: dict) -> dict:
            rid = payload.get("id") or payload.get("review_id")
            if rid is not None and str(rid).isdigit():
                rid = int(rid)
            elif rid is None:
                rid = 0  # 기존 데이터 호환용
            return {
                "id": rid,
                "restaurant_id": int(payload["restaurant_id"]) if payload.get("restaurant_id") and str(payload["restaurant_id"]).isdigit() else payload.get("restaurant_id", 0),
                "content": payload.get("content") or payload.get("review", ""),
            }
        search_results = [
            VectorSearchResult(
                review=_to_review_model(r["payload"]),
                score=r["score"]
            )
            for r in results
        ]
        return VectorSearchResponse(results=search_results, total=len(search_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 검색 중 오류 발생: {str(e)}")


@router.post("/upload", response_model=VectorUploadResponse)
async def upload_vector_data(
    request: VectorUploadRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    벡터 데이터를 벡터 데이터베이스에 업로드합니다 ( 기반).
    
    - **reviews**: 리뷰 리스트 (id, restaurant_id, content, created_at 필수)
    - **restaurants**: 레스토랑 리스트 (id 선택, name, reviews; 선택사항)
    """
    try:
        #  형식으로 변환
        data = {
            "reviews": request.reviews,
            "restaurants": request.restaurants or []
        }
        logger.info(f"포인트 준비 시작: 리뷰 {len(request.reviews)}개, 레스토랑 {len(request.restaurants or [])}개")
        points = vector_search.prepare_points(data)
        logger.info(f"포인트 준비 완료: {len(points)}개 포인트 생성됨")
        
        if not points:
            logger.warning("생성된 포인트가 없습니다.")
            return VectorUploadResponse(
                message="경고: 생성된 포인트가 없습니다",
                points_count=0,
                collection_name=vector_search.collection_name,
            )
        
        vector_search.upload_collection(points)
        logger.info(f"업로드 완료: {len(points)}개 포인트")
        
        # restaurant_vectors 컬렉션 자동 생성 (비교군 검색 최적화)
        try:
            # 업로드된 리뷰에서 모든 레스토랑 ID 추출
            restaurant_ids = set()
            restaurant_info_map = {}  # restaurant_id -> {name, food_category_id}
            
            # 리뷰에서 레스토랑 ID 수집
            for review in request.reviews:
                rid = review.get("restaurant_id") if isinstance(review, dict) else getattr(review, "restaurant_id", None)
                if rid:
                    restaurant_ids.add(str(rid))
            
            # 레스토랑 정보 매핑 (request.restaurants에서)
            for restaurant in request.restaurants or []:
                rid = restaurant.get("id") if isinstance(restaurant, dict) else getattr(restaurant, "id", None)
                if rid:
                    restaurant_info_map[str(rid)] = {
                        "name": restaurant.get("name") if isinstance(restaurant, dict) else getattr(restaurant, "name", f"Restaurant {rid}"),
                        "food_category_id": restaurant.get("food_category_id") if isinstance(restaurant, dict) else getattr(restaurant, "food_category_id", None),
                    }
            
            # 각 레스토랑의 대표 벡터 생성
            vectors_created = 0
            for rid in restaurant_ids:
                try:
                    info = restaurant_info_map.get(rid, {})
                    restaurant_name = info.get("name", f"Restaurant {rid}")
                    food_category_id = info.get("food_category_id")
                    
                    success = vector_search.upsert_restaurant_vector(
                        restaurant_id=rid,
                        restaurant_name=restaurant_name,
                        food_category_id=food_category_id,
                    )
                    if success:
                        vectors_created += 1
                except Exception as e:
                    logger.warning(f"레스토랑 {rid} 대표 벡터 생성 실패: {e}")
                    continue
            
            if vectors_created > 0:
                logger.info(f"restaurant_vectors 컬렉션에 {vectors_created}개 레스토랑 대표 벡터 생성 완료")
        except Exception as e:
            logger.warning(f"restaurant_vectors 자동 생성 중 오류 (비교군 검색은 fallback 사용): {e}")
        
        return VectorUploadResponse(
            message="데이터 업로드 완료",
            points_count=len(points),
            collection_name=vector_search.collection_name,
        )
    except Exception as e:
        logger.error(f"데이터 업로드 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터 업로드 중 오류 발생: {str(e)}")
