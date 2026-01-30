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
    RestaurantReviewsResponse,
    UpsertReviewsRequest,
    UpsertReviewsBatchResponse,
    DeleteReviewRequest,
    DeleteReviewResponse,
    DeleteReviewsBatchRequest,
    DeleteReviewsBatchResponse,
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
    
    모든 메타데이터를 포함하여 반환합니다 ( 컬럼명, subgroup_id·is_recommended 미반환):
    - id, restaurant_id, member_id, group_id
    - content
    - created_at, updated_at (메타)
    - score (유사도 점수)
    
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
        search_results = [
            VectorSearchResult(
                review=r["payload"],
                score=r["score"]
            )
            for r in results
        ]
        return VectorSearchResponse(results=search_results, total=len(search_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 검색 중 오류 발생: {str(e)}")


@router.get("/restaurants/{restaurant_id}/reviews", response_model=RestaurantReviewsResponse)
async def get_restaurant_reviews(
    restaurant_id: str,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    레스토랑 ID로 리뷰를 조회합니다.
    """
    try:
        reviews = vector_search.get_restaurant_reviews(restaurant_id)
        return RestaurantReviewsResponse(
            restaurant_id=restaurant_id,
            reviews=reviews,
            total=len(reviews),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리뷰 조회 중 오류 발생: {str(e)}")


@router.post("/upload", response_model=VectorUploadResponse)
async def upload_vector_data(
    request: VectorUploadRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    벡터 데이터를 벡터 데이터베이스에 업로드합니다 ( 기반).
    
    - **reviews**: 리뷰 리스트 (id, restaurant_id, content; is_recommended, member_id, group_id, subgroup_id, updated_at, images 미사용. created_at은 메타로 선택)
    - **restaurants**: 레스토랑 리스트 (id, name, reviews만; full_address, location, created_at 미사용, 선택사항)
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


@router.post("/reviews/upsert", response_model=UpsertReviewsBatchResponse)
async def upsert_reviews(
    request: UpsertReviewsRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    리뷰를 upsert합니다. 요청 형식은 upload와 동일: {reviews, restaurants?, batch_size?}.
    
    - **reviews**: (id, restaurant_id, content)
    - **restaurants**: (id, name, reviews?) — restaurant_name 해석 및 중첩 리뷰
    - **batch_size**: 벡터 인코딩 배치 크기 (기본 32)
    
    **요청 예시 (upload와 동일 형식):**
    ```json
    {
      "reviews": [
        {"id": 1, "restaurant_id": 1, "content": "맛있어요"}
      ],
      "restaurants": [
        {"id": 1, "name": "테스트 음식점"}
      ],
      "batch_size": 32
    }
    ```
    """
    try:
        data = {
            "reviews": [r.model_dump() if hasattr(r, "model_dump") else r for r in request.reviews],
            "restaurants": [x.model_dump() if hasattr(x, "model_dump") else x for x in (request.restaurants or [])],
        }
        results = vector_search.upsert_reviews_from_data(data, batch_size=request.batch_size or 32)
        success_count = sum(1 for r in results if r.get("action") in ("inserted", "updated"))
        error_count = sum(1 for r in results if r.get("action") == "error")
        return UpsertReviewsBatchResponse(
            results=results,
            total=len(results),
            success_count=success_count,
            error_count=error_count,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리뷰 upsert 중 오류: {str(e)}")


@router.delete("/reviews/delete", response_model=DeleteReviewResponse)
async def delete_review(
    request: DeleteReviewRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    리뷰를 삭제합니다.
    
    **동작 방식:**
    - review_id를 기반으로 Point ID를 생성하여 삭제
    - 리뷰가 존재하지 않으면 "not_found" 반환
    
    **요청 예시:**
    ```json
    {
        "restaurant_id": "res_1234",
        "review_id": "rev_3001"
    }
    ```
    
    **응답:**
    - `action`: "deleted" (삭제됨), "not_found" (찾을 수 없음)
    - `review_id`: 리뷰 ID
    - `point_id`: Point ID
    """
    try:
        result = vector_search.delete_review(
            restaurant_id=request.restaurant_id,
            review_id=request.review_id,
        )
        return DeleteReviewResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리뷰 삭제 중 오류 발생: {str(e)}")


@router.delete("/reviews/delete/batch", response_model=DeleteReviewsBatchResponse)
async def delete_reviews_batch(
    request: DeleteReviewsBatchRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    여러 리뷰를 배치로 삭제합니다.
    
    **특징:**
    - 여러 리뷰를 한 번에 삭제하여 성능 향상
    - 존재하지 않는 리뷰는 자동으로 건너뜀
    
    **요청 예시:**
    ```json
    {
        "restaurant_id": "res_1234",
        "review_ids": ["rev_3001", "rev_3002", "rev_3003"]
    }
    ```
    
    **응답:**
    - `results`: 각 리뷰의 삭제 결과 리스트
    - `total`: 총 처리된 리뷰 수
    - `deleted_count`: 삭제된 리뷰 수
    - `not_found_count`: 찾을 수 없는 리뷰 수
    """
    try:
        result = vector_search.delete_reviews_batch(
            restaurant_id=request.restaurant_id,
            review_ids=request.review_ids,
        )
        return DeleteReviewsBatchResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 삭제 중 오류 발생: {str(e)}")

