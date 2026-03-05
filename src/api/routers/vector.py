"""
벡터 업로드 라우터 (검색 API는 제거됨)

POST /upload: 리뷰·레스토랑 데이터를 벡터 DB에 업로드.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from ...vector_search import VectorSearch
from ...models import VectorUploadRequest, VectorUploadResponse
from ..dependencies import get_vector_search

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=VectorUploadResponse)
async def upload_vector_data(
    request: VectorUploadRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    벡터 데이터를 벡터 DB에 업로드합니다.

    - **reviews**: 리뷰 리스트 (id, restaurant_id, content, created_at 필수)
    - **restaurants**: 레스토랑 리스트 (id 선택, name, reviews; 선택사항)
    """
    try:
        data = {
            "reviews": [r.model_dump() for r in request.reviews],
            "restaurants": [r.model_dump() for r in (request.restaurants or [])],
        }
        # 중복 업로드 스킵: Qdrant count + 요청 리뷰 수 비교 후, 동일하면 기존 포인트 ID와 요청 ID 집합 비교
        expected_ids = vector_search.get_expected_point_ids(data)
        if expected_ids:
            points_count = vector_search.get_collection_points_count()
            if points_count == len(expected_ids):
                existing_ids = vector_search.get_existing_point_ids(limit=points_count + 500)
                if len(existing_ids) == points_count and expected_ids == existing_ids:
                    logger.info(
                        "중복 업로드 방지: 동일 데이터가 이미 존재하여 스킵합니다 (count=%s, id 집합 일치)",
                        points_count,
                    )
                    return VectorUploadResponse(
                        message="중복 업로드 방지: 동일 데이터가 이미 존재하여 스킵합니다",
                        points_count=len(expected_ids),
                        collection_name=vector_search.collection_name,
                    )

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
            restaurant_ids = set()
            restaurant_info_map = {}

            for review in request.reviews:
                rid = review.restaurant_id
                if rid is not None:
                    restaurant_ids.add(str(rid))

            for restaurant in request.restaurants or []:
                rid = restaurant.id
                if rid is not None:
                    restaurant_info_map[str(rid)] = {
                        "name": restaurant.name,
                        "food_category_id": getattr(restaurant, "food_category_id", None),
                    }

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
