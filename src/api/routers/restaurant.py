"""
레스토랑 관련 라우터
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ...models import RestaurantReviewsResponse

router = APIRouter()


# TODO: 레스토랑 이름으로 리뷰 조회 기능은 현재 미구현 상태입니다.
# 필요한 경우 VectorSearch를 사용하여 구현해야 합니다.
# @router.get("/{restaurant_name}/reviews", response_model=dict)
# async def get_reviews_by_name(
#     restaurant_name: str,
#     vector_search: VectorSearch = Depends(get_vector_search),
# ):
#     """
#     레스토랑 이름으로 리뷰를 조회합니다.
#     
#     - **restaurant_name**: 레스토랑 이름
#     """
#     raise HTTPException(status_code=501, detail="이 기능은 아직 구현되지 않았습니다.")

