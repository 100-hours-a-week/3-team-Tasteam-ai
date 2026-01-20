"""
벡터 검색 라우터
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ...vector_search import VectorSearch

logger = logging.getLogger(__name__)
from ...models import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
    ReviewImageSearchRequest,
    ReviewImageSearchResponse,
    VectorUploadRequest,
    VectorUploadResponse,
    RestaurantReviewsResponse,
    UpsertReviewRequest,
    UpsertReviewResponse,
    UpsertReviewsBatchRequest,
    UpsertReviewsBatchResponse,
    DeleteReviewRequest,
    DeleteReviewResponse,
    DeleteReviewsBatchRequest,
    DeleteReviewsBatchResponse,
)
from ..dependencies import get_vector_search, get_llm_utils, get_metrics_collector
from ...llm_utils import LLMUtils
from ...metrics_collector import MetricsCollector

router = APIRouter()


@router.post("/search/similar", response_model=VectorSearchResponse)
async def search_similar_reviews(
    request: VectorSearchRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
    llm_utils: LLMUtils = Depends(get_llm_utils),
):
    """
    의미 기반 검색(벡터검색)을 통해 유사한 리뷰를 검색합니다 ( 기반).
    
    모든 메타데이터를 포함하여 반환합니다 ( 컬럼명):
    - id, restaurant_id, member_id, group_id, subgroup_id
    - content, is_recommended
    - created_at, updated_at, deleted_at
    - score (유사도 점수)
    
    - **query_text**: 검색 쿼리 텍스트
    - **restaurant_id**: 레스토랑 ID 필터 (선택사항, None이면 전체 검색)
    - **limit**: 반환할 최대 개수 (기본값: 3, 최대: 100)
    - **min_score**: 최소 유사도 점수 (기본값: 0.0)
    - **expand_query**: 쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)
    """
    try:
        results = await vector_search.query_similar_reviews_with_expansion(
            query_text=request.query_text,
            restaurant_id=request.restaurant_id,
            limit=request.limit,
            min_score=request.min_score,
            expand_query=request.expand_query,
            llm_utils=llm_utils,
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
    
    - **reviews**: 리뷰 리스트 (REVIEW TABLE -  기반)
    - **restaurants**: 레스토랑 리스트 (선택사항, RESTAURANT TABLE)
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


@router.post("/search/review-images", response_model=ReviewImageSearchResponse)
async def search_review_images(
    request: ReviewImageSearchRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
    llm_utils: LLMUtils = Depends(get_llm_utils),
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """
    리뷰 이미지를 검색합니다 ( 기반, Query Expansion 지원).
    
    프로세스 ():
    1. 벡터 검색으로 리뷰 검색 (query, Query Expansion 적용 가능)
    2. REVIEW TABLE + REVIEW_IMAGE TABLE JOIN
    3. 출력: restaurant_id, 리뷰 id, 리뷰 image_url
    
    - **query**: 검색 쿼리 (예: "분위기 좋다", "데이트하기 좋은")
    - **restaurant_id**: 레스토랑 ID 필터 (선택사항)
    - **limit**: 반환할 최대 개수 (기본값: 10)
    - **min_score**: 최소 유사도 점수 (기본값: 0.0)
    - **expand_query**: 쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)
    
    Returns ():
        - results: 리뷰 이미지 검색 결과 리스트
            - restaurant_id: 레스토랑 ID
            - review_id: 리뷰 ID
            - image_url: 이미지 URL
            - review: 리뷰 정보 (REVIEW TABLE)
        - total: 총 결과 개수
    """
    start_time = time.time()
    original_query = request.query
    expanded_query = request.query
    query_expanded = False
    vllm_metrics = None
    
    try:
        # 1. 쿼리 확장 여부 결정 및 실행 (메트릭 수집)
        should_expand = False
        if request.expand_query is not False:  # None 또는 True
            if request.expand_query is True:
                should_expand = True
            elif request.expand_query is None:
                # 자동 판단
                should_expand = vector_search._should_expand_query(request.query)
            
            if should_expand:
                try:
                    expanded_query, vllm_metrics = await llm_utils.expand_query_for_dense_search_with_metrics(
                        request.query
                    )
                    if expanded_query and expanded_query != request.query:
                        query_expanded = True
                except Exception as e:
                    logger.warning(f"쿼리 확장 실패, 원본 사용: {e}")
                    expanded_query = request.query
        
        # 2. 벡터 검색으로 리뷰 검색 (확장된 쿼리 사용, 확장 비활성화)
        results = await vector_search.get_reviews_with_images(
            query_text=expanded_query,
            limit=request.limit,
            min_score=request.min_score,
            expand_query=False,  # 이미 확장했으므로 False
            llm_utils=llm_utils,
        )
        
        # 3. REVIEW_IMAGE TABLE 정보 추출 (payload의 image_urls 사용)
        review_image_results = []
        for result in results:
            payload = result["payload"]
            image_urls = result.get("image_urls", [])
            
            # 이미지가 있는 경우만 반환 (get_reviews_with_images에서 이미 필터링됨)
            if image_urls:
                review_id = payload.get("id") or payload.get("review_id")
                restaurant_id = payload.get("restaurant_id")
                
                # 각 이미지마다 결과 생성
                for image_url in image_urls:
                    review_image_results.append({
                        "restaurant_id": restaurant_id,
                        "review_id": review_id,
                        "image_url": image_url,
                        "review": payload  # REVIEW TABLE 정보
                    })
        
        # 4. 메트릭 수집
        request_id = metrics.collect_metrics(
            restaurant_id=None,  # 이미지 검색은 레스토랑 ID 없을 수 있음
            analysis_type="image_search",
            start_time=start_time,
            tokens_used=vllm_metrics.get("total_tokens") if vllm_metrics else None,
            additional_info={
                "original_query": original_query,
                "expanded_query": expanded_query,
                "query_expanded": query_expanded,
                "results_count": len(review_image_results),
            },
            ttft_ms=vllm_metrics.get("ttft_ms") if vllm_metrics else None,
        )
        
        # 5. vLLM 메트릭 저장 (쿼리 확장이 실행된 경우)
        if vllm_metrics and query_expanded:
            metrics.collect_vllm_metrics(
                request_id=request_id,
                restaurant_id=None,
                analysis_type="image_search",
                vllm_metrics=vllm_metrics,
            )
        
        return ReviewImageSearchResponse(
            results=review_image_results,
            total=len(review_image_results)
        )
    except Exception as e:
        # 에러 발생 시에도 메트릭 수집
        metrics.collect_metrics(
            restaurant_id=None,
            analysis_type="image_search",
            start_time=start_time,
            error_count=1,
            additional_info={
                "original_query": original_query,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=f"리뷰 이미지 검색 중 오류 발생: {str(e)}")


@router.post("/reviews/upsert", response_model=UpsertReviewResponse)
async def upsert_review(
    request: UpsertReviewRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    리뷰를 upsert합니다 (있으면 업데이트, 없으면 삽입).
    update_filter를 사용하여 낙관적 잠금(Optimistic Locking)을 지원합니다.
    
    **동작 방식:**
    1. `update_version`이 None이면: 항상 업데이트/삽입 (중복 방지)
    2. `update_version`이 지정되면: 해당 버전일 때만 업데이트 (낙관적 잠금)
    
    **사용 시나리오:**
    - **리뷰 추가/수정 (중복 방지)**: `update_version=None`
      - 같은 review_id가 있으면 자동으로 업데이트
      - 없으면 새로 삽입
      
    - **리뷰 수정 (동시성 제어)**: `update_version=3`
      - 현재 버전이 3일 때만 업데이트
      - 다른 사용자가 먼저 수정했다면 (version이 4 이상) 스킵
    
    **요청 예시:**
    ```json
    {
        "restaurant_id": "res_1234",
        "restaurant_name": "비즐",
        "review": {
            "review_id": "rev_3001",
            "review": "맛있어요!",
            "user_id": "user_123",
            "datetime": "2024-01-01T12:00:00",
            "group": "group_1",
            "version": 3
        },
        "update_version": 3  // 이 버전일 때만 업데이트
    }
    ```
    
    **응답:**
    - `action`: "inserted" (새로 삽입), "updated" (업데이트), "skipped" (스킵)
    - `version`: 새로운 버전 번호
    - `reason`: skipped인 경우 이유 ("version_mismatch" 등)
    """
    try:
        #  형식으로 변환
        restaurant_id = request.restaurant.id if hasattr(request.restaurant, 'id') else request.restaurant.get("id")
        restaurant_name = request.restaurant.name if hasattr(request.restaurant, 'name') else request.restaurant.get("name")
        
        # review를 딕셔너리로 변환
        if hasattr(request.review, 'dict'):
            review_dict = request.review.dict()
        elif isinstance(request.review, dict):
            review_dict = request.review
        else:
            review_dict = request.review
        
        result = vector_search.upsert_review(
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
            review=review_dict,
            update_version=request.update_version,
        )
        return UpsertReviewResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리뷰 upsert 중 오류 발생: {str(e)}")


@router.post("/reviews/upsert/batch", response_model=UpsertReviewsBatchResponse)
async def upsert_reviews_batch(
    request: UpsertReviewsBatchRequest,
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    여러 리뷰를 배치로 upsert합니다. (성능 최적화)
    
    **특징:**
    - 배치 벡터 인코딩으로 성능 향상
    - 배치 Qdrant upsert로 효율적인 처리
    - 10개 리뷰를 1번의 API 호출로 처리 가능
    
    **제한사항:**
    - `update_filter`는 지원하지 않습니다 (중복 방지만 가능)
    - 낙관적 잠금이 필요한 경우 개별 upsert 엔드포인트 사용
    
    **요청 예시:**
    ```json
    {
        "restaurant_id": "res_1234",
        "restaurant_name": "비즐",
        "reviews": [
            {
                "review_id": "rev_3001",
                "review": "맛있어요!",
                "user_id": "user_123",
                "datetime": "2024-01-01T12:00:00",
                "group": "group_1",
                "version": 1
            },
            {
                "review_id": "rev_3002",
                "review": "좋아요!",
                "user_id": "user_124",
                "datetime": "2024-01-01T12:01:00",
                "group": "group_1",
                "version": 1
            }
        ],
        "batch_size": 32
    }
    ```
    
    **응답:**
    - `results`: 각 리뷰의 upsert 결과 리스트
    - `total`: 총 처리된 리뷰 수
    - `success_count`: 성공한 리뷰 수 (inserted + updated)
    - `error_count`: 실패한 리뷰 수
    """
    try:
        #  형식으로 변환
        restaurant_id = request.restaurant.id if hasattr(request.restaurant, 'id') else request.restaurant.get("id")
        restaurant_name = request.restaurant.name if hasattr(request.restaurant, 'name') else request.restaurant.get("name")
        
        # restaurant_id를 문자열로 변환 (upsert_reviews_batch가 str을 기대)
        restaurant_id = str(restaurant_id) if restaurant_id is not None else ""
        
        # reviews를 딕셔너리 리스트로 변환
        reviews_list = []
        for review in request.reviews:
            if hasattr(review, 'dict'):
                reviews_list.append(review.dict())
            elif isinstance(review, dict):
                reviews_list.append(review)
            else:
                reviews_list.append(review)
        
        results = vector_search.upsert_reviews_batch(
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
            reviews=reviews_list,
            batch_size=request.batch_size,
        )
        
        # 통계 계산
        success_count = sum(1 for r in results if r.get("action") in ["inserted", "updated"])
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
        raise HTTPException(status_code=500, detail=f"배치 upsert 중 오류 발생: {str(e)}")


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

