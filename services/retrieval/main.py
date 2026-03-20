"""
Retrieval 전용 FastAPI 서비스.

- Qdrant/VectorSearch 접근을 메인 API에서 분리하기 위한 서비스
- 업로드/검색/리뷰 조회/비교군 조회를 HTTP API로 노출
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from src.config import Config
from src.models import (
    HealthResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchResult,
    VectorUploadRequest,
    VectorUploadResponse,
    ReviewModel,
)
from src.vector_search import VectorSearch


app = FastAPI(
    title="Retrieval Service",
    description="Qdrant/VectorSearch 전용 서비스",
    version="1.0.0",
)


def _build_qdrant_client() -> QdrantClient:
    qdrant_url = Config.QDRANT_URL
    if qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    if qdrant_url.startswith(("http://", "https://")):
        return QdrantClient(url=qdrant_url)

    qdrant_path = qdrant_url
    if not os.path.isabs(qdrant_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qdrant_path = os.path.join(project_root, qdrant_path)
    os.makedirs(qdrant_path, exist_ok=True)
    return QdrantClient(path=qdrant_path)


@lru_cache()
def get_vector_search() -> VectorSearch:
    client = _build_qdrant_client()
    return VectorSearch(qdrant_client=client, collection_name=Config.COLLECTION_NAME)


def _to_review_model(payload: Dict[str, Any]) -> ReviewModel:
    return ReviewModel(
        id=int(payload.get("id") or payload.get("review_id") or 0),
        restaurant_id=int(payload.get("restaurant_id") or 0),
        content=payload.get("content") or payload.get("review") or "",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/api/v1/vector/upload", response_model=VectorUploadResponse)
async def upload_vector_data(request: VectorUploadRequest) -> VectorUploadResponse:
    try:
        vector_search = get_vector_search()
        data = {
            "reviews": [r.model_dump() for r in request.reviews],
            "restaurants": [r.model_dump() for r in (request.restaurants or [])],
        }
        points = vector_search.prepare_points(data)
        if not points:
            return VectorUploadResponse(
                message="경고: 생성된 포인트가 없습니다",
                points_count=0,
                collection_name=vector_search.collection_name,
            )
        vector_search.upload_collection(points)
        return VectorUploadResponse(
            message="데이터 업로드 완료",
            points_count=len(points),
            collection_name=vector_search.collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벡터 업로드 실패: {e}")


@app.post("/api/v1/vector/search/hybrid", response_model=VectorSearchResponse)
async def search_hybrid(request: VectorSearchRequest) -> VectorSearchResponse:
    try:
        vector_search = get_vector_search()
        hits = vector_search.query_hybrid_search(
            query_text=request.query_text,
            restaurant_id=request.restaurant_id,
            limit=request.limit,
            fallback_min_score=request.fallback_min_score,
            dense_prefetch_limit=request.dense_prefetch_limit,
            sparse_prefetch_limit=request.sparse_prefetch_limit,
        )
        results: List[VectorSearchResult] = []
        for hit in hits:
            payload = hit.get("payload", {}) or {}
            results.append(
                VectorSearchResult(
                    review=_to_review_model(payload),
                    score=float(hit.get("score", 0.0)),
                )
            )
        return VectorSearchResponse(results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"하이브리드 검색 실패: {e}")


@app.get("/api/v1/vector/reviews/{restaurant_id}")
async def get_restaurant_reviews(restaurant_id: int, limit: int = 10000) -> Dict[str, Any]:
    try:
        vector_search = get_vector_search()
        reviews = vector_search.get_restaurant_reviews(restaurant_id)
        return {"restaurant_id": restaurant_id, "total": min(len(reviews), limit), "reviews": reviews[:limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리뷰 조회 실패: {e}")


@app.get("/api/v1/vector/reviews/{restaurant_id}/recent")
async def get_recent_restaurant_reviews(restaurant_id: int, limit: int = 100) -> Dict[str, Any]:
    try:
        vector_search = get_vector_search()
        reviews = vector_search.get_recent_restaurant_reviews(restaurant_id=restaurant_id, limit=limit)
        return {"restaurant_id": restaurant_id, "total": len(reviews), "reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"최근 리뷰 조회 실패: {e}")


@app.get("/api/v1/vector/restaurants/{restaurant_id}/similar")
async def find_similar_restaurants(
    restaurant_id: int,
    top_n: int = 20,
    food_category_id: Optional[int] = None,
) -> Dict[str, Any]:
    try:
        vector_search = get_vector_search()
        similar = vector_search.find_similar_restaurants(
            target_restaurant_id=restaurant_id,
            top_n=top_n,
            food_category_id=food_category_id,
            exclude_self=True,
        )
        return {"restaurant_id": restaurant_id, "total": len(similar), "results": similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"비교군 조회 실패: {e}")
