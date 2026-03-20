"""
Retrieval 서비스 HTTP 클라이언트.

VectorSearch와 유사한 메서드 시그니처를 제공해 점진 전환에 사용한다.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, Union

import requests

from .config import Config


class RetrievalServiceClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.collection_name = Config.COLLECTION_NAME

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(self._url(path), params=params, timeout=self.timeout_seconds)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self._url(path), json=payload, timeout=self.timeout_seconds)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    # ---- VectorSearch 호환 메서드들 ----
    def get_restaurant_reviews(self, restaurant_id: Union[int, str]) -> List[Dict[str, Any]]:
        data = self._get(f"/api/v1/vector/reviews/{int(restaurant_id)}")
        return data.get("reviews", [])

    def get_recent_restaurant_reviews(self, restaurant_id: Union[int, str], limit: int = 100) -> List[Dict[str, Any]]:
        data = self._get(f"/api/v1/vector/reviews/{int(restaurant_id)}/recent", params={"limit": limit})
        return data.get("reviews", [])

    def get_all_reviews_for_all_average(self, limit: int = 5000) -> List[Dict[str, Any]]:
        data = self._get("/api/v1/vector/reviews/all", params={"limit": limit})
        return data.get("reviews", [])

    def query_hybrid_search(
        self,
        query_text: str,
        restaurant_id: Optional[Union[int, str]] = None,
        limit: int = 20,
        fallback_min_score: float = 0.2,
        dense_prefetch_limit: int = 200,
        sparse_prefetch_limit: int = 300,
        food_category_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "restaurant_id": int(restaurant_id) if restaurant_id is not None else None,
            "limit": limit,
            "fallback_min_score": fallback_min_score,
            "dense_prefetch_limit": dense_prefetch_limit,
            "sparse_prefetch_limit": sparse_prefetch_limit,
        }
        if food_category_id is not None:
            payload["food_category_id"] = food_category_id
        data = self._post("/api/v1/vector/search/hybrid", payload)
        out: List[Dict[str, Any]] = []
        for item in data.get("results", []):
            out.append({"payload": item.get("review", {}), "score": item.get("score", 0.0)})
        return out

    def _query_dense_only(
        self,
        query_text: str,
        restaurant_id: Optional[Union[int, str]] = None,
        limit: int = 20,
        min_score: float = 0.2,
        food_category_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query_text": query_text,
            "restaurant_id": int(restaurant_id) if restaurant_id is not None else None,
            "limit": limit,
            "min_score": min_score,
        }
        if food_category_id is not None:
            payload["food_category_id"] = food_category_id
        data = self._post("/api/v1/vector/search/dense", payload)
        out: List[Dict[str, Any]] = []
        for item in data.get("results", []):
            out.append({"payload": item.get("review", {}), "score": item.get("score", 0.0)})
        return out

    def prepare_points(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 원격 서비스에서는 실제 PointStruct를 다루지 않으므로 길이 계산 용도만 제공.
        return [{} for _ in data.get("reviews", [])]

    def upload_collection(self, points: List[Any]) -> None:
        # 원격 모드에서는 vector router가 직접 upload 엔드포인트를 사용하도록 유도한다.
        raise RuntimeError("retrieval-service 모드에서는 upload_collection 직접 호출을 지원하지 않습니다.")

    def get_expected_point_ids(self, data: Dict[str, Any]) -> Set[str]:
        # 로컬 로직을 그대로 유지해 중복 업로드 스킵 호환.
        ids: Set[str] = set()
        reviews = list(data.get("reviews", []))
        for review in reviews:
            review_id = review.get("id") or review.get("review_id")
            restaurant_id = review.get("restaurant_id")
            content = f"{restaurant_id or ''}:{review_id or ''}"
            ids.add(hashlib.md5(content.encode()).hexdigest())
        return ids

    def get_collection_points_count(self) -> int:
        data = self._get("/api/v1/vector/collection/stats")
        return int(data.get("points_count", 0))

    def get_existing_point_ids(self, limit: int = 10000) -> Set[str]:
        data = self._get("/api/v1/vector/collection/point-ids", params={"limit": limit})
        return set(data.get("point_ids", []))

    def upsert_restaurant_vector(
        self,
        restaurant_id: Union[int, str],
        restaurant_name: str,
        food_category_id: Optional[int] = None,
    ) -> bool:
        payload: Dict[str, Any] = {"restaurant_id": int(restaurant_id), "restaurant_name": restaurant_name}
        if food_category_id is not None:
            payload["food_category_id"] = food_category_id
        data = self._post("/api/v1/vector/restaurants/upsert-vector", payload)
        return bool(data.get("success", False))
