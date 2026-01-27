"""
API 요청/응답 모델 정의 ( 기반)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== 기본 데이터 모델 ( 기반) ====================

class ReviewImageModel(BaseModel):
    """리뷰 이미지 모델 (REVIEW_IMAGE TABLE)"""
    id: Optional[int] = Field(None, description="이미지 ID (BIGINT PK)")
    review_id: int = Field(..., description="리뷰 ID (BIGINT FK)")
    image_url: str = Field(..., description="이미지 URL (VARCHAR(500))")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP)")


class ReviewModel(BaseModel):
    """리뷰 모델 (REVIEW TABLE -  기반)"""
    id: Optional[int] = Field(None, description="리뷰 ID (BIGINT PK)")
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    member_id: Optional[int] = Field(None, description="회원 ID (BIGINT FK)")
    group_id: Optional[int] = Field(None, description="그룹 ID (BIGINT FK, 예: 10234, 12034)")
    subgroup_id: Optional[int] = Field(None, description="서브그룹 ID (BIGINT FK, 예: 10234, 12034)")
    content: str = Field(..., description="리뷰 내용 (VARCHAR(1000))")
    is_recommended: Optional[bool] = Field(None, description="추천 여부 (BOOLEAN)")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP)")
    updated_at: Optional[datetime] = Field(None, description="수정 시간 (TIMESTAMP)")
    deleted_at: Optional[datetime] = Field(None, description="삭제 시간 (TIMESTAMP)")
    
    # REVIEW_IMAGE 관계
    images: Optional[List[ReviewImageModel]] = Field(None, description="리뷰 이미지 리스트")


class RestaurantModel(BaseModel):
    """레스토랑 모델 (RESTAURANT TABLE -  기반)"""
    id: Optional[int] = Field(None, description="레스토랑 ID (BIGINT PK)")
    name: str = Field(..., description="레스토랑 이름 (VARCHAR(100))")
    full_address: Optional[str] = Field(None, description="전체 주소 (VARCHAR(255))")
    location: Optional[Dict[str, Any]] = Field(None, description="위치 정보 (geometry(Point,4326))")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP)")
    deleted_at: Optional[datetime] = Field(None, description="삭제 시간 (TIMESTAMP)")


class FoodCategoryModel(BaseModel):
    """음식 카테고리 모델 (FOOD_CATEGORY TABLE -  기반)"""
    id: int = Field(..., description="카테고리 ID (BIGINT PK)")
    name: str = Field(..., description="카테고리 이름 (VARCHAR(20))")


class RestaurantFoodCategoryModel(BaseModel):
    """레스토랑-음식 카테고리 관계 모델 (RESTAURANT_FOOD_CATEGORY -  기반)"""
    id: Optional[int] = Field(None, description="관계 ID (BIGINT PK)")
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    food_category_id: int = Field(..., description="음식 카테고리 ID (BIGINT FK)")


# ==================== Debug Info ====================

class DebugInfo(BaseModel):
    """디버그 정보 모델"""
    request_id: Optional[str] = Field(None, description="요청 ID")
    processing_time_ms: Optional[float] = Field(None, description="처리 시간 (밀리초)")
    tokens_used: Optional[int] = Field(None, description="사용된 토큰 수")
    model_version: Optional[str] = Field(None, description="모델 버전")
    warnings: Optional[List[str]] = Field(None, description="경고 메시지 리스트")


# ==================== Sentiment Analysis ====================

class SentimentAnalysisRequest(BaseModel):
    """감성 분석 요청 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트 (REVIEW TABLE)")


class SentimentAnalysisDisplayResponse(BaseModel):
    """감성 분석 표시용 응답 모델 (최소 필드)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    positive_ratio: int = Field(..., description="긍정 비율 (%) - 정수값")
    negative_ratio: int = Field(..., description="부정 비율 (%) - 정수값")


class SentimentAnalysisResponse(BaseModel):
    """감성 분석 응답 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    positive_count: int = Field(..., description="긍정 리뷰 개수")
    negative_count: int = Field(..., description="부정 리뷰 개수")
    neutral_count: int = Field(0, description="중립 리뷰 개수")
    total_count: int = Field(..., description="전체 리뷰 개수")
    positive_ratio: int = Field(..., description="긍정 비율 (%) - 정수값")
    negative_ratio: int = Field(..., description="부정 비율 (%) - 정수값")
    neutral_ratio: int = Field(0, description="중립 비율 (%) - 정수값")
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보")


class SentimentAnalysisBatchRequest(BaseModel):
    """배치 감성 분석 요청 모델"""
    restaurants: List[Dict[str, Any]] = Field(
        ...,
        description="레스토랑 데이터 리스트, 각 항목은 restaurant_id와 reviews를 포함"
    )
    max_tokens_per_batch: Optional[int] = Field(
        None,
        ge=1000,
        le=8000,
        description="배치당 최대 토큰 수 (None이면 동적 계산)"
    )


class SentimentAnalysisBatchResponse(BaseModel):
    """배치 감성 분석 응답 모델"""
    results: List[SentimentAnalysisResponse] = Field(..., description="각 레스토랑별 결과")


# ==================== Summary ====================

class SummaryVectorUploadRequest(BaseModel):
    """요약 벡터 업로드 요청 모델 ( 기반)"""
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트 (REVIEW TABLE)")


class SummaryVectorSearchRequest(BaseModel):
    """요약 벡터 검색 요청 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID (단일 ID만 허용)")
    query: str = Field(..., description="검색 쿼리 (예: '맛있다 좋다 친절하다')")
    limit: int = Field(10, ge=1, le=100, description="검색할 최대 리뷰 수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")


class SummaryRequest(BaseModel):
    """리뷰 요약 요청 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    positive_query: Optional[str] = Field("맛있다 좋다 만족", description="긍정 리뷰 검색 쿼리")
    negative_query: Optional[str] = Field("맛없다 별로 불만", description="부정 리뷰 검색 쿼리")
    limit: int = Field(10, ge=1, le=100, description="각 카테고리당 검색할 최대 리뷰 수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")


class SummaryAspect(BaseModel):
    """요약 aspect 모델"""
    aspect: str = Field(..., description="카테고리 (예: '불맛', '서비스', '가격')")
    claim: str = Field(..., description="구체적 주장 (1문장)")
    evidence_quotes: List[str] = Field(default_factory=list, description="근거 인용문 리스트 (최대 3개)")
    evidence_review_ids: List[str] = Field(default_factory=list, description="근거 리뷰 ID 리스트")


class CategorySummary(BaseModel):
    """카테고리별 요약 모델"""
    summary: str = Field(..., description="카테고리 요약")
    bullets: List[str] = Field(default_factory=list, description="핵심 포인트 리스트")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="근거 리스트 (review_id, snippet, rank)")


class SummaryDisplayResponse(BaseModel):
    """리뷰 요약 표시용 응답 모델 (최소 필드)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    overall_summary: str = Field(..., description="전체 요약")
    positive_aspects: List[SummaryAspect] = Field(default_factory=list, description="긍정 aspect 리스트")
    negative_aspects: List[SummaryAspect] = Field(default_factory=list, description="부정 aspect 리스트")
    categories: Optional[Dict[str, CategorySummary]] = Field(None, description="카테고리별 요약 (새 파이프라인)")


class SummaryResponse(BaseModel):
    """리뷰 요약 응답 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    overall_summary: str = Field(..., description="전체 요약 (positive_aspects + negative_aspects 기반)")
    positive_aspects: List[SummaryAspect] = Field(default_factory=list, description="긍정 aspect 리스트")
    negative_aspects: List[SummaryAspect] = Field(default_factory=list, description="부정 aspect 리스트")
    positive_reviews: List[ReviewModel] = Field(default_factory=list, description="긍정 리뷰 메타데이터")
    negative_reviews: List[ReviewModel] = Field(default_factory=list, description="부정 리뷰 메타데이터")
    positive_count: int = Field(0, description="긍정 리뷰 개수")
    negative_count: int = Field(0, description="부정 리뷰 개수")
    categories: Optional[Dict[str, CategorySummary]] = Field(None, description="카테고리별 요약 (새 파이프라인)")
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보")


class SummaryBatchRequest(BaseModel):
    """배치 리뷰 요약 요청 모델"""
    restaurants: List[Dict[str, Any]] = Field(
        ...,
        description="레스토랑 데이터 리스트, 각 항목은 restaurant_id, positive_query, negative_query, limit, min_score를 포함"
    )
    max_tokens_per_batch: Optional[int] = Field(
        None,
        ge=1000,
        le=8000,
        description="배치당 최대 토큰 수 (None이면 동적 계산)"
    )


class SummaryBatchResponse(BaseModel):
    """배치 리뷰 요약 응답 모델"""
    results: List[SummaryResponse] = Field(..., description="각 레스토랑별 요약 결과")


# ==================== Strength ====================

class StrengthVectorUploadRequest(BaseModel):
    """강점 추출 벡터 업로드 요청 모델 ( 기반)"""
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트 (REVIEW TABLE)")
    restaurant_food_categories: List[RestaurantFoodCategoryModel] = Field(
        ..., 
        description="레스토랑-음식 카테고리 관계 리스트"
    )
    food_categories: List[FoodCategoryModel] = Field(
        ..., 
        description="음식 카테고리 리스트"
    )


class StrengthRequestV2(BaseModel):
    """강점 추출 요청 모델 V2 (새로운 파이프라인)"""
    restaurant_id: int = Field(..., description="타겟 레스토랑 ID")
    strength_type: str = Field("both", description="강점 타입: 'representative', 'distinct', 'both'")
    category_filter: Optional[int] = Field(None, description="카테고리 필터")
    region_filter: Optional[str] = Field(None, description="지역 필터")
    price_band_filter: Optional[str] = Field(None, description="가격대 필터")
    top_k: int = Field(10, ge=1, le=50, description="반환할 최대 강점 개수")
    max_candidates: int = Field(300, ge=50, le=1000, description="근거 후보 최대 개수")
    months_back: int = Field(6, ge=1, le=24, description="최근 N개월 리뷰만 사용")
    min_support: int = Field(5, ge=1, le=50, description="최소 support_count (희소 환각 방지)")


class EvidenceSnippet(BaseModel):
    """근거 스니펫 모델"""
    review_id: str = Field(..., description="리뷰 ID")
    snippet: str = Field(..., description="짧은 인용문")
    rating: Optional[float] = Field(None, description="별점")
    created_at: str = Field(..., description="생성 시간")


class StrengthDetail(BaseModel):
    """강점 상세 (Kiwi+lift 파이프라인: category별 lift_percentage만)."""
    category: str = Field(..., description="카테고리: 'service' | 'price'")
    lift_percentage: float = Field(..., description="Lift 퍼센트: (단일−전체)/전체×100")


class StrengthResponseV2(BaseModel):
    """강점 추출 응답 모델 V2"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    strength_type: str = Field(..., description="요청한 강점 타입")
    strengths: List[StrengthDetail] = Field(..., description="강점 리스트")
    total_candidates: int = Field(..., description="근거 후보 총 개수")
    validated_count: int = Field(..., description="검증 통과한 강점 개수")
    category_lift: Optional[Dict[str, float]] = Field(
        None,
        description="카테고리별 lift 퍼센트 (service, price). 통계 근거로 LLM 설명에 사용됨.",
    )
    strength_display: Optional[List[str]] = Field(
        None,
        description="lift 기반 표시 문장 (서비스/가격 만족도, 최신 파이프라인).",
    )
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보")


# ==================== Review Image ====================

class ReviewImageSearchRequest(BaseModel):
    """리뷰 이미지 검색 요청 모델 ( 기반)"""
    query: str = Field(..., description="검색 쿼리 (예: '분위기 좋다', '데이트하기 좋은')")
    restaurant_id: Optional[int] = Field(None, description="레스토랑 ID 필터 (선택사항)")
    limit: int = Field(10, ge=1, le=100, description="반환할 최대 개수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")
    expand_query: Optional[bool] = Field(None, description="쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)")


class ReviewImageResult(BaseModel):
    """리뷰 이미지 검색 결과 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    review_id: int = Field(..., description="리뷰 ID")
    image_url: str = Field(..., description="이미지 URL")
    review: ReviewModel = Field(..., description="리뷰 정보")


class ReviewImageSearchResponse(BaseModel):
    """리뷰 이미지 검색 응답 모델 ( 기반)"""
    results: List[ReviewImageResult] = Field(..., description="검색 결과 리스트")
    total: int = Field(..., description="총 결과 개수")


# ==================== Vector Search (일반) ====================

class VectorSearchRequest(BaseModel):
    """벡터 검색 요청 모델"""
    query_text: str = Field(..., description="검색 쿼리 텍스트")
    restaurant_id: Optional[int] = Field(None, description="레스토랑 ID 필터")
    limit: int = Field(3, ge=1, le=100, description="반환할 최대 개수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")
    expand_query: Optional[bool] = Field(None, description="쿼리 확장 여부 (None: 자동 판단, True: 강제 확장, False: 확장 안함)")


class VectorSearchResult(BaseModel):
    """벡터 검색 결과 모델"""
    review: ReviewModel = Field(..., description="리뷰 정보")
    score: float = Field(..., description="유사도 점수")


class VectorSearchResponse(BaseModel):
    """벡터 검색 응답 모델"""
    results: List[VectorSearchResult] = Field(..., description="검색 결과 리스트")
    total: int = Field(..., description="총 결과 개수")


# ==================== Vector Upload ====================

class VectorUploadRequest(BaseModel):
    """벡터 데이터 업로드 요청 모델"""
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트")
    restaurants: Optional[List[RestaurantModel]] = Field(None, description="레스토랑 리스트 (선택사항)")


class VectorUploadResponse(BaseModel):
    """벡터 데이터 업로드 응답 모델"""
    message: str
    points_count: int
    collection_name: str


# ==================== Review Management ====================

class UpsertReviewRequest(BaseModel):
    """리뷰 Upsert 요청 모델 ( 기반)"""
    restaurant: RestaurantModel = Field(..., description="레스토랑 정보")
    review: ReviewModel = Field(..., description="리뷰 정보 (REVIEW TABLE)")
    update_version: Optional[int] = Field(None, description="업데이트할 버전 (낙관적 잠금용)")


class UpsertReviewResponse(BaseModel):
    """리뷰 Upsert 응답 모델"""
    action: str = Field(..., description="수행된 작업: 'inserted', 'updated', 'skipped'")
    review_id: int = Field(..., description="리뷰 ID")
    version: int = Field(..., description="새로운 버전 번호")
    point_id: str = Field(..., description="Point ID")
    reason: Optional[str] = Field(None, description="skipped인 경우 이유")
    requested_version: Optional[int] = Field(None, description="요청한 버전 (skipped인 경우)")
    current_version: Optional[int] = Field(None, description="현재 버전 (skipped인 경우)")


class UpsertReviewsBatchRequest(BaseModel):
    """리뷰 배치 Upsert 요청 모델"""
    restaurant: RestaurantModel = Field(..., description="레스토랑 정보")
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트")
    batch_size: Optional[int] = Field(32, ge=1, le=100, description="벡터 인코딩 배치 크기")


class UpsertReviewsBatchResponse(BaseModel):
    """리뷰 배치 Upsert 응답 모델"""
    results: List[Dict[str, Any]] = Field(..., description="각 리뷰의 upsert 결과 리스트")
    total: int = Field(..., description="총 처리된 리뷰 수")
    success_count: int = Field(..., description="성공한 리뷰 수 (inserted + updated)")
    error_count: int = Field(..., description="실패한 리뷰 수")


class DeleteReviewRequest(BaseModel):
    """리뷰 삭제 요청 모델"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    review_id: int = Field(..., description="리뷰 ID")


class DeleteReviewResponse(BaseModel):
    """리뷰 삭제 응답 모델"""
    action: str = Field(..., description="수행된 작업: 'deleted', 'not_found'")
    review_id: int = Field(..., description="리뷰 ID")
    point_id: str = Field(..., description="Point ID")


class DeleteReviewsBatchRequest(BaseModel):
    """리뷰 배치 삭제 요청 모델"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    review_ids: List[int] = Field(..., description="리뷰 ID 리스트")


class DeleteReviewsBatchResponse(BaseModel):
    """리뷰 배치 삭제 응답 모델"""
    results: List[Dict[str, Any]] = Field(..., description="각 리뷰의 삭제 결과 리스트")
    total: int = Field(..., description="총 처리된 리뷰 수")
    deleted_count: int = Field(..., description="삭제된 리뷰 수")
    not_found_count: int = Field(..., description="찾을 수 없는 리뷰 수")


# ==================== Restaurant Lookup ====================

class RestaurantReviewsResponse(BaseModel):
    """레스토랑 리뷰 조회 응답 모델"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    reviews: List[ReviewModel] = Field(..., description="리뷰 리스트")
    total: int = Field(..., description="총 리뷰 수")


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    version: str
