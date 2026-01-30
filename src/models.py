"""
API 요청/응답 모델 정의 ( 기반)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== 기본 데이터 모델 ( 기반) ====================

class ErrorResponse(BaseModel):
    """공통 에러 응답 포맷"""
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="에러 메시지(요약)")
    details: Optional[Any] = Field(None, description="에러 상세(검증 오류/추가 정보 등)")
    request_id: str = Field(..., description="요청 추적용 ID")


class ReviewImageModel(BaseModel):
    """리뷰 이미지 모델 (REVIEW_IMAGE TABLE). 스키마에만 존재, API 요청/응답에서는 미사용."""
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
    is_recommended: Optional[bool] = Field(None, description="추천 여부 (BOOLEAN, 메타)")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP, 메타)")
    updated_at: Optional[datetime] = Field(None, description="수정 시간 (TIMESTAMP, 메타)")
    # 스키마에만 존재, API 응답에서는 제외 (member_id와 동일)
    images: Optional[List[ReviewImageModel]] = Field(None, description="리뷰 이미지 리스트 (스키마만, API 미사용)", exclude=True)


class RestaurantModel(BaseModel):
    """레스토랑 모델 (RESTAURANT TABLE -  기반)"""
    id: Optional[int] = Field(None, description="레스토랑 ID (BIGINT PK)")
    name: str = Field(..., description="레스토랑 이름 (VARCHAR(100))")
    full_address: Optional[str] = Field(None, description="전체 주소 (VARCHAR(255))")
    location: Optional[Dict[str, Any]] = Field(None, description="위치 정보 (geometry(Point,4326))")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP)")


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

class SentimentReviewInput(BaseModel):
    """감성 분석용 리뷰 입력 (id, restaurant_id, content)"""
    id: Optional[int] = Field(None, description="리뷰 ID (선택, LLM 재판정 시 매핑용)")
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    content: str = Field(..., description="리뷰 내용 (VARCHAR(1000))")


class SentimentAnalysisRequest(BaseModel):
    """감성 분석 요청 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    reviews: List[SentimentReviewInput] = Field(..., description="리뷰 리스트 (id 선택, restaurant_id, content)")


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


class SentimentRestaurantBatchInput(BaseModel):
    """배치 감성 분석용 레스토랑 입력 (reviews는 SentimentReviewInput: id 선택, restaurant_id, content)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    reviews: List[SentimentReviewInput] = Field(default_factory=list, description="리뷰 리스트 (id 선택, restaurant_id, content)")


class SentimentAnalysisBatchRequest(BaseModel):
    """배치 감성 분석 요청 모델"""
    restaurants: List[SentimentRestaurantBatchInput] = Field(
        ...,
        description="레스토랑 데이터 리스트, 각 항목: restaurant_id, reviews(SentimentReviewInput 리스트)"
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
    """리뷰 요약 요청 모델. 하이브리드 검색 쿼리는 기본 시드(service/price/food)만 사용."""
    restaurant_id: int = Field(..., description="레스토랑 ID")
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
    categories: Optional[Dict[str, CategorySummary]] = Field(None, description="카테고리별 요약 (새 파이프라인)")


class SummaryResponse(BaseModel):
    """리뷰 요약 응답 모델 (디버그용, positive_reviews 등 포함)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    overall_summary: str = Field(..., description="전체 요약")
    positive_reviews: List[ReviewModel] = Field(default_factory=list, description="긍정 리뷰 메타데이터")
    negative_reviews: List[ReviewModel] = Field(default_factory=list, description="부정 리뷰 메타데이터")
    positive_count: int = Field(0, description="긍정 리뷰 개수")
    negative_count: int = Field(0, description="부정 리뷰 개수")
    categories: Optional[Dict[str, CategorySummary]] = Field(None, description="카테고리별 요약 (새 파이프라인)")
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보")


class SummaryBatchRequest(BaseModel):
    """배치 리뷰 요약 요청 모델. 하이브리드 검색 쿼리는 기본 시드만 사용. limit/min_score는 전체 레스토랑 공통."""
    restaurants: List[Dict[str, Any]] = Field(
        ...,
        description="레스토랑 데이터 리스트, 각 항목: restaurant_id(필수)."
    )
    limit: int = Field(10, ge=1, le=100, description="각 카테고리당 검색할 최대 리뷰 수 (전체 레스토랑 공통)")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수 (전체 레스토랑 공통)")


class SummaryBatchResponse(BaseModel):
    """배치 리뷰 요약 응답 모델 (각 항목은 단일 debug=false와 동일한 SummaryDisplayResponse)"""
    results: List[SummaryDisplayResponse] = Field(..., description="각 레스토랑별 요약 결과")


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
    """강점 추출 요청 모델 V2 (Kiwi+lift 파이프라인)"""
    restaurant_id: int = Field(..., description="타겟 레스토랑 ID")
    category_filter: Optional[int] = Field(None, description="카테고리 필터")
    region_filter: Optional[str] = Field(None, description="지역 필터")
    price_band_filter: Optional[str] = Field(None, description="가격대 필터")
    top_k: int = Field(10, ge=1, le=50, description="반환할 최대 강점 개수")
    max_candidates: int = Field(300, ge=50, le=1000, description="근거 후보 최대 개수")
    months_back: int = Field(6, ge=1, le=24, description="최근 N개월 리뷰만 사용")


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


# ==================== Vector Search (일반) ====================

class VectorSearchRequest(BaseModel):
    """벡터 검색 요청 모델"""
    query_text: str = Field(..., description="검색 쿼리 텍스트")
    restaurant_id: Optional[int] = Field(None, description="레스토랑 ID 필터")
    limit: int = Field(3, ge=1, le=100, description="반환할 최대 개수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")


class VectorSearchResult(BaseModel):
    """벡터 검색 결과 모델"""
    review: ReviewModel = Field(..., description="리뷰 정보")
    score: float = Field(..., description="유사도 점수")


class VectorSearchResponse(BaseModel):
    """벡터 검색 응답 모델"""
    results: List[VectorSearchResult] = Field(..., description="검색 결과 리스트")
    total: int = Field(..., description="총 결과 개수")


# ==================== Vector Upload ====================

class VectorUploadReviewInput(BaseModel):
    """벡터 업로드용 리뷰 입력 (is_recommended, member_id, group_id, subgroup_id, updated_at 미사용)"""
    id: Optional[int] = Field(None, description="리뷰 ID (BIGINT PK)")
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    content: str = Field(..., description="리뷰 내용 (VARCHAR(1000))")
    created_at: Optional[datetime] = Field(None, description="생성 시간 (TIMESTAMP, 메타)")
    # 스키마에만 존재, API 요청/응답에서는 미사용 (member_id와 동일)
    images: Optional[List[ReviewImageModel]] = Field(None, description="리뷰 이미지 리스트 (스키마만, API 미사용)", exclude=True)


class VectorUploadRestaurantInput(BaseModel):
    """벡터 업로드용 레스토랑 입력 (full_address, location, created_at 미사용)"""
    id: Optional[int] = Field(None, description="레스토랑 ID (BIGINT PK)")
    name: str = Field(..., description="레스토랑 이름 (VARCHAR(100))")
    reviews: List[VectorUploadReviewInput] = Field(default_factory=list, description="중첩 형식 시 리뷰 리스트")


class VectorUploadRequest(BaseModel):
    """벡터 데이터 업로드 요청 모델"""
    reviews: List[VectorUploadReviewInput] = Field(..., description="리뷰 리스트 (id, restaurant_id, content)")
    restaurants: Optional[List[VectorUploadRestaurantInput]] = Field(None, description="레스토랑 리스트 (id, name, reviews만, 선택사항)")


class VectorUploadResponse(BaseModel):
    """벡터 데이터 업로드 응답 모델"""
    message: str
    points_count: int
    collection_name: str


# ==================== Review Management ====================

class UpsertReviewsRequest(BaseModel):
    """리뷰 Upsert 요청 (upload와 동일 형식: reviews, restaurants)"""
    reviews: List[VectorUploadReviewInput] = Field(..., description="리뷰 리스트 (id, restaurant_id, content)")
    restaurants: Optional[List[VectorUploadRestaurantInput]] = Field(None, description="레스토랑 리스트 (id, name, reviews) - restaurant_name 해석 및 중첩 리뷰")
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
