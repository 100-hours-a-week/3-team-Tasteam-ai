"""
API 요청/응답 모델 정의 ( 기반)
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ==================== 기본 데이터 모델 ( 기반) ====================

class ErrorResponse(BaseModel):
    """공통 에러 응답 포맷"""
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="에러 메시지(요약)")
    details: Optional[Any] = Field(None, description="에러 상세(검증 오류/추가 정보 등)")
    request_id: str = Field(..., description="요청 추적용 ID")


class ReviewModel(BaseModel):
    """리뷰 모델 (API 요청/응답용: id, restaurant_id, content)"""
    id: int = Field(..., description="리뷰 ID")
    restaurant_id: int = Field(..., description="레스토랑 ID")
    content: str = Field(..., description="리뷰 내용")


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
    """감성 분석용 리뷰 입력 (id, restaurant_id, content, created_at)"""
    id: int = Field(..., description="리뷰 ID (LLM 재판정 시 매핑용)")
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    content: str = Field(..., description="리뷰 내용 (VARCHAR(1000))")
    created_at: datetime = Field(..., description="리뷰 작성 시각 (ISO 8601)")


class SentimentAnalysisRequest(BaseModel):
    """감성 분석 요청 모델 ( 기반)"""
    restaurant_id: int = Field(..., description="레스토랑 ID (BIGINT FK)")
    reviews: List[SentimentReviewInput] = Field(..., description="리뷰 리스트 (id, restaurant_id, content, created_at 필수)")


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
    """배치 감성 분석용 레스토랑 입력 (reviews는 SentimentReviewInput: id, restaurant_id, content, created_at 필수)"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    reviews: List[SentimentReviewInput] = Field(default_factory=list, description="리뷰 리스트 (id, restaurant_id, content, created_at 필수)")


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

class SummaryRequest(BaseModel):
    """리뷰 요약 요청 모델. 하이브리드 검색 쿼리는 기본 시드(service/price/food)만 사용."""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    limit: int = Field(10, ge=1, le=100, description="각 카테고리당 검색할 최대 리뷰 수")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="최소 유사도 점수")


class CategorySummary(BaseModel):
    """카테고리별 요약 모델"""
    summary: str = Field(..., description="카테고리 요약")
    bullets: List[str] = Field(default_factory=list, description="핵심 포인트 리스트")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="근거 리스트 (review_id, snippet, rank)")


class SummaryDisplayResponse(BaseModel):
    """리뷰 요약 표시용 응답 모델 (최소 필드). debug=true 시 debug 필드만 추가, positive_reviews 등 미사용."""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    overall_summary: str = Field(..., description="전체 요약")
    categories: Optional[Dict[str, CategorySummary]] = Field(None, description="카테고리별 요약 (새 파이프라인)")
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보 (X-Debug: true 시에만 포함)")


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


# ==================== Comparison ====================

class ComparisonRequest(BaseModel):
    """다른 음식점과의 비교 요청 모델 (Kiwi+lift 파이프라인)"""
    restaurant_id: int = Field(..., description="타겟 레스토랑 ID")


class ComparisonDetail(BaseModel):
    """비교 상세 (Kiwi+lift 파이프라인: category별 lift_percentage만)."""
    category: str = Field(..., description="카테고리: 'service' | 'price'")
    lift_percentage: float = Field(..., description="Lift 퍼센트: (단일−전체)/전체×100")


class ComparisonResponse(BaseModel):
    """다른 음식점과의 비교 응답 모델"""
    restaurant_id: int = Field(..., description="레스토랑 ID")
    comparisons: List[ComparisonDetail] = Field(..., description="비교 항목 리스트")
    total_candidates: int = Field(..., description="근거 후보 총 개수")
    validated_count: int = Field(..., description="검증 통과한 비교 항목 개수")
    category_lift: Optional[Dict[str, float]] = Field(
        None,
        description="카테고리별 lift 퍼센트 (service, price). 통계 근거로 LLM 설명에 사용됨.",
    )
    comparison_display: Optional[List[str]] = Field(
        None,
        description="lift 기반 표시 문장 (서비스/가격 만족도, 최신 파이프라인).",
    )
    debug: Optional[DebugInfo] = Field(None, description="디버그 정보")


class ComparisonBatchRequest(BaseModel):
    """배치 비교 요청 모델"""
    restaurants: List[Dict[str, Any]] = Field(
        ...,
        description="레스토랑 데이터 리스트, 각 항목: restaurant_id(필수)."
    )


class ComparisonBatchResponse(BaseModel):
    """배치 비교 응답 모델 (각 항목은 단일 ComparisonResponse와 동일)"""
    results: List[ComparisonResponse] = Field(..., description="각 레스토랑별 비교 결과")


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
    """벡터 업로드용 리뷰 입력 (id, restaurant_id, content, created_at)"""
    id: int = Field(..., description="리뷰 ID")
    restaurant_id: int = Field(..., description="레스토랑 ID")
    content: str = Field(..., description="리뷰 내용")
    created_at: datetime = Field(..., description="리뷰 작성 시각 (ISO 8601)")


class VectorUploadRestaurantInput(BaseModel):
    """벡터 업로드용 레스토랑 입력 (id 선택, name, reviews)"""
    id: Optional[int] = Field(None, description="레스토랑 ID (BIGINT PK)")
    name: str = Field(..., description="레스토랑 이름 (VARCHAR(100))")
    reviews: List[VectorUploadReviewInput] = Field(default_factory=list, description="중첩 형식 시 리뷰 리스트")


class VectorUploadRequest(BaseModel):
    """벡터 데이터 업로드 요청 모델"""
    reviews: List[VectorUploadReviewInput] = Field(..., description="리뷰 리스트 (id, restaurant_id, content, created_at 필수)")
    restaurants: Optional[List[VectorUploadRestaurantInput]] = Field(None, description="레스토랑 리스트 (id, name, reviews만, 선택사항)")


class VectorUploadResponse(BaseModel):
    """벡터 데이터 업로드 응답 모델"""
    message: str
    points_count: int
    collection_name: str


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str
    version: str
