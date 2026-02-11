"""
감성 분석 라우터 ()
"""

import logging
import time
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ...sentiment_analysis import SentimentAnalyzer
from ...models import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SentimentAnalysisBatchRequest,
    SentimentAnalysisBatchResponse,
    DebugInfo
)
from ..dependencies import get_sentiment_analyzer, get_metrics_collector, get_debug_mode
from ...metrics_collector import MetricsCollector
from ...cache import acquire_lock

router = APIRouter()


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    debug: bool = Depends(get_debug_mode),
):
    """
    단일 레스토랑의 **전체 리뷰**를 sentiment 모델로 분류하여
    **긍/부정 개수 반환 + 코드에서 직접 비율(%) 산출**합니다.
    
    - **reviews**: 리뷰 리스트 (id, restaurant_id, content, created_at 필수)
    - **restaurant_id**: 레스토랑 ID (BIGINT FK)
    
    Returns:
        - restaurant_id: 레스토랑 ID
        - positive_count: 긍정 리뷰 개수
        - negative_count: 부정 리뷰 개수
        - total_count: 전체 리뷰 개수
        - positive_ratio: 긍정 비율 (%)
        - negative_ratio: 부정 비율 (%)
    """
    start_time = time.time()
    
    # 중복 실행 방지: Redis 락 획득 (엔드포인트 진입 시)
    try:
        with acquire_lock(
            restaurant_id=request.restaurant_id,
            analysis_type="sentiment",
            ttl=3600,  # 1시간
        ):
            # SKIP 로직: 최근 성공 실행이면 SKIP (미세한 중복/과호출 흡수)
            if metrics.metrics_db and request.restaurant_id is not None:
                from ...config import Config
                
                if metrics.metrics_db.should_skip_analysis(
                    restaurant_id=request.restaurant_id,
                    analysis_type="sentiment",
                    min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS,
                ):
                    # SKIP 응답 반환 (최근 처리 완료)
                    last_success_at = metrics.metrics_db.get_last_success_at(
                        restaurant_id=request.restaurant_id,
                        analysis_type="sentiment",
                    )
                    
                    # 메트릭 수집 (SKIP도 기록)
                    request_id = metrics.collect_metrics(
                        restaurant_id=request.restaurant_id,
                        analysis_type="sentiment",
                        start_time=start_time,
                        batch_size=len(request.reviews),
                        additional_info={
                            "skipped": True,
                            "reason": "recent_success",
                            "last_success_at": last_success_at.isoformat() if last_success_at else None,
                        },
                        status="skipped",
                    )
                    
                    # SKIP 응답 (항상 SentimentAnalysisResponse, debug 시에만 debug 필드 추가)
                    return SentimentAnalysisResponse(
                        restaurant_id=request.restaurant_id,
                        restaurant_name=getattr(request, "restaurant_name", None),
                        positive_count=0,
                        negative_count=0,
                        neutral_count=0,
                        total_count=len(request.reviews),
                        positive_ratio=0,
                        negative_ratio=0,
                        neutral_ratio=0,
                        debug=DebugInfo(
                            request_id=request_id,
                            processing_time_ms=(time.time() - start_time) * 1000,
                        ) if debug else None,
                    )
            
            # 대표 벡터 기반 TOP-K 방식 사용 (analyze_async: SENTIMENT_CLASSIFIER_USE_THREAD/SENTIMENT_LLM_ASYNC 토글 적용)
            result = await analyzer.analyze_async(
                reviews=request.reviews,
                restaurant_id=request.restaurant_id,
            )

            # 메트릭 수집
            processing_time_ms = (time.time() - start_time) * 1000
            request_id = metrics.collect_metrics(
                restaurant_id=request.restaurant_id,
                analysis_type="sentiment",
                start_time=start_time,
                tokens_used=result.get("tokens_used"),
                batch_size=len(request.reviews),
            )
            # TTFUR = t1 - t0 (요청 수신 시각 t0 → 응답 반환 직전 t1)
            metrics.record_llm_ttft(analysis_type="sentiment", ttft_ms=processing_time_ms)

            # 항상 SentimentAnalysisResponse 반환, debug 시에만 debug 필드 추가
            result["restaurant_name"] = getattr(request, "restaurant_name", None)
            return SentimentAnalysisResponse(
                **result,
                debug=DebugInfo(
                    request_id=request_id,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=result.get("tokens_used"),
                    model_version=result.get("model_version"),
                ) if debug else None,
            )
    except RuntimeError as e:
        # 락 획득 실패 (중복 실행 방지)
        if "중복 실행 방지" in str(e):
            raise HTTPException(
                status_code=409,  # Conflict
                detail=str(e)
            )
        raise
    except Exception as e:
        # 에러 메트릭 수집
        metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="sentiment",
            start_time=start_time,
            error_count=1,
            additional_info={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=f"감성 분석 중 오류 발생: {str(e)}")


@router.post("/analyze/batch", response_model=SentimentAnalysisBatchResponse)
async def analyze_sentiment_batch(
    request: SentimentAnalysisBatchRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    debug: bool = Depends(get_debug_mode),
):
    """
    여러 레스토랑의 **전체 리뷰**를 sentiment 모델로 분류하여 결과를 반환합니다.
    응답은 항상 restaurant_id, positive_count, negative_count 등 동일 구조. X-Debug: true 시에만 각 항목에 debug 필드 추가.
    
    Args:
        request: 배치 감성 분석 요청
            - restaurants: 레스토랑 데이터 리스트
                - restaurant_id: 레스토랑 ID
                - reviews: 리뷰 리스트 (id, restaurant_id, content, created_at 필수)
    
    Returns:
        각 레스토랑별 감성 분석 결과 리스트 (동일 스키마, debug 시에만 debug 필드 포함)
    """
    start_time = time.time()
    restaurant_id = request.restaurants[0].restaurant_id if request.restaurants else None
    try:
        results = await analyzer.analyze_multiple_restaurants_async(restaurants_data=request.restaurants)
        # 각 결과에 restaurant_name 병합 (요청 항목 순서 대응)
        restaurants_list = request.restaurants
        elapsed_ms = (time.time() - start_time) * 1000
        metrics.record_llm_ttft(analysis_type="sentiment", ttft_ms=elapsed_ms)
        metrics.collect_metrics(
            restaurant_id=restaurant_id,
            analysis_type="sentiment",
            start_time=start_time,
            status="success",
            batch_size=len(request.restaurants),
        )

        if debug:
            debug_info = DebugInfo(processing_time_ms=elapsed_ms)
            for r in results:
                r["debug"] = debug_info
        else:
            for r in results:
                r["debug"] = None

        return SentimentAnalysisBatchResponse(results=[
            SentimentAnalysisResponse(**result) for result in results
        ])
    except Exception as e:
        metrics.collect_metrics(
            restaurant_id=restaurant_id,
            analysis_type="sentiment",
            start_time=start_time,
            status="fail",
            error_count=1,
        )
        logging.getLogger(__name__).exception("배치 감성 분석 중 오류")
        raise HTTPException(
            status_code=500,
            detail=f"배치 감성 분석 중 오류 발생: {str(e)}"
            )
