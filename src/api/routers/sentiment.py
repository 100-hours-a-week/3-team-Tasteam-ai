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
)
from ..dependencies import get_sentiment_analyzer, get_metrics_collector
from ...metrics_collector import MetricsCollector
from ...cache import acquire_lock

router = APIRouter()


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer),
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """
    단일 레스토랑 감성 분석. **리뷰는 벡터 DB에서 조회**한 뒤 sentiment 모델로 분류하여
    긍/부정 개수·비율(%)을 반환합니다.
    
    - **restaurant_id**: 레스토랑 ID (벡터 DB에서 해당 레스토랑 리뷰 조회)
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
                        batch_size=0,
                        additional_info={
                            "skipped": True,
                            "reason": "recent_success",
                            "last_success_at": last_success_at.isoformat() if last_success_at else None,
                        },
                        status="skipped",
                    )
                    
                    return SentimentAnalysisResponse(
                        restaurant_id=request.restaurant_id,
                        restaurant_name=getattr(request, "restaurant_name", None),
                        positive_count=0,
                        negative_count=0,
                        neutral_count=0,
                        total_count=0,
                        positive_ratio=0,
                        negative_ratio=0,
                        neutral_ratio=0,
                    )
            
            # 리뷰는 벡터 DB에서 조회 (analyze_async(reviews=None) → vector_search 경로)
            result = await analyzer.analyze_async(
                reviews=None,
                restaurant_id=request.restaurant_id,
            )

            # 메트릭 수집
            processing_time_ms = (time.time() - start_time) * 1000
            request_id = metrics.collect_metrics(
                restaurant_id=request.restaurant_id,
                analysis_type="sentiment",
                start_time=start_time,
                tokens_used=result.get("tokens_used"),
                batch_size=result.get("total_count", 0),
            )
            # TTFUR = t1 - t0 (요청 수신 시각 t0 → 응답 반환 직전 t1)
            metrics.record_llm_ttft(analysis_type="sentiment", ttft_ms=processing_time_ms)

            result["restaurant_name"] = getattr(request, "restaurant_name", None)
            return SentimentAnalysisResponse(**result)
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
):
    """
    여러 레스토랑 감성 분석. 각 레스토랑 리뷰는 **벡터 DB에서 조회**한 뒤 sentiment로 분류.
    응답은 restaurant_id, positive_count, negative_count 등 동일 구조.
    """
    start_time = time.time()
    restaurant_id = request.restaurants[0].restaurant_id if request.restaurants else None
    try:
        results = await analyzer.analyze_multiple_restaurants_async(restaurants_data=request.restaurants)
        elapsed_ms = (time.time() - start_time) * 1000
        metrics.record_llm_ttft(analysis_type="sentiment", ttft_ms=elapsed_ms)
        metrics.collect_metrics(
            restaurant_id=restaurant_id,
            analysis_type="sentiment",
            start_time=start_time,
            status="success",
            batch_size=len(request.restaurants),
        )
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
