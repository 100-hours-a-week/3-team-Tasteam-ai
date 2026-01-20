"""
LLM 관련 라우터 ( - Summary, Strength)
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Union, Dict

from ...llm_utils import LLMUtils
from ...vector_search import VectorSearch
from ...models import (
    SummaryRequest,
    SummaryResponse,
    SummaryDisplayResponse,
    SummaryBatchRequest,
    SummaryBatchResponse,
    StrengthRequestV2,
    StrengthResponseV2,
    DebugInfo
)
from ..dependencies import get_llm_utils, get_vector_search, get_metrics_collector, get_debug_mode
from ...metrics_collector import MetricsCollector
from ...config import Config
from ...strength_extraction import StrengthExtractionPipeline
from ...cache import acquire_lock

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/summarize", response_model=Union[SummaryResponse, SummaryDisplayResponse])
async def summarize_reviews(
    request: SummaryRequest,
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    debug: bool = Depends(get_debug_mode),
):
    """
    벡터 검색을 활용하여 단일 레스토랑의 긍정/부정 리뷰를 자동 검색하고 요약합니다.
    
    vLLM 직접 사용 모드에서는 내부적으로 배치 처리 로직을 재사용하여
    동적 배치 크기와 세마포어 기반 OOM 방지 전략을 적용합니다.
    
    OOM 방지 전략 (vLLM 모드):
    - 각 레스토랑별로 동적 배치 크기 계산 (리뷰 길이에 따라)
    - 세마포어를 통한 동시 처리 수 제한 (VLLM_MAX_CONCURRENT_BATCHES)
    - 각 배치는 독립적으로 처리 가능 (메모리 사용량 예측 가능)
    - vLLM이 자동으로 여러 배치를 효율적으로 처리 (Continuous Batching)
    
    프로세스 (aspect 기반):
    1. 벡터 검색으로 긍정 리뷰 자동 검색
       - query = ["맛있다 좋다 친절하다"]
       - filter( must ( restaurant id = summary 하고자하는 id (단일 id)))
    2. 벡터 검색으로 부정 리뷰 자동 검색
       - query = ["맛없다 싫다 불친절하다"]
       - filter( must ( restaurant id = summary 하고자하는 id (단일 id)))
    3. 긍정 리뷰에서 aspect 단위 장점 추출 (strength 추출의 Step B 로직 재사용)
    4. 부정 리뷰에서 aspect 단위 단점 추출 (동일한 로직)
    5. LLM이 positive_aspects + negative_aspects를 기반으로 overall_summary 생성
    6. 메타데이터와 함께 반환
    
    - **restaurant_id**: 레스토랑 ID (단일 ID만 허용)
    - **positive_query**: 긍정 리뷰 검색 쿼리 (기본값: "맛있다 좋다 만족")
    - **negative_query**: 부정 리뷰 검색 쿼리 (기본값: "맛없다 별로 불만")
    - **limit**: 각 카테고리당 검색할 최대 리뷰 수 (기본값: 10)
    - **min_score**: 최소 유사도 점수 (기본값: 0.0)
    
    Returns:
        - restaurant_id: 레스토랑 ID
        - overall_summary: 전체 요약 (positive_aspects + negative_aspects 기반)
        - positive_aspects: 긍정 aspect 리스트 (aspect, claim, evidence_quotes, evidence_review_ids)
        - negative_aspects: 부정 aspect 리스트 (aspect, claim, evidence_quotes, evidence_review_ids)
        - positive_reviews: 긍정 리뷰 메타데이터
        - negative_reviews: 부정 리뷰 메타데이터
        - positive_count: 긍정 리뷰 개수
        - negative_count: 부정 리뷰 개수
    """
    start_time = time.time()
    
    # 중복 실행 방지: Redis 락 획득 (엔드포인트 진입 시)
    try:
        with acquire_lock(
            restaurant_id=request.restaurant_id,
            analysis_type="summary",
            ttl=3600,  # 1시간
        ):
            # SKIP 로직: 최근 성공 실행이면 SKIP (미세한 중복/과호출 흡수)
            if metrics.metrics_db and request.restaurant_id is not None:
                from ...config import Config
                
                if metrics.metrics_db.should_skip_analysis(
                    restaurant_id=request.restaurant_id,
                    analysis_type="summary",
                    min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS,
                ):
                    # SKIP 응답 반환 (최근 처리 완료)
                    last_success_at = metrics.metrics_db.get_last_success_at(
                        restaurant_id=request.restaurant_id,
                        analysis_type="summary",
                    )
                    
                    # 메트릭 수집 (SKIP도 기록)
                    request_id = metrics.collect_metrics(
                        restaurant_id=request.restaurant_id,
                        analysis_type="summary",
                        start_time=start_time,
                        batch_size=0,
                        additional_info={
                            "skipped": True,
                            "reason": "recent_success",
                            "last_success_at": last_success_at.isoformat() if last_success_at else None,
                        },
                        status="skipped",
                    )
                    
                    # SKIP 응답
                    if debug:
                        return SummaryResponse(
                            restaurant_id=request.restaurant_id,
                            overall_summary="",
                            positive_aspects=[],
                            negative_aspects=[],
                            positive_reviews=[],
                            negative_reviews=[],
                            positive_count=0,
                            negative_count=0,
                            debug=DebugInfo(
                                request_id=request_id,
                                processing_time_ms=(time.time() - start_time) * 1000,
                            ),
                        )
                    else:
                        return SummaryDisplayResponse(
                            overall_summary="",
                            positive_aspects=[],
                            negative_aspects=[],
                        )
            
            # 대표 벡터 기반 TOP-K 리뷰 선택
        # 1. 대표 벡터 주위 TOP-K 리뷰 검색
        top_k_results = vector_search.query_by_restaurant_vector(
            restaurant_id=request.restaurant_id,
            top_k=request.limit * 2,  # 긍정/부정 모두 포함할 수 있도록 더 많이 가져옴
            months_back=None,  # 날짜 필터 없음
        )
        
        # 2. payload 추출
        all_reviews = [r["payload"] for r in top_k_results]
        
        # 3. 대표 벡터 기반 검색 결과가 없으면 예외 발생
        if not all_reviews:
            raise HTTPException(
                status_code=404,
                detail=f"레스토랑 {request.restaurant_id}에 대한 리뷰를 찾을 수 없습니다. "
                       f"대표 벡터를 계산할 수 없거나 리뷰가 존재하지 않습니다."
            )
        
        # 4. 대표 벡터 기반 결과 사용
        # LLM이 aspect 추출 시 자동으로 긍정/부정을 구분
        positive_reviews = all_reviews  # 대표 벡터 기반이므로 대부분 긍정적일 가능성 높음
        negative_reviews = []  # 부정 리뷰는 별도로 검색하지 않음 (대표 벡터 기반이므로)
        
        # 5. LLM 입력 및 처리
        # vLLM 직접 사용 모드인지 확인
        if hasattr(llm_utils, 'use_pod_vllm') and llm_utils.use_pod_vllm:
            # vLLM 모드: summarize_multiple_restaurants_vllm() 재사용 (OOM 방지 전략 포함)
            # 단일 레스토랑 요청을 리스트로 감싸서 전달
            restaurants_data = [{
                "restaurant_id": request.restaurant_id,
                "positive_reviews": positive_reviews,
                "negative_reviews": negative_reviews
            }]
            
            results = await llm_utils.summarize_multiple_restaurants_vllm(
                restaurants_data=restaurants_data,
                max_tokens_per_batch=None,  # 동적 계산
                vector_search=vector_search,  # aspect 검증용
                validate_aspects=True,  # aspect 검증 활성화
            )
            
            # 결과가 비어있지 않으면 첫 번째 결과 반환
            if results:
                result = results[0]
            else:
                # 결과가 없는 경우 빈 결과 반환
                result = {
                    "restaurant_id": request.restaurant_id,
                    "overall_summary": "요약할 리뷰가 없습니다.",
                    "positive_aspects": [],
                    "negative_aspects": [],
                    "positive_reviews": positive_reviews,
                    "negative_reviews": negative_reviews,
                    "positive_count": len(positive_reviews),
                    "negative_count": len(negative_reviews),
                }
        else:
            # 기존 모드: 동기 메서드 사용 (aspect 기반)
            result = llm_utils.summarize_reviews(
                positive_reviews=positive_reviews,
                negative_reviews=negative_reviews,
                vector_search=vector_search,  # aspect 검증용
                validate_aspects=True,  # aspect 검증 활성화
            )
            
            # restaurant_id 추가
            result["restaurant_id"] = request.restaurant_id
        
        # 6. 응답 검증 (overall_summary 포함 확인)
        if not result or "overall_summary" not in result:
            raise HTTPException(
                status_code=500, 
                detail="리뷰 요약 실패: overall_summary가 생성되지 않았습니다."
            )
        
        # positive_aspects와 negative_aspects가 없으면 빈 리스트로 설정
        if "positive_aspects" not in result:
            result["positive_aspects"] = []
        if "negative_aspects" not in result:
            result["negative_aspects"] = []
        
        # evidence_review_ids를 문자열로 변환 (Pydantic 검증 오류 방지)
        for aspect in result.get("positive_aspects", []):
            if "evidence_review_ids" in aspect and aspect["evidence_review_ids"]:
                aspect["evidence_review_ids"] = [str(rid) for rid in aspect["evidence_review_ids"]]
        for aspect in result.get("negative_aspects", []):
            if "evidence_review_ids" in aspect and aspect["evidence_review_ids"]:
                aspect["evidence_review_ids"] = [str(rid) for rid in aspect["evidence_review_ids"]]
        
        # 메트릭 수집
        request_id = metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="summary",
            start_time=start_time,
            tokens_used=result.get("tokens_used"),
            batch_size=len(positive_reviews) + len(negative_reviews),
        )
        
        # 디버그 모드에 따라 응답 반환
        if debug:
            # 디버그 모드: 전체 응답 + 디버그 정보
            return SummaryResponse(
                **result,
                debug=DebugInfo(
                    request_id=request_id,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=result.get("tokens_used"),
                    model_version=result.get("model_version"),
                )
            )
        else:
            # 일반 모드: 최소 필드만 (aspect 포함)
            return SummaryDisplayResponse(
                overall_summary=result["overall_summary"],
                positive_aspects=result.get("positive_aspects", []),
                negative_aspects=result.get("negative_aspects", []),
            )
    except RuntimeError as e:
        # 락 획득 실패 (중복 실행 방지)
        if "중복 실행 방지" in str(e):
            raise HTTPException(
                status_code=409,  # Conflict
                detail=str(e)
            )
        raise
    except HTTPException:
        raise
    except Exception as e:
        # 에러 메트릭 수집
        metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="summary",
            start_time=start_time,
            error_count=1,
            additional_info={"error": str(e)},
        )
        logger.error(f"리뷰 요약 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"리뷰 요약 중 오류 발생: {str(e)}")


@router.post("/extract/strengths", response_model=Union[StrengthResponseV2, Dict])
async def extract_strengths(
    request: StrengthRequestV2,
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    debug: bool = Depends(get_debug_mode),
):
    """
    구조화된 강점 추출 파이프라인
    
    프로세스:
    1. Step A: 타겟 긍정 근거 후보 수집 (Vector Search / Filter)
    2. Step B: 강점 후보 생성 (LLM 구조화 출력)
    3. Step C: 강점별 근거 확장/검증 (Aspect → Qdrant 벡터 검색)
    4. Step D: 의미 중복 제거 (클러스터링)
    5. Step E~H: 비교군 기반 차별 강점 계산 (distinct일 때만)
    
    Args:
        request: 강점 추출 요청
            - restaurant_id: 타겟 레스토랑 ID
            - strength_type: 강점 타입 ('representative', 'distinct', 'both')
            - category_filter: 카테고리 필터 (선택)
            - region_filter: 지역 필터 (선택)
            - price_band_filter: 가격대 필터 (선택)
            - top_k: 반환할 최대 강점 개수
            - max_candidates: 근거 후보 최대 개수
            - months_back: 최근 N개월 리뷰만 사용
            - min_support: 최소 support_count
    
    Returns:
        강점 추출 결과 (구조화된 형식)
    """
    start_time = time.time()
    
    # 중복 실행 방지: Redis 락 획득 (엔드포인트 진입 시)
    try:
        with acquire_lock(
            restaurant_id=request.restaurant_id,
            analysis_type="strength",
            ttl=3600,  # 1시간
        ):
            # SKIP 로직: 최근 성공 실행이면 SKIP (미세한 중복/과호출 흡수)
            if metrics.metrics_db and request.restaurant_id is not None:
                from ...config import Config
                
                if metrics.metrics_db.should_skip_analysis(
                    restaurant_id=request.restaurant_id,
                    analysis_type="strength",
                    min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS,
                ):
                    # SKIP 응답 반환 (최근 처리 완료)
                    last_success_at = metrics.metrics_db.get_last_success_at(
                        restaurant_id=request.restaurant_id,
                        analysis_type="strength",
                    )
                    
                    # 메트릭 수집 (SKIP도 기록)
                    request_id = metrics.collect_metrics(
                        restaurant_id=request.restaurant_id,
                        analysis_type="strength",
                        start_time=start_time,
                        batch_size=0,
                        additional_info={
                            "skipped": True,
                            "reason": "recent_success",
                            "last_success_at": last_success_at.isoformat() if last_success_at else None,
                        },
                        status="skipped",
                    )
                    
                    # SKIP 응답
                    if debug:
                        return StrengthResponseV2(
                            restaurant_id=request.restaurant_id,
                            strength_type=request.strength_type,
                            strengths=[],
                            total_candidates=0,
                            validated_count=0,
                            debug=DebugInfo(
                                request_id=request_id,
                                processing_time_ms=(time.time() - start_time) * 1000,
                            ),
                        )
                    else:
                        return StrengthResponseV2(
                            restaurant_id=request.restaurant_id,
                            strength_type=request.strength_type,
                            strengths=[],
                            total_candidates=0,
                            validated_count=0,
                        )
            
            # 파이프라인 초기화
        pipeline = StrengthExtractionPipeline(
            llm_utils=llm_utils,
            vector_search=vector_search,
        )
        
        # 파이프라인 실행
        result = await pipeline.extract_strengths(
            restaurant_id=request.restaurant_id,
            strength_type=request.strength_type,
            category_filter=request.category_filter,
            region_filter=request.region_filter,
            price_band_filter=request.price_band_filter,
            top_k=request.top_k,
            max_candidates=request.max_candidates,
            months_back=request.months_back,
            min_support=request.min_support,
        )
        
        # 메트릭 수집
        request_id = metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="strength",
            start_time=start_time,
            batch_size=result.get("total_candidates", 0),
            )
            
        # 디버그 정보 추가
        if debug:
            result["debug"] = DebugInfo(
                request_id=request_id,
                processing_time_ms=result.get("processing_time_ms", 0),
            )
        
        return StrengthResponseV2(**result)
    except RuntimeError as e:
        # 락 획득 실패 (중복 실행 방지)
        if "중복 실행 방지" in str(e):
            raise HTTPException(
                status_code=409,  # Conflict
                detail=str(e)
            )
        raise
    except HTTPException:
        raise
    except Exception as e:
        # 에러 메트릭 수집
        metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="strength",
            start_time=start_time,
            error_count=1,
            additional_info={"error": str(e)},
        )
        logger.error(f"강점 추출 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"강점 추출 중 오류 발생: {str(e)}")


@router.post("/summarize/batch", response_model=SummaryBatchResponse)
async def summarize_reviews_batch(
    request: SummaryBatchRequest,
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
):
    """
    여러 레스토랑의 리뷰를 배치로 요약 (aspect 기반, 비동기 처리)
    
    각 레스토랑별로:
    1. 벡터 검색으로 긍정/부정 리뷰 자동 검색
    2. 긍정 리뷰에서 aspect 단위 장점 추출
    3. 부정 리뷰에서 aspect 단위 단점 추출
    4. LLM이 positive_aspects + negative_aspects를 기반으로 overall_summary 생성
    
    OOM 방지 전략:
    - 세마포어를 통한 동시 처리 수 제한 (VLLM_MAX_CONCURRENT_BATCHES)
    - 각 레스토랑은 독립적으로 처리 가능
    
    프로세스 (aspect 기반):
    1. 각 레스토랑별로 벡터 검색 수행 (positive_query, negative_query 사용)
    2. 각 레스토랑별로:
       - 긍정 리뷰에서 aspect 단위 장점 추출
       - 부정 리뷰에서 aspect 단위 단점 추출
       - LLM이 positive_aspects + negative_aspects를 기반으로 overall_summary 생성
    3. 모든 레스토랑을 비동기로 병렬 처리
    4. 레스토랑별로 결과 집계
    
    Args:
        request: 배치 리뷰 요약 요청
            - restaurants: 레스토랑 데이터 리스트
                - restaurant_id: 레스토랑 ID
                - positive_query: 긍정 리뷰 검색 쿼리 (기본값: "맛있다 좋다 만족")
                - negative_query: 부정 리뷰 검색 쿼리 (기본값: "맛없다 별로 불만")
                - limit: 각 카테고리당 검색할 최대 리뷰 수 (기본값: 10)
                - min_score: 최소 유사도 점수 (기본값: 0.0)
            - max_tokens_per_batch: 배치당 최대 토큰 수 (선택사항)
    
    Returns:
        각 레스토랑별 요약 결과 리스트
    """
    try:
        # 각 레스토랑별로 벡터 검색 수행
        restaurants_data = []
        for restaurant_data in request.restaurants:
            restaurant_id = restaurant_data.get("restaurant_id")
            positive_query = restaurant_data.get("positive_query", "맛있다 좋다 만족")
            negative_query = restaurant_data.get("negative_query", "맛없다 별로 불만")
            limit = restaurant_data.get("limit", 10)
            min_score = restaurant_data.get("min_score", 0.0)
            
            # 벡터 검색으로 긍정/부정 리뷰 검색
            positive_results = vector_search.query_similar_reviews(
                query_text=positive_query,
                restaurant_id=restaurant_id,
                limit=limit,
                min_score=min_score,
            )
            negative_results = vector_search.query_similar_reviews(
                query_text=negative_query,
                restaurant_id=restaurant_id,
                limit=limit,
                min_score=min_score,
            )
            
            # payload 추출
            positive_reviews = [r["payload"] for r in positive_results]
            negative_reviews = [r["payload"] for r in negative_results]
            
            restaurants_data.append({
                "restaurant_id": restaurant_id,
                "positive_reviews": positive_reviews,
                "negative_reviews": negative_reviews,
            })
        
        # vLLM 직접 사용 모드인지 확인
        if hasattr(llm_utils, 'use_pod_vllm') and llm_utils.use_pod_vllm:
            results = await llm_utils.summarize_multiple_restaurants_vllm(
                restaurants_data=restaurants_data,
                max_tokens_per_batch=request.max_tokens_per_batch,
                vector_search=vector_search,  # aspect 검증용
                validate_aspects=True,  # aspect 검증 활성화
            )
        else:
            # 기존 방식: 각 레스토랑을 순차 처리
            results = []
            for restaurant_data in restaurants_data:
                result = llm_utils.summarize_reviews(
                    positive_reviews=restaurant_data.get("positive_reviews", []),
                    negative_reviews=restaurant_data.get("negative_reviews", []),
                    vector_search=vector_search,  # aspect 검증용
                    validate_aspects=True,  # aspect 검증 활성화
                )
                result["restaurant_id"] = restaurant_data["restaurant_id"]
                results.append(result)
        
        # evidence_review_ids를 문자열로 변환 (Pydantic 검증 오류 방지)
        for result in results:
            for aspect in result.get("positive_aspects", []):
                if "evidence_review_ids" in aspect and aspect["evidence_review_ids"]:
                    aspect["evidence_review_ids"] = [str(rid) for rid in aspect["evidence_review_ids"]]
            for aspect in result.get("negative_aspects", []):
                if "evidence_review_ids" in aspect and aspect["evidence_review_ids"]:
                    aspect["evidence_review_ids"] = [str(rid) for rid in aspect["evidence_review_ids"]]
        
        return SummaryBatchResponse(results=[
            SummaryResponse(**result) for result in results
        ])
    except Exception as e:
        logger.error(f"배치 리뷰 요약 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"배치 리뷰 요약 중 오류 발생: {str(e)}"
        )
