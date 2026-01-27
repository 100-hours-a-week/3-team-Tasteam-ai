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
from ...summary_pipeline import summarize_aspects_new
from ...aspect_seeds import DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS

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
    새로운 파이프라인: 하이브리드 검색 (Dense + Sparse) + 카테고리별 요약
    
    프로세스:
    1. 기본 시드만 사용 (DEFAULT_SERVICE/PRICE/FOOD_SEEDS, 파일 미사용)
    2. 카테고리별(service, price, food) 하이브리드 검색 수행
    3. 각 카테고리별로 LLM 요약 생성 (summary, bullets, evidence)
    4. overall_summary 생성
    
    주의사항:
    - 이 파이프라인은 긍정/부정 aspect를 세지 않습니다
    - 카테고리별(service/price/food) 하이브리드 검색으로 관련 리뷰를 찾고 요약만 생성합니다
    - 긍정/부정 분류는 sentiment analysis 파이프라인에서 수행됩니다
    
    Args:
        restaurant_id: 레스토랑 ID
        limit: 각 카테고리당 검색할 최대 리뷰 수 (기본값: 10)
        min_score: 최소 유사도 점수 (기본값: 0.0, 사용 안 함)
    
    Returns:
        - restaurant_id: 레스토랑 ID
        - overall_summary: 전체 요약
        - categories: 카테고리별 상세 요약 (service, price, food)
            - summary: 카테고리별 요약
            - bullets: 주요 포인트 리스트
            - evidence: 근거 리뷰 리스트 (review_id, snippet, rank)
        - positive_aspects: [] (새 파이프라인은 긍정/부정 aspect를 세지 않음)
        - negative_aspects: [] (새 파이프라인은 긍정/부정 aspect를 세지 않음)
        - positive_count: 0 (새 파이프라인은 긍정/부정 개수를 세지 않음)
        - negative_count: 0 (새 파이프라인은 긍정/부정 개수를 세지 않음)
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
                            restaurant_id=request.restaurant_id,
                            overall_summary="",
                            positive_aspects=[],
                            negative_aspects=[],
                        )
            
            # 새로운 파이프라인: 하이브리드 검색 + Aspect 기반 카테고리별 요약
        # 1. 기본 시드만 사용 (DEFAULT_*_SEEDS, 파일/ASPECT_SEEDS_FILE 미사용)
        seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]
        name_list = ["service", "price", "food"]
        logger.info("요약: 기본 시드만 사용")
        
        # 2. 카테고리별 하이브리드 검색
        hits_dict = {}
        hits_data_dict = {}
        
        for seeds, name in zip(seed_list, name_list):
            # Seed를 쿼리로 사용 (최대 10개만 사용하여 토큰 절약)
            query_seeds = seeds[:10] if len(seeds) > 10 else seeds
            query_text = " ".join(query_seeds)
            
            # 하이브리드 검색 수행
            hits = vector_search.query_hybrid_search(
                query_text=query_text,
                restaurant_id=request.restaurant_id,
                limit=request.limit,
                min_score=0.0,
            )
            
            # 카테고리별 리스트 초기화
            hits_dict[name] = []
            hits_data_dict[name] = []
            
            for rank, hit in enumerate(hits):
                payload = hit.get("payload", {})
                content = payload.get("content", "")
                review_id = payload.get("review_id") or payload.get("id") or str(hit.get("id", ""))
                
                hits_dict[name].append(content)
                hits_data_dict[name].append({
                    "review_id": str(review_id),
                    "snippet": content,
                    "rank": rank,
                })
        
        # 3. 새로운 파이프라인으로 요약 생성
        result = summarize_aspects_new(
            service_reviews=hits_dict.get("service", []),
            price_reviews=hits_dict.get("price", []),
            food_reviews=hits_dict.get("food", []),
            service_evidence_data=hits_data_dict.get("service", []),
            price_evidence_data=hits_data_dict.get("price", []),
            food_evidence_data=hits_data_dict.get("food", []),
            llm_utils=llm_utils,
            per_category_max=request.limit,
        )
        
        # 4. 응답 형식 변환 (기존 API 형식과 호환)
        # 새로운 형식: {"service": {...}, "price": {...}, "food": {...}, "overall_summary": {...}}
        # 기존 형식: {"overall_summary": "...", "positive_aspects": [...], "negative_aspects": [...]}
        
        # overall_summary 추출
        overall_summary = result.get("overall_summary", {}).get("summary", "")
        if not overall_summary:
            # 카테고리별 summary를 합쳐서 overall_summary 생성
            summaries = []
            for cat in ["service", "price", "food"]:
                cat_summary = result.get(cat, {}).get("summary", "")
                if cat_summary:
                    summaries.append(cat_summary)
            overall_summary = " ".join(summaries) if summaries else "요약할 리뷰가 없습니다."
        
        # positive_aspects와 negative_aspects는 빈 리스트로 설정 (새 파이프라인은 카테고리별 구조 사용)
        # categories를 CategorySummary 모델로 변환
        from ...models import CategorySummary
        categories_dict = {}
        for cat in ["service", "price", "food"]:
            cat_data = result.get(cat, {})
            if cat_data:
                categories_dict[cat] = CategorySummary(
                    summary=cat_data.get("summary", ""),
                    bullets=cat_data.get("bullets", []),
                    evidence=cat_data.get("evidence", []),
                )
        
        result = {
            "restaurant_id": request.restaurant_id,
            "overall_summary": overall_summary,
            "positive_aspects": [],  # 새 파이프라인은 카테고리별 구조 사용
            "negative_aspects": [],
            "positive_reviews": [],  # 새 파이프라인은 카테고리별로 분리
            "negative_reviews": [],
            "positive_count": 0,
            "negative_count": 0,
            "categories": categories_dict if categories_dict else None,
        }
        
        # 메트릭 수집
        total_reviews_count = sum(len(hits_dict.get(cat, [])) for cat in ["service", "price", "food"])
        request_id = metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="summary",
            start_time=start_time,
            tokens_used=result.get("tokens_used"),
            batch_size=total_reviews_count,
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
            # 일반 모드: 최소 필드만 (aspect + categories 포함)
            return SummaryDisplayResponse(
                restaurant_id=result["restaurant_id"],
                overall_summary=result["overall_summary"],
                positive_aspects=result.get("positive_aspects", []),
                negative_aspects=result.get("negative_aspects", []),
                categories=result.get("categories"),
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
    새로운 파이프라인: 통계적 비율 기반 강점 추출 (Kiwi + lift + LLM 설명)
    
    프로세스:
    1. 레스토랑 리뷰 조회 (VectorSearch)
    2. Kiwi 명사 bigram → service/price 긍정 비율 계산 (calculate_single_restaurant_ratios)
    3. 단일 vs 전체 평균 lift 계산 (calculate_strength_lift)
    4. LLM으로 자연어 설명 생성 (generate_strength_descriptions)
    5. 양수 lift만 강점으로 반환 (top_k 제한)
    
    Args:
        request: 강점 추출 요청
            - restaurant_id: 타겟 레스토랑 ID
            - strength_type: 무시됨 (새 파이프라인은 lift 기반 distinct만 출력)
            - top_k: 반환할 최대 강점 개수 (기본 10)
    
    Returns:
        강점 추출 결과 (category_lift, lift_percentage, all_average_ratio, single_restaurant_ratio 포함).
        category_lift: 카테고리별 lift 퍼센트(service, price). 이 수치를 근거로 LLM 설명 생성.
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
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """
    여러 레스토랑의 리뷰를 배치로 요약 (새 파이프라인: 카테고리별 하이브리드 검색 + 요약)
    
    각 레스토랑별로:
    1. 기본 시드만 사용 (DEFAULT_SERVICE/PRICE/FOOD_SEEDS, 파일 미사용)
    2. 카테고리별(service, price, food) 하이브리드 검색 수행
    3. 각 카테고리별로 LLM 요약 생성 (summary, bullets, evidence)
    4. overall_summary 생성
    
    주의사항:
    - 이 파이프라인은 긍정/부정 aspect를 세지 않습니다
    - 카테고리별(service/price/food) 하이브리드 검색으로 관련 리뷰를 찾고 요약만 생성합니다
    - 긍정/부정 분류는 sentiment analysis 파이프라인에서 수행됩니다
    
    Args:
        request: 배치 리뷰 요약 요청
            - restaurants: 레스토랑 데이터 리스트
                - restaurant_id: 레스토랑 ID
                - limit: 각 카테고리당 검색할 최대 리뷰 수 (기본값: 10)
                - min_score: 최소 유사도 점수 (기본값: 0.0, 사용 안 함)
    
    Returns:
        각 레스토랑별 요약 결과 리스트 (categories 기반)
    """
    try:
        results = []
        
        # 기본 시드만 사용 (1회만, 배치 공통)
        seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]
        name_list = ["service", "price", "food"]
        logger.info("요약: 기본 시드만 사용")

        # 각 레스토랑별로 새 파이프라인 적용
        for restaurant_data in request.restaurants:
            restaurant_id = restaurant_data.get("restaurant_id")
            limit = restaurant_data.get("limit", 10)
            
            # 카테고리별 하이브리드 검색
            hits_dict = {}
            hits_data_dict = {}
            
            for seeds, name in zip(seed_list, name_list):
                # Seed를 쿼리로 사용 (최대 10개만 사용하여 토큰 절약)
                query_seeds = seeds[:10] if len(seeds) > 10 else seeds
                query_text = " ".join(query_seeds)
                
                # 하이브리드 검색 수행
                hits = vector_search.query_hybrid_search(
                    query_text=query_text,
                    restaurant_id=restaurant_id,
                    limit=limit,
                    min_score=0.0,
                )
                
                # 카테고리별 리스트 초기화
                hits_dict[name] = []
                hits_data_dict[name] = []
                
                for rank, hit in enumerate(hits):
                    payload = hit.get("payload", {})
                    content = payload.get("content", "")
                    review_id = payload.get("review_id") or payload.get("id") or str(hit.get("id", ""))
                    
                    hits_dict[name].append(content)
                    hits_data_dict[name].append({
                        "review_id": str(review_id),
                        "snippet": content,
                        "rank": rank,
                    })
            
            # 새로운 파이프라인으로 요약 생성
            result = summarize_aspects_new(
                service_reviews=hits_dict.get("service", []),
                price_reviews=hits_dict.get("price", []),
                food_reviews=hits_dict.get("food", []),
                service_evidence_data=hits_data_dict.get("service", []),
                price_evidence_data=hits_data_dict.get("price", []),
                food_evidence_data=hits_data_dict.get("food", []),
                llm_utils=llm_utils,
                per_category_max=limit,
            )
            
            # 응답 형식 변환
            overall_summary = result.get("overall_summary", {}).get("summary", "")
            if not overall_summary:
                # 카테고리별 summary를 합쳐서 overall_summary 생성
                summaries = []
                for cat in ["service", "price", "food"]:
                    cat_summary = result.get(cat, {}).get("summary", "")
                    if cat_summary:
                        summaries.append(cat_summary)
                overall_summary = " ".join(summaries) if summaries else "요약할 리뷰가 없습니다."
            
            # categories를 CategorySummary 모델로 변환
            from ...models import CategorySummary
            categories_dict = {}
            for cat in ["service", "price", "food"]:
                cat_data = result.get(cat, {})
                if cat_data:
                    categories_dict[cat] = CategorySummary(
                        summary=cat_data.get("summary", ""),
                        bullets=cat_data.get("bullets", []),
                        evidence=cat_data.get("evidence", []),
                    )
            
            # 새 파이프라인은 긍정/부정 aspect를 세지 않음
            result_dict = {
                "restaurant_id": restaurant_id,
                "overall_summary": overall_summary,
                "positive_aspects": [],  # 새 파이프라인은 긍정/부정 aspect를 세지 않음
                "negative_aspects": [],  # 새 파이프라인은 긍정/부정 aspect를 세지 않음
                "positive_reviews": [],  # 새 파이프라인은 카테고리별로 분리되어 categories에 포함됨
                "negative_reviews": [],  # 새 파이프라인은 카테고리별로 분리되어 categories에 포함됨
                "positive_count": 0,  # 새 파이프라인은 긍정/부정 개수를 세지 않음
                "negative_count": 0,  # 새 파이프라인은 긍정/부정 개수를 세지 않음
                "categories": categories_dict if categories_dict else None,  # 실제 데이터는 여기에 포함됨
            }
            
            results.append(result_dict)
        
        return SummaryBatchResponse(results=[
            SummaryResponse(**result) for result in results
        ])
    except Exception as e:
        logger.error(f"배치 리뷰 요약 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"배치 리뷰 요약 중 오류 발생: {str(e)}"
        )
