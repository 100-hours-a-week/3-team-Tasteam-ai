"""
LLM 관련 라우터 ( - Summary, Comparison)
"""

import asyncio
import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Union, Dict, Any

from ...llm_utils import LLMUtils
from ...vector_search import VectorSearch
from ...models import (
    SummaryRequest,
    SummaryDisplayResponse,
    SummaryBatchRequest,
    SummaryBatchResponse,
    ComparisonRequest,
    ComparisonResponse,
    DebugInfo
)
from ..dependencies import get_llm_utils, get_vector_search, get_metrics_collector, get_debug_mode
from ...metrics_collector import MetricsCollector
from ...config import Config
from ...comparison import ComparisonPipeline
from ...cache import acquire_lock
from ...summary_pipeline import summarize_aspects_new, summarize_aspects_new_async
from ...aspect_seeds import DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_category_result(result: Dict[str, Any], restaurant_id: int) -> Dict[str, Any]:
    """요약 파이프라인 결과를 SummaryDisplayResponse용 dict로 변환."""
    overall_summary = result.get("overall_summary", {}).get("summary", "")
    if not overall_summary:
        summaries = []
        for cat in ["service", "price", "food"]:
            cat_summary = result.get(cat, {}).get("summary", "")
            if cat_summary:
                summaries.append(cat_summary)
        overall_summary = " ".join(summaries) if summaries else "요약할 리뷰가 없습니다."
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
    return {
        "restaurant_id": restaurant_id,
        "overall_summary": overall_summary,
        "categories": categories_dict if categories_dict else None,
    }


async def _process_one_restaurant_async(
    restaurant_data: Dict[str, Any],
    request: SummaryBatchRequest,
    vector_search: VectorSearch,
    llm_utils: LLMUtils,
    seed_list: List[List[str]],
    name_list: List[str],
    search_sem: asyncio.Semaphore,
    llm_sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """한 레스토랑: search_async에 따라 aspect 3개 병렬/순차 → 요약."""
    restaurant_id = restaurant_data.get("restaurant_id")

    async def do_one_search(seeds: List[str], name: str) -> tuple:
        query_seeds = seeds[:10] if len(seeds) > 10 else seeds
        query_text = " ".join(query_seeds)
        async with search_sem:
            hits = await asyncio.to_thread(
                vector_search.query_hybrid_search,
                query_text,
                restaurant_id=restaurant_id,
                limit=request.limit,
                min_score=request.min_score,
            )
        contents = []
        data_list = []
        for rank, hit in enumerate(hits):
            payload = hit.get("payload", {})
            content = payload.get("content", "")
            review_id = payload.get("review_id") or payload.get("id") or str(hit.get("id", ""))
            contents.append(content)
            data_list.append({"review_id": str(review_id), "snippet": content, "rank": rank})
        return name, contents, data_list

    if Config.SUMMARY_SEARCH_ASYNC:
        search_results = await asyncio.gather(
            *(do_one_search(seeds, name) for seeds, name in zip(seed_list, name_list))
        )
    else:
        search_results = []
        for seeds, name in zip(seed_list, name_list):
            search_results.append(await do_one_search(seeds, name))
    hits_dict = {name: contents for name, contents, _ in search_results}
    hits_data_dict = {name: data_list for name, _, data_list in search_results}

    async with llm_sem:
        if Config.SUMMARY_LLM_ASYNC:
            result = await summarize_aspects_new_async(
                service_reviews=hits_dict.get("service", []),
                price_reviews=hits_dict.get("price", []),
                food_reviews=hits_dict.get("food", []),
                service_evidence_data=hits_data_dict.get("service", []),
                price_evidence_data=hits_data_dict.get("price", []),
                food_evidence_data=hits_data_dict.get("food", []),
                llm_utils=llm_utils,
                per_category_max=request.limit,
            )
        else:
            result = await asyncio.to_thread(
                summarize_aspects_new,
                service_reviews=hits_dict.get("service", []),
                price_reviews=hits_dict.get("price", []),
                food_reviews=hits_dict.get("food", []),
                service_evidence_data=hits_data_dict.get("service", []),
                price_evidence_data=hits_data_dict.get("price", []),
                food_evidence_data=hits_data_dict.get("food", []),
                llm_utils=llm_utils,
                per_category_max=request.limit,
            )
    return _build_category_result(result, restaurant_id)


async def _batch_summarize_async(
    request: SummaryBatchRequest,
    vector_search: VectorSearch,
    llm_utils: LLMUtils,
    seed_list: List[List[str]],
    name_list: List[str],
) -> List[Dict[str, Any]]:
    """배치 요약: search_async=aspect 병렬, restaurant_async=음식점 간 병렬."""
    search_sem = asyncio.Semaphore(Config.BATCH_SEARCH_CONCURRENCY)
    llm_sem = asyncio.Semaphore(Config.BATCH_LLM_CONCURRENCY)

    async def one(rd: Dict[str, Any]) -> Dict[str, Any]:
        return await _process_one_restaurant_async(
            rd, request, vector_search, llm_utils, seed_list, name_list, search_sem, llm_sem
        )

    if Config.SUMMARY_RESTAURANT_ASYNC:
        gathered = await asyncio.gather(
            *(one(rd) for rd in request.restaurants), return_exceptions=True
        )
        results = []
        for i, res in enumerate(gathered):
            if isinstance(res, Exception):
                logger.error(
                    f"배치 요약 restaurant_async: restaurant_id={request.restaurants[i].get('restaurant_id')} 실패: {res}",
                    exc_info=res,
                )
                raise HTTPException(status_code=500, detail=f"배치 요약 중 오류: {res!s}")
            results.append(res)
        return results
    else:
        results = []
        for rd in request.restaurants:
            results.append(await one(rd))
        return results


@router.post("/summarize", response_model=SummaryDisplayResponse)
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
        - debug: X-Debug: true 시에만 포함 (request_id, processing_time_ms 등)
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
                    return SummaryDisplayResponse(
                        restaurant_id=request.restaurant_id,
                        overall_summary="",
                        debug=DebugInfo(
                            request_id=request_id,
                            processing_time_ms=(time.time() - start_time) * 1000,
                        ) if debug else None,
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
        
        # 4. 응답 형식 변환
        # 파이프라인 출력: {"service": {...}, "price": {...}, "food": {...}, "overall_summary": {...}}
        
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
        
        # 메트릭 수집
        total_reviews_count = sum(len(hits_dict.get(cat, [])) for cat in ["service", "price", "food"])
        request_id = metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="summary",
            start_time=start_time,
            tokens_used=None,
            batch_size=total_reviews_count,
        )
        
        # 항상 SummaryDisplayResponse (positive_reviews 등 미사용 필드 제외)
        return SummaryDisplayResponse(
            restaurant_id=request.restaurant_id,
            overall_summary=overall_summary,
            categories=categories_dict if categories_dict else None,
            debug=DebugInfo(
                request_id=request_id,
                processing_time_ms=(time.time() - start_time) * 1000,
                tokens_used=None,
                model_version=None,
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


@router.post("/comparison", response_model=Union[ComparisonResponse, Dict])
async def compare(
    request: ComparisonRequest,
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    debug: bool = Depends(get_debug_mode),
):
    """
    새로운 파이프라인: 통계적 비율 기반 다른 음식점과의 비교 (Kiwi + lift + LLM 설명)
    
    프로세스:
    1. 레스토랑 리뷰 조회 (VectorSearch)
    2. Kiwi 명사 bigram → service/price 긍정 비율 계산 (calculate_single_restaurant_ratios)
    3. 단일 vs 전체 평균 lift 계산 (calculate_comparison_lift)
    4. LLM으로 자연어 설명 생성 (generate_comparison_descriptions)
    5. 양수 lift만 반환 (top_k 제한)
    
    Args:
        request: 비교 요청
            - restaurant_id: 타겟 레스토랑 ID
            - top_k: 반환할 최대 비교 항목 개수 (기본 10)
    
    Returns:
        비교 결과 (category_lift, lift_percentage, all_average_ratio, single_restaurant_ratio 포함).
        category_lift: 카테고리별 lift 퍼센트(service, price). 이 수치를 근거로 LLM 설명 생성.
    """
    start_time = time.time()
    
    # 중복 실행 방지: Redis 락 획득 (엔드포인트 진입 시)
    try:
        with acquire_lock(
            restaurant_id=request.restaurant_id,
            analysis_type="comparison",
            ttl=3600,  # 1시간
        ):
            # SKIP 로직: 최근 성공 실행이면 SKIP (미세한 중복/과호출 흡수)
            if metrics.metrics_db and request.restaurant_id is not None:
                from ...config import Config
                
                if metrics.metrics_db.should_skip_analysis(
                    restaurant_id=request.restaurant_id,
                    analysis_type="comparison",
                    min_interval_seconds=Config.SKIP_MIN_INTERVAL_SECONDS,
                ):
                    # SKIP 응답 반환 (최근 처리 완료)
                    last_success_at = metrics.metrics_db.get_last_success_at(
                        restaurant_id=request.restaurant_id,
                        analysis_type="comparison",
                    )
                    
                    # 메트릭 수집 (SKIP도 기록)
                    request_id = metrics.collect_metrics(
                        restaurant_id=request.restaurant_id,
                        analysis_type="comparison",
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
                        return ComparisonResponse(
                            restaurant_id=request.restaurant_id,
                            comparisons=[],
                            total_candidates=0,
                            validated_count=0,
                            debug=DebugInfo(
                                request_id=request_id,
                                processing_time_ms=(time.time() - start_time) * 1000,
                            ),
                        )
                    else:
                        return ComparisonResponse(
                            restaurant_id=request.restaurant_id,
                            comparisons=[],
                            total_candidates=0,
                            validated_count=0,
                        )
            
            # 파이프라인 초기화
        pipeline = ComparisonPipeline(
            llm_utils=llm_utils,
            vector_search=vector_search,
        )
        
        # 파이프라인 실행
        result = await pipeline.compare(
            restaurant_id=request.restaurant_id,
            top_k=request.top_k,
        )
        
        # 메트릭 수집
        request_id = metrics.collect_metrics(
            restaurant_id=request.restaurant_id,
            analysis_type="comparison",
            start_time=start_time,
            batch_size=result.get("total_candidates", 0),
            )
            
        # 디버그 정보 추가
        if debug:
            result["debug"] = DebugInfo(
                request_id=request_id,
                processing_time_ms=result.get("processing_time_ms", 0),
            )
        
        return ComparisonResponse(**result)
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
            analysis_type="comparison",
            start_time=start_time,
            error_count=1,
            additional_info={"error": str(e)},
        )
        logger.error(f"비교 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"비교 중 오류 발생: {str(e)}")


@router.post("/summarize/batch", response_model=SummaryBatchResponse)
async def summarize_reviews_batch(
    request: SummaryBatchRequest,
    llm_utils: LLMUtils = Depends(get_llm_utils),
    vector_search: VectorSearch = Depends(get_vector_search),
    metrics: MetricsCollector = Depends(get_metrics_collector),
):
    """
    여러 레스토랑의 리뷰를 배치로 요약 (새 파이프라인: 카테고리별 하이브리드 검색 + 요약)
    
    search_async (SUMMARY_SEARCH_ASYNC): aspect(service/price/food) 서치 병렬. restaurant_async (SUMMARY_RESTAURANT_ASYNC): 음식점 간 병렬. 둘 다 false면 완전 순차. SUMMARY_LLM_ASYNC=true면 AsyncOpenAI/httpx, false면 to_thread.
    
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
            - restaurants: 레스토랑 데이터 리스트 (각 항목: restaurant_id)
            - limit: 각 카테고리당 검색할 최대 리뷰 수 (전체 레스토랑 공통, 기본값: 10)
            - min_score: 최소 유사도 점수 (전체 레스토랑 공통, 기본값: 0.0)
    
    Returns:
        각 레스토랑별 요약 결과 리스트 (categories 기반)
    """
    try:
        seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]
        name_list = ["service", "price", "food"]
        logger.info("요약: 기본 시드만 사용")

        if Config.SUMMARY_SEARCH_ASYNC or Config.SUMMARY_RESTAURANT_ASYNC:
            results = await _batch_summarize_async(request, vector_search, llm_utils, seed_list, name_list)
            return SummaryBatchResponse(results=[SummaryDisplayResponse(**r) for r in results])

        # search_async=false, restaurant_async=false: 레스토랑·aspect 완전 순차
        results = []
        for restaurant_data in request.restaurants:
            restaurant_id = restaurant_data.get("restaurant_id")
            
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
                    limit=request.limit,
                    min_score=request.min_score,
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
            results.append(_build_category_result(result, restaurant_id))
        
        return SummaryBatchResponse(results=[
            SummaryDisplayResponse(**r) for r in results
        ])
    except Exception as e:
        logger.error(f"배치 리뷰 요약 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"배치 리뷰 요약 중 오류 발생: {str(e)}"
        )
