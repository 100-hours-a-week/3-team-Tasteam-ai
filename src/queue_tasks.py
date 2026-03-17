"""
RQ 작업 정의 (배치 분리, 재시도, DLQ)

배치 API 호출 시 BATCH_USE_QUEUE=true면 큐에 넣고 job_id 반환.
실패 시 RQ 재시도 + FailedJobRegistry(DLQ)에 자동 이동.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import Config

logger = logging.getLogger(__name__)

# run_id 집계: 오케스트레이터(Lambda)가 완료 판단용으로 사용. job 성공 시 INCR run:{run_id}:done, 최종 실패 시 INCR run:{run_id}:fail
RUN_KEY_TOTAL = "run:{}:total"
RUN_KEY_DONE = "run:{}:done"
RUN_KEY_FAIL = "run:{}:fail"


def _run_id_incr_done(run_id: str) -> None:
    """job 성공 시 호출. Redis run:{run_id}:done INCR."""
    if not run_id:
        return
    try:
        from redis import Redis
        r = Redis.from_url(_get_redis_url())
        r.incr(RUN_KEY_DONE.format(run_id))
    except Exception as e:
        logger.warning("run_id done INCR 실패: %s", e)


def _run_id_incr_fail(run_id: str) -> None:
    """job 최종 실패 시 호출(재시도 소진 후). Redis run:{run_id}:fail INCR."""
    if not run_id:
        return
    try:
        from redis import Redis
        r = Redis.from_url(_get_redis_url())
        r.incr(RUN_KEY_FAIL.format(run_id))
    except Exception as e:
        logger.warning("run_id fail INCR 실패: %s", e)


def run_id_failure_callback(job, exc_type, exc_value, traceback):
    """RQ on_failure 콜백. job.meta에 run_id가 있으면 run:{run_id}:fail INCR."""
    run_id = (job.meta or {}).get("run_id")
    _run_id_incr_fail(run_id or "")


def _to_serializable(obj):
    """Pydantic/중첩 객체를 JSON 직렬화 가능 dict로 변환."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _get_redis_url() -> str:
    """Redis URL 구성 (RQ 연결용)."""
    pw = Config.REDIS_PASSWORD or ""
    auth = f":{pw}@" if pw else ""
    return f"redis://{auth}{Config.REDIS_HOST}:{Config.REDIS_PORT}/{Config.REDIS_DB}"


def _make_offline_meta(run_id: str) -> Dict[str, Any]:
    """오프라인 배치 결과용 meta."""
    return {
        "run_id": run_id,
        "trigger_type": "OFFLINE_SCHEDULED",
        "analysis_scope": "RESTAURANT",
        "schedule_time": datetime.now().isoformat(),
    }


def run_sentiment_batch_job(restaurants_json: str, run_id: Optional[str] = None) -> str:
    """
    감성 분석 배치 (동기 래퍼, RQ worker에서 호출).
    restaurants_json: [{"restaurant_id": 1, "restaurant_name": "..."}, ...]
    run_id: 오케스트레이터가 전달한 실행 ID. 있으면 완료 시 run:{run_id}:done INCR.
    """
    from .api.dependencies import get_qdrant_client, get_vector_search, get_sentiment_analyzer

    if not run_id:
        run_id = f"offline-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:8]}"
    restaurants_data = json.loads(restaurants_json)
    client = get_qdrant_client()
    vs = get_vector_search(qdrant_client=client)
    analyzer = get_sentiment_analyzer(vector_search=vs)

    async def _run():
        return await analyzer.analyze_multiple_restaurants_async(restaurants_data=restaurants_data)

    results = asyncio.run(_run())
    _run_id_incr_done(run_id)
    meta = _make_offline_meta(run_id)
    meta["restaurant_count"] = len(restaurants_data)
    return json.dumps({"meta": meta, "results": results}, ensure_ascii=False)


def run_summary_batch_job(restaurants_json: str, limit: int = 10, run_id: Optional[str] = None) -> str:
    """요약 배치. run_id 있으면 완료 시 run:{run_id}:done INCR."""
    from .api.dependencies import get_qdrant_client, get_vector_search, get_llm_utils
    from .api.routers.llm import _batch_summarize_async
    from .aspect_seeds import DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS
    from .models import SummaryBatchRequest

    if not run_id:
        run_id = f"offline-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:8]}"
    restaurants_data = json.loads(restaurants_json)
    client = get_qdrant_client()
    vs = get_vector_search(qdrant_client=client)
    llm = get_llm_utils()
    req = SummaryBatchRequest(restaurants=restaurants_data, limit=limit)
    seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]
    name_list = ["service", "price", "food"]

    async def _run():
        return await _batch_summarize_async(req, vs, llm, seed_list, name_list)

    results = asyncio.run(_run())
    _run_id_incr_done(run_id)
    meta = _make_offline_meta(run_id)
    meta["restaurant_count"] = len(restaurants_data)
    return json.dumps(
        {"meta": meta, "results": [_to_serializable(r) for r in results]},
        ensure_ascii=False,
        default=str,
    )


def run_comparison_batch_job(restaurants_json: str, run_id: Optional[str] = None) -> str:
    """비교 배치. run_id 있으면 완료 시 run:{run_id}:done INCR."""
    from .api.dependencies import get_qdrant_client, get_vector_search, get_llm_utils
    from .comparison import ComparisonPipeline

    if not run_id:
        run_id = f"offline-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:8]}"
    restaurants_data = json.loads(restaurants_json)
    # 입력 호환: 예전 포맷 [{"restaurant_id": 1, ...}, ...] 또는 신규 포맷 [1,2,3]
    if restaurants_data and isinstance(restaurants_data, list) and isinstance(restaurants_data[0], dict):
        restaurants_data = [r.get("restaurant_id") for r in restaurants_data if r.get("restaurant_id") is not None]
    client = get_qdrant_client()
    vs = get_vector_search(qdrant_client=client)
    llm = get_llm_utils()
    pipeline = ComparisonPipeline(llm_utils=llm, vector_search=vs)

    async def _run():
        return await pipeline.compare_batch(restaurants=restaurants_data)

    results = asyncio.run(_run())
    _run_id_incr_done(run_id)
    meta = _make_offline_meta(run_id)
    meta["restaurant_count"] = len(restaurants_data)
    return json.dumps(
        {"meta": meta, "results": [_to_serializable(r) for r in results]},
        ensure_ascii=False,
        default=str,
    )


def run_all_batch_job(restaurants_json: str, limit: int = 10, run_id: Optional[str] = None) -> str:
    """
    all 배치: sentiment → summary → comparison 순서로 실행 (batch_runner의 all 역할).
    restaurants_json: [{"restaurant_id": 1, "restaurant_name": "..."}, ...]
    run_id 있으면 완료 시 run:{run_id}:done INCR.
    """
    from .api.dependencies import get_qdrant_client, get_vector_search, get_sentiment_analyzer, get_llm_utils
    from .api.routers.llm import _batch_summarize_async
    from .aspect_seeds import DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS
    from .models import SummaryBatchRequest
    from .comparison import ComparisonPipeline

    if not run_id:
        run_id = f"offline-{datetime.now().strftime('%Y%m%d-%H%M')}-{uuid.uuid4().hex[:8]}"
    restaurants_data = json.loads(restaurants_json)
    # 입력 호환: 예전 포맷 [{"restaurant_id": 1, ...}, ...] 또는 신규 포맷 [1,2,3]
    if restaurants_data and isinstance(restaurants_data, list) and isinstance(restaurants_data[0], dict):
        restaurant_ids = [r.get("restaurant_id") for r in restaurants_data if r.get("restaurant_id") is not None]
    else:
        restaurant_ids = [int(x) for x in (restaurants_data or [])]

    results_by_restaurant: Dict[int, Dict[str, Any]] = {}
    for rid in restaurant_ids:
        results_by_restaurant[rid] = {"restaurant_id": rid, "summary": None, "sentiment": None, "comparison": None, "errors": {}}

    client = get_qdrant_client()
    vs = get_vector_search(qdrant_client=client)
    analyzer = get_sentiment_analyzer(vector_search=vs)
    llm = get_llm_utils()

    async def _run_all():
        # 1. sentiment
        try:
            # sentiment는 analyzer가 dict 입력을 기대하므로 구 포맷 유지
            sent_results = await analyzer.analyze_multiple_restaurants_async(
                restaurants_data=[{"restaurant_id": rid} for rid in restaurant_ids]
            )
            for r in sent_results:
                rid = r.get("restaurant_id")
                if rid is not None:
                    results_by_restaurant[rid]["sentiment"] = r
        except Exception as e:
            logger.exception("감성 분석 배치 실패")
            for rid in restaurant_ids:
                results_by_restaurant[rid].setdefault("errors", {})["sentiment"] = str(e)

        # 2. summary
        try:
            req = SummaryBatchRequest(restaurants=[{"restaurant_id": rid} for rid in restaurant_ids], limit=limit)
            seed_list = [DEFAULT_SERVICE_SEEDS, DEFAULT_PRICE_SEEDS, DEFAULT_FOOD_SEEDS]
            name_list = ["service", "price", "food"]
            sum_results = await _batch_summarize_async(req, vs, llm, seed_list, name_list)
            for r in sum_results:
                rid = r.get("restaurant_id")
                if rid is not None:
                    results_by_restaurant[rid]["summary"] = _to_serializable(r)
        except Exception as e:
            logger.exception("요약 배치 실패")
            for rid in restaurant_ids:
                results_by_restaurant[rid].setdefault("errors", {})["summary"] = str(e)

        # 3. comparison
        try:
            pipeline = ComparisonPipeline(llm_utils=llm, vector_search=vs)
            comp_results = await pipeline.compare_batch(restaurants=restaurant_ids)
            for r in comp_results:
                rid = r.get("restaurant_id") if isinstance(r, dict) else None
                if rid is not None and "error" not in r:
                    results_by_restaurant[rid]["comparison"] = _to_serializable(r)
        except Exception as e:
            logger.exception("비교 배치 실패")
            for rid in restaurant_ids:
                results_by_restaurant[rid].setdefault("errors", {})["comparison"] = str(e)

    asyncio.run(_run_all())

    _run_id_incr_done(run_id)
    meta = _make_offline_meta(run_id)
    meta["type"] = "all"
    meta["restaurant_count"] = len(restaurant_ids)
    return json.dumps(
        {"meta": meta, "results": list(results_by_restaurant.values())},
        ensure_ascii=False,
        default=str,
    )
