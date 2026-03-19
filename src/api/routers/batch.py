"""
배치 큐 라우터 (BATCH_USE_QUEUE=true 시 사용)

POST /enqueue: 배치 작업을 큐에 넣고 job_id 반환
GET /status/{job_id}: 작업 상태 조회
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

from ...config import Config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


class BatchRestaurantInput(BaseModel):
    """배치 작업용 레스토랑 입력 (ID만)."""
    restaurant_id: int = Field(..., description="레스토랑 ID")


class BatchEnqueueRequest(BaseModel):
    """배치 enqueue 요청."""
    job_type: str = Field(..., description="sentiment | summary | comparison | all")
    restaurants: List[BatchRestaurantInput] = Field(..., description="레스토랑 ID 리스트 (각 항목: restaurant_id)")
    limit: Optional[int] = 10  # summary용
    run_id: Optional[str] = None  # 오케스트레이터(Lambda) 추적용. 있으면 Redis run:{run_id}:total/done/fail 초기화 및 job meta에 저장


class BatchEnqueueResponse(BaseModel):
    """배치 enqueue 응답."""
    job_id: str
    job_type: str
    queue: str
    run_id: Optional[str] = None


class BatchStatusResponse(BaseModel):
    """배치 작업 상태."""
    job_id: str
    status: str  # queued | started | finished | failed | deferred
    result: Optional[str] = None
    exc_info: Optional[str] = None


class RunStatusResponse(BaseModel):
    """run_id 단위 집계 (오케스트레이터 완료 판단용)."""
    run_id: str
    total: int
    done: int
    fail: int
    completed: bool  # done + fail >= total


def _get_queue():
    """RQ 큐 인스턴스 (Redis 필요)."""
    from redis import Redis
    from rq import Queue
    from ...queue_tasks import _get_redis_url

    redis_conn = Redis.from_url(_get_redis_url())
    return Queue(Config.RQ_QUEUE_NAME, connection=redis_conn)


@router.post("/enqueue", response_model=BatchEnqueueResponse)
async def enqueue_batch(request: BatchEnqueueRequest):
    """
    배치 작업을 큐에 넣고 job_id 반환.
    BATCH_USE_QUEUE=true 이고 Redis 연결 가능할 때만 동작.
    """
    if not Config.BATCH_USE_QUEUE:
        raise HTTPException(status_code=503, detail="BATCH_USE_QUEUE가 비활성화되어 있습니다. 환경 변수 BATCH_USE_QUEUE=true 설정 후 Redis를 기동하세요.")
    try:
        import json
        from redis import Redis
        from ...queue_tasks import (
            _get_redis_url,
            RUN_KEY_TOTAL,
            RUN_KEY_DONE,
            RUN_KEY_FAIL,
            run_sentiment_batch_job,
            run_summary_batch_job,
            run_comparison_batch_job,
            run_all_batch_job,
            run_id_failure_callback,
        )

        # 직렬화: [{"restaurant_id": 1}, ...] (sentiment/summary 호환). comparison/all 쪽에서 id 리스트로 변환
        restaurants_json = json.dumps(
            [{"restaurant_id": r.restaurant_id} for r in request.restaurants],
            ensure_ascii=False,
        )
        queue = _get_queue()
        run_id = request.run_id

        if run_id:
            redis_conn = Redis.from_url(_get_redis_url())
            redis_conn.set(RUN_KEY_TOTAL.format(run_id), 1, ex=86400 * 2)
            redis_conn.set(RUN_KEY_DONE.format(run_id), 0, ex=86400 * 2)
            redis_conn.set(RUN_KEY_FAIL.format(run_id), 0, ex=86400 * 2)

        retry_val = Config.BATCH_JOB_MAX_RETRIES
        enqueue_kw = {
            "job_timeout": "60m",
            "failure_ttl": 86400,
            "result_ttl": 3600,
            "retry": retry_val,
            "on_failure": run_id_failure_callback,
        }
        if run_id:
            enqueue_kw["meta"] = {"run_id": run_id}

        if request.job_type == "sentiment":
            job = queue.enqueue(run_sentiment_batch_job, restaurants_json, run_id, **enqueue_kw)
        elif request.job_type == "summary":
            job = queue.enqueue(run_summary_batch_job, restaurants_json, request.limit or 10, run_id, **enqueue_kw)
        elif request.job_type == "comparison":
            job = queue.enqueue(run_comparison_batch_job, restaurants_json, run_id, **enqueue_kw)
        elif request.job_type == "all":
            job = queue.enqueue(run_all_batch_job, restaurants_json, request.limit or 10, run_id, **enqueue_kw)
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 job_type: {request.job_type}")

        return BatchEnqueueResponse(
            job_id=job.id,
            job_type=request.job_type,
            queue=Config.RQ_QUEUE_NAME,
            run_id=run_id,
        )
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"RQ 미설치: pip install rq. {e}")
    except Exception as e:
        logger.exception("배치 enqueue 실패")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=BatchStatusResponse)
async def get_batch_status(job_id: str):
    """작업 상태 조회."""
    try:
        from rq.job import Job
        from ...queue_tasks import _get_redis_url
        from redis import Redis

        redis_conn = Redis.from_url(_get_redis_url())
        job = Job.fetch(job_id, connection=redis_conn)
        result = None
        exc_info = None
        if job.result is not None:
            result = job.result if isinstance(job.result, str) else str(job.result)
        if job.exc_info:
            exc_info = job.exc_info[:2000]
        return BatchStatusResponse(
            job_id=job_id,
            status=job.get_status(),
            result=result,
            exc_info=exc_info,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/run/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """
    run_id 단위 집계 조회. Lambda 등 오케스트레이터가 done+fail==total 로 완료 판단할 때 사용.
    Redis run:{run_id}:total, run:{run_id}:done, run:{run_id}:fail 읽음.
    """
    try:
        from redis import Redis
        from ...queue_tasks import _get_redis_url, RUN_KEY_TOTAL, RUN_KEY_DONE, RUN_KEY_FAIL

        redis_conn = Redis.from_url(_get_redis_url())
        total = int(redis_conn.get(RUN_KEY_TOTAL.format(run_id)) or 0)
        done = int(redis_conn.get(RUN_KEY_DONE.format(run_id)) or 0)
        fail = int(redis_conn.get(RUN_KEY_FAIL.format(run_id)) or 0)
        return RunStatusResponse(
            run_id=run_id,
            total=total,
            done=done,
            fail=fail,
            completed=(done + fail >= total) if total > 0 else False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
