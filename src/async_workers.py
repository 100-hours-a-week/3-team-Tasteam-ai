"""
비동기 작업 큐 + 워커 (sentiment, LLM, Kiwi, embedding 블로킹 작업 격리).

- asyncio.Queue + 워커 2개.
- 리소스별 세마포 1: sentiment, llm, kiwi, embedding.
- 작업은 to_thread(fn, *args, **kwargs)로 실행되어 이벤트 루프 블로킹 방지.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 리소스별 세마포(1) — 동시에 하나만 실행
sentiment_sem: Optional[asyncio.Semaphore] = None
llm_sem: Optional[asyncio.Semaphore] = None
kiwi_sem: Optional[asyncio.Semaphore] = None
embedding_sem: Optional[asyncio.Semaphore] = None

work_queue: Optional[asyncio.Queue] = None
_worker_tasks: list[asyncio.Task] = []
_worker_count: int = 2

JOB_TYPE = str  # "sentiment" | "llm" | "kiwi" | "embedding"


def _get_worker_count() -> int:
    try:
        from .config import Config
        return max(1, getattr(Config, "ASYNC_WORKER_COUNT", 2))
    except Exception:
        return 2


def _get_semaphores() -> Dict[JOB_TYPE, asyncio.Semaphore]:
    return {
        "sentiment": sentiment_sem,
        "llm": llm_sem,
        "kiwi": kiwi_sem,
        "embedding": embedding_sem,
    }


def _ensure_init() -> None:
    """세마포·큐가 없으면 동기 초기화(루프에서 호출된 뒤여야 함)."""
    global sentiment_sem, llm_sem, kiwi_sem, embedding_sem, work_queue, _worker_count
    if work_queue is not None:
        return
    _worker_count = _get_worker_count()
    sentiment_sem = asyncio.Semaphore(1)
    llm_sem = asyncio.Semaphore(1)
    kiwi_sem = asyncio.Semaphore(1)
    embedding_sem = asyncio.Semaphore(1)
    work_queue = asyncio.Queue()
    logger.info(
        "async_workers 초기화: worker_count=%d, semaphores(sentiment,llm,kiwi,embedding)=1",
        _worker_count,
    )


async def start_workers() -> None:
    """세마포·큐·워커 태스크 시작. lifespan에서 호출."""
    global _worker_tasks
    _ensure_init()
    _worker_tasks = [asyncio.create_task(_worker()) for _ in range(_worker_count)]
    logger.info("async_workers: %d workers started", len(_worker_tasks))


async def stop_workers() -> None:
    """워커 태스크 취소."""
    global _worker_tasks
    for t in _worker_tasks:
        t.cancel()
    if _worker_tasks:
        await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks = []
    logger.info("async_workers: workers stopped")


async def _worker() -> None:
    """큐에서 작업을 꺼내 해당 세마포 획득 후 to_thread로 실행."""
    sems = _get_semaphores()
    while True:
        try:
            job = await work_queue.get()
        except asyncio.CancelledError:
            break
        try:
            job_type, future, fn, args, kwargs = job
            sem = sems.get(job_type)
            if sem is None:
                future.set_exception(ValueError(f"unknown job_type: {job_type}"))
                continue
            try:
                async with sem:
                    result = await asyncio.to_thread(fn, *args, **kwargs)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
        except Exception as e:
            logger.exception("worker task error: %s", e)
            if not future.done():
                future.set_exception(e)
        finally:
            work_queue.task_done()


async def run_via_queue(
    job_type: JOB_TYPE,
    fn: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    블로킹 작업을 큐에 넣고, 해당 리소스 세마포(1) + to_thread로 실행한 뒤 결과를 반환.
    job_type: "sentiment" | "llm" | "kiwi" | "embedding"
    """
    _ensure_init()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    await work_queue.put((job_type, future, fn, args, kwargs))
    return await future
