#!/usr/bin/env python3
"""
RQ 워커 (배치 작업 소비)

BATCH_USE_QUEUE=true 시 API가 enqueue한 작업을 소비.
실패 시 재시도 → 최종 실패 시 FailedJobRegistry(DLQ).

실행:
  python scripts/rq_worker.py
  python scripts/rq_worker.py --burst  # 한 번에 처리 후 종료 (cron용)

필요: pip install rq
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.config import Config
from src.queue_tasks import _get_redis_url


WORKER_READY_KEY = "worker:{}:ready"
WORKER_READY_TTL = 120  # 초. Lambda가 이 시간 안에 ready 폴링하면 됨.


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--burst", action="store_true", help="처리할 작업만 하고 종료 (cron용)")
    args = parser.parse_args()

    from redis import Redis
    from rq import Worker, Queue

    redis_url = _get_redis_url()
    queue_name = Config.RQ_QUEUE_NAME
    redis_conn = Redis.from_url(redis_url)
    queue = Queue(queue_name, connection=redis_conn)

    # RunPod Pod에서 기동 시 오케스트레이터(Lambda)가 "워커 준비" 판단용. 로컬에서는 RUNPOD_POD_ID 미설정 시 스킵
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        try:
            redis_conn.set(WORKER_READY_KEY.format(pod_id), "1", ex=WORKER_READY_TTL)
        except Exception:
            pass

    worker = Worker([queue], connection=redis_conn)
    worker.work(with_scheduler=False, burst=args.burst)


if __name__ == "__main__":
    main()
