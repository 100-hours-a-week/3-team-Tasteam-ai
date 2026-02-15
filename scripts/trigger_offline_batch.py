#!/usr/bin/env python3
"""
오프라인 배치 트리거 스크립트 (EventBridge/cron에서 호출용)

입력 JSON에서 restaurants 추출 후 POST /api/v1/batch/enqueue 호출.
job_type: sentiment | summary | comparison | all

사용 예:
  python scripts/trigger_offline_batch.py -i data.json -t all --base-url http://localhost:8001
  python scripts/trigger_offline_batch.py -i data.json -t summary --limit 5  # 레스토랑 5개만

cron 예:
  0 3 * * * cd /app && python scripts/trigger_offline_batch.py -i /data/restaurants.json -t all -b http://api:8001
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DEFAULT_BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"


def load_restaurants(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """JSON에서 restaurants 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    restaurants = data.get("restaurants") or []
    if limit is not None:
        restaurants = restaurants[:limit]
    return [{"restaurant_id": r.get("id") or r.get("restaurant_id"), "restaurant_name": r.get("name")} for r in restaurants if (r.get("id") or r.get("restaurant_id")) is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="오프라인 배치 enqueue 트리거 (cron/EventBridge에서 호출)")
    parser.add_argument("--input", "-i", type=Path, required=True, help="입력 JSON (restaurants 또는 reviews+restaurants)")
    parser.add_argument("--job-type", "-t", type=str, choices=["sentiment", "summary", "comparison", "all"], default="all", help="job_type (기본 all)")
    parser.add_argument("--base-url", "-b", type=str, default=DEFAULT_BASE_URL, help="API 서버 URL")
    parser.add_argument("--limit", "-n", type=int, default=None, help="처리할 최대 레스토랑 수 (테스트용)")
    parser.add_argument("--summary-limit", type=int, default=10, help="summary용 카테고리당 검색 limit")
    parser.add_argument("--run-id", type=str, default=None, help="오케스트레이터 추적용 run_id (Lambda 등에서 전달 시 사용)")
    parser.add_argument("--timeout", type=int, default=30, help="enqueue 요청 타임아웃(초)")
    args = parser.parse_args()

    if httpx is None:
        print("오류: httpx 패키지가 필요합니다. pip install httpx", file=sys.stderr)
        sys.exit(1)

    if not args.input.is_file():
        print(f"오류: 입력 파일이 없습니다: {args.input}", file=sys.stderr)
        sys.exit(1)

    restaurants = load_restaurants(args.input, limit=args.limit)
    if not restaurants:
        print("경고: restaurants가 비어 있습니다. 종료합니다.", file=sys.stderr)
        sys.exit(0)

    url = f"{args.base_url.rstrip('/')}{API_PREFIX}/batch/enqueue"
    payload = {"job_type": args.job_type, "restaurants": restaurants, "limit": args.summary_limit}
    if args.run_id:
        payload["run_id"] = args.run_id

    try:
        with httpx.Client(timeout=args.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            job_id = data.get("job_id", "")
            print(f"enqueue 완료: job_id={job_id}, job_type={args.job_type}, restaurants={len(restaurants)}")
            print(f"상태 조회: curl {args.base_url.rstrip('/')}{API_PREFIX}/batch/status/{job_id}")
    except httpx.HTTPStatusError as e:
        print(f"enqueue 실패: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"enqueue 실패: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
