#!/usr/bin/env python3
"""
단일 레스토랑 분석 재현 스크립트 (로컬/디버깅용)

한 레스토랑에 대해 sentiment, summary, comparison 동기 API를 순차 호출.
batch_runner의 "한 번에 재현" 역할 대체.

사용 예:
  python scripts/run_single_restaurant.py --restaurant-id 1 --base-url http://localhost:8001
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 레스토랑 sentiment/summary/comparison 동기 API 호출 (재현용)")
    parser.add_argument("--restaurant-id", "-r", type=int, required=True, help="레스토랑 ID")
    parser.add_argument("--base-url", "-b", type=str, default=DEFAULT_BASE_URL, help="API 서버 URL")
    parser.add_argument("--limit", type=int, default=10, help="summary용 카테고리당 검색 limit")
    parser.add_argument("--timeout", type=int, default=120, help="요청 타임아웃(초)")
    args = parser.parse_args()

    if httpx is None:
        print("오류: httpx 패키지가 필요합니다. pip install httpx", file=sys.stderr)
        sys.exit(1)

    base = args.base_url.rstrip("/")
    result: dict = {"restaurant_id": args.restaurant_id, "sentiment": None, "summary": None, "comparison": None, "errors": {}}

    with httpx.Client(timeout=args.timeout) as client:
        # 1. sentiment
        try:
            resp = client.post(f"{base}{API_PREFIX}/sentiment/analyze", json={"restaurant_id": args.restaurant_id})
            resp.raise_for_status()
            result["sentiment"] = resp.json()
            print(f"sentiment OK: restaurant_id={args.restaurant_id}")
        except Exception as e:
            result["errors"]["sentiment"] = str(e)
            print(f"sentiment 실패: {e}", file=sys.stderr)

        # 2. summary
        try:
            resp = client.post(f"{base}{API_PREFIX}/llm/summarize", json={"restaurant_id": args.restaurant_id, "limit": args.limit})
            resp.raise_for_status()
            result["summary"] = resp.json()
            print(f"summary OK: restaurant_id={args.restaurant_id}")
        except Exception as e:
            result["errors"]["summary"] = str(e)
            print(f"summary 실패: {e}", file=sys.stderr)

        # 3. comparison
        try:
            resp = client.post(f"{base}{API_PREFIX}/llm/comparison", json={"restaurant_id": args.restaurant_id})
            resp.raise_for_status()
            result["comparison"] = resp.json()
            print(f"comparison OK: restaurant_id={args.restaurant_id}")
        except Exception as e:
            result["errors"]["comparison"] = str(e)
            print(f"comparison 실패: {e}", file=sys.stderr)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
