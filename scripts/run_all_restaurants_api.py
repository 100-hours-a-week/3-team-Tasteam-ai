#!/usr/bin/env python3
"""
모든 레스토랑에 대해 벡터 업로드 후 Summary / Sentiment / Comparison API를 호출하고 결과를 JSON으로 저장하는 스크립트.

입력: tasteam_app_data.json 형식 (reviews, restaurants 포함)
  - restaurants: [{ id, name }, ...]
  - reviews: [{ id, restaurant_id, content, created_at }, ...]

호출 API:
  1. POST /api/v1/vector/upload  (선택, 기본 수행: 입력 JSON으로 벡터 DB 업로드)
  2. 한 청크 내에서 3개 병렬:
     - POST /api/v1/llm/summarize/batch  (요약)
     - POST /api/v1/sentiment/analyze/batch  (감성 분석, 리뷰 전달)
     - POST /api/v1/llm/comparison/batch  (비교)

출력 JSON:
  {
    "meta": { "base_url", "total_restaurants", "requested_at", "batch_size" },
    "results": [
      {
        "restaurant_id": int,
        "restaurant_name": str,
        "summary": { ... } | null,
        "sentiment": { ... } | null,
        "comparison": { ... } | null,
        "errors": { "summary"?: str, "sentiment"?: str, "comparison"?: str }
      }
    ]
  }

사용 예:
  python scripts/run_all_restaurants_api.py --input tasteam_app_data.json --output all_restaurants_results.json
  python scripts/run_all_restaurants_api.py -i tasteam_app_data.json -o results.json --no-upload  # 업로드 생략
  python scripts/run_all_restaurants_api.py -i tasteam_app_data.json -o results.json --base-url http://localhost:8001 --batch-size 5 --limit 3
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SENTIMENT_REVIEWS_LIMIT = 100
DEFAULT_SUMMARY_LIMIT = 10
DEFAULT_UPLOAD_TIMEOUT = 3600


def load_data(path: Path) -> tuple[List[Dict], List[Dict]]:
    """JSON에서 reviews, restaurants 로드."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reviews = data.get("reviews") or []
    restaurants = data.get("restaurants") or []
    return reviews, restaurants


def build_reviews_by_restaurant(
    reviews: List[Dict],
    max_per_restaurant: int = DEFAULT_SENTIMENT_REVIEWS_LIMIT,
) -> Dict[int, List[Dict]]:
    """restaurant_id별 리뷰 리스트 구성 (SentimentReviewInput 형식)."""
    by_rid: Dict[int, List[Dict]] = {}
    for r in reviews:
        rid = r.get("restaurant_id")
        if rid is None:
            continue
        rid = int(rid) if isinstance(rid, str) and str(rid).isdigit() else rid
        if rid not in by_rid:
            by_rid[rid] = []
        if len(by_rid[rid]) >= max_per_restaurant:
            continue
        item = {
            "id": r.get("id") or r.get("review_id") or len(by_rid[rid]) + 1,
            "restaurant_id": rid,
            "content": (r.get("content") or r.get("review") or "").strip(),
            "created_at": r.get("created_at") or datetime.now().isoformat(),
        }
        if item["content"]:
            by_rid[rid].append(item)
    return by_rid


async def call_upload_async(
    client: httpx.AsyncClient,
    base_url: str,
    payload: Dict[str, Any],
    timeout: float = float(DEFAULT_UPLOAD_TIMEOUT),
) -> tuple[bool, str]:
    """POST /api/v1/vector/upload 호출. (success, message) 반환."""
    url = f"{base_url.rstrip('/')}{API_PREFIX}/vector/upload"
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return False, resp.text or str(resp.status_code)
        data = resp.json()
        count = data.get("points_count", 0)
        return True, f"업로드 완료: {count}개 포인트"
    except Exception as e:
        return False, str(e)


async def call_summary_batch_async(
    client: httpx.AsyncClient,
    base_url: str,
    restaurants_chunk: List[Dict],
    limit: int = DEFAULT_SUMMARY_LIMIT,
    min_score: float = 0.0,
    timeout: float = 600.0,
) -> tuple[List[Dict], List[Optional[str]]]:
    """요약 배치 API 비동기 호출. (results, errors) 반환."""
    url = f"{base_url.rstrip('/')}{API_PREFIX}/llm/summarize/batch"
    payload = {
        "restaurants": [
            {"restaurant_id": r["id"], "restaurant_name": r.get("name")}
            for r in restaurants_chunk
        ],
        "limit": limit,
        "min_score": min_score,
    }
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return [], [resp.text or str(resp.status_code)] * len(restaurants_chunk)
        data = resp.json()
        results = data.get("results") or []
        errors = [None] * len(restaurants_chunk)
        if len(results) < len(restaurants_chunk):
            for i in range(len(results), len(restaurants_chunk)):
                errors[i] = "결과 없음"
        return results, errors
    except Exception as e:
        return [], [str(e)] * len(restaurants_chunk)


async def call_sentiment_batch_async(
    client: httpx.AsyncClient,
    base_url: str,
    restaurants_chunk: List[Dict],
    reviews_by_rid: Dict[int, List[Dict]],
    timeout: float = 600.0,
) -> tuple[List[Dict], List[Optional[str]]]:
    """감성 분석 배치 API 비동기 호출. (results, errors) 반환."""
    url = f"{base_url.rstrip('/')}{API_PREFIX}/sentiment/analyze/batch"
    restaurants_payload = []
    for r in restaurants_chunk:
        rid = r["id"]
        reviews = reviews_by_rid.get(rid) or []
        restaurants_payload.append({
            "restaurant_id": rid,
            "restaurant_name": r.get("name"),
            "reviews": reviews,
        })
    payload = {"restaurants": restaurants_payload}
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return [], [resp.text or str(resp.status_code)] * len(restaurants_chunk)
        data = resp.json()
        results = data.get("results") or []
        errors = [None] * len(restaurants_chunk)
        if len(results) < len(restaurants_chunk):
            for i in range(len(results), len(restaurants_chunk)):
                errors[i] = "결과 없음"
        return results, errors
    except Exception as e:
        return [], [str(e)] * len(restaurants_chunk)


async def call_comparison_batch_async(
    client: httpx.AsyncClient,
    base_url: str,
    restaurants_chunk: List[Dict],
    all_average_data_path: Optional[Path] = None,
    timeout: float = 300.0,
) -> tuple[List[Dict], List[Optional[str]]]:
    """비교 배치 API 비동기 호출. (results, errors) 반환.
    all_average_data_path: 전체 평균·표본 추출용 파일 경로 (--input과 동일)."""
    url = f"{base_url.rstrip('/')}{API_PREFIX}/llm/comparison/batch"
    payload: Dict[str, Any] = {
        "restaurants": [
            {"restaurant_id": r["id"], "restaurant_name": r.get("name")}
            for r in restaurants_chunk
        ],
    }
    if all_average_data_path is not None and all_average_data_path.exists():
        payload["all_average_data_path"] = str(all_average_data_path.resolve())
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return [], [resp.text or str(resp.status_code)] * len(restaurants_chunk)
        data = resp.json()
        results = data.get("results") or []
        errors = [None] * len(restaurants_chunk)
        if len(results) < len(restaurants_chunk):
            for i in range(len(results), len(restaurants_chunk)):
                errors[i] = "결과 없음"
        return results, errors
    except Exception as e:
        return [], [str(e)] * len(restaurants_chunk)


async def run_async(
    input_path: Path,
    output_path: Path,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit_restaurants: Optional[int] = None,
    summary_limit: int = DEFAULT_SUMMARY_LIMIT,
    sentiment_reviews_limit: int = DEFAULT_SENTIMENT_REVIEWS_LIMIT,
    apis: List[str] = None,
    timeout: int = 600,
    do_upload: bool = True,
    upload_timeout: int = DEFAULT_UPLOAD_TIMEOUT,
) -> None:
    """메인: 데이터 로드 → (선택) 벡터 업로드 → 청크별 3 API 병렬 호출 → 결과 JSON 저장."""
    if httpx is None:
        raise RuntimeError("httpx 패키지가 필요합니다: pip install httpx")

    apis = apis or ["summary", "sentiment", "comparison"]
    reviews, restaurants = load_data(input_path)

    async with httpx.AsyncClient() as client:
        if do_upload:
            upload_payload = {"reviews": reviews, "restaurants": restaurants or []}
            print(f"벡터 업로드 중: 리뷰 {len(reviews)}개, 레스토랑 {len(restaurants)}개 (timeout={upload_timeout}s)...")
            ok, msg = await call_upload_async(client, base_url, upload_payload, timeout=float(upload_timeout))
            if ok:
                print(msg)
            else:
                print(f"경고: 벡터 업로드 실패: {msg}", file=sys.stderr)

    if not restaurants:
        print("경고: restaurants가 비어 있습니다. 종료합니다.", file=sys.stderr)
        out = {"meta": {"base_url": base_url, "total_restaurants": 0, "requested_at": datetime.now().isoformat(), "batch_size": batch_size}, "results": []}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return

    if limit_restaurants is not None:
        restaurants = restaurants[:limit_restaurants]
    total = len(restaurants)
    reviews_by_rid = build_reviews_by_restaurant(reviews, max_per_restaurant=sentiment_reviews_limit)
    timeout_float = float(timeout)

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for start in range(0, total, batch_size):
            chunk = restaurants[start : start + batch_size]
            chunk_size = len(chunk)
            summary_res, summary_err = [], [None] * chunk_size
            sentiment_res, sentiment_err = [], [None] * chunk_size
            comparison_res, comparison_err = [], [None] * chunk_size

            async def empty_result(n: int) -> tuple[List[Dict], List[Optional[str]]]:
                return [], [None] * n

            tasks: List[Any] = [
                call_summary_batch_async(client, base_url, chunk, limit=summary_limit, timeout=timeout_float)
                if "summary" in apis else empty_result(chunk_size),
                call_sentiment_batch_async(client, base_url, chunk, reviews_by_rid, timeout=timeout_float)
                if "sentiment" in apis else empty_result(chunk_size),
                call_comparison_batch_async(client, base_url, chunk, all_average_data_path=input_path, timeout=timeout_float)
                if "comparison" in apis else empty_result(chunk_size),
            ]
            gathered = await asyncio.gather(*tasks)
            summary_res, summary_err = gathered[0]
            sentiment_res, sentiment_err = gathered[1]
            comparison_res, comparison_err = gathered[2]

            for i in range(chunk_size):
                r = chunk[i]
                rid = r["id"]
                name = r.get("name") or ""
                errs: Dict[str, Optional[str]] = {}
                if summary_err and summary_err[i]:
                    errs["summary"] = summary_err[i]
                if sentiment_err and sentiment_err[i]:
                    errs["sentiment"] = sentiment_err[i]
                if comparison_err and comparison_err[i]:
                    errs["comparison"] = comparison_err[i]
                results.append({
                    "restaurant_id": rid,
                    "restaurant_name": name,
                    "summary": summary_res[i] if i < len(summary_res) else None,
                    "sentiment": sentiment_res[i] if i < len(sentiment_res) else None,
                    "comparison": comparison_res[i] if i < len(comparison_res) else None,
                    "errors": errs if errs else None,
                })
            print(f"처리: {start + 1}~{start + chunk_size}/{total} 레스토랑")

    meta = {
        "base_url": base_url,
        "total_restaurants": total,
        "requested_at": datetime.now().isoformat(),
        "batch_size": batch_size,
        "apis": apis,
    }
    out = {"meta": meta, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {output_path} (레스토랑 {total}개)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="모든 레스토랑에 대해 Summary/Sentiment/Comparison API 호출 후 JSON 저장 (청크당 3 API 병렬)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, default=PROJECT_ROOT / "tasteam_app_data.json", help="입력 JSON (reviews, restaurants)")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "all_restaurants_results.json", help="출력 JSON 경로")
    parser.add_argument("--base-url", "-b", type=str, default=DEFAULT_BASE_URL, help="API 서버 URL")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="배치당 레스토랑 수")
    parser.add_argument("--limit", "-n", type=int, default=None, help="처리할 최대 레스토랑 수 (테스트용)")
    parser.add_argument("--summary-limit", type=int, default=DEFAULT_SUMMARY_LIMIT, help="요약 API 카테고리당 리뷰 수")
    parser.add_argument("--sentiment-reviews-limit", type=int, default=DEFAULT_SENTIMENT_REVIEWS_LIMIT, help="감성 분석당 최대 리뷰 수")
    parser.add_argument("--apis", type=str, nargs="+", default=["summary", "sentiment", "comparison"], help="호출할 API: summary sentiment comparison")
    parser.add_argument("--timeout", type=int, default=600, help="API 요청 타임아웃(초)")
    parser.add_argument("--no-upload", action="store_true", help="벡터 업로드 생략 (이미 업로드된 경우)")
    parser.add_argument("--upload-timeout", type=int, default=DEFAULT_UPLOAD_TIMEOUT, help="벡터 업로드 요청 타임아웃(초)")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"오류: 입력 파일이 없습니다: {args.input}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_async(
        input_path=args.input,
        output_path=args.output,
        base_url=args.base_url,
        batch_size=args.batch_size,
        limit_restaurants=args.limit,
        summary_limit=args.summary_limit,
        sentiment_reviews_limit=args.sentiment_reviews_limit,
        apis=args.apis,
        timeout=args.timeout,
        do_upload=not args.no_upload,
        upload_timeout=args.upload_timeout,
    ))


if __name__ == "__main__":
    main()
