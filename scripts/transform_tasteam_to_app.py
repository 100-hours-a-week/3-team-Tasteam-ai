#!/usr/bin/env python3
"""
tasteam_seed_reviews.csv를 애플리케이션 벡터 업로드 형식(VectorUploadRequest)으로 변환하는 스크립트.

입력 CSV 컬럼: 도입_URL, 가게이름, 카테고리, 전체평점, 방문자리뷰, 블로그리뷰,
  리뷰작성자, 리뷰작성자링크, 리뷰작성수, 팔로워, 첨부이미지, 리뷰내용,
  이런_점이_좋아요, 방문시간, 방분횟수, 방문인증

출력 JSON: VectorUploadRequest 형식
  - reviews: [{ id, restaurant_id, content, created_at }, ...]
  - restaurants: [{ id, name }, ...]  (선택, --with-restaurants 시 포함)

사용 예:
  python scripts/transform_tasteam_to_app.py --input tasteam_seed_reviews.csv --output tasteam_app_data.json
  python scripts/transform_tasteam_to_app.py --input tasteam_seed_reviews.csv --output tasteam_app_data.json --sample 500 --with-restaurants
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# CSV 컬럼명 (헤더와 일치)
COL_URL = "도입_URL"
COL_RESTAURANT_NAME = "가게이름"
COL_CONTENT = "리뷰내용"
COL_VISIT_TIME = "방문시간"

# 방문시간 예: "25.9.19.금", "25.12.8.월" -> YY.M.D.요일
VISIT_TIME_PATTERN = re.compile(r"^(\d{2})\.(\d{1,2})\.(\d{1,2})")


def extract_restaurant_id_from_url(url: str) -> Optional[int]:
    """URL에서 네이버 플레이스 ID(restaurant_id) 추출. 예: .../place/1303984471?... -> 1303984471"""
    if not url or not isinstance(url, str):
        return None
    m = re.search(r"/place/(\d+)", url.strip())
    return int(m.group(1)) if m else None


def parse_visit_time_to_iso(visit_time: str, default_year: int = 2025) -> str:
    """
    방문시간 문자열을 ISO 8601 형식으로 변환.
    예: "25.9.19.금" -> "2025-09-19T00:00:00"
    파싱 실패 시 default_year-01-01 반환.
    """
    if not visit_time or not isinstance(visit_time, str):
        return f"{default_year}-01-01T00:00:00"
    s = visit_time.strip()
    m = VISIT_TIME_PATTERN.match(s)
    if not m:
        return f"{default_year}-01-01T00:00:00"
    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = 2000 + yy if yy < 100 else yy
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return f"{default_year}-01-01T00:00:00"
    try:
        dt = datetime(year, mm, dd)
        return dt.strftime("%Y-%m-%dT00:00:00")
    except ValueError:
        return f"{default_year}-01-01T00:00:00"


def read_csv_rows(
    path: Path,
    encoding: str = "utf-8",
    sample: Optional[int] = None,
) -> List[Dict[str, str]]:
    """CSV를 읽어 행 리스트 반환. sample이 지정되면 그 수만큼만 읽음."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for i, row in enumerate(reader):
            if sample is not None and i >= sample:
                break
            rows.append(row)
    return rows


def transform_row_to_review(
    row: Dict[str, str],
    review_id: int,
) -> Optional[Tuple[Dict[str, Any], int, str]]:
    """
    CSV 한 행을 (review_dict, restaurant_id, restaurant_name) 또는 None으로 변환.
    content가 비어 있거나 restaurant_id를 추출할 수 없으면 None.
    """
    url = row.get(COL_URL, "")
    restaurant_id = extract_restaurant_id_from_url(url)
    if restaurant_id is None:
        return None

    content = (row.get(COL_CONTENT) or "").strip()
    if not content:
        return None

    name = (row.get(COL_RESTAURANT_NAME) or "").strip() or None
    visit_time = row.get(COL_VISIT_TIME, "")
    created_at = parse_visit_time_to_iso(visit_time)

    review = {
        "id": review_id,
        "restaurant_id": restaurant_id,
        "content": content,
        "created_at": created_at,
    }
    return (review, restaurant_id, name or "")


def build_restaurants_unique(
    review_restaurant_pairs: List[Tuple[Dict[str, Any], int, str]]
) -> List[Dict[str, Any]]:
    """(review, restaurant_id, restaurant_name) 리스트에서 레스토랑 ID별 이름 하나씩 모아 리스트 반환."""
    by_id: Dict[int, str] = {}
    for _, rid, name in review_restaurant_pairs:
        if name and (rid not in by_id or not by_id[rid]):
            by_id[rid] = name
    return [{"id": rid, "name": name} for rid, name in sorted(by_id.items())]


def run(
    input_path: Path,
    output_path: Path,
    encoding: str = "utf-8",
    sample: Optional[int] = None,
    with_restaurants: bool = True,
) -> None:
    """CSV를 읽어 앱용 JSON으로 변환 후 저장."""
    rows = read_csv_rows(input_path, encoding=encoding, sample=sample)
    reviews: List[Dict[str, Any]] = []
    pairs: List[Tuple[Dict[str, Any], int, str]] = []
    next_id = 1
    skipped = 0

    for row in rows:
        result = transform_row_to_review(row, next_id)
        if result is None:
            skipped += 1
            continue
        rev, _rid, _name = result
        reviews.append(rev)
        pairs.append(result)
        next_id += 1

    payload: Dict[str, Any] = {"reviews": reviews}
    if with_restaurants:
        payload["restaurants"] = build_restaurants_unique(pairs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"변환 완료: 입력 {len(rows)}행, 유효 리뷰 {len(reviews)}개, 스킵 {skipped}개 -> {output_path}"
    )
    if with_restaurants and payload.get("restaurants"):
        print(f"레스토랑 수: {len(payload['restaurants'])}개")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tasteam_seed_reviews.csv를 벡터 업로드용 JSON으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("tasteam_seed_reviews.csv"),
        help="입력 CSV 경로 (기본: tasteam_seed_reviews.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tasteam_app_data.json"),
        help="출력 JSON 경로 (기본: tasteam_app_data.json)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV 인코딩 (기본: utf-8)",
    )
    parser.add_argument(
        "--sample",
        "-n",
        type=int,
        default=None,
        help="처리할 최대 행 수 (미지정 시 전체)",
    )
    parser.add_argument(
        "--no-restaurants",
        action="store_true",
        help="restaurants 배열 없이 reviews만 출력",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"오류: 입력 파일이 없습니다: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(
        input_path=args.input,
        output_path=args.output,
        encoding=args.encoding,
        sample=args.sample,
        with_restaurants=not args.no_restaurants,
    )


if __name__ == "__main__":
    main()
