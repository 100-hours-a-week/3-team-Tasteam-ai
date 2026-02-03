#!/usr/bin/env python3
"""
tasteam_seed_reviews.csv와 tasteam_seed_reviews_2.csv를 하나로 합치는 스크립트.

입력: 동일한 컬럼 구조의 CSV (도입_URL, 가게이름, 카테고리, ...)
출력: 합쳐진 CSV

사용 예:
  python scripts/merge_tasteam_csv.py -o tasteam_seed_reviews_merged.csv
  python scripts/merge_tasteam_csv.py -o merged.csv --dedupe
  python scripts/merge_tasteam_csv.py -o merged.csv -1 custom1.csv -2 custom2.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def read_csv_rows(path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """CSV를 읽어 행 리스트 반환."""
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for row in reader:
            rows.append(row)
    return rows


def make_dedup_key(row: dict[str, str]) -> str:
    """동일 리뷰 판별용 키: URL + 작성자 + 내용 + 방문시간."""
    url = (row.get("도입_URL") or "").strip()
    author = (row.get("리뷰작성자") or "").strip()
    content = (row.get("리뷰내용") or "").strip()
    visit = (row.get("방문시간") or "").strip()
    return f"{url}|{author}|{content}|{visit}"


def merge(
    path1: Path,
    path2: Path,
    output: Path,
    dedupe: bool = False,
    encoding: str = "utf-8",
) -> None:
    """두 CSV를 합쳐 출력 파일에 저장."""
    rows1 = read_csv_rows(path1, encoding)
    rows2 = read_csv_rows(path2, encoding)

    merged = rows1 + rows2
    total_before = len(merged)

    if dedupe:
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for row in merged:
            key = make_dedup_key(row)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        merged = unique

    if not merged:
        print("경고: 합칠 행이 없습니다.", file=sys.stderr)
        return

    fieldnames = list(merged[0].keys())
    with open(output, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    dup_removed = total_before - len(merged) if dedupe else 0
    print(f"병합 완료: {path1.name}({len(rows1)}행) + {path2.name}({len(rows2)}행) -> {output.name}({len(merged)}행)")
    if dedupe and dup_removed > 0:
        print(f"  중복 제거: {dup_removed}건")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default1 = project_root / "tasteam_seed_reviews.csv"
    default2 = project_root / "tasteam_seed_reviews_2.csv"
    default_out = project_root / "tasteam_seed_reviews_merged.csv"

    parser = argparse.ArgumentParser(
        description="tasteam_seed_reviews.csv와 tasteam_seed_reviews_2.csv를 합칩니다.",
    )
    parser.add_argument(
        "-1",
        "--input1",
        type=Path,
        default=default1,
        help=f"첫 번째 CSV (기본: {default1.name})",
    )
    parser.add_argument(
        "-2",
        "--input2",
        type=Path,
        default=default2,
        help=f"두 번째 CSV (기본: {default2.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help=f"출력 CSV (기본: {default_out.name})",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="도입_URL+리뷰작성자+리뷰내용+방문시간 기준 중복 제거",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="입출력 인코딩 (기본: utf-8)",
    )

    args = parser.parse_args()

    if not args.input1.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {args.input1}", file=sys.stderr)
        sys.exit(1)
    if not args.input2.exists():
        print(f"오류: 파일을 찾을 수 없습니다: {args.input2}", file=sys.stderr)
        sys.exit(1)

    merge(
        path1=args.input1,
        path2=args.input2,
        output=args.output,
        dedupe=args.dedupe,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
