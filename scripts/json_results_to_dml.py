#!/usr/bin/env python3
"""
tasteam_app_all_restaurants_ai_api_results.json 을 SQL DML(INSERT)로 변환하여 단일 .sql 파일로 저장.

테이블:
  - run_meta: base_url, total_restaurants, requested_at, batch_size, apis_json
  - restaurant_ai_results: restaurant_id, restaurant_name, summary_json, sentiment_json, comparison_json, errors_json

사용 예:
  python scripts/json_results_to_dml.py --input tasteam_app_all_restaurants_ai_api_results.json --output tasteam_app_all_restaurants_ai_api_results.sql
"""
import argparse
import json
import sys
from pathlib import Path


def sql_escape(s: str) -> str:
    """SQL 문자열 리터럴 내 따옴표·백슬래시·개행 이스케이프 (한 줄 유지)."""
    if s is None:
        return ""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("'", "''")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON API 결과를 SQL DML로 변환")
    parser.add_argument("--input", "-i", type=Path, required=True, help="입력 JSON 경로")
    parser.add_argument("--output", "-o", type=Path, required=True, help="출력 .sql 경로")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"오류: 입력 파일이 없습니다: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta") or {}
    results = data.get("results") or []

    lines = [
        "-- tasteam_app_all_restaurants_ai_api_results.json -> DML",
        "-- 테이블 생성 (선택, 이미 있으면 생략)",
        "",
        "CREATE TABLE IF NOT EXISTS run_meta (",
        "  id SERIAL PRIMARY KEY,",
        "  base_url TEXT,",
        "  total_restaurants INTEGER,",
        "  requested_at TEXT,",
        "  batch_size INTEGER,",
        "  apis_json TEXT",
        ");",
        "",
        "CREATE TABLE IF NOT EXISTS restaurant_ai_results (",
        "  id SERIAL PRIMARY KEY,",
        "  restaurant_id INTEGER NOT NULL,",
        "  restaurant_name TEXT,",
        "  summary_json TEXT,",
        "  sentiment_json TEXT,",
        "  comparison_json TEXT,",
        "  errors_json TEXT",
        ");",
        "",
        "-- run_meta 1건",
        "INSERT INTO run_meta (base_url, total_restaurants, requested_at, batch_size, apis_json) VALUES (",
        f"  '{sql_escape(meta.get('base_url', ''))}',",
        f"  {int(meta.get('total_restaurants') or 0)},",
        f"  '{sql_escape(meta.get('requested_at', ''))}',",
        f"  {int(meta.get('batch_size') or 0)},",
        f"  '{sql_escape(json.dumps(meta.get('apis') or [], ensure_ascii=False))}'",
        ");",
        "",
        "-- restaurant_ai_results",
    ]

    for r in results:
        rid = int(r.get("restaurant_id") or 0)
        name = sql_escape(r.get("restaurant_name") or "")
        summary_json = sql_escape(json.dumps(r.get("summary"), ensure_ascii=False) if r.get("summary") is not None else "")
        sentiment_json = sql_escape(json.dumps(r.get("sentiment"), ensure_ascii=False) if r.get("sentiment") is not None else "")
        comparison_json = sql_escape(json.dumps(r.get("comparison"), ensure_ascii=False) if r.get("comparison") is not None else "")
        errors_json = sql_escape(json.dumps(r.get("errors"), ensure_ascii=False) if r.get("errors") is not None else "")
        lines.append(
            "INSERT INTO restaurant_ai_results (restaurant_id, restaurant_name, summary_json, sentiment_json, comparison_json, errors_json) VALUES ("
        )
        lines.append(f"  {rid}, '{name}', '{summary_json}', '{sentiment_json}', '{comparison_json}', '{errors_json}'")
        lines.append(");")

    sql_content = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(sql_content)

    print(f"DML 저장 완료: {args.output} (run_meta 1건, restaurant_ai_results {len(results)}건)")


if __name__ == "__main__":
    main()
