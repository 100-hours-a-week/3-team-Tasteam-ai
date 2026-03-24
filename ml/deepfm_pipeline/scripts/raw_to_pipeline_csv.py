"""
S3 Raw 다운로드 디렉터리 → 파이프라인용 단일 CSV 변환 CLI.

사용: python scripts/raw_to_pipeline_csv.py --raw-dir /data/raw_download --out /data/raw/training_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# pipeline 루트
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.raw_to_pipeline import raw_dir_to_pipeline_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Raw(events/restaurants/menus) → 파이프라인용 CSV")
    p.add_argument("--raw-dir", type=str, required=True, help="s3_raw_poll_download.py 출력 디렉터리 (raw/ 하위 포함)")
    p.add_argument("--out", type=str, required=True, help="출력 CSV 경로 (예: training_dataset.csv)")
    p.add_argument("--base-prefix", type=str, default="raw", help="raw 데이터 루트 prefix (기본: raw)")
    args = p.parse_args()
    n = raw_dir_to_pipeline_csv(args.raw_dir, args.out, base_prefix=args.base_prefix)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
