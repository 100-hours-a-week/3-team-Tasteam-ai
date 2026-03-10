"""
DeepFM 추천 결과를 서비스 계약(S3 파티션 + _SUCCESS)에 맞게 생성/업로드하는 배치 스크립트.

계약: docs/service_extraction/service_constract.md

저장 경로:
  s3://tasteam-{env}-analytics/recommendations/pipeline_version=VERSION/dt=YYYY-MM-DD/part-00001.csv
  s3://tasteam-{env}-analytics/recommendations/pipeline_version=VERSION/dt=YYYY-MM-DD/_SUCCESS
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _default_output_dir() -> Path:
    # 이 파일: ml/deepfm_pipeline/scripts/score_batch_to_s3.py
    return Path(__file__).resolve().parents[1] / "output"


def _find_run_dir_by_version(pipeline_version: str) -> Path | None:
    out = _default_output_dir()
    if not out.exists():
        return None
    for d in out.iterdir():
        if not d.is_dir():
            continue
        pv_file = d / "pipeline_version.txt"
        if pv_file.exists() and pv_file.read_text(encoding="utf-8").strip() == pipeline_version:
            return d
    return None


def _parse_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"Not an s3 url: {url}")
    rest = url[len("s3://") :]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _upload_to_s3(local_path: Path, s3_url: str) -> None:
    import boto3

    if not local_path.exists():
        raise FileNotFoundError(local_path)
    bucket, key = _parse_s3_url(s3_url)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def main() -> None:
    p = argparse.ArgumentParser(description="DeepFM score_batch → 계약된 S3 경로 업로드 + _SUCCESS")
    p.add_argument("--pipeline-version", required=True, help="pipeline_version (output/ 하위에서 run_dir 자동 탐색)")
    p.add_argument("--run-dir", default=None, help="선택. run 디렉터리 경로. 없으면 pipeline_version으로 탐색")
    p.add_argument("--candidates-path", required=True, help="후보 feature CSV 경로")
    p.add_argument("--meta-path", default=None, help="선택. user_id/anonymous_id/restaurant_id/context_snapshot 메타 CSV 경로")
    p.add_argument("--env", required=True, choices=["dev", "stg", "prod"], help="tasteam-{env}-analytics 버킷 선택")
    p.add_argument("--dt", default=None, help="선택. YYYY-MM-DD (기본: UTC 오늘)")
    p.add_argument("--ttl-hours", type=float, default=24.0, help="expires_at TTL(시간)")
    p.add_argument("--batch-size", type=int, default=256, help="추론 배치 크기")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _find_run_dir_by_version(args.pipeline_version)
    if not run_dir or not run_dir.exists():
        raise SystemExit(f"Run not found for pipeline_version={args.pipeline_version}. Use --run-dir or ensure output/ contains the run.")

    pv = (run_dir / "pipeline_version.txt").read_text(encoding="utf-8").strip() if (run_dir / "pipeline_version.txt").exists() else args.pipeline_version
    dt = args.dt or datetime.now(timezone.utc).date().isoformat()
    bucket = f"tasteam-{args.env}-analytics"
    key_prefix = f"recommendations/pipeline_version={pv}/dt={dt}"
    out_url = f"s3://{bucket}/{key_prefix}/part-00001.csv"
    success_url = f"s3://{bucket}/{key_prefix}/_SUCCESS"

    tmp_dir = Path(tempfile.mkdtemp(prefix="deepfm-score-batch-"))
    out_path = tmp_dir / "part-00001.csv"

    # 로컬 CSV 생성
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.score_batch import run as score_batch_run

    score_batch_run(
        run_dir=run_dir,
        candidates_path=Path(args.candidates_path),
        output_path=out_path,
        meta_path=Path(args.meta_path) if args.meta_path else None,
        ttl_hours=args.ttl_hours,
        batch_size=args.batch_size,
    )

    # S3 업로드 + 마커
    _upload_to_s3(out_path, out_url)
    marker = tmp_dir / "_SUCCESS"
    marker.write_text("", encoding="utf-8")
    _upload_to_s3(marker, success_url)

    print(out_url)


if __name__ == "__main__":
    main()

