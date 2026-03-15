"""
Prefect 기반 DeepFM 배치 추론 워크플로우 (S3 폴링 → 추론 → S3 업로드).

한 번에: S3 Raw 다운로드 → 후보 CSV 변환 → DeepFM 추론 → S3 업로드.
(목록 조회만/다운로드만은 CLI 스크립트: scripts/s3_raw_poll_download.py --list-only 등)

실행 (deepfm_pipeline 디렉터리 또는 repo 루트에서):
  python ml/deepfm_pipeline/inference_flow.py --env dev --out-dir ./data/raw_download --run-dir ./output/run_xxx
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_deepfm_root = Path(__file__).resolve().parent
if str(_deepfm_root) not in sys.path:
    sys.path.insert(0, str(_deepfm_root))

from prefect import flow, task


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------


@task(name="deepfm-s3-download-raw", log_prints=True)
def task_s3_download_raw(
    bucket: str,
    out_dir: str | Path,
    data_types: list[str],
    dt_filter: str | None = None,
    profile_name: str | None = None,
) -> dict[str, list[Path]]:
    """S3 Raw 다운로드 (_SUCCESS 있는 파티션만)."""
    from scripts.s3_raw_poll_download import poll_and_download

    result = poll_and_download(
        bucket=bucket,
        out_dir=Path(out_dir),
        data_types=data_types,
        dt_filter=dt_filter,
        profile_name=profile_name,
    )
    total = sum(len(paths) for paths in result.values())
    print(f"Downloaded {total} file(s) -> {out_dir}")
    return result


@task(name="deepfm-raw-to-candidates-csv", log_prints=True)
def task_raw_to_candidates_csv(
    raw_dir: str | Path,
    output_csv_path: str | Path,
    data_types: list[str] | None = None,
) -> int:
    """Raw(events/restaurants/menus) → 파이프라인용 후보 CSV 하나로 변환."""
    from utils.raw_to_pipeline import raw_dir_to_pipeline_csv

    types = tuple(data_types) if data_types else ("events", "restaurants", "menus")
    n = raw_dir_to_pipeline_csv(raw_dir, output_csv_path, data_types=types)
    print(f"Wrote {n} rows to {output_csv_path}")
    return n


@task(name="deepfm-inference-and-upload", log_prints=True)
def task_inference_and_upload(
    run_dir: str | Path,
    candidates_path: str | Path,
    s3_env: str,
    dt: str | None = None,
    output_format: str = "csv",
    profile_name: str | None = None,
    ttl_hours: float = 24.0,
    batch_size: int = 256,
) -> str:
    """추론 실행 후 추천 결과를 S3에 업로드 (part-00001.csv/json.gz + _SUCCESS)."""
    from scripts.score_batch_to_s3 import _upload_to_s3
    from utils.score_batch import run as score_batch_run

    run_dir = Path(run_dir)
    candidates_path = Path(candidates_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidates_path not found: {candidates_path}")

    pv_path = run_dir / "pipeline_version.txt"
    pv = pv_path.read_text(encoding="utf-8").strip() if pv_path.exists() else "deepfm-1.0.unknown"
    dt = dt or datetime.now(timezone.utc).date().isoformat()
    bucket = f"tasteam-{s3_env}-analytics"
    key_prefix = f"recommendations/pipeline_version={pv}/dt={dt}"
    ext = "json.gz" if output_format.lower() == "json.gz" else "csv"
    out_filename = f"part-00001.{ext}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="deepfm-inference-"))
    out_path = tmp_dir / out_filename

    score_batch_run(
        run_dir=run_dir,
        candidates_path=candidates_path,
        output_path=out_path,
        raw_candidates_path=candidates_path,
        ttl_hours=ttl_hours,
        batch_size=batch_size,
        output_format=output_format,
    )

    out_url = f"s3://{bucket}/{key_prefix}/{out_filename}"
    success_url = f"s3://{bucket}/{key_prefix}/_SUCCESS"
    _upload_to_s3(out_path, out_url, profile_name=profile_name)
    (tmp_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _upload_to_s3(tmp_dir / "_SUCCESS", success_url, profile_name=profile_name)
    print(f"Uploaded {out_url}")
    return out_url


# -----------------------------------------------------------------------------
# Flow
# -----------------------------------------------------------------------------


@flow(name="DeepFM Batch Inference Pipeline", log_prints=True)
def deepfm_inference_flow(
    env: str | None = None,
    bucket: str | None = None,
    out_dir: str = "",
    run_dir: str = "",
    data_types: list[str] | None = None,
    dt_filter: str | None = None,
    dt_upload: str | None = None,
    profile_name: str | None = None,
    output_format: str = "csv",
    ttl_hours: float = 24.0,
    batch_size: int = 256,
) -> dict:
    """
    DeepFM 배치 추론 워크플로우: S3 Raw 다운로드 → 후보 CSV 변환 → 추론 → S3 업로드.

    - env: dev | stg | prod → tasteam-{env}-analytics (bucket 미지정 시)
    - bucket: 버킷 직접 지정 (env 대신)
    - out_dir: Raw 다운로드/후보 CSV 경로 (필수)
    - run_dir: 모델 run 디렉터리 (필수)
    - dt_filter: Raw 파티션 필터 (YYYY-MM-DD). None이면 _SUCCESS 있는 전부
    - dt_upload: 업로드 파티션 dt. None이면 UTC 오늘
    - profile_name: AWS CLI 프로필 (미지정 시 AWS_PROFILE)
    """
    _data_types = data_types or ["events", "restaurants", "menus"]
    _bucket = bucket or (f"tasteam-{env}-analytics" if env else None)
    if not _bucket:
        raise ValueError("env or bucket required")
    if not out_dir:
        raise ValueError("out_dir required")
    if not run_dir:
        raise ValueError("run_dir required")

    profile = profile_name or os.environ.get("AWS_PROFILE") or "jayvi"

    task_s3_download_raw(
        bucket=_bucket,
        out_dir=out_dir,
        data_types=_data_types,
        dt_filter=dt_filter,
        profile_name=profile,
    )

    if env:
        upload_env = env
    elif _bucket.startswith("tasteam-") and _bucket.endswith("-analytics"):
        upload_env = _bucket.replace("tasteam-", "").replace("-analytics", "") or "dev"
    else:
        upload_env = "dev"

    candidates_csv = Path(out_dir) / "raw_candidates.csv"
    task_raw_to_candidates_csv(out_dir, candidates_csv, data_types=_data_types)

    if not candidates_csv.exists() or candidates_csv.stat().st_size == 0:
        print("No candidates generated; skipping inference and upload.")
        return {"out_dir": out_dir, "run_dir": run_dir, "skipped": True}

    out_url = task_inference_and_upload(
        run_dir=run_dir,
        candidates_path=candidates_csv,
        s3_env=upload_env,
        dt=dt_upload,
        output_format=output_format,
        profile_name=profile,
        ttl_hours=ttl_hours,
        batch_size=batch_size,
    )
    return {"out_dir": out_dir, "run_dir": run_dir, "recommendation_url": out_url}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="DeepFM 배치 추론 Prefect flow (S3 → 추론 → S3 업로드)")
    p.add_argument("--env", choices=["dev", "stg", "prod"], default=None, help="환경 → tasteam-{env}-analytics")
    p.add_argument("--bucket", type=str, default=None, help="버킷 직접 지정 (--env 대신)")
    p.add_argument("--out-dir", type=str, required=True, help="Raw 다운로드/후보 CSV 경로")
    p.add_argument("--run-dir", type=str, required=True, help="모델 run 디렉터리")
    p.add_argument(
        "--data-types",
        type=str,
        default="events,restaurants,menus",
        help="쉼표 구분 Raw 데이터 타입",
    )
    p.add_argument("--dt", type=str, default=None, help="Raw 파티션 필터 및 업로드 dt (YYYY-MM-DD)")
    p.add_argument("--profile", type=str, default=None, help="AWS CLI 프로필 (미지정 시 AWS_PROFILE)")
    p.add_argument("--output-format", choices=["csv", "json.gz"], default="csv", help="추천 결과 파일 형식")
    p.add_argument("--ttl-hours", type=float, default=24.0, help="expires_at TTL")
    p.add_argument("--batch-size", type=int, default=256, help="추론 배치 크기")
    args = p.parse_args()

    data_types = [s.strip() for s in args.data_types.split(",") if s.strip()]
    if not data_types:
        data_types = ["events", "restaurants", "menus"]

    result = deepfm_inference_flow(
        env=args.env,
        bucket=args.bucket,
        out_dir=args.out_dir,
        run_dir=args.run_dir,
        data_types=data_types,
        dt_filter=args.dt,
        dt_upload=args.dt,
        profile_name=args.profile,
        output_format=args.output_format,
        ttl_hours=args.ttl_hours,
        batch_size=args.batch_size,
    )
    print("Result:", result)
