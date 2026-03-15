"""
S3 Raw 데이터 폴링·다운로드. service_constract.md §4 준수.

기준: 파티션(dt=YYYY-MM-DD)에 _SUCCESS 마커가 있을 때만 해당 파티션을 유효로 보고
      데이터 파일(part-*.csv 등)을 다운로드한다.

경로 계약:
  s3://tasteam-{env}-analytics/raw/events/dt=YYYY-MM-DD/part-00001.csv, _SUCCESS
  s3://tasteam-{env}-analytics/raw/restaurants/dt=YYYY-MM-DD/...
  s3://tasteam-{env}-analytics/raw/menus/dt=YYYY-MM-DD/...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SUCCESS_MARKER = "_SUCCESS"
RAW_DATA_TYPES = ("events", "restaurants", "menus")


def _s3_list_partition_keys(client, bucket: str, prefix: str) -> list[str]:
    """prefix 하위의 객체 키 목록 (최대 1000)."""
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            keys.append(obj["Key"])
    return keys


def _partitions_with_success(client, bucket: str, data_type: str) -> list[str]:
    """
    raw/{data_type}/ 하위에서 _SUCCESS가 있는 dt= 파티션만 반환.
    반환: ['dt=2025-03-01', 'dt=2025-03-02', ...] (정렬됨)
    """
    prefix = f"raw/{data_type}/"
    keys = _s3_list_partition_keys(client, bucket, prefix)
    # dt=YYYY-MM-DD/ 형태의 파티션만 추출
    partition_dirs: set[str] = set()
    success_partitions: set[str] = set()
    for k in keys:
        parts = k[len(prefix) :].split("/")
        if not parts or parts[0] == "":
            continue
        part_name = parts[0]  # dt=YYYY-MM-DD
        if part_name.startswith("dt="):
            partition_dirs.add(part_name)
            if len(parts) >= 2 and parts[1] == SUCCESS_MARKER:
                success_partitions.add(part_name)
    # _SUCCESS가 있는 파티션만, 정렬
    return sorted(success_partitions & partition_dirs)


def _download_partition(
    client,
    bucket: str,
    s3_prefix: str,
    local_dir: Path,
    exclude_marker: bool = True,
) -> list[Path]:
    """
    s3_prefix (예: raw/events/dt=2025-03-01/) 하위의 객체를 local_dir에 다운로드.
    exclude_marker=True면 _SUCCESS는 다운로드하지 않음.
    반환: 다운로드된 로컬 파일 경로 목록.
    """
    keys = _s3_list_partition_keys(client, bucket, s3_prefix)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key in keys:
        name = key.split("/")[-1]
        if exclude_marker and name == SUCCESS_MARKER:
            continue
        local_path = local_dir / name
        client.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)
    return downloaded


def _s3_client(profile_name: str | None = None):
    """AWS_PROFILE 또는 profile_name이 있으면 해당 프로필로 S3 클라이언트 생성."""
    import boto3

    if profile_name:
        return boto3.Session(profile_name=profile_name).client("s3")
    return boto3.client("s3")


def list_partitions_only(
    bucket: str,
    data_types: list[str] | None = None,
    dt_filter: str | None = None,
    profile_name: str | None = None,
) -> dict[str, list[str]]:
    """
    _SUCCESS가 있는 파티션만 조회 (다운로드 없음).
    반환: { "events": ["dt=2025-03-01", ...], "restaurants": [...], "menus": [...] }
    """
    types = data_types or list(RAW_DATA_TYPES)
    for t in types:
        if t not in RAW_DATA_TYPES:
            raise ValueError(f"data_type must be one of {RAW_DATA_TYPES}, got {t}")
    client = _s3_client(profile_name)
    result: dict[str, list[str]] = {}
    for data_type in types:
        all_ready = _partitions_with_success(client, bucket, data_type)
        if dt_filter:
            want = f"dt={dt_filter}"
            result[data_type] = [want] if want in all_ready else []
        else:
            result[data_type] = all_ready
    return result


def poll_and_download(
    bucket: str,
    out_dir: str | Path,
    data_types: list[str] | None = None,
    dt_filter: str | None = None,
    profile_name: str | None = None,
) -> dict[str, list[Path]]:
    """
    _SUCCESS가 있는 파티션만 찾아서 해당 파티션의 데이터 파일을 out_dir에 다운로드.

    - bucket: S3 버킷명 (예: tasteam-dev-analytics)
    - out_dir: 로컬 기준 디렉터리. 하위에 raw/events/dt=.../ 등으로 저장됨
    - data_types: ['events', 'restaurants', 'menus'] 중 다운로드할 타입. None이면 전부
    - dt_filter: 지정 시 해당 dt= 하나만 (예: 2025-03-01). None이면 _SUCCESS 있는 모든 dt
    - profile_name: AWS CLI 프로필 이름 (미지정 시 AWS_PROFILE 환경변수 또는 기본 자격증명)

    반환: { "events": [Path, ...], "restaurants": [...], "menus": [...] } (다운로드된 파일 목록)
    """
    out_dir = Path(out_dir)
    types = data_types or list(RAW_DATA_TYPES)
    for t in types:
        if t not in RAW_DATA_TYPES:
            raise ValueError(f"data_type must be one of {RAW_DATA_TYPES}, got {t}")

    client = _s3_client(profile_name)
    result: dict[str, list[Path]] = {t: [] for t in types}

    for data_type in types:
        prefix_base = f"raw/{data_type}/"
        all_ready = _partitions_with_success(client, bucket, data_type)
        if dt_filter:
            want = f"dt={dt_filter}"
            partitions = [want] if want in all_ready else []
        else:
            partitions = all_ready

        for part in partitions:
            s3_prefix = prefix_base + part + "/"
            local_part_dir = out_dir / "raw" / data_type / part
            files = _download_partition(client, bucket, s3_prefix, local_part_dir)
            result[data_type].extend(files)

    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="S3 Raw 폴링·다운로드 (_SUCCESS 있는 파티션만). service_constract §4."
    )
    p.add_argument("--env", choices=["dev", "stg", "prod"], help="환경 → tasteam-{env}-analytics 버킷")
    p.add_argument("--bucket", type=str, default=None, help="버킷 직접 지정 (--env 대신)")
    p.add_argument("--out-dir", type=str, default=None, help="다운로드 기준 디렉터리 (--list-only 시 불필요)")
    p.add_argument(
        "--data-types",
        type=str,
        default=",".join(RAW_DATA_TYPES),
        help=f"쉼표 구분. 기본: {','.join(RAW_DATA_TYPES)}",
    )
    p.add_argument("--dt", type=str, default=None, help="특정 dt만 (YYYY-MM-DD). 없으면 _SUCCESS 있는 전부")
    p.add_argument("--profile", type=str, default=None, help="AWS CLI 프로필 이름 (미지정 시 AWS_PROFILE 사용)")
    p.add_argument("--list-only", action="store_true", help="다운로드 없이 _SUCCESS 있는 파티션만 목록 출력")
    args = p.parse_args()

    profile = args.profile or os.environ.get("AWS_PROFILE") or "jayvi"
    bucket = args.bucket
    if not bucket and args.env:
        bucket = f"tasteam-{args.env}-analytics"
    if not bucket:
        p.error("--bucket 또는 --env 필요")

    data_types = [s.strip() for s in args.data_types.split(",") if s.strip()]
    if not data_types:
        data_types = list(RAW_DATA_TYPES)

    if args.list_only:
        result = list_partitions_only(
            bucket=bucket,
            data_types=data_types,
            dt_filter=args.dt,
            profile_name=profile,
        )
        for t, partitions in result.items():
            print(f"{t}: {partitions}")
        print(f"Total partitions: {sum(len(v) for v in result.values())}")
        return

    if not args.out_dir:
        p.error("--out-dir required unless --list-only")

    result = poll_and_download(
        bucket=bucket,
        out_dir=args.out_dir,
        data_types=data_types,
        dt_filter=args.dt,
        profile_name=profile,
    )
    total = 0
    for t, paths in result.items():
        print(f"{t}: {len(paths)} file(s)")
        total += len(paths)
    print(f"Total: {total} file(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
