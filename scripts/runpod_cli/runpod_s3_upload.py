"""
RunPod Network Volume S3 호환 API로 로컬 파일/디렉터리 업로드.

환경변수:
  RUNPOD_S3_ACCESS_KEY   - RunPod S3 API Access Key (user_...)
  RUNPOD_S3_SECRET_ACCESS_KEY - RunPod S3 API Secret (rps_...)
  RUNPOD_NETWORK_VOLUME_ID    - 업로드 대상 볼륨 ID (기본: 4rlm64f9lv, train용)
  RUNPOD_S3_REGION            - 데이터센터 ID (기본: eu-ro-1)
  RUNPOD_S3_ENDPOINT_URL      - S3 엔드포인트 (기본: https://s3api-eu-ro-1.runpod.io)

참고: docs/runpod_cli/runpod_net_vol_s3_api.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc, assignment]

# EU-RO-1 기본값 (vllm_net_vol.md, distill_train_net_vol.md)
DEFAULT_REGION = "eu-ro-1"
DEFAULT_ENDPOINT_URL = "https://s3api-eu-ro-1.runpod.io"
DEFAULT_VOLUME_ID = "4rlm64f9lv"  # train용 볼륨 (labeled 데이터 업로드 대상)


def _get_credentials() -> tuple[str, str]:
    access = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    secret = os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise ValueError(
            "RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY must be set for RunPod S3 upload."
        )
    return access, secret


def get_runpod_s3_client(
    region: str | None = None,
    endpoint_url: str | None = None,
) -> Any:
    """RunPod S3 호환 API용 boto3 S3 클라이언트 생성."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for RunPod S3 upload: pip install boto3")
    access, secret = _get_credentials()
    region = region or os.environ.get("RUNPOD_S3_REGION", DEFAULT_REGION)
    endpoint_url = endpoint_url or os.environ.get("RUNPOD_S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL)
    return boto3.client(
        "s3",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        endpoint_url=endpoint_url.rstrip("/"),
    )


def upload_file(
    s3_client: Any,
    bucket: str,
    local_path: str | Path,
    object_key: str,
) -> None:
    """단일 파일을 네트워크 볼륨에 업로드."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    s3_client.upload_file(str(local_path), bucket, object_key)


def upload_directory(
    s3_client: Any,
    bucket: str,
    local_dir: str | Path,
    remote_prefix: str = "",
) -> int:
    """
    로컬 디렉터리 전체를 네트워크 볼륨에 업로드.
    remote_prefix: 볼륨 내 상대 경로 (예: labeled/20260216_123456).
    반환: 업로드한 파일 수.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)
    prefix = remote_prefix.rstrip("/")
    count = 0
    for f in local_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(local_dir)
            object_key = f"{prefix}/{rel}" if prefix else str(rel).replace("\\", "/")
            s3_client.upload_file(str(f), bucket, object_key)
            count += 1
    return count


def upload_labeled_dir_to_runpod(
    labeled_dir: str | Path,
    volume_id: str | None = None,
    remote_prefix: str | None = None,
) -> int:
    """
    라벨링 결과 디렉터리를 RunPod 네트워크 볼륨에 업로드.
    RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY 필수.
    volume_id 미지정 시 RUNPOD_NETWORK_VOLUME_ID 또는 기본값(4rlm64f9lv) 사용.
    remote_prefix 미지정 시 labeled/ 디렉터리 이름(버전) 사용.
    """
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY are required.")
    labeled_dir = Path(labeled_dir)
    bucket = volume_id or os.environ.get("RUNPOD_NETWORK_VOLUME_ID", DEFAULT_VOLUME_ID)
    prefix = remote_prefix if remote_prefix is not None else f"labeled/{labeled_dir.name}"
    client = get_runpod_s3_client()
    return upload_directory(client, bucket, labeled_dir, prefix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload a directory to RunPod Network Volume (S3 API).")
    parser.add_argument("local_dir", type=Path, help="Local directory to upload")
    parser.add_argument("--volume-id", default=None, help=f"Network volume ID (default: env or {DEFAULT_VOLUME_ID})")
    parser.add_argument("--prefix", default="", help="Remote path prefix (e.g. labeled/20260216_123456)")
    args = parser.parse_args()
    bucket = args.volume_id or os.environ.get("RUNPOD_NETWORK_VOLUME_ID", DEFAULT_VOLUME_ID)
    client = get_runpod_s3_client()
    n = upload_directory(client, bucket, args.local_dir, args.prefix or args.local_dir.name)
    print(f"Uploaded {n} files to s3://{bucket}/{args.prefix or args.local_dir.name}")
