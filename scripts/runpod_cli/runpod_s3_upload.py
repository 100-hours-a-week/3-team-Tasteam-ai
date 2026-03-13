"""
RunPod Network Volume S3 호환 API로 로컬 파일/디렉터리 업로드.

환경변수:
  RUNPOD_S3_ACCESS_KEY / RUNPOD_S3_SECRET_ACCESS_KEY - 필수. 시크릿은 config 파일에 넣지 말 것.
  RUNPOD_NETWORK_VOLUME_ID, RUNPOD_S3_REGION, RUNPOD_S3_ENDPOINT_URL - 선택. 미설정 시 config/runpod.yaml 기본값 사용.

참고: docs/runpod_cli/runpod_net_vol_s3_api.md
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore[assignment]
    ClientError = Exception  # type: ignore[misc, assignment]

from . import runpod_config


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
    """RunPod S3 호환 API용 boto3 S3 클라이언트 생성. 기본값은 config/runpod.yaml·환경 변수에서 로드."""
    if boto3 is None:
        raise RuntimeError("boto3 is required for RunPod S3 upload: pip install boto3")
    access, secret = _get_credentials()
    region = region or runpod_config.get_s3_region()
    endpoint_url = endpoint_url or runpod_config.get_s3_endpoint_url()
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


def object_exists(s3_client: Any, bucket: str, key: str) -> bool:
    """볼륨(S3)에 해당 키가 존재하는지 확인. merge 완료 폴링용."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        raise


def prefix_has_objects(
    volume_id: str,
    prefix: str,
    *,
    s3_client: Any | None = None,
) -> bool:
    """볼륨에서 prefix 하위에 객체가 하나라도 있는지 확인. 다운로드 전 ls 체크용."""
    client = s3_client or get_runpod_s3_client()
    prefix_norm = prefix.rstrip("/") + "/" if prefix else ""
    resp = client.list_objects_v2(Bucket=volume_id, Prefix=prefix_norm, MaxKeys=1)
    return len(resp.get("Contents", [])) > 0


def upload_file_to_volume(
    local_path: str | Path,
    volume_id: str | None = None,
    object_key: str | None = None,
) -> None:
    """
    단일 파일을 RunPod 네트워크 볼륨에 업로드.
    object_key 미지정 시 파일명만 사용 (볼륨 루트에 업로드).
    """
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY are required.")
    local_path = Path(local_path)
    bucket = volume_id or runpod_config.get_volume_id_train()
    key = object_key if object_key else local_path.name
    client = get_runpod_s3_client()
    upload_file(client, bucket, local_path, key)


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
    skip_if_exists: bool = True,
) -> int:
    """
    라벨링 결과 디렉터리를 RunPod 네트워크 볼륨에 업로드.
    RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY 필수.
    volume_id 미지정 시 RUNPOD_NETWORK_VOLUME_ID 또는 기본값(v3i546pkrz) 사용.
    remote_prefix 미지정 시 labeled/ 디렉터리 이름(버전) 사용.
    skip_if_exists: True면 해당 버전(prefix/train_labeled.json)이 이미 볼륨에 있으면 업로드 생략하고 0 반환.
    """
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY are required.")
    labeled_dir = Path(labeled_dir)
    bucket = volume_id or runpod_config.get_volume_id_train()
    prefix = remote_prefix if remote_prefix is not None else f"labeled/{labeled_dir.name}"
    client = get_runpod_s3_client()
    if skip_if_exists:
        key = f"{prefix}/train_labeled.json"
        if object_exists(client, bucket, key):
            return 0
    return upload_directory(client, bucket, labeled_dir, prefix)


def list_run_ids_with_adapter(volume_id: str, runs_prefix: str = "distill_pipeline_output/runs") -> list[str]:
    """볼륨 내 runs/ 하위에서 adapter가 있는 run_id 목록 반환 (최신 순)."""
    client = get_runpod_s3_client()
    bucket = volume_id or runpod_config.get_volume_id_train()
    prefix = f"{runs_prefix}/"
    run_ids: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            run_prefix = cp.get("Prefix", "").rstrip("/")
            if not run_prefix:
                continue
            run_id = run_prefix.split("/")[-1]
            # adapter 폴더 존재 여부
            adj = client.list_objects_v2(Bucket=bucket, Prefix=f"{run_prefix}/adapter/", MaxKeys=1)
            if adj.get("KeyCount", 0) > 0:
                run_ids.append(run_id)
    run_ids.sort(reverse=True)
    return run_ids


def download_directory_from_runpod(
    volume_id: str,
    remote_prefix: str,
    local_dir: str | Path,
) -> int:
    """볼륨의 remote_prefix 하위를 로컬 local_dir로 다운로드. 반환: 파일 수.

    RunPod S3 list pagination 비호환(동일 NextContinuationToken 반복) 때문에
    paginator 대신 list_objects_v2를 수동 루프로 호출한다.
    동일 토큰이 반복되면 잠시 대기 후 재시도하고, 계속 반복되면 중단한다.
    키 중복 제거 후 다운로드하여 PaginationError 없이 동작한다.
    """
    client = get_runpod_s3_client()
    bucket = volume_id
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = remote_prefix.rstrip("/") + "/"
    seen_keys: set[str] = set()
    count = 0

    continuation_token: str | None = None
    last_token: str | None = None
    same_token_retries = 0
    max_same_token_retries = 5
    retry_sleep_sec = 2

    while True:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj.get("Key")
            if not key or not key.startswith(prefix) or key in seen_keys:
                continue
            seen_keys.add(key)
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            count += 1

        next_token = resp.get("NextContinuationToken")
        if not next_token:
            break

        if next_token == last_token:
            same_token_retries += 1
            if same_token_retries > max_same_token_retries:
                break
            time.sleep(retry_sleep_sec)
            continuation_token = next_token
            continue

        same_token_retries = 0
        last_token = next_token
        continuation_token = next_token

    return count


def delete_prefix_from_volume(
    volume_id: str,
    prefix: str,
    s3_client: Any | None = None,
) -> int:
    """볼륨에서 prefix 하위 객체를 모두 삭제. 반환: 삭제한 객체 수.

    RunPod S3 API pagination 비호환(동일 ContinuationToken 반복)을 피하기 위해
    paginator 대신 list_objects_v2를 수동 루프로 호출하며, 같은 token이 두 번 나오면
    그 시점에서 list를 중단하고 그때까지 수집한 키만 삭제한다.
    """
    client = s3_client or get_runpod_s3_client()
    prefix_norm = prefix.rstrip("/") + "/" if prefix else ""
    deleted = 0
    seen_tokens: set[str] = set()
    continuation_token: str | None = None
    while True:
        if continuation_token is not None and continuation_token in seen_tokens:
            # 동일 token이 두 번 나온 경우(API 비호환) 루프 종료
            break
        if continuation_token is not None:
            seen_tokens.add(continuation_token)
        kwargs: dict[str, Any] = {
            "Bucket": volume_id,
            "Prefix": prefix_norm,
            "MaxKeys": 1000,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = client.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        if contents:
            keys = [{"Key": obj["Key"]} for obj in contents]
            client.delete_objects(Bucket=volume_id, Delete={"Objects": keys})
            deleted += len(keys)
        next_token = resp.get("NextContinuationToken")
        if not next_token:
            break
        continuation_token = next_token
    return deleted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload a directory to RunPod Network Volume (S3 API).")
    parser.add_argument("local_dir", type=Path, help="Local directory to upload")
    parser.add_argument("--volume-id", default=None, help="Network volume ID (default: config/runpod.yaml or env)")
    parser.add_argument("--prefix", default="", help="Remote path prefix (e.g. labeled/20260216_123456)")
    args = parser.parse_args()
    bucket = args.volume_id or runpod_config.get_volume_id_train()
    client = get_runpod_s3_client()
    n = upload_directory(client, bucket, args.local_dir, args.prefix or args.local_dir.name)
    print(f"Uploaded {n} files to s3://{bucket}/{args.prefix or args.local_dir.name}")
