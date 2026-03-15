#!/usr/bin/env python3
"""
RunPod 네트워크 볼륨(S3 호환) 내부 목록 조회. ls처럼 prefix 하위 키를 출력한다.

사용:
  python -m scripts.runpod_cli.volume_ls [--volume-id ID] [--prefix PREFIX] [--long]
  python scripts/runpod_cli/volume_ls.py [--volume-id ID] [--prefix PREFIX] [--long]

환경변수: RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY 필수.
"""

from __future__ import annotations

import argparse
from typing import Any

from . import runpod_config
from .runpod_s3_upload import get_runpod_s3_client


def list_volume_prefix(
    volume_id: str,
    prefix: str = "",
    *,
    s3_client: Any = None,
) -> list[dict]:
    """볼륨에서 prefix 하위 객체 목록 반환. 각 항목: Key, Size, LastModified.

    RunPod S3 list pagination 비호환(동일 NextContinuationToken 반복 등) 때문에
    paginator 대신 list_objects_v2를 수동 루프로 호출한다.
    동일 토큰이 반복되면 잠시 대기 후 재시도하고, 계속 반복되면 중단한다.
    """
    client = s3_client or get_runpod_s3_client()
    prefix_norm = prefix.rstrip("/") + "/" if prefix else ""
    out: list[dict] = []
    seen_keys: set[str] = set()

    continuation_token: str | None = None
    last_token: str | None = None
    same_token_retries = 0
    max_same_token_retries = 5
    retry_sleep_sec = 2

    while True:
        kwargs: dict[str, Any] = {
            "Bucket": volume_id,
            "Prefix": prefix_norm,
            "MaxKeys": 1000,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            out.append(
                {
                    "Key": key,
                    "Size": obj.get("Size", 0),
                    "LastModified": obj.get("LastModified"),
                }
            )

        next_token = resp.get("NextContinuationToken")
        if not next_token:
            break

        if next_token == last_token:
            same_token_retries += 1
            if same_token_retries > max_same_token_retries:
                break
            # list 인덱스 지연/비호환 완화: 잠깐 대기 후 같은 token으로 재시도
            import time as _time

            _time.sleep(retry_sleep_sec)
            continuation_token = next_token
            continue

        same_token_retries = 0
        last_token = next_token
        continuation_token = next_token

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List objects inside a RunPod network volume (S3 API).",
    )
    parser.add_argument(
        "--volume-id",
        default=None,
        help="Volume ID (default: eval volume from config/env RUNPOD_NETWORK_VOLUME_ID)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix to list under (e.g. distill_pipeline_output/eval_output)",
    )
    parser.add_argument(
        "--long",
        "-l",
        action="store_true",
        help="Show size and last modified time",
    )
    args = parser.parse_args()

    vol_id = (
        args.volume_id
        or runpod_config.get_volume_id_eval()
        or runpod_config.get_volume_id_train()
    )
    if not vol_id:
        parser.error("--volume-id required or set RUNPOD_NETWORK_VOLUME_ID / config")

    items = list_volume_prefix(vol_id, args.prefix)
    for item in items:
        key = item["Key"]
        if args.long:
            size = item["Size"]
            lm = item["LastModified"]
            lm_str = lm.isoformat() if lm else ""
            print(f"{size:>12}  {lm_str}  {key}")
        else:
            print(key)


if __name__ == "__main__":
    main()
