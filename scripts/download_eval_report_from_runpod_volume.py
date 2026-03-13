#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.runpod_cli.runpod_s3_upload import get_runpod_s3_client


def _build_key_candidates(version: str, run_id: str, eval_group: str) -> list[str]:
    # Pod 구현/실행 환경에 따라 eval 그룹 디렉터리 이름이 달라질 수 있어 후보를 둔다.
    groups = [eval_group] if eval_group else ["eva1", "eval"]
    keys: list[str] = []
    for g in groups:
        keys.append(f"distill_pipeline_output/eval_output/{version}/{g}/{run_id}/report.json")
    return keys


def main() -> int:
    p = argparse.ArgumentParser(
        description="RunPod Network Volume(S3 API)에서 eval report.json 단일 파일만 다운로드",
    )
    p.add_argument("--eval-version", required=True, help="eval_output 버전 (예: 20260313_040811)")
    p.add_argument(
        "--run-id",
        required=True,
        help="eval run_id (예: 20260313_040905). 보통 report.json이 들어있는 디렉터리명",
    )
    p.add_argument(
        "--eval-group",
        default="",
        help="eval 그룹 디렉터리명 (예: eva1 또는 eval). 미지정 시 ['eva1','eval'] 순서로 시도",
    )
    p.add_argument(
        "--out",
        default="",
        help="다운로드 저장 경로. 미지정 시 distill_pipeline_output/eval_from_pod/<eval-version>_report.json",
    )
    p.add_argument(
        "--volume-id",
        default="",
        help="RunPod volume id. 미지정 시 RUNPOD_NETWORK_VOLUME_ID 환경변수 사용",
    )
    args = p.parse_args()

    vol_id = args.volume_id or os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    if not vol_id:
        raise SystemExit("RUNPOD_NETWORK_VOLUME_ID not set (or pass --volume-id).")

    out = (
        Path(args.out)
        if args.out
        else Path("distill_pipeline_output/eval_from_pod") / f"{args.eval_version}_report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    client = get_runpod_s3_client()
    key_candidates = _build_key_candidates(args.eval_version, args.run_id, args.eval_group)

    last_err: Exception | None = None
    for key in key_candidates:
        try:
            client.head_object(Bucket=vol_id, Key=key)
            client.download_file(vol_id, key, str(out))
            print(f"downloaded: s3://{vol_id}/{key} -> {out}")
            return 0
        except Exception as e:
            last_err = e

    msg = "report.json not found. Tried keys:\n" + "\n".join(f"- {k}" for k in key_candidates)
    if last_err:
        msg += f"\nLast error: {type(last_err).__name__}: {last_err}"
    raise SystemExit(msg)


if __name__ == "__main__":
    raise SystemExit(main())

