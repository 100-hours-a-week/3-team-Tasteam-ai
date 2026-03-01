#!/usr/bin/env python3
"""
merged 모델을 API 추론용 RunPod 네트워크 볼륨(API_LLM)으로 S3 sync 업로드.

사용:
  # 로컬 merged 디렉터리 지정
  python scripts/sync_merged_to_api_llm_volume.py --source-dir ./distill_pipeline_output/merged_for_serving/20260216_123456

  # distill_pipeline_output/merged_for_serving 중 최신 버전 자동 선택
  python scripts/sync_merged_to_api_llm_volume.py --latest

  # 로컬 merged_for_serving 기준 최신 (latest_merged_path.json 무시)
  python scripts/sync_merged_to_api_llm_volume.py --latest --from-dir ./distill_pipeline_output

환경변수:
  RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY  필수.
  RUNPOD_S3_REGION / RUNPOD_S3_ENDPOINT_URL         미설정 시 API_LLM 볼륨(EU-SE-1) 사용.

참고: docs/network_volume/api_llm_net_vol.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# API_LLM 볼륨 (api_llm_net_vol.md)
API_LLM_VOLUME_ID = "n5e655a7w5"
API_LLM_S3_PREFIX = "merged"
API_LLM_REGION = "eu-se-1"
API_LLM_ENDPOINT_URL = "https://s3api-eu-se-1.runpod.io"


def _get_latest_merged_dir(base_dir: Path) -> Path | None:
    """merged_for_serving/ 아래 YYYYMMDD_HHMMSS 형식 디렉터리 중 최신 하나 반환."""
    merged_root = base_dir / "merged_for_serving"
    if not merged_root.is_dir():
        return None
    candidates = [d for d in merged_root.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def _get_latest_from_pointer(base_dir: Path) -> Path | None:
    """latest_merged_path.json 의 merged_model_path 반환 (로컬 경로일 때만)."""
    pointer = base_dir / "merged_for_serving" / "latest_merged_path.json"
    if not pointer.is_file():
        return None
    try:
        import json
        data = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(data.get("merged_model_path", ""))
        if path.is_absolute() and path.exists():
            return path
        # 상대 경로면 base_dir 기준
        resolved = (base_dir / path).resolve()
        return resolved if resolved.exists() else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync merged model directory to API_LLM RunPod network volume (S3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Merged model local directory (e.g. .../merged_for_serving/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest merged dir from base dir (by timestamp or latest_merged_path.json)",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Base dir for --latest (default: PROJECT_ROOT/distill_pipeline_output)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print source and target, do not upload",
    )
    args = parser.parse_args()

    base_dir = args.from_dir or (_PROJECT_ROOT / "distill_pipeline_output")

    if args.source_dir is not None:
        source = Path(args.source_dir).resolve()
    elif args.latest:
        source = _get_latest_from_pointer(base_dir) or _get_latest_merged_dir(base_dir)
        if source is None:
            print("No latest merged dir found. Run with --source-dir or merge first.", file=sys.stderr)
            return 1
    else:
        parser.error("Provide either --source-dir or --latest")

    if not source.is_dir():
        print(f"Source is not a directory: {source}", file=sys.stderr)
        return 1

    # config.json 등 merged 모델 필수 파일 존재 여부 간단 체크
    if not (source / "config.json").exists():
        print(f"Warning: {source}/config.json not found. Is this a merged HuggingFace model dir?", file=sys.stderr)

    target = f"s3://{API_LLM_VOLUME_ID}/{API_LLM_S3_PREFIX}/"
    print(f"Source:  {source}")
    print(f"Target:  {target} (API_LLM volume, container path /workspace/merged)")
    if args.dry_run:
        print("Dry-run: skipping upload.")
        return 0

    try:
        from runpod_cli.runpod_s3_upload import get_runpod_s3_client, upload_directory
    except ImportError as e:
        print(f"Import error: {e}. Need runpod_cli.runpod_s3_upload (RUNPOD_S3_* env set).", file=sys.stderr)
        return 1

    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        print("Set RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY.", file=sys.stderr)
        return 1

    client = get_runpod_s3_client(region=API_LLM_REGION, endpoint_url=API_LLM_ENDPOINT_URL)
    n = upload_directory(client, API_LLM_VOLUME_ID, source, API_LLM_S3_PREFIX)
    print(f"Uploaded {n} files to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
