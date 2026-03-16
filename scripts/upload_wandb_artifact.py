#!/usr/bin/env python3
"""
로컬 파일 또는 디렉터리를 W&B artifact로 업로드하는 범용 스크립트.

사용:
  # 디렉터리 업로드 (artifact 이름 + 타입 지정)
  python scripts/upload_wandb_artifact.py --path ./data/service_raw/v1 --name service_raw_v1 --type raw_data

  # 파일 하나 업로드
  python scripts/upload_wandb_artifact.py --path ./report.json --name my-report --type report

  # 프로젝트/엔티티 지정
  python scripts/upload_wandb_artifact.py --path ./out --name my-artifact --project deepfm-pipeline --type dataset

환경변수:
  WANDB_API_KEY   필수.
  WANDB_PROJECT   선택. --project 미지정 시 사용 (기본: tasteam-artifacts).
  WANDB_ENTITY    선택. --entity 미지정 시 사용.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def upload_artifact(
    *,
    path: str | Path,
    name: str,
    type: str = "dataset",
    project: str | None = None,
    entity: str | None = None,
    description: str | None = None,
    metadata: dict | None = None,
) -> str:
    """
    path(파일 또는 디렉터리)를 wandb artifact로 업로드.
    반환: qualified name (entity/project/name:version).
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required. Set it in the environment.")

    project = (project or os.environ.get("WANDB_PROJECT") or "tasteam-artifacts").strip()
    entity = (entity or os.environ.get("WANDB_ENTITY") or "").strip()

    import wandb

    run = wandb.init(project=project, entity=entity or None, job_type="artifact_upload")
    artifact = wandb.Artifact(
        name=name,
        type=type,
        description=description or None,
        metadata=metadata or {},
    )
    if path.is_file():
        artifact.add_file(str(path), name=path.name)
    else:
        artifact.add_dir(str(path), name=path.name)
    run.log_artifact(artifact)
    version = getattr(artifact, "version", None) or "latest"
    qualified = f"{run.entity}/{run.project}/{name}:{version}"
    wandb.finish()
    return qualified


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload a local file or directory as a Weights & Biases artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--path", type=str, required=True, help="로컬 파일 또는 디렉터리 경로")
    ap.add_argument("--name", type=str, required=True, help="Artifact 이름 (동일 이름 재업로드 시 wandb가 v0, v1, ... 버저닝)")
    ap.add_argument("--type", type=str, default="dataset", help="Artifact 타입 (기본: dataset)")
    ap.add_argument("--project", type=str, default=None, help="W&B 프로젝트 (기본: WANDB_PROJECT 또는 tasteam-artifacts)")
    ap.add_argument("--entity", type=str, default=None, help="W&B 엔티티 (기본: WANDB_ENTITY)")
    ap.add_argument("--description", type=str, default=None, help="Artifact 설명")
    args = ap.parse_args()

    try:
        qualified = upload_artifact(
            path=args.path,
            name=args.name,
            type=args.type,
            project=args.project,
            entity=args.entity,
            description=args.description,
        )
        print(qualified)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
