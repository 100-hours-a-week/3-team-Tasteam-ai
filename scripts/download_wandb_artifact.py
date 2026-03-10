#!/usr/bin/env python3
"""
wandb artifact를 지정하여 로컬로 다운로드하는 범용 스크립트.

사용:
  # qualified name 한 번에 (entity/project/artifact_name:version)
  python scripts/download_wandb_artifact.py "my-entity/my-project/my-artifact:v1" --out-dir ./downloaded

  # entity, project, name, version 분리
  python scripts/download_wandb_artifact.py --entity my-entity --project my-project --name my-artifact --version latest --out-dir ./downloaded

  # entity 생략 시 WANDB_ENTITY 환경변수 사용 (비어 있으면 qualified에서 제외)
  python scripts/download_wandb_artifact.py --project tasteam-distill --name distill-eval-report --version v1 --out-dir ./eval_artifacts

환경변수:
  WANDB_API_KEY  필수.
  WANDB_ENTITY   선택. --entity 미지정 시 사용.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_qualified(qualified: str) -> tuple[str, str, str, str]:
    """'entity/project/name:version' 또는 'project/name:version' 파싱. version 없으면 'latest'."""
    parts = qualified.strip().split(":")
    version = parts[1].strip() if len(parts) > 1 else "latest"
    path = parts[0].strip().rstrip("/")
    segs = path.split("/")
    if len(segs) < 2:
        raise ValueError(
            f"Qualified name must be at least 'project/artifact_name' or 'entity/project/artifact_name': got {qualified!r}"
        )
    if len(segs) == 2:
        entity = ""
        project, name = segs[0], segs[1]
    else:
        entity, project, name = segs[0], segs[1], "/".join(segs[2:])
    return entity, project, name, version


def _build_qualified(entity: str, project: str, name: str, version: str) -> str:
    if entity:
        return f"{entity}/{project}/{name}:{version}"
    return f"{project}/{name}:{version}"


def download_artifact(
    *,
    qualified: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    name: str | None = None,
    version: str = "latest",
    out_dir: str | Path = ".",
) -> str:
    """
    wandb artifact를 out_dir에 다운로드. 다운로드 루트 경로 반환.
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required. Set it in the environment.")

    if qualified:
        entity, project, name, version = _parse_qualified(qualified)
    else:
        if not project or not name:
            raise ValueError("Either qualified or (--project and --name) is required.")
        entity = (entity or os.environ.get("WANDB_ENTITY") or "").strip()
        version = version or "latest"

    qualified_name = _build_qualified(entity, project, name, version)

    import wandb

    api = wandb.Api()
    art = api.artifact(qualified_name)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    art.download(root=str(out_path))
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download a Weights & Biases artifact to a local directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "qualified",
        nargs="?",
        default=None,
        help="Full qualified name: entity/project/artifact_name:version (version optional, default latest)",
    )
    ap.add_argument("--entity", default=None, help="W&B entity (optional; uses WANDB_ENTITY if not set)")
    ap.add_argument("--project", default=None, help="W&B project name")
    ap.add_argument("--name", default=None, help="Artifact name")
    ap.add_argument("--version", default="latest", help="Artifact version (default: latest)")
    ap.add_argument("--out-dir", type=str, default="./artifacts_download", help="Download root directory (default: ./artifacts_download)")
    args = ap.parse_args()

    try:
        root = download_artifact(
            qualified=args.qualified,
            entity=args.entity,
            project=args.project,
            name=args.name,
            version=args.version,
            out_dir=args.out_dir,
        )
        print(root)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
