"""
추론 시 run_dir 확보: 로컬 검색 → 없으면 wandb artifact(model_checkpoint)에서 pipeline_version으로 다운로드.

training_flow에서 model_checkpoint(type=model) artifact에 pipeline_version 메타데이터로 올리므로,
같은 pipeline_version 메타를 가진 버전을 찾아 다운로드해 run_dir로 사용.
"""
from __future__ import annotations

import os
from pathlib import Path


# 학습 시 업로드하는 artifact 이름/타입 (training_flow.py와 일치)
MODEL_ARTIFACT_NAME = "model_checkpoint"
MODEL_ARTIFACT_TYPE = "model"
DEFAULT_WANDB_PROJECT = "deepfm-pipeline"


def _find_run_dir_by_version(pipeline_version: str, search_dir: Path) -> Path | None:
    """search_dir 아래에서 pipeline_version.txt 내용이 일치하는 run 디렉터리 반환."""
    if not search_dir.exists():
        return None
    for d in search_dir.iterdir():
        if not d.is_dir():
            continue
        pv_file = d / "pipeline_version.txt"
        if pv_file.exists() and pv_file.read_text(encoding="utf-8").strip() == pipeline_version:
            return d
    return None


def _run_dir_valid(run_dir: Path) -> bool:
    """run_dir에 model.pt, feature_sizes.txt가 있으면 True."""
    return (run_dir / "model.pt").exists() and (run_dir / "feature_sizes.txt").exists()


def _download_run_dir_from_artifact(
    pipeline_version: str,
    cache_dir: Path,
    *,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
) -> Path:
    """
    wandb artifact(model_checkpoint) 중 metadata.pipeline_version이 일치하는 버전을 찾아
    cache_dir 아래에 다운로드하고 그 경로 반환.
    """
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required to download model from artifact.")

    try:
        import wandb
    except ImportError:
        raise ValueError("wandb is required to download model from artifact. Install with: pip install wandb")

    api = wandb.Api()
    entity = (wandb_entity or os.environ.get("WANDB_ENTITY") or "").strip()
    # artifact_collection: name은 "entity/project/artifact_name" 또는 "project/artifact_name"
    if entity:
        collection_ref = f"{entity}/{wandb_project}/{MODEL_ARTIFACT_NAME}"
    else:
        collection_ref = f"{wandb_project}/{MODEL_ARTIFACT_NAME}"

    try:
        # api.artifacts(type_name, name) → 버전 목록; api.artifact_collection은 단일 컬렉션
        artifacts_iter = api.artifacts(
            type_name=MODEL_ARTIFACT_TYPE,
            name=collection_ref,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find artifact collection {MODEL_ARTIFACT_TYPE}/{MODEL_ARTIFACT_NAME}: {e}"
        ) from e

    # 버전 순회하며 metadata.pipeline_version 일치하는 것 찾기
    out_dir = cache_dir / pipeline_version
    out_dir.mkdir(parents=True, exist_ok=True)

    for art in artifacts_iter:
        if not getattr(art, "metadata", None):
            continue
        if art.metadata.get("pipeline_version") == pipeline_version:
            art.download(root=str(out_dir))
            if _run_dir_valid(out_dir):
                return out_dir
            # artifact가 add_dir로 올라갔으면 루트에 바로 model.pt 등이 있음. 한 단계 더 있을 수 있음.
            for sub in out_dir.iterdir():
                if sub.is_dir() and _run_dir_valid(sub):
                    return sub
            return out_dir
    # 루프에서 못 찾음

    raise FileNotFoundError(
        f"No artifact version with metadata.pipeline_version={pipeline_version!r} in {collection_ref}. "
        "Train and upload a model first, or use --run-dir with a local path."
    )


def resolve_run_dir(
    run_dir: Path | None,
    pipeline_version: str | None,
    *,
    search_output_dir: Path,
    cache_dir: Path,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
) -> Path:
    """
    추론에 쓸 run_dir 경로를 반환. 다음 순서로 시도:
    1. run_dir이 주어졌고 존재하며 model.pt/feature_sizes.txt 있음 → run_dir 반환
    2. pipeline_version이 주어짐 → search_output_dir에서 해당 버전 run 검색 → 있으면 반환
    3. pipeline_version + WANDB_API_KEY → artifact에서 다운로드 후 cache_dir 아래 경로 반환
    4. 그 외 → FileNotFoundError / ValueError
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        if run_dir.exists() and _run_dir_valid(run_dir):
            return run_dir
        if run_dir.exists():
            raise FileNotFoundError(
                f"run_dir exists but missing model.pt or feature_sizes.txt: {run_dir}"
            )
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    if not pipeline_version:
        raise ValueError("Either run_dir or pipeline_version is required.")

    local_run = _find_run_dir_by_version(pipeline_version, search_output_dir)
    if local_run is not None and _run_dir_valid(local_run):
        return local_run

    return _download_run_dir_from_artifact(
        pipeline_version,
        cache_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
