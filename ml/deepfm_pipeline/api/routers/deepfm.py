"""
Admin DeepFM API (api_design.md).

- POST /admin/deepfm/train
- GET  /admin/deepfm/models
- POST /admin/deepfm/activate
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ActivateRequestDto,
    ActivateResponseDto,
    ModelInfoDto,
    ModelsResponseDto,
    TrainRequestDto,
    TrainResponseDto,
)

router = APIRouter(prefix="/admin/deepfm", tags=["admin-deepfm"])

# 프로젝트 루트 (api 패키지 기준 상위 2단계)
_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_output_dir() -> Path:
    return _ROOT / "output"


def _active_version_file() -> Path:
    return _default_output_dir() / "active_pipeline_version.txt"


def _find_run_dir_by_version(pipeline_version: str) -> Path | None:
    out = _default_output_dir()
    if not out.exists():
        return None
    dirs: list[Path] = list(out.iterdir())
    cache_dir = out / "artifact_cache"
    if cache_dir.exists():
        dirs.extend(cache_dir.iterdir())
    for d in dirs:
        if not d.is_dir():
            continue
        pv_file = d / "pipeline_version.txt"
        if pv_file.exists() and pv_file.read_text(encoding="utf-8").strip() == pipeline_version:
            return d
    return None


@router.post("/train", response_model=TrainResponseDto)
def trigger_train(body: TrainRequestDto | None = None) -> TrainResponseDto:
    """
    Prefect 학습 플로우 실행 트리거.
    산출물: 모델 artifact + pipeline_version.
    """
    body = body or TrainRequestDto()
    import sys
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from training_flow import deepfm_training_flow, _default_raw_data_dir, _default_processed_data_dir, _default_output_dir as _flow_out

    raw = body.raw_data_dir or str(_ROOT / "data" / "raw")
    processed = body.processed_data_dir or str(_ROOT / "data")
    out = body.output_dir or str(_flow_out())

    try:
        result = deepfm_training_flow(
            raw_data_dir=raw,
            processed_data_dir=processed,
            source_dataset_path=body.source_dataset_path,
            test_ratio=body.test_ratio,
            random_state=body.random_state,
            num_train_sample=body.num_train_sample,
            num_test_sample=body.num_test_sample,
            num_val=body.num_val or 1000,
            epochs=body.epochs or 5,
            batch_size=body.batch_size or 100,
            lr=body.lr or 1e-4,
            output_dir=out,
            use_cuda=body.use_cuda,
            skip_preprocess=body.skip_preprocess,
            use_sample_weight=body.use_sample_weight,
            time_column=body.time_column,
            train_end=body.train_end,
            valid_end=body.valid_end,
            test_end=body.test_end,
            group_column=body.group_column,
            negative_sampling_ratio=body.negative_sampling_ratio,
            negative_sampling_seed=body.negative_sampling_seed,
            eval_list_size=body.eval_list_size,
            eval_num_neg=body.eval_num_neg,
            eval_num_popular_neg=body.eval_num_popular_neg,
            eval_popular_top_k=body.eval_popular_top_k,
            eval_list_seed=body.eval_list_seed,
            use_wandb=body.use_wandb,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrainResponseDto(
        pipeline_version=result["pipeline_version"],
        model_path=result["model_path"],
        run_manifest_path=result["run_manifest_path"],
        metrics=result.get("metrics") if result.get("metrics") and "error" not in result.get("metrics", {}) else None,
    )


@router.get("/models", response_model=ModelsResponseDto)
def list_models() -> ModelsResponseDto:
    """모델/버전 목록 조회. output/ 및 output/artifact_cache/ 하위 run 포함."""
    out = _default_output_dir()
    models: list[ModelInfoDto] = []
    if not out.exists():
        return ModelsResponseDto(models=[], active_version=_get_active_version())

    dirs: list[Path] = list(out.iterdir())
    cache_dir = out / "artifact_cache"
    if cache_dir.exists():
        dirs.extend(cache_dir.iterdir())
    for d in sorted((d for d in dirs if d.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        pv_file = d / "pipeline_version.txt"
        if not pv_file.exists():
            continue
        pv = pv_file.read_text(encoding="utf-8").strip()
        manifest = d / "run_manifest.json"
        created_at = None
        metrics = None
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                created_at = data.get("timestamp_utc")
                metrics = data.get("metrics")
            except Exception:
                pass
        models.append(
            ModelInfoDto(
                pipeline_version=pv,
                run_dir=str(d),
                created_at=created_at,
                metrics=metrics,
            )
        )

    return ModelsResponseDto(models=models, active_version=_get_active_version())


def _get_active_version() -> str | None:
    f = _active_version_file()
    if not f.exists():
        return None
    return f.read_text(encoding="utf-8").strip() or None


@router.post("/activate", response_model=ActivateResponseDto)
def activate_model(body: ActivateRequestDto) -> ActivateResponseDto:
    """서빙용 pipeline_version 활성화. 로컬에 없으면 wandb artifact에서 다운로드 후 활성화."""
    run_dir = _find_run_dir_by_version(body.pipeline_version)
    if not run_dir or not run_dir.exists():
        try:
            import sys
            if str(_ROOT) not in sys.path:
                sys.path.insert(0, str(_ROOT))
            from utils.run_dir_resolver import resolve_run_dir
            cache_dir = _default_output_dir() / "artifact_cache"
            run_dir = resolve_run_dir(
                run_dir=None,
                pipeline_version=body.pipeline_version,
                search_output_dir=_default_output_dir(),
                cache_dir=cache_dir,
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {e}",
            ) from e
    out = _default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    _active_version_file().write_text(body.pipeline_version, encoding="utf-8")
    return ActivateResponseDto(active_version=body.pipeline_version)
