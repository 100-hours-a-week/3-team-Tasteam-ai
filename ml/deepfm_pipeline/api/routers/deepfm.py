"""
Admin DeepFM API (api_design.md).

- POST /admin/deepfm/train
- POST /admin/deepfm/score-batch
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
    ScoreBatchRequestDto,
    ScoreBatchResponseDto,
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
    for d in out.iterdir():
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


@router.post("/score-batch", response_model=ScoreBatchResponseDto)
def trigger_score_batch(body: ScoreBatchRequestDto) -> ScoreBatchResponseDto:
    """
    배치 스코어링/추천 생성 트리거.
    입력: pipeline_version, 후보 경로, TTL 등.
    출력: recommendation CSV. INSERT는 호출 측(ETL/DB)에서 수행.
    """
    run_dir: Path | None = None
    if body.run_dir:
        run_dir = Path(body.run_dir)
    else:
        run_dir = _find_run_dir_by_version(body.pipeline_version)
    if not run_dir or not run_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Run not found for pipeline_version={body.pipeline_version}. Set run_dir or ensure version exists under output/.",
        )

    out_path = Path(body.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import sys
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from utils.score_batch import run as score_batch_run

    try:
        score_batch_run(
            run_dir=run_dir,
            candidates_path=Path(body.candidates_path),
            output_path=out_path,
            meta_path=Path(body.meta_path) if body.meta_path else None,
            ttl_hours=body.ttl_hours,
            batch_size=body.batch_size,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    n_rows = 0
    if out_path.exists():
        n_rows = sum(1 for _ in open(out_path, encoding="utf-8")) - 1  # minus header
    pv = (run_dir / "pipeline_version.txt").read_text(encoding="utf-8").strip() if (run_dir / "pipeline_version.txt").exists() else body.pipeline_version

    return ScoreBatchResponseDto(
        pipeline_version=pv,
        output_path=str(out_path),
        rows_written=max(0, n_rows),
    )


@router.get("/models", response_model=ModelsResponseDto)
def list_models() -> ModelsResponseDto:
    """모델/버전 목록 조회. 현재 활성(서빙) pipeline_version 포함."""
    out = _default_output_dir()
    models: list[ModelInfoDto] = []
    if not out.exists():
        return ModelsResponseDto(models=[], active_version=_get_active_version())

    for d in sorted(out.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
    """서빙용 pipeline_version 활성화. 어떤 모델이 현재 서빙인지 관리."""
    run_dir = _find_run_dir_by_version(body.pipeline_version)
    if not run_dir or not run_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: pipeline_version={body.pipeline_version}",
        )
    out = _default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    _active_version_file().write_text(body.pipeline_version, encoding="utf-8")
    return ActivateResponseDto(active_version=body.pipeline_version)
