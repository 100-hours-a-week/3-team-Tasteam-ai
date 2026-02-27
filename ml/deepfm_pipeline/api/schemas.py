"""
Admin DeepFM API DTO (api_design.md).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ----- POST /admin/deepfm/train -----
class TrainRequestDto(BaseModel):
    """학습 트리거 요청 (전부 선택). 미지정 시 파이프라인 기본값 사용."""

    raw_data_dir: str | None = None
    processed_data_dir: str | None = None
    num_train_sample: int | None = None
    num_test_sample: int | None = None
    num_val: int | None = None
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    output_dir: str | None = None
    use_cuda: bool = False
    skip_preprocess: bool = False
    use_sample_weight: bool = True
    time_column: str | None = None
    train_end: str | None = None
    valid_end: str | None = None
    test_end: str | None = None
    group_column: str | None = None
    use_wandb: bool = True


class TrainResponseDto(BaseModel):
    """학습 트리거 응답."""

    pipeline_version: str = Field(..., description="발급된 pipeline_version")
    model_path: str = Field(..., description="저장된 model.pt 경로")
    run_manifest_path: str = Field(..., description="run_manifest.json 경로")
    metrics: dict[str, Any] | None = Field(None, description="오프라인 지표 (NDCG@K, AUC 등)")


# ----- POST /admin/deepfm/score-batch -----
class ScoreBatchRequestDto(BaseModel):
    """배치 스코어링/추천 생성 트리거 요청."""

    pipeline_version: str = Field(..., description="사용할 모델 pipeline_version (또는 run_dir에 해당하는 버전)")
    run_dir: str | None = Field(None, description="run 디렉터리 경로. 없으면 pipeline_version으로 output 하위에서 탐색")
    candidates_path: str = Field(..., description="후보 CSV 경로 (전처리된 feature 열)")
    output_path: str = Field(..., description="recommendation 형식 CSV 출력 경로")
    meta_path: str | None = Field(None, description="선택. user_id, anonymous_id, restaurant_id, context_snapshot 메타 CSV")
    ttl_hours: float = Field(24.0, description="expires_at TTL(시간)")
    batch_size: int = Field(256, description="추론 배치 크기")


class ScoreBatchResponseDto(BaseModel):
    """배치 스코어링 응답. recommendation INSERT는 호출 측(ETL/DB)에서 수행."""

    pipeline_version: str
    output_path: str
    rows_written: int = Field(..., description="출력된 recommendation 행 수")


# ----- GET /admin/deepfm/models -----
class ModelInfoDto(BaseModel):
    """단일 모델(버전) 정보."""

    pipeline_version: str
    run_dir: str
    created_at: str | None = None
    metrics: dict[str, Any] | None = None


class ModelsResponseDto(BaseModel):
    """모델/버전 목록 조회 응답."""

    models: list[ModelInfoDto] = Field(default_factory=list)
    active_version: str | None = Field(None, description="현재 활성(서빙) pipeline_version")


# ----- POST /admin/deepfm/activate -----
class ActivateRequestDto(BaseModel):
    """서빙용 pipeline_version 활성화 요청."""

    pipeline_version: str = Field(..., description="활성화할 pipeline_version")


class ActivateResponseDto(BaseModel):
    """활성화 응답."""

    active_version: str = Field(..., description="현재 활성 pipeline_version")
