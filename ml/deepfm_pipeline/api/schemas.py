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
    source_dataset_path: str | None = Field(None, description="단일 CSV 경로. 지정 시 train/test로 나눈 뒤 전처리·학습")
    test_ratio: float = Field(0.2, description="source_dataset_path 사용 시 test 비율 (0~1)")
    random_state: int | None = Field(42, description="source_dataset_path 사용 시 분할 시드 (time_column 사용 시 무시)")
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
    negative_sampling_ratio: float = Field(1.0, description="positive 1건당 추가할 음성 샘플 수, 0이면 미적용")
    negative_sampling_seed: int = Field(42, description="음성 샘플링 시드")
    eval_list_size: int = Field(101, description="test/val 리스트당 행 수 (1 pos + eval_num_neg neg). 0이면 미적용")
    eval_num_neg: int = Field(100, description="리스트당 음성 개수")
    eval_num_popular_neg: int = Field(50, description="리스트당 인기 아이템 기반 음성 개수 (나머지는 랜덤)")
    eval_popular_top_k: int = Field(1000, description="인기 아이템 풀 크기 (positive count 상위 K)")
    eval_list_seed: int = Field(42, description="eval 리스트 구성 시드")
    use_wandb: bool = True


class TrainResponseDto(BaseModel):
    """학습 트리거 응답."""

    pipeline_version: str = Field(..., description="발급된 pipeline_version")
    model_path: str = Field(..., description="저장된 model.pt 경로")
    run_manifest_path: str = Field(..., description="run_manifest.json 경로")
    metrics: dict[str, Any] | None = Field(None, description="오프라인 지표 (NDCG@K, AUC 등)")

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
