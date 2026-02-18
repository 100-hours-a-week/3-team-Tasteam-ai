"""
Prefect 기반 DeepFM 학습 파이프라인.

실행 (프로젝트 루트에서):
  python deepfm_training/training_flow.py

또는 deepfm_training 디렉터리에서:
  python training_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# deepfm_training 루트를 path에 넣어 model/data/utils import 가능하게 함 (repo root에서 실행 시)
_deepfm_root = Path(__file__).resolve().parent
if str(_deepfm_root) not in sys.path:
    sys.path.insert(0, str(_deepfm_root))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from prefect import flow, task
from prefect.context import get_run_context

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset


# --- 기본 경로 (deepfm_training 기준) ---
def _default_raw_data_dir() -> str:
    return str(_deepfm_root / "data" / "raw")


def _default_processed_data_dir() -> str:
    return str(_deepfm_root / "data")


def _default_output_dir() -> str:
    return str(_deepfm_root / "output")


@task(name="deepfm-preprocess", log_prints=True)
def preprocess_task(
    raw_data_dir: str,
    processed_data_dir: str,
    num_train_sample: int = 9000,
    num_test_sample: int = 1000,
) -> str:
    """
    Criteo raw 데이터를 전처리하여 train.txt, test.txt, feature_sizes.txt 생성.
    """
    from utils.dataPreprocess import preprocess

    raw = Path(raw_data_dir)
    out = Path(processed_data_dir)
    if not raw.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    out.mkdir(parents=True, exist_ok=True)

    preprocess(
        datadir=str(raw),
        outdir=str(out),
        num_train_sample=num_train_sample,
        num_test_sample=num_test_sample,
    )
    print(f"Preprocess done: {processed_data_dir}")
    return processed_data_dir


@task(name="deepfm-train", log_prints=True)
def train_task(
    processed_data_dir: str,
    num_train: int = 9000,
    num_val: int = 1000,
    epochs: int = 5,
    batch_size: int = 100,
    lr: float = 1e-4,
    output_dir: str | None = None,
    use_cuda: bool = False,
    verbose: bool = True,
) -> dict:
    """
    전처리된 데이터로 DeepFM 학습 후 모델 저장.
    반환: {"model_path": ..., "feature_sizes_path": ..., "processed_data_dir": ...}
    """
    data_path = Path(processed_data_dir)
    if not (data_path / "train.txt").exists():
        raise FileNotFoundError(f"Processed train.txt not found in {processed_data_dir}")

    out_path = Path(output_dir or _default_output_dir())
    run_ctx = get_run_context()
    run_id = getattr(run_ctx, "flow_run", None) and getattr(run_ctx.flow_run, "id", None)
    run_name = str(run_id) if run_id else "manual"
    out_run = out_path / run_name
    out_run.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    train_data = CriteoDataset(str(data_path), train=True)
    loader_train = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(num_train)),
    )
    val_data = CriteoDataset(str(data_path), train=True)
    loader_val = DataLoader(
        val_data,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val)),
    )

    feature_sizes_path = data_path / "feature_sizes.txt"
    feature_sizes = np.loadtxt(str(feature_sizes_path), delimiter=",")
    feature_sizes = [int(x) for x in feature_sizes]

    model = DeepFM(feature_sizes, use_cuda=use_cuda)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=epochs, verbose=verbose)

    # 모델 및 메타 저장
    model_path = out_run / "model.pt"
    torch.save(model.state_dict(), model_path)
    meta_path = out_run / "feature_sizes.txt"
    np.savetxt(str(meta_path), feature_sizes, fmt="%d", delimiter=",")
    print(f"Model saved: {model_path}")

    return {
        "model_path": str(model_path),
        "feature_sizes_path": str(meta_path),
        "processed_data_dir": processed_data_dir,
    }


@flow(name="DeepFM Training Pipeline", log_prints=True)
def deepfm_training_flow(
    raw_data_dir: str | None = None,
    processed_data_dir: str | None = None,
    num_train_sample: int = 9000,
    num_test_sample: int = 1000,
    num_val: int = 1000,
    epochs: int = 5,
    batch_size: int = 100,
    lr: float = 1e-4,
    output_dir: str | None = None,
    use_cuda: bool = False,
    skip_preprocess: bool = False,
) -> dict:
    """
    전처리 → 학습을 순서대로 실행하는 DeepFM 파이프라인.

    - skip_preprocess=True 이면 전처리를 건너뛰고 기존 processed_data_dir만 사용.
    """
    raw = raw_data_dir or _default_raw_data_dir()
    processed = processed_data_dir or _default_processed_data_dir()
    out = output_dir or _default_output_dir()

    if not skip_preprocess:
        preprocess_task(
            raw_data_dir=raw,
            processed_data_dir=processed,
            num_train_sample=num_train_sample,
            num_test_sample=num_test_sample,
        )

    result = train_task(
        processed_data_dir=processed,
        num_train=num_train_sample,
        num_val=num_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=out,
        use_cuda=use_cuda,
        verbose=True,
    )
    return result


if __name__ == "__main__":
    # 기본 인자로 한 번 실행 (전처리 포함)
    deepfm_training_flow()
