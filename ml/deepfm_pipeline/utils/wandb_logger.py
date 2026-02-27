"""
W&B 연동 (wandb_design.md).

- Artifacts: split 메타, feature_sizes, dataset 스냅샷, model checkpoint, evaluation report, scoring CSV
- pipeline_version ↔ wandb run/artifact id 매핑
- wandb 미설치/미로그인 시 no-op
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def is_available() -> bool:
    return _WANDB_AVAILABLE


def init_run(
    project: str = "deepfm-pipeline",
    config: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> bool:
    """W&B run 시작. config는 flow/task 인자 등."""
    if not _WANDB_AVAILABLE:
        return False
    try:
        wandb.init(project=project, config=config or {}, name=run_name)
        return True
    except Exception:
        return False


def log_artifact(
    name: str,
    type: str,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """파일 또는 디렉터리를 artifact로 로깅."""
    if not _WANDB_AVAILABLE:
        return False
    try:
        path = Path(path)
        if not path.exists():
            return False
        art = wandb.Artifact(name=name, type=type, metadata=metadata or {})
        if path.is_file():
            art.add_file(str(path))
        else:
            art.add_dir(str(path))
        wandb.log_artifact(art)
        return True
    except Exception:
        return False


def log_metrics(metrics: dict[str, Any]) -> bool:
    """메트릭을 현재 run에 로깅 (e.g. NDCG@K, AUC)."""
    if not _WANDB_AVAILABLE:
        return False
    try:
        wandb.log(metrics)
        return True
    except Exception:
        return False


def set_summary(key: str, value: Any) -> bool:
    """run summary에 pipeline_version, artifact id 등 저장."""
    if not _WANDB_AVAILABLE:
        return False
    try:
        if wandb.run:
            wandb.run.summary[key] = value
        return True
    except Exception:
        return False


def finish_run() -> bool:
    if not _WANDB_AVAILABLE:
        return False
    try:
        wandb.finish()
        return True
    except Exception:
        return False


def get_run_id() -> str | None:
    """현재 wandb run id (pipeline_version 매핑용)."""
    if not _WANDB_AVAILABLE or not wandb.run:
        return None
    try:
        return wandb.run.id
    except Exception:
        return None


def dataset_stats_from_processed_dir(processed_data_dir: str | Path) -> dict[str, Any]:
    """train/val/test 행 수 + split_meta 요약 + train 상위 N행 해시 (스냅샷)."""
    root = Path(processed_data_dir)
    stats: dict[str, Any] = {}

    for name in ("train", "val", "test"):
        p = root / f"{name}.txt"
        if p.exists():
            lines = p.read_text(encoding="utf-8").splitlines()
            stats[f"n_{name}"] = len(lines)
            if name == "train" and lines:
                sample = "\n".join(lines[: min(100, len(lines))])
                stats["train_sample_sha256"] = hashlib.sha256(sample.encode()).hexdigest()
        else:
            stats[f"n_{name}"] = 0

    meta_path = root / "split_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        stats["split_meta"] = {
            k: v for k, v in meta.items()
            if k in ("train_end", "valid_end", "test_end", "time_column", "group_column", "use_sample_weight")
        }
    return stats
