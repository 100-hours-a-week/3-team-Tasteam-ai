#!/usr/bin/env python3
"""
wandb Sweep 에이전트: QLoRA 하이퍼파라미터 탐색.

sweep 등록:
  wandb sweep scripts/wandb_sweep_qlora.yaml

에이전트 실행 (경로는 환경변수로):
  export WANDB_SWEEP_LABELED_PATH=distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/train_labeled.json
  export WANDB_SWEEP_OUTPUT_DIR=distill_pipeline_output
  wandb agent <sweep_id>

또는 에이전트만 N회 실행:
  wandb agent <sweep_id> --count 5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# 프로젝트 루트
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def train() -> None:
    """wandb.agent가 호출하는 학습 함수. wandb.config에서 하이퍼파라미터를 읽어 run_train 실행."""
    import wandb
    from scripts.train_qlora import run_train

    wandb.init(project=os.environ.get("WANDB_PROJECT", "tasteam-distill"))

    labeled_path = os.environ.get("WANDB_SWEEP_LABELED_PATH")
    output_dir = os.environ.get("WANDB_SWEEP_OUTPUT_DIR")
    if not labeled_path or not output_dir:
        raise RuntimeError(
            "WANDB_SWEEP_LABELED_PATH and WANDB_SWEEP_OUTPUT_DIR environment variables are required. "
            "Example: export WANDB_SWEEP_LABELED_PATH=path/to/train_labeled.json "
            "WANDB_SWEEP_OUTPUT_DIR=distill_pipeline_output"
        )

    cfg = wandb.config
    args = SimpleNamespace(
        labeled_path=Path(labeled_path),
        output_dir=Path(output_dir),
        student_model=cfg.get("student_model", "Qwen/Qwen2.5-0.5B-Instruct"),
        gold_oversample_ratio=float(cfg.get("gold_oversample_ratio", 0.25)),
        target_modules=cfg.get("target_modules", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"),
        r=int(cfg.get("r", 8)),
        alpha=int(cfg.get("alpha", 16)),
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        num_epochs=int(cfg.get("num_epochs", 2)),
        batch_size=int(cfg.get("batch_size", 2)),
        grad_accum=int(cfg.get("grad_accum", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        eval_ratio=float(cfg.get("eval_ratio", 0.1)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 3)),
    )
    run_train(args)


if __name__ == "__main__":
    import wandb

    # wandb agent가 실행할 때 sweep_id를 첫 인자로 넘기는 경우 처리
    sweep_id = None
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        sweep_id = sys.argv[1].strip()
    if not sweep_id:
        sweep_id = os.environ.get("WANDB_SWEEP_ID")

    if sweep_id:
        wandb.agent(sweep_id, function=train, project=os.environ.get("WANDB_PROJECT", "tasteam-distill"))
    else:
        # sweep 없이 단일 run으로 테스트 (config 기본값)
        wandb.init(project=os.environ.get("WANDB_PROJECT", "tasteam-distill"), config={})
        train()
