#!/usr/bin/env python3
"""
요약 KD 파이프라인용 Prefect flows.

전략·수치는 docs/easydistill/distill_strategy.md 를 따름:
  - 식당 단위 split (train/val/test 식당 분리, 누수 방지)
  - OpenAI 골드 300~800개 → train에서만, 정답/포맷 기준
  - self-hosted teacher로 나머지 train 라벨 → 품질 필터 필수 (JSON/길이/금지표현/근거)
  - 학습 데이터: 골드 + 필터된 teacher → 2500~3200, 골드 oversample 20~30%
  - val/test 라벨: 가능하면 OpenAI (각 300~500, 400~600)

Flow (docs/easydistill/distill_by_prefect.md):
  1. build_dataset_flow — 식당 단위 split, 윈도우/샘플 생성, train/val/test 저장 + 버전 태깅
  2. labeling_flow      — OpenAI 골드(학습용) + self-hosted teacher(학습용) + 품질 필터 (JSON/길이/근거/반복/dedup)
  3. train_student_flow — QLoRA SFT (cross-entropy only, ROUGE/BERTScore 학습에 미사용)
  4. evaluate_flow      — val/test: OpenAI 평가 라벨로 ROUGE/BERTScore/GPT-judge + 휴먼 평가 50~100개 (human_labels 스키마)

실행:
  python scripts/distill_flows.py build_dataset [--input path] [--out-dir dir]
  python scripts/distill_flows.py all
"""

# distill_strategy.md 권장 수치
OPENAI_GOLD_MIN, OPENAI_GOLD_MAX = 300, 800  # 골드 라벨 개수
TRAIN_TARGET_MIN, TRAIN_TARGET_MAX = 2500, 3200  # 최종 학습 데이터 목표
GOLD_OVERSAMPLE_RATIO = 0.25  # 골드 비중 20~30%
VAL_SAMPLES_TARGET = (300, 500)
TEST_SAMPLES_TARGET = (400, 600)
# 품질 필터 (distill_by_prefect §2): ①JSON구조 ②길이 ③근거기반성휴리스틱 ④OpenAI골드비교(선택) ⑤반복/붕괴감지
# 휴먼 평가 라벨 스키마: sample_id, model_name, relevance(1~5), faithfulness(1~5), structure_consistency(1~5), hallucination(0/1), overall_score(1~5), comment

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from prefect import flow, task
    from prefect.context import get_run_context
except ImportError:
    sys.exit("prefect is required: pip install prefect")

# 프로젝트 루트 (스크립트 기준)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


@task(name="build-dataset-task", log_prints=True)
def build_dataset_task(
    input_path: Path,
    out_dir: Path,
    window_configs: list[tuple[int, int]] | None = None,
    add_full_restaurant: bool = True,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    """데이터 증강 실행 (scripts/data_augmentation.py). 반환: dataset_version, train_path, val_path, test_path, stats_path."""
    out_dir = Path(out_dir)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dataset_dir = out_dir / "datasets" / version
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "data_augmentation.py"),
        "--input", str(input_path),
        "--out-dir", str(dataset_dir),
        "--train-ratio", str(train_ratio),
        "--val-ratio", str(val_ratio),
        "--test-ratio", str(test_ratio),
        "--seed", str(seed),
    ]
    if add_full_restaurant:
        cmd.append("--add-full")
    if window_configs:
        for w, s in window_configs:
            cmd.extend(["--window", str(w), str(s)])

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"data_augmentation.py exited with {result.returncode}")

    train_path = dataset_dir / "train.json"
    val_path = dataset_dir / "val.json"
    test_path = dataset_dir / "test.json"
    stats_path = dataset_dir / "stats.json"
    return {
        "dataset_version": version,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
        "stats_path": str(stats_path),
        "dataset_dir": str(dataset_dir),
    }


@flow(name="build_dataset_flow", log_prints=True)
def build_dataset_flow(
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    window_configs: list[tuple[int, int]] | None = None,
    add_full_restaurant: bool = True,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict:
    """식당 단위 split, 윈도우/샘플 생성, train/val/test 저장 + 버전 태깅."""
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    return build_dataset_task(
        input_path=input_path,
        out_dir=out_dir,
        window_configs=window_configs,
        add_full_restaurant=add_full_restaurant,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


@task(name="labeling-task", log_prints=True)
def labeling_task(
    train_path: str,
    openai_cap: int = 500,
    output_labeled_dir: str | None = None,
) -> dict:
    """라벨링 스텁. 실제 구현 시: OpenAI 골드(학습용) + self-hosted teacher(학습용) + 품질 필터 5종(JSON/길이/근거휴리스틱/골드비교/반복감지) 필수 (distill_by_prefect §2)."""
    print("labeling_flow: stub — implement OpenAI + self-hosted teacher labeling")
    out_dir = Path(output_labeled_dir or _PROJECT_ROOT / "distill_pipeline_output")
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    labeled_dir = out_dir / "labeled" / version
    labeled_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = labeled_dir / "train_labeled.json"
    # placeholder
    json.dump({"samples": [], "meta": {"openai_cap": openai_cap, "stub": True}}, open(labeled_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"labeled_version": version, "labeled_path": str(labeled_path)}


@flow(name="labeling_flow", log_prints=True)
def labeling_flow(
    train_path: str,
    openai_cap: int = 500,
    output_labeled_dir: str | Path | None = None,
) -> dict:
    """OpenAI 골드 라벨링(cap, 전략 300~800) + self-hosted teacher 나머지 + 품질 필터/dedup + 라벨 버전 저장 (distill_strategy.md)."""
    return labeling_task(
        train_path=train_path,
        openai_cap=openai_cap,
        output_labeled_dir=str(output_labeled_dir) if output_labeled_dir else None,
    )


@task(name="train-student-task", log_prints=True)
def train_student_task(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | None = None,
) -> dict:
    """QLoRA 학습 스텁. 실제 구현 시: 골드 oversample 20~30%, 총 2500~3200 학습 데이터로 PEFT+bitsandbytes (distill_strategy.md)."""
    print("train_student_flow: stub — implement QLoRA training")
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "training_meta.json"
    json.dump({
        "labeled_path": labeled_path,
        "student_model": student_model,
        "stub": True,
        "adapter_path": str(run_dir / "adapter"),
    }, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"adapter_path": str(run_dir / "adapter"), "training_meta_path": str(meta_path), "run_id": run_id}


@flow(name="train_student_flow", log_prints=True)
def train_student_flow(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | Path | None = None,
) -> dict:
    """QLoRA 학습 실행, artifact 저장, 학습 메타 기록."""
    return train_student_task(
        labeled_path=labeled_path,
        student_model=student_model,
        output_dir=str(output_dir) if output_dir else None,
    )


@task(name="evaluate-task", log_prints=True)
def evaluate_task(
    val_path: str,
    test_path: str,
    adapter_path: str,
    output_dir: str | None = None,
) -> dict:
    """평가 스텁. 실제 구현 시: val/test는 OpenAI 평가 라벨 → ROUGE/BERTScore/GPT-judge (학습에 미사용). 휴먼 50~100개 (sample_id, model_name, relevance, faithfulness, structure_consistency, hallucination, overall_score, comment) (distill_by_prefect)."""
    print("evaluate_flow: stub — implement val/test eval and report")
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    eval_dir = out_dir / "eval" / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "report.json"
    human_path = eval_dir / "human_eval_samples.json"
    json.dump({"val_path": val_path, "test_path": test_path, "adapter_path": adapter_path, "stub": True}, open(report_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"sample_ids": []}, open(human_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return {"report_path": str(report_path), "human_eval_sample_path": str(human_path)}


@flow(name="evaluate_flow", log_prints=True)
def evaluate_flow(
    val_path: str,
    test_path: str,
    adapter_path: str,
    output_dir: str | Path | None = None,
) -> dict:
    """val/test 평가, 휴먼 평가 샘플 뽑기, 리포트 생성."""
    return evaluate_task(
        val_path=val_path,
        test_path=test_path,
        adapter_path=adapter_path,
        output_dir=str(output_dir) if output_dir else None,
    )


@flow(name="distill_pipeline_all", log_prints=True)
def distill_pipeline_all(
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    openai_cap: int = 500,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """build_dataset → labeling → train_student → evaluate 순차 실행."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_flow(train_path=ds["train_path"], openai_cap=openai_cap, output_labeled_dir=out_dir)
    tr = train_student_flow(labeled_path=lb["labeled_path"], student_model=student_model, output_dir=out_dir)
    ev = evaluate_flow(val_path=ds["val_path"], test_path=ds["test_path"], adapter_path=tr["adapter_path"], output_dir=out_dir)
    return {"build_dataset": ds, "labeling": lb, "train_student": tr, "evaluate": ev}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefect flows for summary KD pipeline (distill_by_prefect.md)")
    parser.add_argument("flow", choices=["build_dataset", "all"], help="Flow to run")
    parser.add_argument("--input", type=Path, default=None, help="Input reviews JSON (default: tasteam_app_all_review_data.json)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output root (default: distill_pipeline_output)")
    parser.add_argument("--openai-cap", type=int, default=500, help="OpenAI labeling cap (for 'all')")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Student model (for 'all')")
    args = parser.parse_args()

    if args.flow == "build_dataset":
        result = build_dataset_flow(
            input_path=args.input,
            out_dir=args.out_dir,
        )
        print("Result:", result)
    elif args.flow == "all":
        result = distill_pipeline_all(
            input_path=args.input,
            out_dir=args.out_dir,
            openai_cap=args.openai_cap,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))


if __name__ == "__main__":
    main()
