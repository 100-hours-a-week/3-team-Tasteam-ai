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
  python scripts/distill_flows.py labeling --train-path datasets/xxx/train.json
  python scripts/distill_flows.py labeling_with_pod --train-path datasets/xxx/train.json  # Pod 생성→라벨링→삭제
  python scripts/distill_flows.py train_student --labeled-path .../train_labeled.json --output-dir ...
  python scripts/distill_flows.py run_sweep --sweep-id <sweep_id> --labeled-path .../train_labeled.json [--out-dir ...]  # wandb sweep (subprocess)
  python scripts/distill_flows.py all        # build_dataset → labeling → 단일 train_student → evaluate
  python scripts/distill_flows.py all_sweep --sweep-id <sweep_id>  # build_dataset → labeling → sweep → best run으로 evaluate

예시:
데이터셋 생성(아직 없다면)
  python scripts/distill_flows.py build_dataset --input tasteam_app_all_review_data.json --out-dir distill_pipeline_output
라벨링 실행
  python scripts/distill_flows.py labeling --train-path distill_pipeline_output/datasets/YYYYMMDD_HHMMSS/train.json --out-dir distill_pipeline_output --openai-cap 500
학습 실행
  python scripts/distill_flows.py train_student --labeled-path distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/train_labeled.json --output-dir distill_pipeline_output
평가 실행
  python scripts/distill_flows.py evaluate --adapter-path distill_pipeline_output/adapters/YYYYMMDD_HHMMSS/adapter.safetensors --output-dir distill_pipeline_output --val-labeled-path distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/val_labeled.json --test-labeled-path distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/test_labeled.json
  
전체 파이프라인 한번에 실행
  python scripts/distill_flows.py all --out-dir distill_pipeline_output --openai-cap 500
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
import logging
import os
import subprocess
import sys
import time
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

# RunPodClient import (scripts/runpod_cli)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
try:
    from runpod_cli.pod_create_delete_cli import RunPodClient
except ImportError:
    RunPodClient = None

logger = logging.getLogger(__name__)


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
    val_path: str | None = None,
    test_path: str | None = None,
    openai_cap: int = 500,
    output_labeled_dir: str | None = None,
) -> dict:
    """label_for_distill.py subprocess: OpenAI 골드 + self-hosted teacher + 품질 필터 (distill_strategy.md)."""
    out_dir = Path(output_labeled_dir or _PROJECT_ROOT / "distill_pipeline_output")
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    labeled_dir = out_dir / "labeled" / version
    labeled_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "label_for_distill.py"),
        "--train-path", str(train_path),
        "--openai-cap", str(openai_cap),
        "--output-dir", str(labeled_dir),
    ]
    if val_path and Path(val_path).exists():
        cmd.extend(["--val-path", str(val_path)])
    if test_path and Path(test_path).exists():
        cmd.extend(["--test-path", str(test_path)])

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"label_for_distill.py exited with {result.returncode}")

    labeled_path = labeled_dir / "train_labeled.json"
    out = {"labeled_version": version, "labeled_path": str(labeled_path)}
    val_lbl = labeled_dir / "val_labeled.json"
    test_lbl = labeled_dir / "test_labeled.json"
    if val_lbl.exists():
        out["val_labeled_path"] = str(val_lbl)
    if test_lbl.exists():
        out["test_labeled_path"] = str(test_lbl)
    return out


@flow(name="labeling_flow", log_prints=True)
def labeling_flow(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    openai_cap: int = 500,
    output_labeled_dir: str | Path | None = None,
) -> dict:
    """OpenAI 골드 라벨링(cap, 전략 300~800) + self-hosted teacher 나머지 + 품질 필터/dedup (distill_strategy.md)."""
    return labeling_task(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        openai_cap=openai_cap,
        output_labeled_dir=str(output_labeled_dir) if output_labeled_dir else None,
    )


def _wait_for_vllm_ready(base_url: str, timeout_sec: int = 180, poll_interval: int = 10) -> None:
    """vLLM /v1/models 가 응답할 때까지 대기."""
    import requests
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                logger.info("vLLM ready: %s", url)
                return
        except Exception as e:
            logger.debug("vLLM not ready yet: %s", e)
        time.sleep(poll_interval)
    raise TimeoutError(f"vLLM at {base_url} did not become ready within {timeout_sec}s")


@task(name="labeling-with-pod-task", log_prints=True)
def labeling_with_pod_task(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    openai_cap: int = 500,
    output_labeled_dir: str | None = None,
    pod_wait_timeout_sec: int = 600,
    vllm_ready_timeout_sec: int = 180,
) -> dict:
    """
    Pod 생성 → vLLM 준비 대기 → 라벨링 → Pod 삭제.
    RUNPOD_API_KEY 필요. self-hosted teacher용 vLLM Pod를 생성 후 label_for_distill 실행.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available. Check runpod_cli import.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY environment variable is required for labeling_with_pod")

    client = RunPodClient(token=token)
    payload = RunPodClient.get_default_pod_payload()
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    print("Pod created:", pod_id)

    try:
        ready = client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        public_ip = ready.get("publicIp")
        if not public_ip:
            raise RuntimeError(f"Pod {pod_id} has no publicIp. Response: {ready}")

        base_url = f"http://{public_ip}:8000/v1"
        print("Pod ready:", pod_id, "base_url:", base_url)

        _wait_for_vllm_ready(base_url, timeout_sec=vllm_ready_timeout_sec)

        env = os.environ.copy()
        env["VLLM_POD_BASE_URL"] = base_url
        env["USE_POD_VLLM"] = "true"
        env["LLM_PROVIDER"] = "runpod"

        out_dir = Path(output_labeled_dir or _PROJECT_ROOT / "distill_pipeline_output")
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        labeled_dir = out_dir / "labeled" / version
        labeled_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "label_for_distill.py"),
            "--train-path", str(train_path),
            "--openai-cap", str(openai_cap),
            "--output-dir", str(labeled_dir),
        ]
        if val_path and Path(val_path).exists():
            cmd.extend(["--val-path", str(val_path)])
        if test_path and Path(test_path).exists():
            cmd.extend(["--test-path", str(test_path)])

        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"label_for_distill.py exited with {result.returncode}")

        labeled_path = labeled_dir / "train_labeled.json"
        out = {"labeled_version": version, "labeled_path": str(labeled_path)}
        for name, fn in [("val_labeled_path", "val_labeled.json"), ("test_labeled_path", "test_labeled.json")]:
            p = labeled_dir / fn
            if p.exists():
                out[name] = str(p)
        return out
    finally:
        print("Cleaning up pod:", pod_id)
        client.delete_pod(pod_id)


@flow(name="labeling_with_pod_flow", log_prints=True)
def labeling_with_pod_flow(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    openai_cap: int = 500,
    output_labeled_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    vllm_ready_timeout_sec: int = 180,
) -> dict:
    """
    Pod 생성 → 라벨링(OpenAI 골드 + vLLM teacher) → Pod 삭제.
    docs/runpod_cli/cli_strategy.md: "Pod 생성 → 대기 → VLLM_POD_BASE_URL 설정 후 label_for_distill 실행 → 작업 완료 후 Pod 삭제"
    """
    return labeling_with_pod_task(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        openai_cap=openai_cap,
        output_labeled_dir=str(output_labeled_dir) if output_labeled_dir else None,
        pod_wait_timeout_sec=pod_wait_timeout_sec,
        vllm_ready_timeout_sec=vllm_ready_timeout_sec,
    )


@task(name="train-student-task", log_prints=True)
def train_student_task(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | None = None,
    gold_oversample_ratio: float = GOLD_OVERSAMPLE_RATIO,
) -> dict:
    """train_qlora.py subprocess: QLoRA SFT, report_to=wandb (kd_qlora_prefect_wandb_strategy.md)."""
    import os
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    env = os.environ.copy()
    try:
        ctx = get_run_context()
        if hasattr(ctx, "flow_run") and ctx.flow_run:
            flow_run_id = str(ctx.flow_run.id)
            env["WANDB_RUN_ID"] = flow_run_id
            env["PREFECT_FLOW_RUN_ID"] = flow_run_id
    except Exception:
        pass

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "train_qlora.py"),
        "--labeled-path", str(labeled_path),
        "--student-model", student_model,
        "--output-dir", str(out_dir),
        "--gold-oversample-ratio", str(gold_oversample_ratio),
    ]
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"train_qlora.py exited with {result.returncode}\n{result.stderr or ''}")

    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            out = json.loads(line)
            return {
                "adapter_path": out["adapter_path"],
                "training_meta_path": out["training_meta_path"],
                "run_id": out["run_id"],
            }
    # fallback: infer from output dir
    runs_dir = out_dir / "runs"
    if runs_dir.exists():
        subdirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if subdirs:
            run_dir = subdirs[0]
            return {
                "adapter_path": str(run_dir / "adapter"),
                "training_meta_path": str(run_dir / "training_meta.json"),
                "run_id": run_dir.name,
            }
    raise RuntimeError("train_qlora.py did not produce expected output")


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


@task(name="run-sweep-agent-task", log_prints=True)
def run_sweep_agent_task(
    sweep_id: str,
    labeled_path: str,
    output_dir: str,
) -> dict:
    """run_qlora_sweep.py를 subprocess로 실행. sweep 전체가 끝날 때까지 대기 (프로세스 격리)."""
    env = os.environ.copy()
    env["WANDB_SWEEP_LABELED_PATH"] = labeled_path
    env["WANDB_SWEEP_OUTPUT_DIR"] = output_dir
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_qlora_sweep.py"),
        sweep_id,
    ]
    logger.info("Running sweep agent (subprocess): %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"run_qlora_sweep.py exited with {result.returncode}")
    return {"sweep_id": sweep_id, "labeled_path": labeled_path, "output_dir": output_dir}


@flow(name="run_sweep_flow", log_prints=True)
def run_sweep_flow(
    sweep_id: str,
    labeled_path: str,
    output_dir: str | Path | None = None,
) -> dict:
    """wandb sweep 에이전트를 subprocess로 실행 (프로세스 격리, docs/wandb/wandb_subprocess_oneprocess.md)."""
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    return run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)


@task(name="get-best-adapter-from-sweep-task", log_prints=True)
def get_best_adapter_path_from_sweep_task(
    sweep_id: str,
    output_dir: str,
    metric_name: str = "train/loss",
) -> str | None:
    """wandb API로 sweep의 best run 조회 후 로컬 adapter 경로 반환 (run name = run_id)."""
    try:
        import wandb
        api = wandb.Api()
        sweep = api.sweep(sweep_id)
        runs = list(sweep.runs)
    except Exception as e:
        logger.warning("wandb sweep best run 조회 실패: %s", e)
        return None
    runs_with_metric = [r for r in runs if r.summary.get(metric_name) is not None]
    if not runs_with_metric:
        logger.warning("sweep에 메트릭 %s가 있는 run이 없음", metric_name)
        return None
    best = min(runs_with_metric, key=lambda r: float(r.summary[metric_name]))
    run_name = best.name or getattr(best, "id", None)
    if not run_name:
        logger.warning("best run에 name 없음")
        return None
    adapter_path = str(Path(output_dir) / "runs" / run_name / "adapter")
    if not Path(adapter_path).exists():
        logger.warning("best run adapter 경로가 없음: %s", adapter_path)
        return None
    logger.info("sweep best run: %s, adapter_path=%s", run_name, adapter_path)
    return adapter_path


@flow(name="run_sweep_and_evaluate_flow", log_prints=True)
def run_sweep_and_evaluate_flow(
    sweep_id: str,
    labeled_path: str,
    output_dir: str | Path | None = None,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """sweep subprocess 실행 → best run adapter 경로 조회 → evaluate 실행."""
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)
    best_adapter = get_best_adapter_path_from_sweep_task(
        sweep_id=sweep_id,
        output_dir=out_dir,
        metric_name="train/loss",
    )
    if not best_adapter:
        return {"sweep_id": sweep_id, "best_adapter_path": None, "evaluate": None}
    ev = evaluate_flow(
        adapter_path=best_adapter,
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        output_dir=out_dir,
        base_model=base_model,
    )
    return {"sweep_id": sweep_id, "best_adapter_path": best_adapter, "evaluate": ev}


@task(name="evaluate-task", log_prints=True)
def evaluate_task(
    adapter_path: str,
    output_dir: str | None = None,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """eval_distill.py subprocess: ROUGE/BERTScore on val/test labeled (OpenAI ground truth)."""
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    if not val_labeled_path and not test_labeled_path:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        eval_dir = out_dir / "eval" / run_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        report_path = eval_dir / "report.json"
        human_path = eval_dir / "human_eval_samples.json"
        json.dump(
            {"skipped": True, "reason": "no val/test labeled paths"},
            open(report_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        json.dump({"sample_ids": []}, open(human_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return {"report_path": str(report_path), "human_eval_sample_path": str(human_path)}

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "eval_distill.py"),
        "--adapter-path", str(adapter_path),
        "--base-model", base_model,
        "--output-dir", str(out_dir),
    ]
    if val_labeled_path and Path(val_labeled_path).exists():
        cmd.extend(["--val-labeled", str(val_labeled_path)])
    if test_labeled_path and Path(test_labeled_path).exists():
        cmd.extend(["--test-labeled", str(test_labeled_path)])

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"eval_distill.py exited with {result.returncode}\n{result.stderr or ''}")

    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            out = json.loads(line)
            return {"report_path": out["report_path"], "human_eval_sample_path": out["human_eval_sample_path"]}
    raise RuntimeError("eval_distill.py did not produce expected output")


@flow(name="evaluate_flow", log_prints=True)
def evaluate_flow(
    adapter_path: str,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    output_dir: str | Path | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """val/test 평가 (ROUGE/BERTScore), 휴먼 평가 샘플 뽑기."""
    return evaluate_task(
        adapter_path=adapter_path,
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        output_dir=str(output_dir) if output_dir else None,
        base_model=base_model,
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
    lb = labeling_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
    )
    tr = train_student_flow(labeled_path=lb["labeled_path"], student_model=student_model, output_dir=out_dir)
    ev = evaluate_flow(
        adapter_path=tr["adapter_path"],
        val_labeled_path=lb.get("val_labeled_path"),
        test_labeled_path=lb.get("test_labeled_path"),
        output_dir=out_dir,
    )
    return {"build_dataset": ds, "labeling": lb, "train_student": tr, "evaluate": ev}


@flow(name="distill_pipeline_all_sweep", log_prints=True)
def distill_pipeline_all_sweep(
    sweep_id: str,
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    openai_cap: int = 500,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """build_dataset → labeling → run_sweep (subprocess) → best run adapter로 evaluate 순차 실행."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
    )
    sweep_ev = run_sweep_and_evaluate_flow(
        sweep_id=sweep_id,
        labeled_path=lb["labeled_path"],
        output_dir=out_dir,
        val_labeled_path=lb.get("val_labeled_path"),
        test_labeled_path=lb.get("test_labeled_path"),
        base_model=student_model,
    )
    return {"build_dataset": ds, "labeling": lb, "run_sweep_and_evaluate": sweep_ev}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefect flows for summary KD pipeline (distill_by_prefect.md)")
    parser.add_argument(
        "flow",
        choices=["build_dataset", "labeling", "labeling_with_pod", "train_student", "run_sweep", "evaluate", "all", "all_sweep"],
        help="Flow to run (run_sweep: wandb sweep subprocess, all_sweep: build_dataset→labeling→sweep→best run으로 evaluate)",
    )
    parser.add_argument("--sweep-id", type=str, default=None, help="wandb sweep id (for run_sweep)")
    parser.add_argument("--input", type=Path, default=None, help="Input reviews JSON (default: tasteam_app_all_review_data.json)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output root (default: distill_pipeline_output)")
    parser.add_argument("--train-path", type=Path, default=None, help="train.json (for labeling)")
    parser.add_argument("--val-path", type=Path, default=None, help="val.json (for labeling)")
    parser.add_argument("--test-path", type=Path, default=None, help="test.json (for labeling)")
    parser.add_argument("--labeled-path", type=Path, default=None, help="train_labeled.json (for train_student)")
    parser.add_argument("--adapter-path", type=Path, default=None, help="adapter path (for evaluate)")
    parser.add_argument("--val-labeled-path", type=Path, default=None, help="val_labeled.json (for evaluate)")
    parser.add_argument("--test-labeled-path", type=Path, default=None, help="test_labeled.json (for evaluate)")
    parser.add_argument("--openai-cap", type=int, default=500, help="OpenAI labeling cap (for labeling/all)")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Student model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or _PROJECT_ROOT / "distill_pipeline_output")

    if args.flow == "build_dataset":
        result = build_dataset_flow(input_path=args.input, out_dir=out_dir)
        print("Result:", result)
    elif args.flow == "labeling":
        if not args.train_path:
            parser.error("labeling requires --train-path")
        ds_dir = out_dir / "datasets"
        val_p, test_p = None, None
        if args.val_path:
            val_p = str(args.val_path)
        elif ds_dir.exists():
            for d in sorted(ds_dir.iterdir(), reverse=True):
                v = d / "val.json"
                if v.exists():
                    val_p = str(v)
                    break
        if args.test_path:
            test_p = str(args.test_path)
        elif ds_dir.exists():
            for d in sorted(ds_dir.iterdir(), reverse=True):
                t = d / "test.json"
                if t.exists():
                    test_p = str(t)
                    break
        result = labeling_flow(
            train_path=str(args.train_path),
            val_path=val_p,
            test_path=test_p,
            openai_cap=args.openai_cap,
            output_labeled_dir=out_dir,
        )
        print("Result:", result)
    elif args.flow == "labeling_with_pod":
        if not args.train_path:
            parser.error("labeling_with_pod requires --train-path")
        ds_dir = out_dir / "datasets"
        val_p, test_p = None, None
        if args.val_path:
            val_p = str(args.val_path)
        elif ds_dir.exists():
            for d in sorted(ds_dir.iterdir(), reverse=True):
                v = d / "val.json"
                if v.exists():
                    val_p = str(v)
                    break
        if args.test_path:
            test_p = str(args.test_path)
        elif ds_dir.exists():
            for d in sorted(ds_dir.iterdir(), reverse=True):
                t = d / "test.json"
                if t.exists():
                    test_p = str(t)
                    break
        result = labeling_with_pod_flow(
            train_path=str(args.train_path),
            val_path=val_p,
            test_path=test_p,
            openai_cap=args.openai_cap,
            output_labeled_dir=out_dir,
        )
        print("Result:", result)
    elif args.flow == "train_student":
        if not args.labeled_path:
            parser.error("train_student requires --labeled-path")
        result = train_student_flow(
            labeled_path=str(args.labeled_path),
            student_model=args.student_model,
            output_dir=out_dir,
        )
        print("Result:", result)
    elif args.flow == "run_sweep":
        if not args.sweep_id:
            parser.error("run_sweep requires --sweep-id")
        if not args.labeled_path:
            parser.error("run_sweep requires --labeled-path")
        result = run_sweep_flow(
            sweep_id=args.sweep_id,
            labeled_path=str(args.labeled_path),
            output_dir=out_dir,
        )
        print("Result:", result)
    elif args.flow == "evaluate":
        if not args.adapter_path:
            parser.error("evaluate requires --adapter-path")
        result = evaluate_flow(
            adapter_path=str(args.adapter_path),
            val_labeled_path=str(args.val_labeled_path) if args.val_labeled_path else None,
            test_labeled_path=str(args.test_labeled_path) if args.test_labeled_path else None,
            output_dir=out_dir,
            base_model=args.student_model,
        )
        print("Result:", result)
    elif args.flow == "all":
        result = distill_pipeline_all(
            input_path=args.input,
            out_dir=out_dir,
            openai_cap=args.openai_cap,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))
    elif args.flow == "all_sweep":
        if not args.sweep_id:
            parser.error("all_sweep requires --sweep-id")
        result = distill_pipeline_all_sweep(
            sweep_id=args.sweep_id,
            input_path=args.input,
            out_dir=out_dir,
            openai_cap=args.openai_cap,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))


if __name__ == "__main__":
    main()
