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
  2. labeling_with_pod_flow — OpenAI 골드 후 Pod에서 teacher 라벨링 + 품질 필터
  3. train_student_with_pod_flow — 학습용 Pod에서 QLoRA SFT
  4. evaluate_flow      — val/test: OpenAI 평가 라벨로 ROUGE/BERTScore/GPT-judge + 휴먼 평가

실행:
  python scripts/distill_flows.py build_dataset [--input path] [--out-dir dir]
  python scripts/distill_flows.py labeling_with_pod --train-path datasets/xxx/train.json
  python scripts/distill_flows.py train_student_with_pod --labeled-path .../train_labeled.json --output-dir ...
  python scripts/distill_flows.py run_sweep [--sweep-id <sweep_id>] --labeled-path .../train_labeled.json [--out-dir ...]
  python scripts/distill_flows.py all        # build_dataset → labeling(Pod) → train(Pod) → evaluate → merge for serving
  python scripts/distill_flows.py all_sweep [--sweep-id <sweep_id>]  # sweep-id 없으면 flow 내부에서 sweep 등록 후 실행
  python scripts/distill_flows.py merge_for_serving --adapter-path .../adapter [--out-dir ...]  # adapter만 merge하여 서빙 경로 생성
  python scripts/distill_flows.py merge_for_serving_with_pod --adapter-path .../adapter  # 로컬 adapter 업로드 후 Pod merge
  python scripts/distill_flows.py merge_for_serving_with_pod --run-id RUN_ID  # 볼륨의 runs/RUN_ID/adapter로 merge
  python scripts/distill_flows.py merge_for_serving_with_pod  # 볼륨에서 최신 학습 adapter로 merge

예시:
  python scripts/distill_flows.py build_dataset --input tasteam_app_all_review_data.json --out-dir distill_pipeline_output
  python scripts/distill_flows.py labeling_with_pod --train-path distill_pipeline_output/datasets/YYYYMMDD_HHMMSS/train.json --out-dir distill_pipeline_output --openai-cap 500
  python scripts/distill_flows.py train_student_with_pod --labeled-path distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/train_labeled.json --output-dir distill_pipeline_output
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
import re
import subprocess
import sys
import tempfile
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
try:
    from runpod_cli.runpod_s3_upload import (
        upload_labeled_dir_to_runpod,
        list_run_ids_with_adapter,
        download_directory_from_runpod,
        upload_file_to_volume,
        upload_directory,
        get_runpod_s3_client,
        object_exists,
        DEFAULT_VOLUME_ID,
    )
except ImportError:
    upload_labeled_dir_to_runpod = None
    list_run_ids_with_adapter = None
    download_directory_from_runpod = None
    upload_file_to_volume = None
    upload_directory = None
    get_runpod_s3_client = None
    object_exists = None
    DEFAULT_VOLUME_ID = "4rlm64f9lv"

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
    OpenAI 골드 먼저(Pod 없이) → Pod 기동 → self-hosted teacher로 나머지 라벨링 → Pod 삭제.
    RUNPOD_API_KEY 필요.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available. Check runpod_cli import.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY environment variable is required for labeling_with_pod")

    out_dir = Path(output_labeled_dir or _PROJECT_ROOT / "distill_pipeline_output")
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    labeled_dir = out_dir / "labeled" / version
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # 1) OpenAI 골드만 먼저 (Pod 없이)
    cmd_gold = [
        sys.executable,
        str(_SCRIPT_DIR / "label_for_distill.py"),
        "--phase", "openai_first",
        "--train-path", str(train_path),
        "--openai-cap", str(openai_cap),
        "--output-dir", str(labeled_dir),
        "--seed", "42",
    ]
    if val_path and Path(val_path).exists():
        cmd_gold.extend(["--val-path", str(val_path)])
    if test_path and Path(test_path).exists():
        cmd_gold.extend(["--test-path", str(test_path)])
    result = subprocess.run(cmd_gold, cwd=str(_PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"label_for_distill.py openai_first exited with {result.returncode}")
    gold_path = labeled_dir / "train_labeled_gold_only.json"
    if not gold_path.exists():
        raise RuntimeError(f"Expected {gold_path} after openai_first")

    # 2) Pod 생성 → vLLM 준비
    client = RunPodClient(token=token)
    payload = RunPodClient.get_default_pod_payload(use="labeling")
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

        # 3) 나머지 teacher 라벨링 + 병합 → train_labeled.json
        cmd_teacher = [
            sys.executable,
            str(_SCRIPT_DIR / "label_for_distill.py"),
            "--phase", "teacher_rest",
            "--train-path", str(train_path),
            "--gold-labeled-path", str(gold_path),
            "--output-dir", str(labeled_dir),
            "--seed", "42",
        ]
        result = subprocess.run(cmd_teacher, cwd=str(_PROJECT_ROOT), env=env, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"label_for_distill.py teacher_rest exited with {result.returncode}")

        labeled_path = labeled_dir / "train_labeled.json"
        out = {"labeled_version": version, "labeled_path": str(labeled_path)}
        for name, fn in [("val_labeled_path", "val_labeled.json"), ("test_labeled_path", "test_labeled.json")]:
            p = labeled_dir / fn
            if p.exists():
                out[name] = str(p)
        # RunPod 네트워크 볼륨 업로드 (RUNPOD_S3_ACCESS_KEY 설정 시)
        if upload_labeled_dir_to_runpod and os.environ.get("RUNPOD_S3_ACCESS_KEY"):
            try:
                n = upload_labeled_dir_to_runpod(labeled_dir)
                out["runpod_upload"] = {"uploaded": True, "count": n, "labeled_dir": str(labeled_dir)}
            except Exception as e:
                logger.warning("RunPod S3 upload failed: %s", e)
                out["runpod_upload"] = {"uploaded": False, "error": str(e)}
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
    OpenAI 골드 먼저(Pod 없이) → Pod 기동 → self-hosted teacher 나머지 → Pod 삭제.
    docs/runpod_cli/cli_strategy.md
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


def _get_train_volume_id() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN", os.environ.get("RUNPOD_NETWORK_VOLUME_ID", DEFAULT_VOLUME_ID))


@task(name="train-student-with-pod-task", log_prints=True)
def train_student_with_pod_task(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    train_poll_interval_sec: int = 90,
    train_timeout_sec: int = 7200,
) -> dict:
    """
    학습용 Pod 생성 → 볼륨에 라벨 업로드 → Pod에서 train_qlora 실행 →
    S3로 완료 감지 후 adapter 다운로드 → Pod 삭제.
    RUNPOD_API_KEY, RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY 필요.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available.")
    if not upload_labeled_dir_to_runpod or not list_run_ids_with_adapter or not download_directory_from_runpod:
        raise RuntimeError("runpod_s3_upload (upload/list/download) required for train_student_with_pod.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY required for train_student_with_pod")
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY required for train_student_with_pod")

    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    labeled_dir = Path(labeled_path).parent
    version = labeled_dir.name
    path_on_volume = f"/workspace/labeled/{version}/train_labeled.json"

    try:
        upload_labeled_dir_to_runpod(labeled_dir, volume_id=_get_train_volume_id())
    except Exception as e:
        raise RuntimeError(f"Upload labeled dir to volume failed: {e}") from e

    vol_id = _get_train_volume_id()
    try:
        existing_runs = set(list_run_ids_with_adapter(vol_id))
    except Exception:
        existing_runs = set()

    docker_start_cmd = [
        "--labeled-path", path_on_volume,
        "--output-dir", "/workspace/distill_pipeline_output",
    ]
    payload = RunPodClient.get_default_pod_payload(use="train", docker_start_cmd=docker_start_cmd)
    client = RunPodClient(token=token)
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    print("Train Pod created:", pod_id)

    try:
        client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        deadline = time.time() + train_timeout_sec
        new_run_id: str | None = None
        while time.time() < deadline:
            try:
                runs = list_run_ids_with_adapter(vol_id)
                for rid in runs:
                    if rid not in existing_runs:
                        new_run_id = rid
                        break
                if new_run_id:
                    break
            except Exception as e:
                logger.debug("Poll runs: %s", e)
            time.sleep(train_poll_interval_sec)
        if not new_run_id:
            raise TimeoutError(f"Train Pod did not produce adapter within {train_timeout_sec}s. Check Pod logs.")

        remote_prefix = f"distill_pipeline_output/runs/{new_run_id}/adapter"
        local_adapter_dir = out_dir / "adapters" / new_run_id / "adapter"
        local_adapter_dir.mkdir(parents=True, exist_ok=True)
        n = download_directory_from_runpod(vol_id, remote_prefix, local_adapter_dir)
        print("Downloaded adapter files:", n)

        return {
            "adapter_path": str(local_adapter_dir),
            "run_id": new_run_id,
            "training_meta_path": str(out_dir / "adapters" / new_run_id / "training_meta.json"),
        }
    finally:
        print("Cleaning up train pod:", pod_id)
        client.delete_pod(pod_id)


@flow(name="train_student_with_pod_flow", log_prints=True)
def train_student_with_pod_flow(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    train_timeout_sec: int = 7200,
) -> dict:
    """학습을 RunPod 학습용 Pod에서만 실행 (볼륨 업로드 → Pod 생성 → 완료 시 adapter 다운로드)."""
    return train_student_with_pod_task(
        labeled_path=labeled_path,
        student_model=student_model,
        output_dir=str(output_dir) if output_dir else None,
        pod_wait_timeout_sec=pod_wait_timeout_sec,
        train_timeout_sec=train_timeout_sec,
    )


@task(name="register-sweep-task", log_prints=True)
def register_sweep_task(sweep_yaml_path: str | Path) -> str:
    """wandb sweep <yaml> 를 subprocess로 실행하고 stdout에서 sweep id를 파싱해 반환."""
    path = Path(sweep_yaml_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"sweep yaml not found: {path}")
    cmd = ["wandb", "sweep", str(path)]
    logger.info("Registering sweep: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"wandb sweep failed (code={result.returncode}): {out}")
    # wandb 출력 예: "Create sweep with ID: entity/project/xxxx" 또는 "Sweep ID: entity/project/xxxx"
    match = re.search(r"(?:sweep with )?ID:\s*(\S+/\S+/\S+)", out, re.IGNORECASE)
    if not match:
        match = re.search(r"(\w+/\w+/[a-zA-Z0-9]+)", out)
    if not match:
        raise RuntimeError(f"Could not parse sweep id from wandb output: {out}")
    sweep_id = match.group(1).strip()
    logger.info("Parsed sweep_id: %s", sweep_id)
    return sweep_id


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


DEFAULT_SWEEP_YAML = _SCRIPT_DIR / "wandb_sweep_qlora.yaml"


@flow(name="run_sweep_flow", log_prints=True)
def run_sweep_flow(
    labeled_path: str,
    sweep_id: str | None = None,
    sweep_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """wandb sweep 에이전트를 subprocess로 실행. sweep_id가 없으면 sweep_yaml으로 먼저 등록 후 실행."""
    if sweep_id is None:
        yaml_path = sweep_yaml if sweep_yaml is not None else DEFAULT_SWEEP_YAML
        sweep_id = register_sweep_task(yaml_path)
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    return run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)


@task(name="get-best-adapter-from-sweep-task", log_prints=True)
def get_best_adapter_path_from_sweep_task(
    sweep_id: str,
    output_dir: str,
    metric_name: str = "train/loss",
) -> str | None:
    """wandb API로 sweep의 best run 조회 후 로컬 adapter 경로 반환 (run name = run_id). 로컬 디스크에 이미 있을 때만 유효."""
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


@task(name="get-best-adapter-from-artifact-task", log_prints=True)
def get_best_adapter_from_artifact_task(
    sweep_id: str,
    download_dir: str | Path,
    metric_name: str = "train/loss",
) -> str | None:
    """
    wandb API로 sweep의 best run 조회 후, 해당 run의 adapter artifact(qlora-adapter-{run_id})를
    다운로드하여 로컬 adapter 경로 반환. 확장성·모듈화용 전용 task (sweep 없이 run_id만으로 호출 시에는 별도 wrapper 사용).
    """
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
    artifact_name = f"qlora-adapter-{run_name}"
    try:
        art = api.artifact(f"{best.entity}/{best.project}/{artifact_name}:latest")
    except Exception as e:
        logger.warning("wandb artifact 조회 실패 (%s): %s", artifact_name, e)
        return None
    root = Path(download_dir) / "artifacts" / run_name
    root.mkdir(parents=True, exist_ok=True)
    art.download(root=str(root))
    adapter_path = root / "adapter"
    if not adapter_path.is_dir():
        logger.warning("artifact 내 adapter 디렉터리가 없음: %s", adapter_path)
        return None
    logger.info("sweep best run: %s, adapter_path=%s (from artifact)", run_name, adapter_path)
    return str(adapter_path)


@flow(name="run_sweep_and_evaluate_flow", log_prints=True)
def run_sweep_and_evaluate_flow(
    labeled_path: str,
    sweep_id: str | None = None,
    sweep_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """sweep subprocess 실행 → best run adapter를 wandb artifact에서 다운로드 → evaluate 실행. sweep_id 없으면 sweep_yaml으로 등록 후 실행."""
    if sweep_id is None:
        yaml_path = sweep_yaml if sweep_yaml is not None else DEFAULT_SWEEP_YAML
        sweep_id = register_sweep_task(yaml_path)
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)
    best_adapter = get_best_adapter_from_artifact_task(
        sweep_id=sweep_id,
        download_dir=out_dir,
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


@task(name="merge-adapter-for-serving-task", log_prints=True)
def merge_adapter_for_serving_task(
    adapter_path: str,
    base_model: str,
    output_dir: str | Path,
    merge_subdir: str = "merged_for_serving",
) -> dict:
    """Adapter + base를 merge하여 서빙용 단일 디렉터리로 저장. merge_adapter_for_serving.py subprocess."""
    out_dir = Path(output_dir)
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    merged_dir = out_dir / merge_subdir / version
    merged_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "merge_adapter_for_serving.py"),
        "--adapter-path", str(adapter_path),
        "--base-model", base_model,
        "--output-dir", str(merged_dir),
    ]
    logger.info("Merge adapter for serving: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"merge_adapter_for_serving.py exited with {result.returncode}\n{result.stderr or ''}")
    merged_path = merged_dir.resolve()
    pointer_dir = out_dir / merge_subdir
    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer = {
        "merged_model_path": str(merged_path),
        "merged_at": version,
        "adapter_path": str(adapter_path),
        "base_model": base_model,
    }
    pointer_path = pointer_dir / "latest_merged_path.json"
    with open(pointer_path, "w", encoding="utf-8") as f:
        json.dump(pointer, f, ensure_ascii=False, indent=2)
    logger.info("Wrote serving pointer: %s -> %s", pointer_path, merged_path)
    return {"merged_model_path": str(merged_path), "pointer_path": str(pointer_path), "pointer": pointer}


@flow(name="merge_for_serving_flow", log_prints=True)
def merge_for_serving_flow(
    adapter_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | Path | None = None,
    merge_subdir: str = "merged_for_serving",
) -> dict:
    """Adapter를 base와 merge하여 서빙용 경로로 저장하고, latest_merged_path.json 포인터 기록. API는 LLM_MODEL=<merged_model_path> 사용."""
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    return merge_adapter_for_serving_task(
        adapter_path=adapter_path,
        base_model=base_model,
        output_dir=out_dir,
        merge_subdir=merge_subdir,
    )


@task(name="merge-adapter-with-pod-task", log_prints=True)
def merge_adapter_with_pod_task(
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str | None = None,
    run_id: str | None = None,
    volume_id: str | None = None,
    merge_poll_interval_sec: int = 30,
    merge_timeout_sec: int = 1800,
    pod_wait_timeout_sec: int = 600,
) -> dict:
    """
    볼륨이 마운트된 Pod에서 merge 스크립트 실행 → 결과를 볼륨에 직접 저장.
    - run_id 지정 시: 볼륨에 이미 있는 runs/<run_id>/adapter 사용 (로컬 업로드 없음).
    - adapter_path 지정 시: 로컬 adapter 디렉터리를 볼륨에 업로드 후 Pod에서 merge 실행.
    - 둘 다 미지정 시: 볼륨에서 adapter가 있는 run 중 최신(run_id 타임스탬프 역순 첫 번째) 사용.
    완료 시 볼륨에 merged_for_serving/<version>/ 및 latest_merged_path.json 기록.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available.")
    if not upload_file_to_volume or not upload_directory or not get_runpod_s3_client or not object_exists:
        raise RuntimeError("runpod_s3_upload (upload_file_to_volume, upload_directory, object_exists) required.")
    if list_run_ids_with_adapter is None:
        raise RuntimeError("list_run_ids_with_adapter required for merge_adapter_with_pod (latest run fallback).")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY required for merge_adapter_with_pod")
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY required.")

    vol_id = volume_id or _get_train_volume_id()
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if run_id and adapter_path:
        raise ValueError("Provide either run_id or adapter_path, not both.")
    if not run_id and not adapter_path:
        # 볼륨에서 adapter가 있는 run 중 최신 사용
        run_ids = list_run_ids_with_adapter(vol_id)
        if not run_ids:
            raise ValueError("No run with adapter found on volume. Train first or provide --adapter-path or --run-id.")
        run_id = run_ids[0]
        logger.info("Using latest run on volume: run_id=%s", run_id)

    client = get_runpod_s3_client()

    if run_id:
        # 볼륨에 이미 있는 runs/<run_id>/adapter 사용
        adapter_on_volume = f"/workspace/distill_pipeline_output/runs/{run_id}/adapter"
        logger.info("Using adapter on volume: %s", adapter_on_volume)
    else:
        # 로컬 adapter를 볼륨에 업로드
        adapter_dir = Path(adapter_path).resolve()
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"adapter path not a directory: {adapter_dir}")
        adapter_prefix = f"distill_pipeline_output/merge_input/{version}/adapter"
        logger.info("Uploading adapter to volume prefix %s", adapter_prefix)
        upload_directory(client, vol_id, adapter_dir, adapter_prefix)
        adapter_on_volume = f"/workspace/distill_pipeline_output/merge_input/{version}/adapter"

    # Merge 스크립트를 볼륨에 업로드
    merge_script = _SCRIPT_DIR / "merge_adapter_for_serving.py"
    if not merge_script.is_file():
        raise FileNotFoundError(f"merge script not found: {merge_script}")
    upload_file_to_volume(merge_script, volume_id=vol_id, object_key="distill_pipeline_output/merge_scripts/merge_adapter_for_serving.py")

    # Pod에서 merge 실행 (볼륨 경로: /workspace)
    output_on_volume = f"/workspace/distill_pipeline_output/merged_for_serving/{version}"
    docker_start_cmd = [
        "/workspace/distill_pipeline_output/merge_scripts/merge_adapter_for_serving.py",
        "--adapter-path", adapter_on_volume,
        "--base-model", base_model,
        "--output-dir", output_on_volume,
    ]
    payload = RunPodClient.get_default_pod_payload(use="merge", docker_start_cmd=docker_start_cmd)
    runpod_client = RunPodClient(token=token)
    pod = runpod_client.create_pod(payload)
    pod_id = pod["id"]
    logger.info("Merge Pod created: %s", pod_id)

    try:
        runpod_client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        meta_key = f"distill_pipeline_output/merged_for_serving/{version}/merge_meta.json"
        deadline = time.time() + merge_timeout_sec
        while time.time() < deadline:
            if object_exists(client, vol_id, meta_key):
                logger.info("Merge completed: %s", meta_key)
                break
            time.sleep(merge_poll_interval_sec)
        else:
            raise TimeoutError(f"Merge did not produce merge_meta.json within {merge_timeout_sec}s. Check Pod logs.")

        # 볼륨에 latest_merged_path.json 기록 (추론 Pod가 읽을 수 있도록)
        pointer = {
            "merged_model_path": output_on_volume,
            "merged_at": version,
            "adapter_path": adapter_on_volume,
            "base_model": base_model,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(pointer, f, ensure_ascii=False, indent=2)
            tmp_path = f.name
        try:
            upload_file_to_volume(tmp_path, volume_id=vol_id, object_key="distill_pipeline_output/merged_for_serving/latest_merged_path.json")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return {
            "merged_model_path_on_volume": output_on_volume,
            "pointer_path_on_volume": "/workspace/distill_pipeline_output/merged_for_serving/latest_merged_path.json",
            "version": version,
            "run_id_used": run_id,
            "pointer": pointer,
        }
    finally:
        logger.info("Cleaning up merge pod: %s", pod_id)
        runpod_client.delete_pod(pod_id)


@flow(name="merge_for_serving_with_pod_flow", log_prints=True)
def merge_for_serving_with_pod_flow(
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path: str | None = None,
    run_id: str | None = None,
    volume_id: str | None = None,
    merge_timeout_sec: int = 1800,
) -> dict:
    """볼륨이 마운트된 Pod에서 adapter+base merge 실행 후 결과를 볼륨에 저장.
    run_id: 볼륨에 이미 있는 학습 run (runs/<run_id>/adapter 사용). adapter_path와 둘 중 하나만 지정.
    adapter_path: 로컬 adapter 디렉터리 (볼륨에 업로드 후 merge).
    둘 다 미지정 시: 볼륨에서 adapter가 있는 run 중 최신 사용."""
    return merge_adapter_with_pod_task(
        base_model=base_model,
        adapter_path=adapter_path,
        run_id=run_id,
        volume_id=volume_id,
        merge_timeout_sec=merge_timeout_sec,
    )


@flow(name="distill_pipeline_all", log_prints=True)
def distill_pipeline_all(
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    openai_cap: int = 500,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """build_dataset → labeling(Pod) → train_student(Pod) → evaluate 순차 실행."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_with_pod_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
    )
    tr = train_student_with_pod_flow(
        labeled_path=lb["labeled_path"],
        student_model=student_model,
        output_dir=out_dir,
    )
    ev = evaluate_flow(
        adapter_path=tr["adapter_path"],
        val_labeled_path=lb.get("val_labeled_path"),
        test_labeled_path=lb.get("test_labeled_path"),
        output_dir=out_dir,
    )
    merge_result = merge_for_serving_flow(
        adapter_path=tr["adapter_path"],
        base_model=student_model,
        output_dir=out_dir,
    )
    return {"build_dataset": ds, "labeling": lb, "train_student": tr, "evaluate": ev, "merge_for_serving": merge_result}


@flow(name="distill_pipeline_all_sweep", log_prints=True)
def distill_pipeline_all_sweep(
    sweep_id: str | None = None,
    sweep_yaml: str | Path | None = None,
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    openai_cap: int = 500,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """build_dataset → labeling(Pod) → run_sweep → best run adapter로 evaluate. sweep_id 없으면 sweep_yaml으로 등록 후 실행."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_with_pod_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
    )
    sweep_ev = run_sweep_and_evaluate_flow(
        sweep_id=sweep_id,
        sweep_yaml=sweep_yaml,
        labeled_path=lb["labeled_path"],
        output_dir=out_dir,
        val_labeled_path=lb.get("val_labeled_path"),
        test_labeled_path=lb.get("test_labeled_path"),
        base_model=student_model,
    )
    merge_result = None
    if sweep_ev.get("best_adapter_path"):
        merge_result = merge_for_serving_flow(
            adapter_path=sweep_ev["best_adapter_path"],
            base_model=student_model,
            output_dir=out_dir,
        )
    return {"build_dataset": ds, "labeling": lb, "run_sweep_and_evaluate": sweep_ev, "merge_for_serving": merge_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefect flows for summary KD pipeline (distill_by_prefect.md)")
    parser.add_argument(
        "flow",
        choices=["build_dataset", "labeling_with_pod", "train_student_with_pod", "run_sweep", "evaluate", "merge_for_serving", "merge_for_serving_with_pod", "all", "all_sweep"],
        help="Flow to run. all/all_sweep: 라벨링·학습 Pod 기준. merge_for_serving_with_pod: Pod에서 merge 후 볼륨에 저장.",
    )
    parser.add_argument("--sweep-id", type=str, default=None, help="wandb sweep id (optional; 없으면 --sweep-yaml으로 flow 내부에서 등록)")
    parser.add_argument("--sweep-yaml", type=Path, default=None, help="sweep 설정 yaml (sweep-id 없을 때 사용, 기본: scripts/wandb_sweep_qlora.yaml)")
    parser.add_argument("--input", type=Path, default=None, help="Input reviews JSON (default: tasteam_app_all_review_data.json)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output root (default: distill_pipeline_output)")
    parser.add_argument("--train-path", type=Path, default=None, help="train.json (for labeling_with_pod)")
    parser.add_argument("--val-path", type=Path, default=None, help="val.json (for labeling_with_pod)")
    parser.add_argument("--test-path", type=Path, default=None, help="test.json (for labeling_with_pod)")
    parser.add_argument("--labeled-path", type=Path, default=None, help="train_labeled.json (for train_student_with_pod, run_sweep)")
    parser.add_argument("--adapter-path", type=Path, default=None, help="adapter path (for evaluate, merge_for_serving; merge_for_serving_with_pod 시 로컬 adapter)")
    parser.add_argument("--run-id", type=str, default=None, help="볼륨 상의 학습 run id (merge_for_serving_with_pod 시 --adapter-path 대신 사용)")
    parser.add_argument("--val-labeled-path", type=Path, default=None, help="val_labeled.json (for evaluate)")
    parser.add_argument("--test-labeled-path", type=Path, default=None, help="test_labeled.json (for evaluate)")
    parser.add_argument("--openai-cap", type=int, default=500, help="OpenAI labeling cap (for labeling_with_pod/all)")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Student model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or _PROJECT_ROOT / "distill_pipeline_output")

    if args.flow == "build_dataset":
        result = build_dataset_flow(input_path=args.input, out_dir=out_dir)
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
    elif args.flow == "train_student_with_pod":
        if not args.labeled_path:
            parser.error("train_student_with_pod requires --labeled-path")
        result = train_student_with_pod_flow(
            labeled_path=str(args.labeled_path),
            student_model=args.student_model,
            output_dir=out_dir,
        )
        print("Result:", result)
    elif args.flow == "run_sweep":
        if not args.labeled_path:
            parser.error("run_sweep requires --labeled-path")
        sweep_yaml = args.sweep_yaml if args.sweep_yaml is not None else DEFAULT_SWEEP_YAML
        result = run_sweep_flow(
            sweep_id=args.sweep_id,
            sweep_yaml=sweep_yaml,
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
    elif args.flow == "merge_for_serving":
        if not args.adapter_path:
            parser.error("merge_for_serving requires --adapter-path")
        result = merge_for_serving_flow(
            adapter_path=str(args.adapter_path),
            base_model=args.student_model,
            output_dir=out_dir,
        )
        print("Result:", result)
        print("API 사용: LLM_MODEL=" + result["merged_model_path"])
    elif args.flow == "merge_for_serving_with_pod":
        if args.run_id and args.adapter_path:
            parser.error("merge_for_serving_with_pod: use either --run-id or --adapter-path, not both")
        result = merge_for_serving_with_pod_flow(
            base_model=args.student_model,
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
            run_id=args.run_id,
        )
        print("Result:", result)
        print("추론 Pod에서: LLM_MODEL=" + result["merged_model_path_on_volume"])
        print("또는 볼륨의", result["pointer_path_on_volume"], "에서 merged_model_path 읽기")
    elif args.flow == "all":
        result = distill_pipeline_all(
            input_path=args.input,
            out_dir=out_dir,
            openai_cap=args.openai_cap,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))
    elif args.flow == "all_sweep":
        sweep_yaml = args.sweep_yaml if args.sweep_yaml is not None else DEFAULT_SWEEP_YAML
        result = distill_pipeline_all_sweep(
            sweep_id=args.sweep_id,
            sweep_yaml=sweep_yaml,
            input_path=args.input,
            out_dir=out_dir,
            openai_cap=args.openai_cap,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))


if __name__ == "__main__":
    main()
