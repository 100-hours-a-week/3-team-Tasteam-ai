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
  2. labeling_openai_only — 전부 OpenAI(gpt-4o-mini)로 라벨링, Pod 미사용
  3. labeling_with_pod — OpenAI 골드 후 Pod에서 teacher 라벨링 + 품질 필터
  4. train_student_with_pod_flow — 학습용 Pod에서 QLoRA SFT
  5. evaluate_flow      — val/test: OpenAI 평가 라벨로 ROUGE/BERTScore/GPT-judge + 휴먼 평가

실행:
  python scripts/distill_flows.py build_dataset [--input path] [--out-dir dir]
  python scripts/distill_flows.py labeling_openai_only --train-path datasets/xxx/train.json  # 전부 OpenAI
  python scripts/distill_flows.py labeling_with_pod --train-path datasets/xxx/train.json  # OpenAI 골드 + Pod teacher
  python scripts/distill_flows.py labeling_pod_only --train-path .../train.json --gold-path .../train_labeled_gold_only.json  # OpenAI 생략, Pod teacher만
  python scripts/distill_flows.py train_student_with_pod --labeled-path .../train_labeled.json --output-dir ...
  python scripts/distill_flows.py run_sweep [--sweep-id <sweep_id>] --labeled-path .../train_labeled.json [--out-dir ...]
  python scripts/distill_flows.py train_and_evaluate --labeled-path .../train_labeled.json [--val-labeled-path ...] [--test-labeled-path ...]  # 학습(Pod) → 평가만
  python scripts/distill_flows.py sweep_eval_merge --labeled-path .../train_labeled.json [--sweep-id ...]  # Pod sweep → evaluate → merge (build_dataset/labeling 생략)
  python scripts/distill_flows.py upload_labeled_artifact --labeled-path .../train_labeled.json  # 기존 labeled만 wandb artifact로 업로드
  python scripts/distill_flows.py upload_dataset_artifact --train-path .../datasets/YYYYMMDD_HHMMSS/train.json  # 기존 dataset만 wandb artifact로 업로드
  python scripts/distill_flows.py upload_eval_artifact --eval-path .../eval/YYYYMMDD_HHMMSS/report.json  # 기존 eval 디렉터리만 wandb artifact로 업로드
  python scripts/distill_flows.py all        # build_dataset → labeling_openai_only(기본; --use-pod 시 labeling_with_pod) → train → evaluate → merge
  python scripts/distill_flows.py all_sweep [--sweep-id <sweep_id>]  # sweep-id 없으면 flow 내부에서 sweep 등록 후 실행
  python scripts/distill_flows.py merge_for_serving --adapter-path .../adapter [--out-dir ...]  # 로컬 merge (파이프라인 기본)
  python scripts/distill_flows.py merge_for_serving_with_pod --adapter-path .../adapter  # 수동: Pod에서 merge (볼륨 사용)
  python scripts/distill_flows.py evaluate_on_pod --adapter-path .../adapter --val-labeled-path .../val_labeled.json [--test-labeled-path ...]  # Pod에서 평가 후 artifact 업로드·로컬 다운로드
  python scripts/distill_flows.py download_eval_artifact --artifact-version v1 [--out-dir ...]  # 평가 artifact를 지정 버전으로 다운로드

예시:
  python scripts/distill_flows.py build_dataset --input tasteam_app_all_review_data.json --out-dir distill_pipeline_output
  python scripts/distill_flows.py labeling_openai_only --train-path distill_pipeline_output/datasets/YYYYMMDD_HHMMSS/train.json --out-dir distill_pipeline_output
  python scripts/distill_flows.py labeling_with_pod --train-path distill_pipeline_output/datasets/YYYYMMDD_HHMMSS/train.json --out-dir distill_pipeline_output --openai-cap 500
  python scripts/distill_flows.py labeling_pod_only --train-path .../train.json --gold-path .../labeled/YYYYMMDD_HHMMSS/train_labeled_gold_only.json --out-dir distill_pipeline_output
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

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

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
    from runpod_cli import runpod_config
except ImportError:
    runpod_config = None
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
    )
except ImportError:
    upload_labeled_dir_to_runpod = None
    list_run_ids_with_adapter = None
    download_directory_from_runpod = None
    upload_file_to_volume = None
    upload_directory = None
    get_runpod_s3_client = None
    object_exists = None

logger = logging.getLogger(__name__)

# Ctrl+C/SIGTERM 시 생성된 RunPod Pod 정리 (control_c_not_shutdown_error.md)
_CREATED_POD_IDS: list[str] = []
_POD_IDS_LOCK = threading.Lock()
_old_sigint_handler = None
_old_sigterm_handler = None


def _cleanup_pods_on_signal(signum, frame):
    """SIGINT/SIGTERM 수신 시 이번 프로세스에서 생성한 Pod 일괄 삭제 후 기존 핸들러 호출."""
    with _POD_IDS_LOCK:
        ids = _CREATED_POD_IDS[:]
        _CREATED_POD_IDS.clear()
    if ids:
        token = os.environ.get("RUNPOD_API_KEY")
        if token and RunPodClient is not None:
            client = RunPodClient(token=token)
            for pod_id in ids:
                try:
                    client.delete_pod(pod_id)
                    logger.info("Cleaned up pod %s (signal)", pod_id)
                except Exception as e:
                    logger.warning("Failed to delete pod %s: %s", pod_id, e)
    if signum == signal.SIGINT:
        if callable(_old_sigint_handler):
            _old_sigint_handler(signum, frame)
        else:
            raise KeyboardInterrupt()
    elif signum == signal.SIGTERM:
        if callable(_old_sigterm_handler):
            _old_sigterm_handler(signum, frame)
        else:
            raise SystemExit(128 + signum)


def _register_pod_signal_handlers() -> None:
    global _old_sigint_handler, _old_sigterm_handler
    if RunPodClient is None:
        return
    _old_sigint_handler = signal.signal(signal.SIGINT, _cleanup_pods_on_signal)
    try:
        _old_sigterm_handler = signal.signal(signal.SIGTERM, _cleanup_pods_on_signal)
    except (ValueError, OSError):
        pass  # Windows 등에서 SIGTERM 미지원


_register_pod_signal_handlers()


def _track_pod_created(pod_id: str) -> None:
    with _POD_IDS_LOCK:
        _CREATED_POD_IDS.append(pod_id)


def _untrack_pod(pod_id: str) -> None:
    with _POD_IDS_LOCK:
        if pod_id in _CREATED_POD_IDS:
            _CREATED_POD_IDS.remove(pod_id)


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
    """식당 단위 split, 윈도우/샘플 생성, train/val/test 저장 + 버전 태깅 + dataset artifact 업로드."""
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    result = build_dataset_task(
        input_path=input_path,
        out_dir=out_dir,
        window_configs=window_configs,
        add_full_restaurant=add_full_restaurant,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    art = upload_dataset_to_artifact_task(
        dataset_dir=result["dataset_dir"],
        project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        entity=os.environ.get("WANDB_ENTITY"),
    )
    result["artifact"] = art
    return result


def _vllm_base_url_from_runpod_proxy(pod_id: str, internal_http_port: int = 8000) -> str:
    """RunPod HTTP 프록시 URL: https://{pod_id}-{internal_port}.proxy.runpod.net (publicIp/portMappings 불필요)."""
    base = f"https://{pod_id}-{internal_http_port}.proxy.runpod.net/v1"
    logger.info("Using RunPod HTTP proxy: %s", base)
    return base


def _vllm_base_url_from_pod(pod: dict, internal_http_port: int = 8000) -> str:
    """RunPod pod 응답에서 publicIp와 portMappings(내부→외부)를 사용해 vLLM base_url 생성."""
    public_ip = (pod.get("publicIp") or "").strip()
    if not public_ip:
        raise ValueError("Pod has no publicIp")
    port_mappings = pod.get("portMappings") or {}
    # API는 내부 포트를 키로(문자열 또는 숫자), 외부 포트를 값으로 반환
    external_port = port_mappings.get(str(internal_http_port)) or port_mappings.get(internal_http_port)
    if external_port is None:
        external_port = internal_http_port  # 매핑 없으면 내부 포트 그대로 시도(대칭/레거시)
    logger.info(
        "portMappings=%s, internal_http_port=%s -> external_port=%s, base_url=http://%s:%s/v1",
        port_mappings,
        internal_http_port,
        external_port,
        public_ip,
        external_port,
    )
    return f"http://{public_ip}:{external_port}/v1"


def _wait_for_vllm_ready(base_url: str, timeout_sec: int = 180, poll_interval: int = 10) -> None:
    """vLLM /v1/models 가 응답할 때까지 대기."""
    import requests
    url = base_url.rstrip("/") + "/models"
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
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
    openai_only: bool = True,
) -> dict:
    """
    openai_only=True (기본): teacher를 4o mini로 단일화 — 전부 OpenAI(gpt-4o-mini)로만 라벨링, Pod 미사용.
    openai_only=False: OpenAI 골드 먼저(Pod 없이) → Pod 기동 → self-hosted teacher로 나머지 라벨링 → Pod 삭제.
    RUNPOD_API_KEY는 openai_only=False일 때만 필요.
    """
    out_dir = Path(output_labeled_dir or _PROJECT_ROOT / "distill_pipeline_output")
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    labeled_dir = out_dir / "labeled" / version
    labeled_dir.mkdir(parents=True, exist_ok=True)

    if openai_only:
        # teacher 4o mini 단일화: 전부 OpenAI로 라벨링, Pod 없음
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "label_for_distill.py"),
            "--phase", "single",
            "--openai-only",
            "--train-path", str(train_path),
            "--openai-cap", "999999",
            "--output-dir", str(labeled_dir),
            "--seed", "42",
        ]
        if val_path and Path(val_path).exists():
            cmd.extend(["--val-path", str(val_path)])
        if test_path and Path(test_path).exists():
            cmd.extend(["--test-path", str(test_path)])
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"label_for_distill.py openai_only exited with {result.returncode}")
        labeled_path = labeled_dir / "train_labeled.json"
        out = {"labeled_version": version, "labeled_path": str(labeled_path)}
        for name, fn in [("val_labeled_path", "val_labeled.json"), ("test_labeled_path", "test_labeled.json")]:
            p = labeled_dir / fn
            if p.exists():
                out[name] = str(p)
        if upload_labeled_dir_to_runpod and os.environ.get("RUNPOD_S3_ACCESS_KEY"):
            try:
                n = upload_labeled_dir_to_runpod(labeled_dir)
                out["runpod_upload"] = {"uploaded": True, "count": n, "labeled_dir": str(labeled_dir)}
            except Exception as e:
                logger.warning("RunPod S3 upload failed: %s", e)
                out["runpod_upload"] = {"uploaded": False, "error": str(e)}
        art = upload_labeled_to_artifact_task(labeled_dir=labeled_dir)
        out["artifact"] = art
        return out

    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available. Check runpod_cli import.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY environment variable is required for labeling_with_pod")

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
    _track_pod_created(pod_id)
    print("Pod created:", pod_id)

    try:
        client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        base_url = _vllm_base_url_from_runpod_proxy(pod_id)
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
        art = upload_labeled_to_artifact_task(labeled_dir=labeled_dir)
        out["artifact"] = art
        return out
    finally:
        print("Cleaning up pod:", pod_id)
        client.delete_pod(pod_id)
        _untrack_pod(pod_id)


@flow(name="labeling_with_pod_flow", log_prints=True)
def labeling_with_pod_flow(
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
    openai_cap: int = 500,
    output_labeled_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
    openai_only: bool = True,
) -> dict:
    """
    openai_only=True (기본): teacher를 4o mini로 단일화(전부 OpenAI, Pod 미사용).
    openai_only=False: OpenAI 골드 먼저 → Pod 기동 → self-hosted teacher 나머지 → Pod 삭제.
    docs/runpod_cli/cli_strategy.md
    """
    return labeling_with_pod_task(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        openai_cap=openai_cap,
        output_labeled_dir=str(output_labeled_dir) if output_labeled_dir else None,
        pod_wait_timeout_sec=pod_wait_timeout_sec,
        public_ip_wait_timeout_sec=public_ip_wait_timeout_sec,
        vllm_ready_timeout_sec=vllm_ready_timeout_sec,
        openai_only=openai_only,
    )


@task(name="labeling-pod-only-task", log_prints=True)
def labeling_pod_only_task(
    train_path: str,
    gold_path: str,
    output_labeled_dir: str | None = None,
    pod_wait_timeout_sec: int = 600,
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
) -> dict:
    """
    이미 생성된 train_labeled_gold_only.json이 있을 때, Pod만 띄워 teacher_rest 라벨링 후 Pod 삭제.
    OpenAI 단계는 건너뜀. RUNPOD_API_KEY 필요.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available. Check runpod_cli import.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY environment variable is required for labeling_pod_only")

    gold_path_resolved = Path(gold_path)
    if not gold_path_resolved.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")
    labeled_dir = Path(output_labeled_dir) if output_labeled_dir else gold_path_resolved.parent
    labeled_dir.mkdir(parents=True, exist_ok=True)

    client = RunPodClient(token=token)
    payload = RunPodClient.get_default_pod_payload(use="labeling")
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    _track_pod_created(pod_id)
    print("Pod created:", pod_id)

    try:
        client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        base_url = _vllm_base_url_from_runpod_proxy(pod_id)
        print("Pod ready:", pod_id, "base_url:", base_url)

        _wait_for_vllm_ready(base_url, timeout_sec=vllm_ready_timeout_sec)

        env = os.environ.copy()
        env["VLLM_POD_BASE_URL"] = base_url
        env["USE_POD_VLLM"] = "true"
        env["LLM_PROVIDER"] = "runpod"

        cmd_teacher = [
            sys.executable,
            str(_SCRIPT_DIR / "label_for_distill.py"),
            "--phase", "teacher_rest",
            "--train-path", str(train_path),
            "--gold-labeled-path", str(gold_path_resolved),
            "--output-dir", str(labeled_dir),
            "--seed", "42",
        ]
        result = subprocess.run(cmd_teacher, cwd=str(_PROJECT_ROOT), env=env, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"label_for_distill.py teacher_rest exited with {result.returncode}")

        labeled_path = labeled_dir / "train_labeled.json"
        out = {"labeled_path": str(labeled_path)}
        for name, fn in [("val_labeled_path", "val_labeled.json"), ("test_labeled_path", "test_labeled.json")]:
            p = labeled_dir / fn
            if p.exists():
                out[name] = str(p)
        if upload_labeled_dir_to_runpod and os.environ.get("RUNPOD_S3_ACCESS_KEY"):
            try:
                n = upload_labeled_dir_to_runpod(labeled_dir)
                out["runpod_upload"] = {"uploaded": True, "count": n, "labeled_dir": str(labeled_dir)}
            except Exception as e:
                logger.warning("RunPod S3 upload failed: %s", e)
                out["runpod_upload"] = {"uploaded": False, "error": str(e)}
        art = upload_labeled_to_artifact_task(labeled_dir=labeled_dir)
        out["artifact"] = art
        return out
    finally:
        print("Cleaning up pod:", pod_id)
        client.delete_pod(pod_id)
        _untrack_pod(pod_id)


@flow(name="labeling_pod_only_flow", log_prints=True)
def labeling_pod_only_flow(
    train_path: str,
    gold_path: str,
    output_labeled_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
) -> dict:
    """
    기존 train_labeled_gold_only.json으로 Pod에서 teacher_rest만 실행 (OpenAI 단계 생략).
    """
    return labeling_pod_only_task(
        train_path=train_path,
        gold_path=gold_path,
        output_labeled_dir=str(output_labeled_dir) if output_labeled_dir else None,
        pod_wait_timeout_sec=pod_wait_timeout_sec,
        public_ip_wait_timeout_sec=public_ip_wait_timeout_sec,
        vllm_ready_timeout_sec=vllm_ready_timeout_sec,
    )


def _get_train_volume_id() -> str:
    if runpod_config is not None:
        return runpod_config.get_volume_id_train()
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN", os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "v3i546pkrz"))


@task(name="train-student-with-pod-task", log_prints=True)
def train_student_with_pod_task(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir: str | Path | None = None,
    pod_wait_timeout_sec: int = 600,
    train_timeout_sec: int = 7200,
) -> dict:
    """
    학습용 Pod 생성 → 볼륨에 라벨 업로드(이미 있으면 스킵) → Pod에서 train_qlora 실행(/tmp에 출력, artifact 업로드) →
    Pod 종료 대기 → wandb artifact에서 최신 run adapter 다운로드 → Pod 삭제.
    adapter는 볼륨에 쓰지 않고 artifact만 사용.
    RUNPOD_API_KEY, RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_ACCESS_KEY, WANDB_API_KEY 필요.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available.")
    if not upload_labeled_dir_to_runpod:
        raise RuntimeError("runpod_s3_upload (upload_labeled_dir_to_runpod) required for train_student_with_pod.")
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

    docker_start_cmd = [
        "--labeled-path", path_on_volume,
        "--output-dir", "/tmp/distill_pipeline_output",
    ]
    payload = RunPodClient.get_default_pod_payload(use="train", docker_start_cmd=docker_start_cmd)
    client = RunPodClient(token=token)
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    _track_pod_created(pod_id)
    print("Train Pod created:", pod_id)

    try:
        client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        client.wait_until_stopped(pod_id, timeout_sec=train_timeout_sec, poll_interval_sec=120)
        logger.info("Train Pod finished: %s", pod_id)

        adapter_path = get_latest_run_adapter_from_artifact_task(
            download_dir=out_dir,
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
        )
        if not adapter_path:
            raise RuntimeError("Train Pod finished but no adapter artifact found in wandb. Check Pod logs and WANDB_API_KEY.")

        return {
            "adapter_path": adapter_path,
            "run_id": Path(adapter_path).parent.name,
            "training_meta_path": str(out_dir / "artifacts" / Path(adapter_path).parent.name / "training_meta.json"),
        }
    finally:
        print("Cleaning up train pod:", pod_id)
        client.delete_pod(pod_id)
        _untrack_pod(pod_id)


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


DEFAULT_WANDB_PROJECT = "tasteam-distill"


@task(name="ensure-wandb-project-task", log_prints=True)
def ensure_wandb_project_task(
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
) -> None:
    """wandb 프로젝트가 없으면 wandb.init으로 생성 (sweep/agent 404 방지)."""
    import wandb
    os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)
    wandb.init(project=project, entity=entity or os.environ.get("WANDB_ENTITY"))
    wandb.finish()
    logger.info("W&B project ensured: %s (entity=%s)", project, entity or "default")


@task(name="upload-labeled-to-artifact-task", log_prints=True)
def upload_labeled_to_artifact_task(
    labeled_dir: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = "labeled-data",
) -> dict:
    """
    labeled 디렉터리(train_labeled.json, val_labeled.json 등)를 wandb artifact로 업로드해 버전 관리.
    artifact 이름은 동일하게 두면 wandb가 v0, v1, ... 으로 버저닝.
    """
    import wandb

    labeled_dir = Path(labeled_dir)
    if not labeled_dir.is_dir():
        raise FileNotFoundError(f"labeled_dir is not a directory: {labeled_dir}")
    version = labeled_dir.name
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set; skipping artifact upload")
        return {"skipped": True, "reason": "WANDB_API_KEY not set", "labeled_dir": str(labeled_dir)}

    try:
        run = wandb.init(
            project=project,
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=f"upload-labeled-{version}",
            job_type="data_upload",
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata={"version": version, "labeled_dir": str(labeled_dir)},
        )
        artifact.add_dir(str(labeled_dir), name="labeled")
        wandb.log_artifact(artifact)
        wandb.finish()
        logger.info("Uploaded labeled dir to artifact %s (version=%s)", artifact_name, version)
        return {
            "artifact_name": artifact_name,
            "version": version,
            "labeled_dir": str(labeled_dir),
        }
    except Exception as e:
        logger.warning("wandb artifact upload failed: %s", e)
        return {"skipped": True, "reason": str(e), "labeled_dir": str(labeled_dir)}


@flow(name="upload_labeled_artifact_flow", log_prints=True)
def upload_labeled_artifact_flow(
    labeled_path: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
) -> dict:
    """
    기존 labeled 디렉터리를 wandb artifact로만 업로드 (라벨링/학습 없이 올리기만 실행).
    labeled_path: train_labeled.json 경로 또는 labeled 디렉터리 경로.
    """
    path = Path(labeled_path)
    if path.is_file():
        labeled_dir = path.parent
    else:
        labeled_dir = path
    return upload_labeled_to_artifact_task(
        labeled_dir=labeled_dir,
        project=project,
        entity=entity,
    )


@task(name="upload-dataset-to-artifact-task", log_prints=True)
def upload_dataset_to_artifact_task(
    dataset_dir: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = "dataset",
) -> dict:
    """
    dataset 디렉터리(train.json, val.json, test.json, stats.json)를 wandb artifact로 업로드.
    분할 전략 변경 시 추적·비교용.
    """
    import wandb

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir is not a directory: {dataset_dir}")
    version = dataset_dir.name
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set; skipping dataset artifact upload")
        return {"skipped": True, "reason": "WANDB_API_KEY not set", "dataset_dir": str(dataset_dir)}

    try:
        run = wandb.init(
            project=project,
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=f"upload-dataset-{version}",
            job_type="data_upload",
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            metadata={"version": version, "dataset_dir": str(dataset_dir)},
        )
        artifact.add_dir(str(dataset_dir), name="dataset")
        wandb.log_artifact(artifact)
        wandb.finish()
        logger.info("Uploaded dataset dir to artifact %s (version=%s)", artifact_name, version)
        return {
            "artifact_name": artifact_name,
            "version": version,
            "dataset_dir": str(dataset_dir),
        }
    except Exception as e:
        logger.warning("wandb dataset artifact upload failed: %s", e)
        return {"skipped": True, "reason": str(e), "dataset_dir": str(dataset_dir)}


@flow(name="upload_dataset_artifact_flow", log_prints=True)
def upload_dataset_artifact_flow(
    dataset_path: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
) -> dict:
    """
    기존 dataset 디렉터리를 wandb artifact로만 업로드 (build_dataset 없이 올리기만 실행).
    dataset_path: train.json 경로 또는 datasets/YYYYMMDD_HHMMSS 디렉터리 경로.
    """
    path = Path(dataset_path)
    if path.is_file():
        dataset_dir = path.parent
    else:
        dataset_dir = path
    return upload_dataset_to_artifact_task(
        dataset_dir=dataset_dir,
        project=project,
        entity=entity,
    )


@task(name="register-sweep-task", log_prints=True)
def register_sweep_task(sweep_yaml_path: str | Path) -> str:
    """wandb.sweep()로 sweep 등록 후 sweep id를 반환 (stdout 파싱 없음)."""
    import wandb

    path = Path(sweep_yaml_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"sweep yaml not found: {path}")
    with open(path, encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)
    if not sweep_config:
        raise ValueError(f"Empty or invalid sweep yaml: {path}")
    project = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    entity = os.environ.get("WANDB_ENTITY") or None
    logger.info("Registering sweep via wandb.sweep (project=%s, entity=%s)", project, entity)
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    logger.info("Registered sweep_id: %s", sweep_id)
    return sweep_id


@task(name="upload-labeled-to-volume-for-sweep-task", log_prints=True)
def upload_labeled_to_volume_for_sweep_task(labeled_path: str) -> dict:
    """멀티 Pod 시 라벨 디렉터리를 볼륨에 한 번만 업로드. run_sweep_on_pod_task(skip_upload=True)와 함께 사용."""
    if not upload_labeled_dir_to_runpod:
        raise RuntimeError("upload_labeled_dir_to_runpod required.")
    labeled_dir = Path(labeled_path).parent
    upload_labeled_dir_to_runpod(labeled_dir, volume_id=_get_train_volume_id())
    return {"labeled_path": labeled_path}


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


@task(name="run-sweep-on-pod-task", log_prints=True, retries=0)
def run_sweep_on_pod_task(
    sweep_id: str,
    labeled_path: str,
    output_dir: str,
    pod_wait_timeout_sec: int = 3600,
    sweep_timeout_sec: int = 90000,
    sweep_poll_interval_sec: int = 60,
    pod_index: int | None = None,
    skip_upload: bool = False,
) -> dict:
    """
    학습용 Pod에서 wandb sweep 에이전트 실행 (run_qlora_sweep.py).
    볼륨에 라벨 업로드(skip_upload=False 시) → Pod 생성(ENTRYPOINT/CMD 오버라이드) → Pod 종료까지 대기 → Pod 삭제.
    완료 후 adapter는 wandb artifact에서 get_best_adapter_from_artifact_task로 수급.
    pod_index: 멀티 Pod 시 Pod 이름 고유화용 (sweep-pod-0, sweep-pod-1, ...).
    skip_upload: True면 업로드 생략 (멀티 Pod에서 업로드는 flow에서 한 번만 수행).
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available.")
    if not upload_labeled_dir_to_runpod:
        raise RuntimeError("upload_labeled_dir_to_runpod required for run_sweep_on_pod.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY required for run_sweep_on_pod")
    out_dir = Path(output_dir)
    labeled_dir = Path(labeled_path).parent
    version = labeled_dir.name
    path_on_volume = f"/workspace/labeled/{version}/train_labeled.json"
    if not skip_upload:
        try:
            upload_labeled_dir_to_runpod(labeled_dir, volume_id=_get_train_volume_id())
        except Exception as e:
            raise RuntimeError(f"Upload labeled dir to volume failed: {e}") from e

    # sweep_id = entity/project/sweep_run_id → Pod에 동일 project/entity 전달해 404 방지
    # 경로(Users/js/tasteam 등)가 넘어오면 파싱하지 않고 env/기본값 사용
    _PATH_LIKE_ENTITIES = frozenset({"users", "home", "tmp", "opt", "var", "root"})
    parts = sweep_id.split("/")
    if (
        len(parts) >= 3
        and not sweep_id.startswith("/")
        and parts[0].lower() not in _PATH_LIKE_ENTITIES
    ):
        wandb_project = parts[1]
        wandb_entity = parts[0]
    else:
        wandb_project = DEFAULT_WANDB_PROJECT
        wandb_entity = os.environ.get("WANDB_ENTITY", "")
    payload = RunPodClient.get_default_pod_payload(use="train", docker_start_cmd=[sweep_id])
    payload["name"] = f"sweep-pod-{pod_index}" if pod_index is not None else "sweep-pod"
    payload["dockerEntrypoint"] = ["python", "/app/scripts/run_qlora_sweep.py"]
    payload["dockerStartCmd"] = [sweep_id]
    base_env = payload.get("env") or {}
    payload["env"] = {
        **base_env,
        "WANDB_SWEEP_LABELED_PATH": path_on_volume,
        "WANDB_SWEEP_OUTPUT_DIR": "/tmp/distill_pipeline_output",
        "WANDB_SWEEP_ID": sweep_id,
        "WANDB_PROJECT": wandb_project,
    }
    if wandb_entity:
        payload["env"]["WANDB_ENTITY"] = wandb_entity
    client = RunPodClient(token=token)
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    _track_pod_created(pod_id)
    logger.info("Sweep Pod created: %s (sweep_id=%s)", pod_id, sweep_id)
    try:
        client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        client.wait_until_stopped(
            pod_id,
            timeout_sec=sweep_timeout_sec,
            poll_interval_sec=sweep_poll_interval_sec,
        )
        logger.info("Sweep Pod finished: %s", pod_id)
    finally:
        logger.info("Cleaning up sweep pod: %s", pod_id)
        client.delete_pod(pod_id)
        _untrack_pod(pod_id)
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
        ensure_wandb_project_task(project=DEFAULT_WANDB_PROJECT, entity=os.environ.get("WANDB_ENTITY"))
        yaml_path = sweep_yaml if sweep_yaml is not None else DEFAULT_SWEEP_YAML
        sweep_id = register_sweep_task(yaml_path)
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    return run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)


def _get_run_metric_value(run: Any, metric_name: str = "eval_loss") -> float | None:
    """run.summary에서 메트릭 값 반환. eval_loss / eval/loss 둘 다 시도."""
    summary = getattr(run, "summary", None) or {}
    candidates = [metric_name]
    if metric_name == "eval_loss":
        candidates.append("eval/loss")
    elif metric_name == "eval/loss":
        candidates.append("eval_loss")
    for key in candidates:
        try:
            v = summary.get(key)
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            continue
    return None


@task(name="get-best-adapter-from-sweep-task", log_prints=True)
def get_best_adapter_path_from_sweep_task(
    sweep_id: str,
    output_dir: str,
    metric_name: str = "eval_loss",
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
    runs_with_metric = [r for r in runs if _get_run_metric_value(r, metric_name) is not None]
    if not runs_with_metric:
        logger.warning("sweep에 메트릭 %s(또는 eval/loss)가 있는 run이 없음", metric_name)
        return None
    best = min(runs_with_metric, key=lambda r: _get_run_metric_value(r, metric_name))
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


@task(name="check-sweep-complete-task", log_prints=True)
def check_sweep_complete_task(sweep_id: str) -> bool:
    """wandb sweep가 이미 완료(Finished)였는지 확인. run_cap 도달 또는 state=Finished."""
    try:
        import wandb
        api = wandb.Api()
        sweep = api.sweep(sweep_id)
        state = (getattr(sweep, "state", None) or "").lower()
        if state == "finished":
            logger.info("sweep %s 이미 완료(state=%s), Pod 생성 건너뜀", sweep_id, state)
            return True
        run_cap = 50
        try:
            cfg = getattr(sweep, "config", None) or {}
            run_cap = int(cfg.get("run_cap", 50))
        except (TypeError, ValueError):
            pass
        runs = list(sweep.runs)
        if len(runs) >= run_cap:
            logger.info("sweep %s run %d/%d 도달, Pod 생성 건너뜀", sweep_id, len(runs), run_cap)
            return True
        return False
    except Exception as e:
        logger.warning("sweep 완료 여부 조회 실패 (%s), Pod 생성 진행: %s", sweep_id, e)
        return False


@task(name="get-best-adapter-from-artifact-task", log_prints=True)
def get_best_adapter_from_artifact_task(
    sweep_id: str,
    download_dir: str | Path,
    metric_name: str = "eval_loss",
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
    runs_with_metric = [r for r in runs if _get_run_metric_value(r, metric_name) is not None]
    if not runs_with_metric:
        logger.warning("sweep에 메트릭 %s(또는 eval/loss)가 있는 run이 없음", metric_name)
        return None
    best = min(runs_with_metric, key=lambda r: _get_run_metric_value(r, metric_name))
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


@task(name="get-latest-run-adapter-from-artifact-task", log_prints=True)
def get_latest_run_adapter_from_artifact_task(
    download_dir: str | Path,
    project: str | None = None,
    entity: str | None = None,
    per_page: int = 20,
) -> str | None:
    """
    wandb 프로젝트에서 가장 최근 run의 adapter artifact를 다운로드해 로컬 경로 반환.
    단일 run 학습(train_student_with_pod) 후 Pod가 종료되면 artifact만으로 adapter 수급할 때 사용.
    """
    try:
        import wandb
        api = wandb.Api()
        project = project or os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
        entity = entity or os.environ.get("WANDB_ENTITY") or ""
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, order="-created_at", per_page=per_page)
    except Exception as e:
        logger.warning("wandb runs 조회 실패: %s", e)
        return None
    download_dir = Path(download_dir)
    for run in runs:
        run_name = run.name or getattr(run, "id", None)
        if not run_name:
            continue
        artifact_name = f"qlora-adapter-{run_name}"
        try:
            art = api.artifact(f"{run.entity}/{run.project}/{artifact_name}:latest")
        except Exception:
            continue
        root = download_dir / "artifacts" / run_name
        root.mkdir(parents=True, exist_ok=True)
        art.download(root=str(root))
        adapter_path = root / "adapter"
        if adapter_path.is_dir():
            logger.info("latest run adapter: %s -> %s", run_name, adapter_path)
            return str(adapter_path)
    logger.warning("프로젝트 %s 에서 adapter artifact를 가진 run이 없음", path)
    return None


@flow(name="run_sweep_and_evaluate_flow", log_prints=True)
def run_sweep_and_evaluate_flow(
    labeled_path: str,
    sweep_id: str | None = None,
    sweep_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    use_pod: bool = True,
    num_pods: int = 2,
) -> dict:
    """sweep 실행(use_pod=True 시 Pod에서, False 시 로컬 subprocess) → best adapter를 wandb artifact에서 다운로드 → evaluate. sweep_id 없으면 sweep_yaml으로 등록. num_pods>1이면 동일 sweep_id로 N개 Pod 동시 실행(멀티 Pod)."""
    if sweep_id is None:
        ensure_wandb_project_task(project=DEFAULT_WANDB_PROJECT, entity=os.environ.get("WANDB_ENTITY"))
        yaml_path = sweep_yaml if sweep_yaml is not None else DEFAULT_SWEEP_YAML
        sweep_id = register_sweep_task(yaml_path)
    out_dir = str(output_dir) if output_dir else str(_PROJECT_ROOT / "distill_pipeline_output")
    sweep_complete = check_sweep_complete_task(sweep_id)
    if not sweep_complete:
        if use_pod:
            if num_pods > 1:
                # 멀티 Pod: 라벨 한 번만 업로드 후 Pod 생성 순차화(스태거)로 동시 500 방지 (runpod_api_500.md)
                upload_labeled_to_volume_for_sweep_task(labeled_path)
                _SWEEP_POD_STAGGER_SEC = 15  # Pod 생성 요청 간격(초)
                futures = []
                for i in range(num_pods):
                    if i > 0:
                        time.sleep(_SWEEP_POD_STAGGER_SEC)
                    futures.append(
                        run_sweep_on_pod_task.submit(
                            sweep_id=sweep_id,
                            labeled_path=labeled_path,
                            output_dir=out_dir,
                            pod_index=i,
                            skip_upload=True,
                        )
                    )
                for f in futures:
                    f.result()
            else:
                run_sweep_on_pod_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)
        else:
            run_sweep_agent_task(sweep_id=sweep_id, labeled_path=labeled_path, output_dir=out_dir)
    best_adapter = get_best_adapter_from_artifact_task(
        sweep_id=sweep_id,
        download_dir=out_dir,
        metric_name="eval_loss",
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
        json.dump(
            {"val": {"skipped": True}, "test": {"skipped": True}, "meta": {"reason": "no val/test labeled paths", "llm_judge_sample_ids": []}},
            open(report_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        art = upload_eval_to_artifact_task(eval_dir=eval_dir)
        return {"report_path": str(report_path), "eval_dir": str(eval_dir), "artifact": art}

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

    report_path = None
    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            out = json.loads(line)
            report_path = out.get("report_path")
            break
    if not report_path:
        raise RuntimeError("eval_distill.py did not produce report_path")

    eval_dir = Path(report_path).parent
    llm_judge_path = eval_dir / "llm_as_a_judge_results.json"
    kd_report_path = eval_dir / "kd_sft_analysis_report.json"

    # report.json의 meta.llm_judge_sample_ids로 LLM-as-a-Judge 실행 (별도 samples 파일 없음)
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)
    sample_ids = report_data.get("meta", {}).get("llm_judge_sample_ids", [])
    if sample_ids and val_labeled_path and Path(val_labeled_path).exists():
        cmd_judge = [
            sys.executable,
            str(_SCRIPT_DIR / "eval_llm_as_judge.py"),
            "--report", str(report_path),
            "--val-labeled", str(val_labeled_path),
            "--adapter-path", str(adapter_path),
            "--base-model", base_model,
            "--output", str(llm_judge_path),
        ]
        rj = subprocess.run(cmd_judge, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
        if rj.returncode != 0:
            logger.warning("eval_llm_as_judge failed: %s", rj.stderr or rj.stdout)
        elif llm_judge_path.exists():
            # kd_sft_analysis
            cmd_kd = [
                sys.executable,
                str(_SCRIPT_DIR / "kd_sft_analysis.py"),
                "--input", str(llm_judge_path),
                "--output-dir", str(eval_dir),
            ]
            rk = subprocess.run(cmd_kd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True)
            if rk.returncode != 0:
                logger.warning("kd_sft_analysis failed: %s", rk.stderr or rk.stdout)

    art = upload_eval_to_artifact_task(eval_dir=eval_dir)
    return {
        "report_path": str(report_path),
        "eval_dir": str(eval_dir),
        "llm_as_a_judge_results_path": str(llm_judge_path) if llm_judge_path.exists() else None,
        "kd_sft_analysis_report_path": str(kd_report_path) if kd_report_path.exists() else None,
        "artifact": art,
    }


@flow(name="evaluate_flow", log_prints=True)
def evaluate_flow(
    adapter_path: str,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    output_dir: str | Path | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """val/test 평가 (ROUGE/BERTScore) → LLM-as-a-Judge → kd_sft_analysis → eval 아티팩트 업로드."""
    return evaluate_task(
        adapter_path=adapter_path,
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        output_dir=str(output_dir) if output_dir else None,
        base_model=base_model,
    )


EVAL_ARTIFACT_NAME_DEFAULT = "distill-eval-report"


@task(name="upload-eval-to-artifact-task", log_prints=True)
def upload_eval_to_artifact_task(
    eval_dir: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
) -> dict:
    """
    eval 디렉터리(report.json, llm_as_a_judge_results.json, kd_sft_analysis_report.json 등)를 wandb artifact로 업로드.
    """
    import wandb

    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"eval_dir is not a directory: {eval_dir}")
    version = eval_dir.name
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set; skipping eval artifact upload")
        return {"skipped": True, "reason": "WANDB_API_KEY not set", "eval_dir": str(eval_dir)}

    try:
        run = wandb.init(
            project=project,
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=f"upload-eval-{version}",
            job_type="eval_upload",
        )
        artifact = wandb.Artifact(
            name=artifact_name,
            type="eval-report",
            metadata={"version": version, "eval_dir": str(eval_dir)},
        )
        artifact.add_dir(str(eval_dir), name="eval")
        wandb.log_artifact(artifact)
        wandb.finish()
        logger.info("Uploaded eval dir to artifact %s (version=%s)", artifact_name, version)
        return {
            "artifact_name": artifact_name,
            "version": version,
            "eval_dir": str(eval_dir),
        }
    except Exception as e:
        logger.warning("wandb eval artifact upload failed: %s", e)
        return {"skipped": True, "reason": str(e), "eval_dir": str(eval_dir)}


@flow(name="upload_eval_artifact_flow", log_prints=True)
def upload_eval_artifact_flow(
    eval_path: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
) -> dict:
    """
    기존 eval 디렉터리를 wandb artifact로만 업로드 (evaluate 없이 올리기만 실행).
    eval_path: report.json 경로 또는 eval/YYYYMMDD_HHMMSS 디렉터리 경로.
    """
    path = Path(eval_path)
    if path.is_file():
        eval_dir = path.parent
    else:
        eval_dir = path
    return upload_eval_to_artifact_task(
        eval_dir=eval_dir,
        project=project,
        entity=entity,
        artifact_name=artifact_name,
    )


@task(name="download-eval-artifact-task", log_prints=True)
def download_eval_artifact_task(
    artifact_version: str,
    output_dir: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
) -> dict:
    """wandb artifact에서 평가 결과를 지정 버전으로 다운로드. artifact_version 예: latest, v0, v1."""
    import wandb

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY required for download_eval_artifact_task")
    api = wandb.Api()
    qualified = f"{entity or os.environ.get('WANDB_ENTITY', '')}/{project}/{artifact_name}:{artifact_version}"
    qualified = qualified.lstrip("/")
    art = api.artifact(qualified)
    art.download(root=str(output_dir))
    # artifact는 add_dir(..., name="eval")로 올렸으므로 report는 output_dir/eval/report.json
    eval_subdir = output_dir / "eval"
    report_path = eval_subdir / "report.json"
    llm_judge_path = eval_subdir / "llm_as_a_judge_results.json"
    kd_report_path = eval_subdir / "kd_sft_analysis_report.json"
    return {
        "report_path": str(report_path) if report_path.exists() else None,
        "llm_as_a_judge_results_path": str(llm_judge_path) if llm_judge_path.exists() else None,
        "kd_sft_analysis_report_path": str(kd_report_path) if kd_report_path.exists() else None,
        "artifact_version": artifact_version,
        "qualified_name": qualified,
        "download_root": str(output_dir),
    }


@task(name="evaluate-on-pod-task", log_prints=True)
def evaluate_on_pod_task(
    adapter_path: str,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    output_dir: str | Path | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    volume_id: str | None = None,
    eval_poll_interval_sec: int = 60,
    eval_timeout_sec: int = 7200,
    pod_wait_timeout_sec: int = 600,
) -> dict:
    """
    Pod에서 eval_distill 실행 → 결과를 wandb artifact로 업로드 → 로컬에서 해당 버전으로 다운로드.
    adapter_path, val_labeled_path, test_labeled_path 필수(평가 입력). output_dir는 다운로드 받을 로컬 경로.
    """
    if RunPodClient is None:
        raise RuntimeError("RunPodClient not available.")
    if not upload_file_to_volume or not upload_directory or not get_runpod_s3_client or not object_exists:
        raise RuntimeError("runpod_s3_upload (upload_file_to_volume, upload_directory, object_exists) required.")
    token = os.environ.get("RUNPOD_API_KEY")
    if not token:
        raise ValueError("RUNPOD_API_KEY required for evaluate_on_pod_task")
    if not os.environ.get("RUNPOD_S3_ACCESS_KEY") or not os.environ.get("RUNPOD_S3_SECRET_ACCESS_KEY"):
        raise ValueError("RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_ACCESS_KEY required.")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY required for Pod to upload eval result artifact")

    vol_id = volume_id or (runpod_config.get_volume_id_eval() if runpod_config else None) or os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    if not vol_id:
        raise ValueError("volume_id or RUNPOD_NETWORK_VOLUME_ID required (eval uses volumes.eval / pod.eval from runpod.yaml)")
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    prefix = f"distill_pipeline_output/eval_input/{version}"
    eval_output_on_volume = f"/workspace/distill_pipeline_output/eval_output/{version}"
    client = get_runpod_s3_client()

    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter path is not a directory: {adapter_dir}")
    upload_directory(client, vol_id, adapter_dir, f"{prefix}/adapter")
    adapter_on_volume = f"/workspace/{prefix}/adapter"

    val_path = Path(val_labeled_path) if val_labeled_path else None
    test_path = Path(test_labeled_path) if test_labeled_path else None
    if not val_path or not val_path.exists():
        raise ValueError("val_labeled_path is required and must exist for evaluate_on_pod_task")
    upload_file_to_volume(val_path, volume_id=vol_id, object_key=f"{prefix}/val_labeled.json")
    if test_path and test_path.exists():
        upload_file_to_volume(test_path, volume_id=vol_id, object_key=f"{prefix}/test_labeled.json")
    val_on_volume = f"/workspace/{prefix}/val_labeled.json"
    test_on_volume = f"/workspace/{prefix}/test_labeled.json" if (test_path and test_path.exists()) else None

    eval_script = _SCRIPT_DIR / "eval_distill.py"
    wrapper_script = _SCRIPT_DIR / "run_eval_and_upload_artifact.py"
    if not eval_script.is_file() or not wrapper_script.is_file():
        raise FileNotFoundError("eval_distill.py and run_eval_and_upload_artifact.py must exist in scripts/")
    upload_file_to_volume(eval_script, volume_id=vol_id, object_key="distill_pipeline_output/eval_scripts/eval_distill.py")
    upload_file_to_volume(wrapper_script, volume_id=vol_id, object_key="distill_pipeline_output/eval_scripts/run_eval_and_upload_artifact.py")

    cmd = [
        "/workspace/distill_pipeline_output/eval_scripts/run_eval_and_upload_artifact.py",
        "--adapter-path", adapter_on_volume,
        "--val-labeled", val_on_volume,
        "--output-dir", eval_output_on_volume,
        "--base-model", base_model,
        "--artifact-name", artifact_name,
        "--wandb-project", project,
    ]
    if test_on_volume:
        cmd.extend(["--test-labeled", test_on_volume])
    if entity:
        cmd.extend(["--wandb-entity", entity])

    payload = RunPodClient.get_default_pod_payload(use="eval", docker_start_cmd=cmd)
    runpod_client = RunPodClient(token=token)
    pod = runpod_client.create_pod(payload)
    pod_id = pod["id"]
    _track_pod_created(pod_id)
    logger.info("Eval Pod created: %s", pod_id)

    try:
        runpod_client.wait_until_running(pod_id, timeout_sec=pod_wait_timeout_sec)
        done_key = f"distill_pipeline_output/eval_output/{version}/eval_done.json"
        deadline = time.time() + eval_timeout_sec
        while time.time() < deadline:
            if object_exists(client, vol_id, done_key):
                break
            time.sleep(eval_poll_interval_sec)
        else:
            raise TimeoutError(f"Eval Pod did not produce {done_key} within {eval_timeout_sec}s. Check Pod logs.")

        resp = client.get_object(Bucket=vol_id, Key=done_key)
        done = json.loads(resp["Body"].read().decode("utf-8"))
        qualified_name = done.get("qualified_name")
        artifact_version_str = done.get("version") or "latest"
        if not qualified_name:
            logger.warning("eval_done.json has no qualified_name; cannot download artifact")
            return {"report_path": None, "artifact_version": artifact_version_str, "eval_done": done}

        out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output" / "eval_from_pod")
        out_dir.mkdir(parents=True, exist_ok=True)
        import wandb as _wandb
        _api = _wandb.Api()
        _art = _api.artifact(qualified_name)
        _art.download(root=str(out_dir))
        eval_subdir = out_dir / "eval"
        report_path = eval_subdir / "report.json"
        llm_judge_path = eval_subdir / "llm_as_a_judge_results.json"
        kd_report_path = eval_subdir / "kd_sft_analysis_report.json"
        return {
            "report_path": str(report_path) if report_path.exists() else None,
            "llm_as_a_judge_results_path": str(llm_judge_path) if llm_judge_path.exists() else None,
            "kd_sft_analysis_report_path": str(kd_report_path) if kd_report_path.exists() else None,
            "artifact_version": artifact_version_str,
            "qualified_name": qualified_name,
            "download_root": str(out_dir),
        }
    finally:
        logger.info("Cleaning up eval pod: %s", pod_id)
        runpod_client.delete_pod(pod_id)
        _untrack_pod(pod_id)


@flow(name="evaluate_on_pod_flow", log_prints=True)
def evaluate_on_pod_flow(
    adapter_path: str,
    val_labeled_path: str,
    test_labeled_path: str | None = None,
    output_dir: str | Path | None = None,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    volume_id: str | None = None,
    eval_timeout_sec: int = 7200,
) -> dict:
    """Pod에서 평가 실행 후 결과를 artifact로 올리고, 로컬에서 해당 버전으로 다운로드."""
    return evaluate_on_pod_task(
        adapter_path=adapter_path,
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        output_dir=str(output_dir) if output_dir else None,
        base_model=base_model,
        artifact_name=artifact_name,
        project=project,
        entity=entity,
        volume_id=volume_id,
        eval_timeout_sec=eval_timeout_sec,
    )


@flow(name="download_eval_artifact_flow", log_prints=True)
def download_eval_artifact_flow(
    artifact_version: str,
    output_dir: str | Path,
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str | None = None,
    artifact_name: str = EVAL_ARTIFACT_NAME_DEFAULT,
) -> dict:
    """기존 평가 artifact를 지정 버전(예: latest, v0, v1)으로 다운로드."""
    return download_eval_artifact_task(
        artifact_version=artifact_version,
        output_dir=output_dir,
        project=project,
        entity=entity,
        artifact_name=artifact_name,
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
    _track_pod_created(pod_id)
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
        _untrack_pod(pod_id)


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
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    openai_only: bool = True,
) -> dict:
    """build_dataset → labeling(기본 OpenAI only; --use-pod 시 Pod) → train_student(Pod) → evaluate → merge(로컬) 순차 실행."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_with_pod_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
        public_ip_wait_timeout_sec=public_ip_wait_timeout_sec,
        vllm_ready_timeout_sec=vllm_ready_timeout_sec,
        openai_only=openai_only,
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
    public_ip_wait_timeout_sec: int = 180,
    vllm_ready_timeout_sec: int = 180,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_pods: int = 2,
    openai_only: bool = True,
) -> dict:
    """build_dataset → labeling(기본 OpenAI only; --use-pod 시 Pod) → run_sweep → best run adapter로 evaluate → merge(로컬). sweep_id 없으면 sweep_yaml으로 등록. num_pods>1이면 멀티 Pod."""
    out_dir = Path(out_dir or _PROJECT_ROOT / "distill_pipeline_output")
    input_path = Path(input_path or _PROJECT_ROOT / "tasteam_app_all_review_data.json")

    ds = build_dataset_flow(input_path=input_path, out_dir=out_dir)
    lb = labeling_with_pod_flow(
        train_path=ds["train_path"],
        val_path=ds["val_path"],
        test_path=ds["test_path"],
        openai_cap=openai_cap,
        output_labeled_dir=out_dir,
        public_ip_wait_timeout_sec=public_ip_wait_timeout_sec,
        vllm_ready_timeout_sec=vllm_ready_timeout_sec,
        openai_only=openai_only,
    )
    sweep_ev = run_sweep_and_evaluate_flow(
        sweep_id=sweep_id,
        sweep_yaml=sweep_yaml,
        labeled_path=lb["labeled_path"],
        output_dir=out_dir,
        val_labeled_path=lb.get("val_labeled_path"),
        test_labeled_path=lb.get("test_labeled_path"),
        base_model=student_model,
        use_pod=True,
        num_pods=num_pods,
    )
    merge_result = None
    if sweep_ev.get("best_adapter_path"):
        merge_result = merge_for_serving_flow(
            adapter_path=sweep_ev["best_adapter_path"],
            base_model=student_model,
            output_dir=out_dir,
        )
    return {"build_dataset": ds, "labeling": lb, "run_sweep_and_evaluate": sweep_ev, "merge_for_serving": merge_result}


@flow(name="train_and_evaluate_flow", log_prints=True)
def train_and_evaluate_flow(
    labeled_path: str,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    output_dir: str | Path | None = None,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> dict:
    """학습(Pod) → 평가만 실행. build_dataset, labeling, merge_for_serving은 건너뜀."""
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    tr = train_student_with_pod_flow(
        labeled_path=labeled_path,
        student_model=student_model,
        output_dir=out_dir,
    )
    ev = evaluate_flow(
        adapter_path=tr["adapter_path"],
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        output_dir=out_dir,
        base_model=student_model,
    )
    return {"train_student": tr, "evaluate": ev}


@flow(name="sweep_eval_merge_flow", log_prints=True)
def sweep_eval_merge_flow(
    labeled_path: str,
    val_labeled_path: str | None = None,
    test_labeled_path: str | None = None,
    sweep_id: str | None = None,
    sweep_yaml: str | Path | None = None,
    output_dir: str | Path | None = None,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_pods: int = 2,
) -> dict:
    """Pod에서 sweep → best adapter로 evaluate → merge(로컬). build_dataset, labeling은 건너뜀. num_pods>1이면 멀티 Pod."""
    out_dir = Path(output_dir or _PROJECT_ROOT / "distill_pipeline_output")
    sweep_ev = run_sweep_and_evaluate_flow(
        labeled_path=labeled_path,
        sweep_id=sweep_id,
        sweep_yaml=sweep_yaml,
        output_dir=out_dir,
        val_labeled_path=val_labeled_path,
        test_labeled_path=test_labeled_path,
        base_model=student_model,
        use_pod=True,
        num_pods=num_pods,
    )
    merge_result = None
    if sweep_ev.get("best_adapter_path"):
        merge_result = merge_for_serving_flow(
            adapter_path=sweep_ev["best_adapter_path"],
            base_model=student_model,
            output_dir=out_dir,
        )
    return {"run_sweep_and_evaluate": sweep_ev, "merge_for_serving": merge_result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefect flows for summary KD pipeline (distill_by_prefect.md)")
    parser.add_argument(
        "flow",
        choices=["build_dataset", "labeling_openai_only", "labeling_with_pod", "labeling_pod_only", "train_student_with_pod", "run_sweep", "train_and_evaluate", "sweep_eval_merge", "evaluate", "evaluate_on_pod", "download_eval_artifact", "merge_for_serving", "merge_for_serving_with_pod", "upload_labeled_artifact", "upload_dataset_artifact", "upload_eval_artifact", "all", "all_sweep"],
        help="Flow to run. evaluate_on_pod: Pod에서 평가 후 결과를 wandb artifact로 올리고 로컬에 다운로드. download_eval_artifact: 평가 artifact를 지정 버전으로 다운로드.",
    )
    parser.add_argument("--gold-path", type=Path, default=None, help="train_labeled_gold_only.json 경로 (labeling_pod_only 필수)")
    parser.add_argument("--sweep-id", type=str, default=None, help="wandb sweep id (optional; 없으면 --sweep-yaml으로 flow 내부에서 등록)")
    parser.add_argument("--sweep-yaml", type=Path, default=None, help="sweep 설정 yaml (sweep-id 없을 때 사용, 기본: scripts/wandb_sweep_qlora.yaml)")
    parser.add_argument("--input", type=Path, default=None, help="Input reviews JSON (default: tasteam_app_all_review_data.json)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output root (default: distill_pipeline_output)")
    parser.add_argument("--train-path", type=Path, default=None, help="train.json (for labeling_openai_only, labeling_with_pod)")
    parser.add_argument("--val-path", type=Path, default=None, help="val.json (for labeling_with_pod)")
    parser.add_argument("--test-path", type=Path, default=None, help="test.json (for labeling_with_pod)")
    parser.add_argument("--labeled-path", type=Path, default=None, help="train_labeled.json (for train_student_with_pod, run_sweep)")
    parser.add_argument("--adapter-path", type=Path, default=None, help="adapter path (for evaluate, merge_for_serving; merge_for_serving_with_pod 시 로컬 adapter)")
    parser.add_argument("--run-id", type=str, default=None, help="볼륨 상의 학습 run id (merge_for_serving_with_pod 시 --adapter-path 대신 사용)")
    parser.add_argument("--val-labeled-path", type=Path, default=None, help="val_labeled.json (for evaluate, evaluate_on_pod)")
    parser.add_argument("--test-labeled-path", type=Path, default=None, help="test_labeled.json (for evaluate, evaluate_on_pod)")
    parser.add_argument("--artifact-version", type=str, default="latest", help="For download_eval_artifact: artifact version (latest, v0, v1, ...)")
    parser.add_argument("--eval-path", type=Path, default=None, help="report.json 또는 eval/YYYYMMDD_HHMMSS 디렉터리 (upload_eval_artifact 필수)")
    parser.add_argument("--openai-cap", type=int, default=500, help="OpenAI labeling cap (for labeling_with_pod, all)")
    parser.add_argument("--use-pod", action="store_true", help="all/all_sweep에서 labeling_with_pod 사용 (기본: labeling_openai_only)")
    parser.add_argument("--public-ip-wait-timeout", type=int, default=180, help="publicIp 할당 대기 초 (labeling_with_pod, labeling_pod_only)")
    parser.add_argument("--vllm-ready-timeout", type=int, default=180, help="vLLM /v1/models 준비 대기 초 (labeling_with_pod, labeling_pod_only)")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Student model")
    parser.add_argument("--num-pods", type=int, default=2, help="sweep 시 동시 Pod 개수 (sweep_eval_merge, all_sweep; 기본 2)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or _PROJECT_ROOT / "distill_pipeline_output")

    if args.flow == "build_dataset":
        result = build_dataset_flow(input_path=args.input, out_dir=out_dir)
        print("Result:", result)
    elif args.flow == "labeling_openai_only":
        if not args.train_path:
            parser.error("labeling_openai_only requires --train-path")
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
            openai_cap=999999,
            output_labeled_dir=out_dir,
            public_ip_wait_timeout_sec=args.public_ip_wait_timeout,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
            openai_only=True,
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
            public_ip_wait_timeout_sec=args.public_ip_wait_timeout,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
            openai_only=False,
        )
        print("Result:", result)
    elif args.flow == "labeling_pod_only":
        if not args.train_path:
            parser.error("labeling_pod_only requires --train-path")
        if not args.gold_path or not args.gold_path.exists():
            parser.error("labeling_pod_only requires --gold-path (path to train_labeled_gold_only.json)")
        result = labeling_pod_only_flow(
            train_path=str(args.train_path),
            gold_path=str(args.gold_path),
            output_labeled_dir=str(args.gold_path.parent),
            public_ip_wait_timeout_sec=args.public_ip_wait_timeout,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
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
    elif args.flow == "train_and_evaluate":
        if not args.labeled_path:
            parser.error("train_and_evaluate requires --labeled-path")
        labeled_dir = Path(args.labeled_path).parent
        val_p = str(args.val_labeled_path) if args.val_labeled_path else (str(labeled_dir / "val_labeled.json") if (labeled_dir / "val_labeled.json").exists() else None)
        test_p = str(args.test_labeled_path) if args.test_labeled_path else (str(labeled_dir / "test_labeled.json") if (labeled_dir / "test_labeled.json").exists() else None)
        result = train_and_evaluate_flow(
            labeled_path=str(args.labeled_path),
            val_labeled_path=val_p,
            test_labeled_path=test_p,
            output_dir=out_dir,
            student_model=args.student_model,
        )
        print("Result keys:", list(result.keys()))
    elif args.flow == "sweep_eval_merge":
        if not args.labeled_path:
            parser.error("sweep_eval_merge requires --labeled-path")
        labeled_dir = Path(args.labeled_path).parent
        val_p = str(args.val_labeled_path) if args.val_labeled_path else (str(labeled_dir / "val_labeled.json") if (labeled_dir / "val_labeled.json").exists() else None)
        test_p = str(args.test_labeled_path) if args.test_labeled_path else (str(labeled_dir / "test_labeled.json") if (labeled_dir / "test_labeled.json").exists() else None)
        sweep_yaml = args.sweep_yaml if args.sweep_yaml is not None else DEFAULT_SWEEP_YAML
        result = sweep_eval_merge_flow(
            labeled_path=str(args.labeled_path),
            val_labeled_path=val_p,
            test_labeled_path=test_p,
            sweep_id=args.sweep_id,
            sweep_yaml=sweep_yaml,
            output_dir=out_dir,
            student_model=args.student_model,
            num_pods=args.num_pods,
        )
        print("Result keys:", list(result.keys()))
        if result.get("merge_for_serving"):
            print("API 사용: LLM_MODEL=" + result["merge_for_serving"]["merged_model_path"])
    elif args.flow == "upload_labeled_artifact":
        if not args.labeled_path:
            parser.error("upload_labeled_artifact requires --labeled-path (train_labeled.json or labeled dir)")
        result = upload_labeled_artifact_flow(
            labeled_path=args.labeled_path,
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
        )
        print("Result:", result)
    elif args.flow == "upload_dataset_artifact":
        if not args.train_path:
            parser.error("upload_dataset_artifact requires --train-path (e.g. .../datasets/YYYYMMDD_HHMMSS/train.json)")
        result = upload_dataset_artifact_flow(
            dataset_path=args.train_path,
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
        )
        print("Result:", result)
    elif args.flow == "upload_eval_artifact":
        if not args.eval_path or not args.eval_path.exists():
            parser.error("upload_eval_artifact requires --eval-path (e.g. .../eval/YYYYMMDD_HHMMSS/report.json or .../eval/YYYYMMDD_HHMMSS)")
        result = upload_eval_artifact_flow(
            eval_path=args.eval_path,
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
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
    elif args.flow == "evaluate_on_pod":
        if not args.adapter_path:
            parser.error("evaluate_on_pod requires --adapter-path")
        if not args.val_labeled_path or not args.val_labeled_path.exists():
            parser.error("evaluate_on_pod requires --val-labeled-path (path to val_labeled.json)")
        result = evaluate_on_pod_flow(
            adapter_path=str(args.adapter_path),
            val_labeled_path=str(args.val_labeled_path),
            test_labeled_path=str(args.test_labeled_path) if args.test_labeled_path else None,
            output_dir=out_dir,
            base_model=args.student_model,
        )
        print("Result:", result)
    elif args.flow == "download_eval_artifact":
        result = download_eval_artifact_flow(
            artifact_version=args.artifact_version,
            output_dir=out_dir,
            project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            entity=os.environ.get("WANDB_ENTITY"),
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
            public_ip_wait_timeout_sec=args.public_ip_wait_timeout,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
            student_model=args.student_model,
            openai_only=not args.use_pod,
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
            public_ip_wait_timeout_sec=args.public_ip_wait_timeout,
            vllm_ready_timeout_sec=args.vllm_ready_timeout,
            student_model=args.student_model,
            num_pods=args.num_pods,
            openai_only=not args.use_pod,
        )
        print("Result keys:", list(result.keys()))


if __name__ == "__main__":
    main()
