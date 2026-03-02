"""
RunPod 설정 로드. config/runpod.yaml 사용, 환경 변수로 오버라이드.

- API 키·S3 시크릿: 환경 변수만 사용 (RUNPOD_API_KEY, RUNPOD_S3_ACCESS_KEY 등).
- 볼륨 ID, 이미지명, region 등: runpod.yaml 기본값, env로 오버라이드.

설정 파일 경로: RUNPOD_CONFIG_PATH 미설정 시 프로젝트 루트 config/runpod.yaml
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "runpod.yaml"

_cached: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _cached
    if _cached is not None:
        return _cached
    path = os.environ.get("RUNPOD_CONFIG_PATH", "")
    if path and Path(path).is_file():
        config_path = Path(path)
    elif _DEFAULT_CONFIG_PATH.is_file():
        config_path = _DEFAULT_CONFIG_PATH
    else:
        _cached = {}
        return _cached
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            _cached = yaml.safe_load(f) or {}
    except Exception:
        _cached = {}
    return _cached


def _get(category: str, *keys: str, default: Any = None) -> Any:
    d = _load()
    for k in (category, *keys):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


# --- S3 ---
def get_s3_region() -> str:
    return os.environ.get("RUNPOD_S3_REGION") or _get("s3", "region") or "eu-ro-1"


def get_s3_endpoint_url() -> str:
    return os.environ.get("RUNPOD_S3_ENDPOINT_URL") or _get("s3", "endpoint_url") or "https://s3api-eu-ro-1.runpod.io"


def get_s3_volume_id_default() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or _get("s3", "volume_id") or "v3i546pkrz"


def get_volume_id_train() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN") or os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or _get("volumes", "train") or _get("s3", "volume_id") or "v3i546pkrz"


def get_volume_id_labeling() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_LABELING") or _get("volumes", "labeling") or "o3a3ya7flt"


def get_volume_id_merge() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN") or os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or _get("volumes", "merge") or "v3i546pkrz"


# --- Pod (이미지, 볼륨, payload 공통) ---
def get_pod_image_train() -> str:
    return os.environ.get("RUNPOD_POD_IMAGE_NAME_TRAIN") or _get("pod", "train", "image_name") or "jinsoo1218/train-llm:latest"


def get_pod_image_labeling() -> str:
    return os.environ.get("RUNPOD_POD_IMAGE_NAME_LABELING") or _get("pod", "labeling", "image_name") or "jinsoo1218/runpod-pod-vllm:latest"


def get_pod_network_volume_id_train() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN") or os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or _get("pod", "train", "network_volume_id") or "v3i546pkrz"


def get_pod_network_volume_id_labeling() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_LABELING") or _get("pod", "labeling", "network_volume_id") or "o3a3ya7flt"


def get_pod_network_volume_id_merge() -> str:
    return os.environ.get("RUNPOD_NETWORK_VOLUME_ID_TRAIN") or os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or _get("pod", "merge", "network_volume_id") or "v3i546pkrz"


def get_pod_gpu_type_ids() -> list[str]:
    val = _get("pod", "gpu_type_ids")
    if isinstance(val, list) and val:
        return val
    return ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"]


def get_pod_data_center_ids() -> list[str]:
    val = _get("pod", "data_center_ids")
    if isinstance(val, list) and val:
        return val
    return [
        "EU-RO-1", "CA-MTL-1", "EU-SE-1", "US-IL-1", "EUR-IS-1", "EU-CZ-1", "US-TX-3", "EUR-IS-2",
        "US-KS-2", "US-GA-2", "US-WA-1", "US-TX-1", "CA-MTL-3", "EU-NL-1", "US-TX-4", "US-CA-2",
        "US-NC-1", "OC-AU-1", "US-DE-1", "EUR-IS-3", "CA-MTL-2", "AP-JP-1", "EUR-NO-1", "EU-FR-1",
        "US-KS-3", "US-GA-1",
    ]


def get_pod_allowed_cuda_versions() -> list[str]:
    val = _get("pod", "allowed_cuda_versions")
    if isinstance(val, list) and val:
        return val
    return ["13.0"]


def get_pod_common_int(key: str, default: int) -> int:
    val = _get("pod", key)
    if isinstance(val, int):
        return val
    return default


# --- API ---
def get_api_base_url() -> str:
    return os.environ.get("RUNPOD_API_BASE_URL") or _get("api", "base_url") or "https://rest.runpod.io/v1"


def get_api_timeout() -> int:
    if os.environ.get("RUNPOD_API_TIMEOUT"):
        try:
            return int(os.environ["RUNPOD_API_TIMEOUT"])
        except ValueError:
            pass
    return _get("api", "timeout") or 120


def reload() -> None:
    """캐시 초기화 (테스트 또는 설정 파일 변경 시)."""
    global _cached
    _cached = None
