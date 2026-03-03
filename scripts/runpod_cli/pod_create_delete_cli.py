import os
import time
import requests
from typing import Any, Dict, Literal, Optional

from . import runpod_config


class RunPodClient:
    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or runpod_config.get_api_base_url()).rstrip("/")
        self.timeout = timeout if timeout is not None else runpod_config.get_api_timeout()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def create_pod(
        self,
        payload: Dict[str, Any],
        max_retries: int = 4,
        retry_statuses: tuple[int, ...] = (500, 502, 503),
        base_delay_sec: float = 2.0,
    ) -> Dict[str, Any]:
        """Pod 생성. 500/502/503 또는 ConnectionError/ReadTimeout/ChunkedEncodingError 시 지수 백오프 재시도."""
        url = f"{self.base_url}/pods"
        last_exc = None
        retryable = (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
        )
        for attempt in range(max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                return self._handle_json_response(resp)
            except requests.HTTPError as e:
                last_exc = e
                status = e.response.status_code if e.response is not None else None
                if status in retry_statuses and attempt < max_retries - 1:
                    time.sleep(base_delay_sec * (2**attempt))
                    continue
                raise
            except retryable as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(base_delay_sec * (2**attempt))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return {}

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/pods/{pod_id}"
        resp = self.session.get(url, timeout=self.timeout)
        return self._handle_json_response(resp)

    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/pods/{pod_id}"
        resp = self.session.delete(url, timeout=self.timeout)

        if resp.status_code == 404:
            return {"status": "already_deleted"}

        # DELETE는 응답 body 비는 경우가 있어서 처리
        if resp.status_code >= 400:
            return self._handle_json_response(resp)

        return resp.json() if resp.text.strip() else {"status": "deleted"}

    def wait_until_running(
        self,
        pod_id: str,
        timeout_sec: int = 600,
        poll_interval_sec: int = 5,
    ) -> Dict[str, Any]:
        """
        desiredStatus / status / publicIp 같은 필드를 폴링해서
        'RUNNING'에 도달할 때까지 대기.
        """
        deadline = time.time() + timeout_sec
        last = None

        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod

            # 응답 필드명은 환경/버전에 따라 다를 수 있어서 방어적으로 체크
            desired = (pod.get("desiredStatus") or "").upper()
            status = (pod.get("status") or pod.get("runtimeStatus") or "").upper()

            # 원하는 조건을 더 엄격히 잡고 싶으면 여기를 조정
            if desired == "RUNNING" and (status in ("RUNNING", "", None) or "RUN" in status):
                return pod

            time.sleep(poll_interval_sec)

        raise TimeoutError(f"Pod {pod_id} did not reach RUNNING within {timeout_sec}s. Last: {last}")

    def wait_for_public_ip(
        self,
        pod_id: str,
        timeout_sec: int = 180,
        poll_interval_sec: int = 5,
    ) -> Dict[str, Any]:
        """
        wait_until_running 이후 publicIp가 할당될 때까지 폴링.
        labeling Pod처럼 publicIp로 접속해야 하는 경우에 사용.
        """
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod
            public_ip = (pod.get("publicIp") or "").strip()
            if public_ip:
                return pod
            time.sleep(poll_interval_sec)
        raise TimeoutError(f"Pod {pod_id} publicIp not assigned within {timeout_sec}s. Last: {last}")

    def wait_for_port_mappings(
        self,
        pod_id: str,
        internal_port: int = 8000,
        timeout_sec: int = 120,
        poll_interval_sec: int = 5,
    ) -> Dict[str, Any]:
        """
        publicIp 할당 후 portMappings에 내부 포트가 채워질 때까지 폴링.
        RunPod는 초기화 중에는 portMappings가 비어 있을 수 있음.
        """
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod
            port_mappings = pod.get("portMappings") or {}
            if port_mappings.get(str(internal_port)) is not None or port_mappings.get(internal_port) is not None:
                return pod
            time.sleep(poll_interval_sec)
        raise TimeoutError(
            f"Pod {pod_id} portMappings[{internal_port}] not assigned within {timeout_sec}s. Last portMappings: {last.get('portMappings') if last else None}"
        )

    def wait_until_stopped(
        self,
        pod_id: str,
        timeout_sec: int = 14400,
        poll_interval_sec: int = 60,
    ) -> Dict[str, Any]:
        """
        Pod가 RUNNING이 아닌 상태(컨테이너 종료 등)가 될 때까지 폴링.
        sweep 등 장시간 작업 후 종료 감지용.
        """
        deadline = time.time() + timeout_sec
        last = None
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod
            pod_status = (pod.get("status") or "").upper()
            if pod_status in ("ALREADY_DELETED", "DELETED"):
                return pod
            desired = (pod.get("desiredStatus") or "").upper()
            status = (pod.get("status") or pod.get("runtimeStatus") or "").upper()
            if desired != "RUNNING" or ("EXIT" in status or "STOP" in status or status == "COMPLETED" or status == "DELETED"):
                return pod
            time.sleep(poll_interval_sec)
        raise TimeoutError(f"Pod {pod_id} did not stop within {timeout_sec}s. Last: {last}")

    @staticmethod
    def get_default_pod_payload(
        use: Literal["labeling", "train", "merge"] = "labeling",
        docker_start_cmd: list[str] | None = None,
    ) -> Dict[str, Any]:
        """Pod 생성용 기본 payload.
        use: "labeling" → vLLM 이미지/볼륨, "train" → 학습 이미지/볼륨, "merge" → 학습 이미지/볼륨에서 merge 스크립트 실행.
        docker_start_cmd: 지정 시 컨테이너 CMD 오버라이드 (train 시 --labeled-path 등 전달용, merge 시 merge 스크립트 경로+인자).
        기본값은 config/runpod.yaml 및 환경 변수에서 로드.
        """
        if use == "train":
            image_name = runpod_config.get_pod_image_train()
            network_volume_id = runpod_config.get_pod_network_volume_id_train()
            name = "train-pod"
        elif use == "merge":
            image_name = runpod_config.get_pod_image_train()
            network_volume_id = runpod_config.get_pod_network_volume_id_merge()
            name = "merge-pod"
        else:
            image_name = runpod_config.get_pod_image_labeling()
            network_volume_id = runpod_config.get_pod_network_volume_id_labeling()
            name = "vllm-pod"
        payload = {
            "allowedCudaVersions": runpod_config.get_pod_allowed_cuda_versions(),
            "cloudType": "SECURE",
            "computeType": "GPU",
            "containerDiskInGb": runpod_config.get_pod_common_int("container_disk_in_gb", 50),
            "cpuFlavorPriority": "availability",
            "dataCenterIds": runpod_config.get_pod_data_center_ids(),
            "dataCenterPriority": "availability",
            "dockerEntrypoint": [],
            "dockerStartCmd": docker_start_cmd if docker_start_cmd is not None else [],
            "env": {"ENV_VAR": "value",**({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),},
            "globalNetworking": False,
            "gpuCount": runpod_config.get_pod_common_int("gpu_count", 1),
            "gpuTypeIds": runpod_config.get_pod_gpu_type_ids(),
            "gpuTypePriority": "availability",
            "imageName": image_name,
            "interruptible": False,
            "locked": False,
            "minDiskBandwidthMBps": 123,
            "minDownloadMbps": 123,
            "minRAMPerGPU": 8,
            "minUploadMbps": 123,
            "minVCPUPerGPU": 2,
            "name": name,
            "networkVolumeId": network_volume_id,
            "ports": ["8000/http", "22/tcp"],
            "supportPublicIp": True,
            "vcpuCount": runpod_config.get_pod_common_int("vcpu_count", 2),
            "volumeInGb": 20,
            "volumeMountPath": "/workspace",
        }
        if use == "merge":
            payload["dockerEntrypoint"] = ["python"]
        return payload

    def _handle_json_response(self, resp: requests.Response) -> Dict[str, Any]:
        # 에러 메시지를 보기 좋게
        try:
            data = resp.json() if resp.text.strip() else {}
        except Exception:
            data = {"raw": resp.text}

        if resp.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {resp.status_code} for {resp.request.method} {resp.url}: {data}",
                response=resp,
            )
        return data


if __name__ == "__main__":
    token = os.environ["RUNPOD_API_KEY"]
    client = RunPodClient(token)
    payload = RunPodClient.get_default_pod_payload()

    pod = client.create_pod(payload)
    pod_id = pod["id"]
    print("created:", pod_id, pod.get("desiredStatus"))


    try:
        ready = client.wait_until_running(pod_id)
        print("Pod ready:", ready.get("id"), ready.get("publicIp"), ready.get("ports"))

        # ✅ 여기서 실제 작업 수행
        # 예) vLLM health check / 요청 테스트 등

    finally:
        print("Cleaning up pod...")
        print(client.delete_pod(pod_id))
