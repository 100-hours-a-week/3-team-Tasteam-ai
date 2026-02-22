import os
import time
import requests
from typing import Any, Dict, Optional


class RunPodClient:
    def __init__(self, token: str, base_url: str = "https://rest.runpod.io/v1", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def create_pod(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/pods"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        return self._handle_json_response(resp)

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
    token = os.environ["RUNPOD_API_KEY"]  # 너가 쓰는 키 이름으로 통일
    client = RunPodClient(token)

    payload = {
        "allowedCudaVersions": ["13.0"],
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": 50,
        "cpuFlavorPriority": "availability",
        "dataCenterIds": [
            "EU-RO-1","CA-MTL-1","EU-SE-1","US-IL-1","EUR-IS-1","EU-CZ-1","US-TX-3","EUR-IS-2",
            "US-KS-2","US-GA-2","US-WA-1","US-TX-1","CA-MTL-3","EU-NL-1","US-TX-4","US-CA-2",
            "US-NC-1","OC-AU-1","US-DE-1","EUR-IS-3","CA-MTL-2","AP-JP-1","EUR-NO-1","EU-FR-1",
            "US-KS-3","US-GA-1"
        ],
        "dataCenterPriority": "availability",
        "dockerEntrypoint": [],
        "dockerStartCmd": [],
        "env": {"ENV_VAR": "value"},
        "globalNetworking": False,
        "gpuCount": 1,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
        "gpuTypePriority": "availability",
        "imageName": "jinsoo1218/vllm-pod:latest",
        "interruptible": False,
        "locked": False,
        "minDiskBandwidthMBps": 123,
        "minDownloadMbps": 123,
        "minRAMPerGPU": 8,
        "minUploadMbps": 123,
        "minVCPUPerGPU": 2,
        "name": "vllm-pod",
        "networkVolumeId": "2kn4qj6rql",
        "ports": ["8000/http", "22/tcp"],
        "supportPublicIp": True,
        "vcpuCount": 2,
        "volumeInGb": 20,
        "volumeMountPath": "/workspace",
    }

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
