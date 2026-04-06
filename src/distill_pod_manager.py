from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple

import requests

from .config import Config

logger = logging.getLogger(__name__)


class DistillPodManager:
    """Distill Pod lifecycle manager (MVP).

    상태:
    - STOPPED: pod 없음
    - CREATING: 생성/기동 중
    - READY: 사용 가능
    - UNHEALTHY: 장애 감지(쿨다운 중)
    - DELETING: 삭제 중
    """

    STOPPED = "STOPPED"
    CREATING = "CREATING"
    READY = "READY"
    UNHEALTHY = "UNHEALTHY"
    DELETING = "DELETING"

    def __init__(self) -> None:
        self.state: str = self.STOPPED
        self.current_pod_id: Optional[str] = None
        self.current_base_url: Optional[str] = None
        self.last_failure_at: Optional[float] = None
        self._pending_delete_ids: Set[str] = set()
        self._op_lock = asyncio.Lock()

    def _in_cooldown(self) -> bool:
        if not self.last_failure_at:
            return False
        return (time.time() - self.last_failure_at) < max(0, Config.DISTILL_POD_COOLDOWN_SECONDS)

    @staticmethod
    def _headers() -> Dict[str, str]:
        token = (Config.RUNPOD_API_KEY or "").strip()
        if not token:
            raise ValueError("RUNPOD_API_KEY is required for distill pod auto-create")
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _runpod_api_base() -> str:
        return "https://rest.runpod.io/v1"

    @staticmethod
    def _proxy_base_url(pod_id: str) -> str:
        port = Config.DISTILL_POD_PROXY_PORT
        return f"https://{pod_id}-{port}.proxy.runpod.net/v1"

    @staticmethod
    def _build_create_payload() -> Dict[str, Any]:
        image = (Config.DISTILL_POD_IMAGE_NAME or "").strip()
        vol_id = (Config.DISTILL_POD_NETWORK_VOLUME_ID or "").strip()
        if not image:
            raise ValueError("DISTILL_POD_IMAGE_NAME is required when DISTILL_POD_AUTO_CREATE=true")
        if not vol_id:
            raise ValueError("DISTILL_POD_NETWORK_VOLUME_ID is required when DISTILL_POD_AUTO_CREATE=true")
        return {
            "computeType": "GPU",
            "cloudType": "SECURE",
            "name": Config.DISTILL_POD_NAME,
            "imageName": image,
            "gpuCount": 1,
            "gpuTypeIds": [Config.DISTILL_POD_GPU_TYPE_ID],
            "networkVolumeId": vol_id,
            "volumeMountPath": "/workspace",
            "ports": [f"{Config.DISTILL_POD_PROXY_PORT}/http"],
            "supportPublicIp": True,
            "containerDiskInGb": 50,
            "vcpuCount": 2,
            "volumeInGb": 20,
            "dockerEntrypoint": [],
            "dockerStartCmd": [],
            "env": {},
        }

    def _create_pod_sync(self) -> Tuple[str, str]:
        payload = self._build_create_payload()
        resp = requests.post(
            f"{self._runpod_api_base()}/pods",
            headers=self._headers(),
            json=payload,
            timeout=Config.DISTILL_POD_HEALTHCHECK_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json() if resp.text.strip() else {}
        pod_id = data.get("id")
        if not pod_id:
            raise RuntimeError(f"RunPod create response missing id: {data}")

        deadline = time.time() + max(1, Config.DISTILL_POD_CREATE_TIMEOUT_SECONDS)
        last_data: Dict[str, Any] = {}
        while time.time() < deadline:
            s = requests.get(
                f"{self._runpod_api_base()}/pods/{pod_id}",
                headers=self._headers(),
                timeout=Config.DISTILL_POD_HEALTHCHECK_TIMEOUT_SECONDS,
            )
            s.raise_for_status()
            last_data = s.json() if s.text.strip() else {}
            desired = (last_data.get("desiredStatus") or "").upper()
            status = (last_data.get("status") or last_data.get("runtimeStatus") or "").upper()
            if desired == "RUNNING" and (status in ("RUNNING", "") or "RUN" in status):
                break
            time.sleep(5)
        else:
            raise TimeoutError(f"Pod did not reach RUNNING in time. last={last_data}")

        base_url = self._proxy_base_url(pod_id)
        h = requests.get(
            f"{base_url}/models",
            timeout=Config.DISTILL_POD_HEALTHCHECK_TIMEOUT_SECONDS,
        )
        h.raise_for_status()
        return pod_id, base_url

    def _delete_pod_sync(self, pod_id: str) -> None:
        r = requests.delete(
            f"{self._runpod_api_base()}/pods/{pod_id}",
            headers=self._headers(),
            timeout=Config.DISTILL_POD_HEALTHCHECK_TIMEOUT_SECONDS,
        )
        if r.status_code not in (200, 202, 204, 404):
            r.raise_for_status()

    async def _ensure_ready_locked(self) -> Optional[str]:
        if self.current_base_url and self.state == self.READY:
            return self.current_base_url

        if self._in_cooldown():
            return None

        if Config.DISTILL_POD_BASE_URL:
            self.current_base_url = Config.DISTILL_POD_BASE_URL.rstrip("/")
            self.current_pod_id = None
            self.state = self.READY
            return self.current_base_url

        if not Config.DISTILL_POD_AUTO_CREATE:
            return None

        self.state = self.CREATING
        pod_id, base_url = await asyncio.to_thread(self._create_pod_sync)
        self.current_pod_id = pod_id
        self.current_base_url = base_url
        self.state = self.READY
        logger.info("Distill Pod READY: pod_id=%s base_url=%s", pod_id, base_url)
        return base_url

    async def get_ready_base_url(self) -> Optional[str]:
        async with self._op_lock:
            try:
                return await self._ensure_ready_locked()
            except Exception:
                self.state = self.UNHEALTHY
                self.last_failure_at = time.time()
                raise

    async def mark_unhealthy_and_schedule_delete(self, reason: str) -> None:
        async with self._op_lock:
            logger.warning("Distill Pod unhealthy: %s", reason)
            self.state = self.UNHEALTHY
            self.last_failure_at = time.time()
            if self.current_pod_id:
                self._pending_delete_ids.add(self.current_pod_id)
            self.current_pod_id = None
            self.current_base_url = None

    async def flush_pending_deletes(self) -> None:
        async with self._op_lock:
            if not self._pending_delete_ids:
                return
            self.state = self.DELETING
            to_delete = list(self._pending_delete_ids)
            for pod_id in to_delete:
                try:
                    await asyncio.to_thread(self._delete_pod_sync, pod_id)
                    self._pending_delete_ids.discard(pod_id)
                    logger.info("Distill Pod deleted: %s", pod_id)
                except Exception as e:
                    logger.warning("Distill Pod delete failed (retry later): pod_id=%s err=%s", pod_id, e)
            if self._pending_delete_ids:
                self.state = self.UNHEALTHY
            else:
                self.state = self.STOPPED

    async def handle_failure(self, reason: str) -> None:
        await self.mark_unhealthy_and_schedule_delete(reason)
        if Config.DISTILL_POD_DELETE_ON_FAILURE:
            await self.flush_pending_deletes()

    def _run_coro_sync(self, coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Sync wrapper cannot run while event loop is running")
        except RuntimeError as e:
            if "cannot run while event loop is running" in str(e):
                raise
            return asyncio.run(coro)

    def get_ready_base_url_sync(self) -> Optional[str]:
        return self._run_coro_sync(self.get_ready_base_url())

    def handle_failure_sync(self, reason: str) -> None:
        self._run_coro_sync(self.handle_failure(reason))

    def flush_pending_deletes_sync(self) -> None:
        self._run_coro_sync(self.flush_pending_deletes())


_manager: Optional[DistillPodManager] = None


def get_distill_pod_manager() -> DistillPodManager:
    global _manager
    if _manager is None:
        _manager = DistillPodManager()
    return _manager
