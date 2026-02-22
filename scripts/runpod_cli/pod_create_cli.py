import os
import requests

RUNPOD_TOKEN = os.environ["RUNPOD_API_KEY"]  # 환경변수로 넣는 걸 추천
URL = "https://rest.runpod.io/v1/pods"

headers = {
    "Authorization": f"Bearer {RUNPOD_TOKEN}",
    "Content-Type": "application/json",
}

payload = {
    "allowedCudaVersions": ["13.0"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "containerDiskInGb": 50,
    "containerRegistryAuthId": "",
    # "countryCodes": ["KR"],  # 예: 필요하면 실제 값으로
    # "cpuFlavorIds": ["cpu3c"],  # GPU면 보통 불필요/자동인 경우도 있음 (스펙 확인)
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
    "networkVolumeId": "2kn4qj6rql",   # <-- 여기 실제 ID로
    "ports": ["8888/http", "22/tcp"],
    "supportPublicIp": True,
    "vcpuCount": 2,
    "volumeInGb": 20,
    "volumeMountPath": "/workspace",
}

resp = requests.post(URL, headers=headers, json=payload, timeout=60)
print("status:", resp.status_code)

# 에러면 원인 보기 좋게
try:
    data = resp.json()
except Exception:
    print(resp.text)
    raise

print(data)

# 실패 시 예외
resp.raise_for_status()