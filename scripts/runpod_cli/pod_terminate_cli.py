import os
import requests

class RunPodClient:
    def __init__(self, token: str):
        self.base_url = "https://rest.runpod.io/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def delete_pod(self, pod_id: str):
        url = f"{self.base_url}/pods/{pod_id}"
        resp = requests.delete(url, headers=self.headers, timeout=30)

        if resp.status_code == 404:
            return {"status": "already_deleted"}

        resp.raise_for_status()

        if resp.text.strip():
            return resp.json()
        return {"status": "deleted"}


if __name__ == "__main__":
    token = os.environ["RUNPOD_API_KEY"]  # 하나로 통일
    client = RunPodClient(token)
    print(client.delete_pod("0hs9up7f8mdl41"))