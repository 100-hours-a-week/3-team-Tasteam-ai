RunPod Network Volume이 **S3-compatible API**를 제공해서, `aws s3` CLI로 **업로드/다운로드/삭제/목록 보기** 같은 기본 파일 작업을 할 수 있어. ([Runpod Documentation][1])

네가 쓴 `aws s3 ls ...`는 “목록 보기”고, 업로드는 `cp` 또는 `sync`를 쓰면 돼.

---

## 1) 업로드 예시 (`aws s3 cp`)

로컬 파일을 RunPod Network Volume(=버킷)로 업로드:

```bash
aws s3 cp ./adapter_model.safetensors \
  s3://v3i546pkrz/models/adapter_model.safetensors \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io
```

* `s3://v3i546pkrz/` → **볼륨 ID가 버킷 이름** 역할 ([Runpod Documentation][1])
* `--endpoint-url ...runpod.io` → AWS가 아니라 **RunPod S3 API 엔드포인트로 접속** ([Runpod Documentation][1])
* 경로는 볼륨 루트 기준 “객체 키”로 들어가고, Pod에서는 `/workspace/...`로 보이게 돼 ([Runpod Documentation][1])

---

## 2) 폴더 통째로 업로드 (`aws s3 sync`)

```bash
aws s3 sync ./distill_pipeline_output/ \
  s3://v3i546pkrz/distill_pipeline_output/ \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io
```

문서상 `sync`도 동작하긴 하는데, 파일이 아주 많거나(예: 10,000+), 복잡한 구조면 이슈가 있을 수 있다고 언급돼. ([Runpod Documentation][1])

---

## 3) 다운로드 예시

```bash
aws s3 cp \
  s3://v3i546pkrz/distill_pipeline_output/runs/xxx/adapter/adapter_model.safetensors \
  ./adapter_model.safetensors \
  --region eu-ro-1 \
  --endpoint-url https://s3api-eu-ro-1.runpod.io
```

---

## 4) 중요한 전제: “인증키”가 있어야 함

이게 Public 버킷처럼 열려 있는 게 아니라, **RunPod에서 제공하는 S3 Access/Secret 키로 인증**하고 써야 해. (AWS CLI가 그 키를 사용하도록 설정해야 함) ([Runpod Documentation][1])

---

## 5) Python으로 업로드 (Prefect 라벨링 연동)

프로젝트에서 라벨링 결과를 RunPod 네트워크 볼륨에 올리려면 환경변수를 설정한 뒤 라벨링 flow를 실행하면 된다.

* **환경변수**: `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_ACCESS_KEY` (필수), `RUNPOD_NETWORK_VOLUME_ID` (선택, 기본 v3i546pkrz (distill_train_net_vol.md))
* **동작**: `labeling_flow` / `labeling_with_pod_flow` 실행 후, 위 환경변수가 있으면 라벨 디렉터리(`train_labeled.json` 등)를 자동으로 네트워크 볼륨 `labeled/YYYYMMDD_HHMMSS/` 에 업로드한다.
* **스크립트**: `scripts/runpod_cli/runpod_s3_upload.py` — 디렉터리 업로드용. CLI 예: `python -m runpod_cli.runpod_s3_upload ./labeled/20260216_123456 --volume-id v3i546pkrz`

---

### 한 줄 결론

네가 보여준 `aws s3 ls ...` 방식 그대로, **`aws s3 cp` / `aws s3 sync`로 RunPod 네트워크 볼륨에 파일 올릴 수 있어.** ([Runpod Documentation][1])

[1]: https://docs.runpod.io/storage/s3-api?utm_source=chatgpt.com "S3-compatible API"
