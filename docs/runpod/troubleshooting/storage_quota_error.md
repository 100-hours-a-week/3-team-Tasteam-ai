boto3.exceptions.S3UploadFailedError: Failed to upload distill_pipeline_output/labeled/20260226_051037/test_labeled.json to v3i546pkrz/labeled/20260226_051037/test_labeled.json: An error occurred (InsufficientStorage) when calling the PutObject operation: bucket storage quota exceeded
16:24:02.002 | INFO    | prefect - Stopping temporary server on http://127.0.0.1:8950

---


RunPod **Network Volume**(S3 호환) 쪽에서 **저장 공간 한도를 넘었다**는 뜻입니다.

- **의미**: `v3i546pkrz` 볼륨(학습용으로 쓰는 그 볼륨)의 **스토리지 할당량이 초과**해서 `PutObject`가 실패했습니다.
- **발생 위치**: `distill_pipeline_output/labeled/...` 를 볼륨에 업로드할 때(예: sweep 전 `upload_labeled_to_volume_for_sweep_task` 또는 Pod 학습 전 라벨 업로드).

**할 수 있는 것:**

1. **볼륨 용량 늘리기**  
   RunPod 콘솔에서 해당 Network Volume(`v3i546pkrz`)의 **용량(GB)을 늘립니다**. 유료 옵션이라면 플랜/요금 확인이 필요합니다.

2. **볼륨 안의 불필요한 데이터 지우기**  
   같은 볼륨에 예전 run 결과, adapter, 캐시 등이 쌓여 있을 수 있으니:
   - RunPod 콘솔에서 볼륨 내용을 보고,  
   - 또는 S3 API로 `v3i546pkrz` 버킷 리스트 후  
   오래된 `runs/`, `labeled/`, `merge_input/` 등 **삭제해도 되는 경로**를 지워서 공간을 비웁니다.

3. **업로드하는 데이터 줄이기**  
   - `test_labeled.json` 등이 매우 크면, sweep 시에는 **train만** 올리고 val/test는 제외하는 식으로 업로드 범위를 줄일 수 있습니다(코드/플로우에서 “볼륨에 올리는 파일 목록”을 조정해야 함).

4. **다른 볼륨 사용**  
   새 Network Volume을 만들어 용량을 더 주고, 그 볼륨 ID를 `RUNPOD_NETWORK_VOLUME_ID_TRAIN`(또는 sweep/학습에서 쓰는 볼륨 설정)에 넣어 사용하는 방법입니다.

**우선 확인할 것:**  
RunPod 대시보드에서 Volume `v3i546pkrz` 의 **사용량 / 최대 용량**을 보고, 필요하면 위 1·2번으로 공간을 확보하면 됩니다.

---


코드 기준으로 **훈련용 네트워크 볼륨**(`v3i546pkrz` / `RUNPOD_NETWORK_VOLUME_ID_TRAIN`)에는 아래 구조로 적재됩니다. (S3 키 = Pod 안 `/workspace/` 아래 경로와 동일)

---

## 1. **labeled/** — 라벨링 결과

- **누가**: `upload_labeled_dir_to_runpod()` (sweep 전 업로드, 또는 `train_student_with_pod` 시)
- **경로**: `labeled/{version}/`  
  예: `labeled/20260226_051037/`
- **파일**:  
  - `train_labeled.json`  
  - `val_labeled.json` (있을 때)  
  - `test_labeled.json` (있을 때)  
  - `train_labeled_gold_only.json` (있을 때)
- **용도**: Pod에서 학습/스윕 시 읽는 라벨 데이터

---

## 2. **distill_pipeline_output/runs/{run_id}/adapter/** — 학습된 어댑터

- **누가**: Pod 안에서 돌리는 `train_qlora.py` / `run_qlora_sweep.py` (출력 디렉터리 `/workspace/distill_pipeline_output`)
- **경로**: `distill_pipeline_output/runs/{run_id}/adapter/`
- **용도**: run별 QLoRA 어댑터. `train_student_with_pod` 후 여기서 다운로드하거나, `merge_for_serving_with_pod --run-id RUN_ID` 시 사용

---

## 3. **distill_pipeline_output/merge_input/**, **merge_scripts/**, **merged_for_serving/**

- **merge_input/{version}/adapter/**  
  로컬 어댑터를 Pod merge용으로 올릴 때 사용 (`merge_for_serving_with_pod --adapter-path ...`).
- **merge_scripts/merge_adapter_for_serving.py**  
  Merge Pod에서 실행하는 스크립트 1개.
- **merged_for_serving/{version}/**  
  Merge 결과 (merged 모델, `merge_meta.json` 등).
- **merged_for_serving/latest_merged_path.json**  
  “최신 merge 결과 경로” 포인터 파일.

---

## 요약 표

| S3 prefix (볼륨 내 경로) | 내용 |
|--------------------------|------|
| `labeled/{version}/` | 라벨 JSON (train/val/test_labeled.json 등) |
| `distill_pipeline_output/runs/{run_id}/adapter/` | run별 QLoRA 어댑터 |
| `distill_pipeline_output/merge_input/{version}/adapter/` | merge용으로 올린 어댑터 |
| `distill_pipeline_output/merge_scripts/` | merge 스크립트 |
| `distill_pipeline_output/merged_for_serving/{version}/` | merge 결과물 |
| `distill_pipeline_output/merged_for_serving/latest_merged_path.json` | 최신 merge 경로 |

용량을 줄이려면 **오래된 `labeled/` 버전**과 **더 이상 쓰지 않는 `distill_pipeline_output/runs/`**, 필요 없어진 **`merge_input/`·`merged_for_serving/`** 를 정리하면 됩니다.

---


가능 여부만 짧게 정리하면 아래와 같습니다.

---

## 1. Label – 한 번만 저장 (이미 있으면 업로드 스킵)

**가능합니다.**

- `upload_labeled_dir_to_runpod` / `upload_labeled_to_volume_for_sweep_task` 호출 전에,  
  볼륨에 `labeled/{version}/train_labeled.json`(또는 같은 버전의 대표 파일)이 **이미 있는지** S3 `head_object` 등으로 확인합니다.
- 있으면 업로드를 건너뛰고, 없을 때만 업로드하면 됩니다.
- `runpod_s3_upload.py`의 `object_exists`를 쓰거나, 같은 방식으로 “이 버전 이미 있음”만 검사하면 됩니다.

---

## 2. Adapter – 볼륨에는 두지 않고 artifact에만

**가능하지만, 학습 스크립트/플로우 수정이 필요합니다.**

- **Sweep**: 이미 best adapter는 `get_best_adapter_from_artifact_task`로 **artifact만** 쓰고 있어서, “결과 사용” 관점에서는 볼륨에 adapter가 없어도 됩니다.
- **문제**: Pod 안에서 돌아가는 `train_qlora.py` / `run_qlora_sweep.py`가 **지금은** `output_dir`(기본적으로 `/workspace/distill_pipeline_output`)에 `runs/{run_id}/adapter`를 쓰고 있어서, 그대로 두면 계속 볼륨에 쌓입니다.
- **가능한 방법**  
  - **A)** 학습 스크립트에서 `output_dir`를 **볼륨이 아닌 경로**로 두기  
    (예: Pod 로컬 디스크 `/tmp/distill_pipeline_output` 등).  
    artifact 업로드는 그대로 두고, 볼륨에는 runs/를 만들지 않게 합니다.  
  - **B)** 지금처럼 볼륨에 쓰되, artifact 업로드가 끝난 뒤 해당 run의 `runs/{run_id}/` (또는 adapter만)를 **삭제**하는 단계를 스크립트/플로우에 넣어, 볼륨에는 오래 두지 않기.

그리고 **단일 run** (`train_student_with_pod`, sweep 아님)은 지금 adapter를 **볼륨에서 다운로드**하고 있으므로, “adapter는 볼륨에만 두지 않고 artifact만”으로 가려면:

- 학습이 끝난 run의 adapter를 artifact로 올리는 건 그대로 두고,
- `train_student_with_pod` 쪽은 “해당 run_id의 adapter를 **artifact에서** 받아오는” 방식으로 바꿔야 합니다 (wandb artifact API로 `qlora-adapter-{run_id}` 다운로드).

정리하면, “adapter는 볼륨에 저장하지 않고 artifact에만”은 **설계상 가능**하고, 그렇게 하려면  
- Pod 출력 경로를 볼륨 밖으로 바꾸거나,  
- 볼륨에 쓴 뒤 artifact 업로드 후 삭제하고,  
- 단일 run 플로우는 adapter를 artifact에서만 가져오도록  
코드 변경이 필요합니다.

---

## 3. Merge 스크립트 – 볼륨에 저장하지 않기

**가능합니다.**

- 지금은 `merge_for_serving_with_pod`가 `merge_adapter_for_serving.py`를 볼륨의 `distill_pipeline_output/merge_scripts/`에 업로드한 뒤, Pod에서 그 경로를 실행합니다.
- **대안**: 이 스크립트를 **이미 Docker 이미지 안에 포함**해 둡니다 (예: `jinsoo1218/train-llm` 이미지의 `/app/scripts/merge_adapter_for_serving.py`).  
  그러면 Pod는 “볼륨에 올린 스크립트” 대신 **이미지 안 경로**를 실행하도록 하고, 볼륨에는 merge 스크립트를 업로드하지 않으면 됩니다.
- 이미지에 스크립트가 없다면, 이미지 빌드 시 추가하거나, “merge는 로컬에서만 수행”(`merge_for_serving_flow`, Pod 미사용)으로 제한하면 볼륨에 스크립트를 둘 필요가 없습니다.

---

## 요약

| 항목 | 가능 여부 | 필요한 변경 |
|------|-----------|-------------|
| Label 한 번만 저장 | ✅ 가능 | 업로드 전에 `labeled/{version}/` 존재 여부 확인 후 있으면 스킵 |
| Adapter는 artifact만, 볼륨엔 저장 안 함 | ✅ 가능 (변경 필요) | 학습 스크립트 output를 볼륨 밖으로 두거나, artifact 업로드 후 볼륨에서 삭제; 단일 run은 adapter를 artifact에서만 다운로드 |
| Merge 스크립트 볼륨에 안 둠 | ✅ 가능 | 스크립트를 이미지에 포함하고, Pod는 이미지 내 경로만 실행 (볼륨 업로드 제거) |

원하시면 에이전트 모드에서 “label 스킵 / adapter 볼륨 미사용 / merge 스크립트 미업로드”를 위한 구체적인 수정 위치(파일·함수 이름)까지 짚어서 패치 계획을 잡아 드리겠습니다.

---


**아니요. 볼륨에 꼭 쓸 필요는 없습니다.** artifact만 올리고, 볼륨에는 adapter를 안 써도 됩니다.

- **Sweep**: 이미 best adapter를 artifact에서만 가져오므로, Pod가 adapter를 볼륨에 쓰지 않아도 동작합니다.
- **단일 run (`train_student_with_pod`)**: 지금은 “볼륨에 adapter 생겼나?”로 완료를 감지하고, 볼륨에서 다운로드합니다. 여기만 artifact 기준으로 바꾸면 됩니다.

**필요한 변경 요지:**

1. **Pod 안 학습 스크립트**  
   - `output_dir`을 **볼륨이 아닌 경로**로 두기 (예: `/tmp/distill_pipeline_output` 또는 컨테이너 로컬 디스크).  
   - adapter는 그 경로에만 쓰고, **지금처럼 wandb artifact 업로드는 그대로** 두기.  
   - 그러면 “볼륨에 쓰기 + artifact 업로드”가 아니라 **artifact 업로드만** 하게 됩니다.

2. **단일 run 플로우 (`train_student_with_pod`)**  
   - 완료 감지: 볼륨 폴링 대신, Pod 프로세스 종료 또는 wandb run 완료 등으로 판단.  
   - adapter 수급: 해당 run의 `run_id`를 알 수 있으면 (Pod 로그, wandb API 등), `get_best_adapter_from_artifact_task`와 같은 방식으로 **그 run_id의 artifact만** 받아오면 됩니다.

---

## 적용된 변경 (볼륨 사용 최소화)

다음이 코드에 반영되어 있습니다.

| 항목 | 구현 |
|------|------|
| **Label** | `upload_labeled_dir_to_runpod(skip_if_exists=True)` 기본. `labeled/{version}/train_labeled.json`이 이미 볼륨에 있으면 업로드 생략. |
| **Adapter** | Sweep·단일 run 모두 Pod 출력을 `/tmp/distill_pipeline_output`으로 두어 **볼륨에 adapter 미기록**. adapter는 **wandb artifact**만 사용. 단일 run은 Pod 종료 후 `get_latest_run_adapter_from_artifact_task`로 최신 run adapter 다운로드. |
| **Merge** | 파이프라인(all, all_sweep, sweep_eval_merge)에서는 **로컬 merge만** 사용(`merge_for_serving_flow`). Pod merge(`merge_for_serving_with_pod`)는 수동 실행용으로만 유지. |

정리하면, **“볼륨에 꼭 쓰고 artifact에도 올린다”가 아니라 “artifact 업로드만 하고 볼륨에는 adapter를 안 쓴다”**로 바꾸는 건 가능하고, 위 두 부분만 맞추면 됩니다.