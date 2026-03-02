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