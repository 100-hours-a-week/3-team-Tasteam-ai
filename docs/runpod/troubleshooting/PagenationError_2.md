
---

## 원인

`download_eval_from_volume` 실행 시 **다운로드** 단계에서 같은 Pagination 에러가 났습니다.

- **에러**: `PaginationError: The same next token was received twice`
- **위치**: `scripts/runpod_cli/runpod_s3_upload.py` **183행**  
  `download_directory_from_runpod()` 안의  
  `for page in paginator.paginate(Bucket=bucket, Prefix=prefix)`

즉, **list 할 때만** paginator를 쓰는 부분에서 RunPod S3가 같은 `ContinuationToken`을 두 번 주면서 botocore가 `PaginationError`를 던진 상황입니다.  
(이미 수정한 `delete_prefix_from_volume`은 paginator를 쓰지 않고 수동 루프라서 이 에러가 나지 않습니다.)

---

## 해결 방향

`download_directory_from_runpod`도 **paginator 대신 수동 `list_objects_v2` 루프**로 바꾸면 됩니다.  
`delete_prefix_from_volume`에 적용한 것과 같은 방식으로:

- `list_objects_v2`를 반복 호출하고 `NextContinuationToken`으로 다음 페이지 요청
- **이미 사용한 token이 다시 나오면** list 루프만 중단
- **그때까지 수집한 키들만** `download_file`로 받기

이렇게 하면 RunPod이 같은 token을 다시 줘도, 받은 객체들까지는 정상적으로 다운로드됩니다.

---

## 수정할 코드 위치

- **파일**: `scripts/runpod_cli/runpod_s3_upload.py`
- **함수**: `download_directory_from_runpod` (170행 근처)
- **변경**: 182–183행의 `paginator.paginate(...)` 제거 후,  
  `list_objects_v2` + `MaxKeys` + `ContinuationToken` + `seen_tokens` 로 수동 페이지네이션 구현 (위 “해결 방향”대로).

이 변경을 코드에 적용해 주길 원하시면 **Agent 모드**로 전환한 뒤, “`download_directory_from_runpod`에 delete와 같은 수동 pagination 적용해줘”라고 요청하시면 됩니다.