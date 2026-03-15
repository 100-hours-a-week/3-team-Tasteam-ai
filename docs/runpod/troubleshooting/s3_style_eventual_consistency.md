
로그를 보면 **원인은 “볼륨이 원래 비어 있었다”가 아니라, 타이밍/일관성 문제에 가깝습니다.**

---

## 상황 정리

1. **eval_done.json은 읽혔음**  
   → `object_exists(done_key)` 성공 → 폴링 종료 → `get_object`로 `eval_done` 내용 읽음.  
   → 그래서 Result에 Pod가 쓴 `report_path`, `eval_dir`가 들어 있음.

2. **같은 prefix로 list는 비어 있음**  
   → 직후 `prefix_has_objects(vol_id, "distill_pipeline_output/eval_output/20260312_200521")` → **False**  
   → "No objects under prefix ...; download may be empty" 경고.

3. **다운로드도 0건**  
   → `download_directory_from_runpod`도 같은 prefix로 `list_objects_v2`를 쓰므로, list가 비어 있으면 받을 파일이 없음.

즉, **같은 prefix 아래에 있는 `eval_done.json`은 get으로 보이는데, list에는 아직 안 나오는 상태**입니다.

---

## 원인 추정: S3 스타일 eventual consistency

- **get_object / head_object**는 키 하나만 보므로, 쓰기가 반영되면 곧바로 성공할 수 있고,
- **list_objects_v2**는 인덱스/목록을 쓰기 때문에, 쓰기 직후에는 같은 prefix가 “비어 있는 것처럼” 나올 수 있습니다.

그래서 “수정하면 정상 다운로드될 가능성이 높나?”에 대한 답은:

- **지금 수정(rglob 등)은 올바르게 되어 있고**,  
- 이번 실패는 **list가 아직 반영 전이라 0건으로 다운로드된 것**이라서,
- **eval_done.json을 찾은 직후에 list/download를 하지 말고, 잠시 기다렸다가 list/download 하면** 정상 다운로드될 가능성이 높습니다.

---

## 수정 제안 (코드 변경)

- **eval_done.json을 발견한 직후**, `prefix_has_objects` / `download_directory_from_runpod` 호출 전에 **5~15초 정도 sleep**을 넣기.  
  또는  
- list 결과가 0건인데 `eval_done.json`은 존재하면, **짧은 간격으로 2~3회 재시도** (sleep 후 다시 list/download).

이렇게 하면 같은 RunPod 볼륨이라도 list가 따라오는 시간을 주어서, 지금처럼 “eval_done은 있는데 No objects / report not found”가 나오는 경우를 줄일 수 있습니다.

원하면 Agent 모드에서 **eval_done 확인 직후 sleep 또는 list/download 재시도**를 넣는 패치 위치까지 짚어서 수정안 적어줄 수 있습니다.