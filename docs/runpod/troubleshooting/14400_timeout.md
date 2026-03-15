
터미널 로그 기준으로 정리하면 아래와 같습니다.

---

## 무슨 일이 있었는지

- **Flow**: `evaluate_on_pod`  
- **에러**: `TimeoutError` — Pod가 **4시간(14400초)** 안에 볼륨에  
  `distill_pipeline_output/eval_output/20260311_135015/eval_done.json`  
  을 만들지 못했다고 판단하고 종료했습니다.

동작 방식은 다음과 같습니다.

1. Eval용 Pod를 띄우고, Pod 안에서 `eval_distill` 실행 후 `run_eval_and_upload_artifact.py`가 끝나면 위 경로에 `eval_done.json`을 씁니다.
2. 로컬 flow는 **볼륨에 그 파일이 생길 때까지** 일정 간격으로 폴링합니다.
3. **14400초(4시간)** 안에 파일이 없으면 `evaluate_on_pod_task`에서 위 `TimeoutError`를 던집니다.

관련 코드: `scripts/distill_flows.py` 1545–1552행 근처  
(`done_key` 설정, `deadline = time.time() + eval_timeout_sec`, 폴링 루프, 타임아웃 시 `raise TimeoutError(...)`).

---

## 가능한 원인

- Pod 안에서 eval이 **4시간 안에 끝나지 않음** (샘플 수 많음, GPU/인스턴스 느림 등)
- Pod/컨테이너가 **중간에 죽거나 재시작**됨
- eval은 끝났는데 **`eval_done.json`을 쓰기 전에** 크래시 또는 스크립트 오류
- **볼륨 마운트/경로 문제**로 Pod가 해당 경로에 파일을 쓰지 못함

---

## 다음에 할 수 있는 것

1. **Pod 로그 확인**  
   RunPod 대시보드에서 해당 Eval Pod의 로그를 열어,  
   - `eval_distill` / `run_eval_and_upload_artifact` 가 실제로 끝까지 돌았는지  
   - 에러 메시지나 스택트레이스가 있는지  
   확인하는 것이 가장 중요합니다. 에러 메시지를 알려주시면 원인 추적이 훨씬 수월합니다.

2. **이미 볼륨에 eval 결과만 있는 경우**  
   `eval_done.json`은 없지만  
   `distill_pipeline_output/eval_output/20260311_135015/` 아래에 `report.json` 등이 이미 있다면,  
   **같은 버전으로** `download_eval_from_volume` 을 쓰면 됩니다.  
   ```bash
   python scripts/distill_flows.py download_eval_from_volume \
     --eval-version 20260311_135015 \
     --val-labeled-path distill_pipeline_output/labeled/20260308_102036/val_labeled.json \
     --adapter-path distill_pipeline_output/artifacts/g4nca6b2/adapter
   ```
   이렇게 하면 해당 버전 디렉터리만 받아서 로컬에서 judge → kd_sft → 아티팩트까지 이어갈 수 있습니다.

3. **타임아웃을 늘리고 다시 시도**  
   지금은 `eval_timeout_sec`이 14400으로 하드코딩되어 있고, CLI 옵션은 없습니다.  
   타임아웃을 늘리려면 `evaluate_on_pod_task` / `evaluate_on_pod_flow` 에서 `eval_timeout_sec` 기본값을 키우거나, `main()`에서 `evaluate_on_pod_flow(..., eval_timeout_sec=...)` 로 넘기도록 하고, 필요하면 `--eval-timeout` 같은 인자를 추가해야 합니다. (원하시면 Agent 모드에서 그 변경 위치와 예시 패치를 구체적으로 적어드리겠습니다.)

요약하면, **우선 RunPod에서 해당 Eval Pod 로그를 확인**하고,  
- 로그에 에러가 있으면 그 내용을 기준으로 수정하고  
- 로그상으로는 성공인데 `eval_done.json`만 없다면 `download_eval_from_volume --eval-version 20260311_135015` 로 마무리하는 흐름을 추천합니다.