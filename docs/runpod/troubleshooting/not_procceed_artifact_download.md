
상황이 이렇게 정리됩니다.

---

## 무슨 일이 있었는지

1. **그 시점의 상태**
   - wandb에 run 50개 올라감 (sweep 완료)
   - RunPod 콘솔에서 Pod는 이미 종료
   - 마지막 run이 2시간 전이었으므로, Prefect는 **`wait_until_stopped`** 구간에서 RunPod API를 계속 폴링 중이었음
   - 아직 `get_best_adapter_from_artifact_task`(다운로드) 단계에는 도달하지 않은 상태

2. **맥북을 닫고 이동**
   - `caffeinate -i`로 재우지는 않았지만, **랩탑을 닫고 와이파이 없는 구간을 약 1시간 통과**
   - 그 사이 네트워크 끊김 → `rest.runpod.io` 연결 실패 (DNS 실패 또는 연결 끊김)

3. **에러가 난 위치**
   - **`wait_until_stopped`** 안의 `get_pod()` 호출에서  
     `RemoteDisconnected` / `ConnectionError` 발생
   - 그 다음 **`finally`** 블록의 **`client.delete_pod(pod_id)`** 호출에서  
     (재시도 시점에 네트워크가 없어) **`NameResolutionError`** (rest.runpod.io DNS 실패) 발생
   - 그 결과 해당 Pod task가 예외로 끝나고, flow 전체가 실패
   - **best adapter 다운로드 단계는 실행되지 않음**

즉, **“Pod는 이미 죽었는데, 로컬에서는 네트워크가 끊긴 상태로 RunPod API를 계속 호출하다가 실패한 것”**이 맞습니다.

---

## 정리

- **원인**: 이동 중 네트워크 단절로 RunPod API(`rest.runpod.io`) 호출 실패 → `wait_until_stopped` 및 `delete_pod`에서 ConnectionError/NameResolutionError.
- **이미 끝난 것**: sweep 50 run, Pod 종료.  
- **안 된 것**: best adapter 다운로드 → evaluate → merge.

---

## 다음에 할 일

Pod는 이미 없고 run도 50개 올라갔으므로, **sweep을 다시 돌릴 필요 없이** best adapter만 받아서 이어가면 됩니다.

1. **네트워크가 되는 환경**에서 아래 중 하나 실행.

2. **방법 A – sweep_id만 넣고 한 번에**
   ```bash
   python scripts/distill_flows.py sweep_eval_merge \
     --labeled-path distill_pipeline_output/labeled/20260226_051037/train_labeled.json \
     --sweep-id <당시_sweep_id> \
     --out-dir distill_pipeline_output --num-pods 2
   ```
   - sweep이 이미 끝났기 때문에 Pod는 금방 끝나고, 이어서 best adapter 다운로드 → evaluate → merge가 진행됩니다.

3. **방법 B – best adapter만 수동 다운로드 후 evaluate / merge만**
   - 이전에 안내한 Python 한 줄로 best adapter 다운로드.
   - 그 다음 `evaluate`, `merge_for_serving`만 `--adapter-path`로 실행.

요약하면, **“닫고 이동하는 동안 네트워크가 끊겨 RunPod API 호출이 실패했고, 그 때문에 다운로드 단계까지 가지 못한 것”**이 맞고, 지금은 네트워크 복구 후 위처럼 sweep 이후 단계만 다시 실행하면 됩니다.

---


**다운로드가 2시간 동안 안 된 이유**는, flow가 **다운로드 단계에 아예 도달하지 못했기 때문**입니다.

## Flow가 진행되는 순서

1. **run_sweep_on_pod_task**를 Pod 2개에 대해 **병렬**로 실행  
2. **각 태스크**는  
   - Pod 생성 → `wait_until_running` → **`wait_until_stopped(pod_id)`** → `delete_pod`  
   여기서 **`wait_until_stopped`가 return 할 때까지** 그 태스크는 끝나지 않습니다.  
3. **두 태스크가 모두** `return` 한 다음에만  
   `get_best_adapter_from_artifact_task`(다운로드)가 실행됩니다.

즉, **“마지막 run 이후 2시간” 동안은 다운로드 코드가 실행될 기회가 없고**, 그 전 단계인 **`wait_until_stopped` 루프 안**에 계속 있었을 가능성이 큽니다.

## 왜 `wait_until_stopped`가 2시간이나 안 끝났을까

`wait_until_stopped`는 대략 이렇게 동작합니다:

- **60초마다** `get_pod(pod_id)`로 RunPod API에 Pod 상태 요청  
- 아래 조건 중 하나가 되면 **그때서야** 루프를 빠져나와 다음 단계(그 다음엔 다운로드)로 진행:
  - `status == "already_deleted"`
  - `desiredStatus != "RUNNING"`
  - `status`에 `"EXIT"` / `"STOP"` 포함 또는 `status == "COMPLETED"`

그래서 **RunPod 콘솔에서는 Pod가 “종료”로 보여도**,  
API가 계속 `desiredStatus: "RUNNING"`, `status: "RUNNING"`(또는 코드가 “종료”로 보지 않는 다른 값)을 돌려주면,  
**코드는 “아직 안 끝났다”고 보고 60초 sleep → 다시 `get_pod`를 2시간 동안 반복**하게 됩니다.

가능한 원인은 대략 두 가지입니다.

1. **RunPod API가 실제로 “종료” 상태를 우리가 기대하는 형태로 안 준 경우**  
   - 예: `status: "TERMINATED"`, `"STOPPED"`, `"EXITED"` 등 다른 문자열/다른 필드명  
   - 우리 코드는 `"EXIT" in status` / `"STOP" in status` / `status == "COMPLETED"`만 보므로, 값이 다르면 절대 “종료”로 인식하지 못함.  
2. **두 Pod 중 하나라도** 위 조건을 한 번도 만족하지 못해,  
   그 Pod에 대한 `wait_until_stopped`가 2시간 내내 빠져나오지 못한 경우.

어느 쪽이든 **“마지막 run 이후 2시간은 네트워크가 괜찮았는데도 다운로드가 안 된 이유”**는  
- 네트워크 문제가 아니라  
- **flow가 “다운로드” 전 단계인 `wait_until_stopped`에서, RunPod API 응답이 “종료”로 인식되지 않아 2시간 동안 계속 대기했기 때문**으로 보는 것이 맞습니다.  
그 후 맥북을 닫고 와이파이 없는 곳으로 이동하면서 `get_pod()`가 실패(ConnectionError / DNS 오류)한 것이 겹쳐, 지금 보신 에러가 난 것입니다.