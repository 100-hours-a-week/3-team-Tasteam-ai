(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python scripts/distill_flows.py sweep_eval_merge --labeled-path distill_pipeline_output/labeled/20260226_051037/train_labeled.json --out-dir distill_pipeline_output --num-pods 2

20:47:17.062 | INFO    | prefect - Starting temporary server on http://127.0.0.1:8593
See https://docs.prefect.io/v3/concepts/server#how-to-guides for more information on running a dedicated Prefect server.
20:47:19.538 | INFO    | Flow run 'armored-bear' - Beginning flow run 'armored-bear' for flow 'sweep_eval_merge_flow'
20:47:19.612 | INFO    | Flow run 'classic-aardwark' - Beginning subflow run 'classic-aardwark' for flow 'run_sweep_and_evaluate_flow'
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: jin-soo (jin-soo-none) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /Users/js/tasteam-new-async/wandb/run-20260301_204720-pkwb9j13
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run fanciful-wind-18
wandb: ⭐️ View project at https://wandb.ai/jin-soo-none/tasteam-distill
wandb: 🚀 View run at https://wandb.ai/jin-soo-none/tasteam-distill/runs/pkwb9j13
wandb: 🚀 View run fanciful-wind-18 at: https://wandb.ai/jin-soo-none/tasteam-distill/runs/pkwb9j13
wandb: ⭐️ View project at: https://wandb.ai/jin-soo-none/tasteam-distill
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260301_204720-pkwb9j13/logs
20:47:23.416 | INFO    | Task run 'ensure-wandb-project-task-86c' - Finished in state Completed()
20:47:24.265 | INFO    | Task run 'register-sweep-task-ebc' - Create sweep with ID: ollyx0fg
20:47:24.266 | INFO    | Task run 'register-sweep-task-ebc' - Sweep URL: https://wandb.ai/jin-soo-none/tasteam-distill/sweeps/ollyx0fg
20:47:24.268 | INFO    | Task run 'register-sweep-task-ebc' - Finished in state Completed()
20:47:32.018 | INFO    | Task run 'upload-labeled-to-volume-for-sweep-task-c17' - Finished in state Completed()
20:47:33.378 | ERROR   | Task run 'run-sweep-on-pod-task-1d8' - Task run failed with exception: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}")
Traceback (most recent call last):
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1014, in run_context
    yield self
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1673, in run_task_sync
    engine.call_task_fn(txn)
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1031, in call_task_fn
    result = call_with_parameters(self.task.fn, parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/utilities/callables.py", line 210, in call_with_parameters
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 843, in run_sweep_on_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 207, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}
20:47:33.384 | ERROR   | Task run 'run-sweep-on-pod-task-1d8' - Finished in state Failed("Task run encountered an exception HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}")

---

이번 실행에서 일어난 일
Sweep 1개
sweep_id를 안 넘겼기 때문에 run_sweep_and_evaluate_flow 안에서 register_sweep_task가 한 번만 호출됨.
로그: Create sweep with ID: 52u6zlxl → sweep 1개 생성이 정상 동작입니다.
라벨 업로드 1번
upload-labeled-to-volume-for-sweep-task-1af 한 번만 실행되고 완료.
Pod 2개 중 첫 번째에서 실패
num_pods=2라서 run_sweep_on_pod_task가 2개 submit됨 (sweep-pod-0, sweep-pod-1).
for f in futures: f.result() 로 먼저 첫 번째 future를 기다림.
그 태스크(run-sweep-on-pod-task-4b7)가 RunPod API로 Pod 생성 요청을 보냈을 때 HTTP 500이 나와서 예외가 발생하고, flow 전체가 그 시점에 실패.
그래서 두 번째 Pod 태스크는 결과를 기다리기 전에 flow가 끝나서, 로그에는 Pod 생성 시도가 1번만 보이고, 실제로 생성된 Pod는 0개입니다.

이번 로그에서 해결할 점
이번 실패의 직접 원인은 RunPod API 500입니다.
create pod: Something went wrong. Please try again later or contact support.
우리 코드나 sweep/Pod 개수 설계 문제가 아니라, RunPod 쪽 일시 오류나 리소스/할당 문제일 가능성이 큽니다.
조치 제안:
잠시 뒤 다시 실행
--num-pods 2 그대로 한 번 더 실행해 보기.
RunPod 대시보드
해당 계정/리전에서 GPU/인스턴스 한도, 에러 메시지가 있는지 확인.
필요하면 RunPod 지원에 500 응답과 요청 시각을 알려 문의.

---

no instace error

---


터미널 로그 기준으로 정리하면 아래와 같습니다.

## 실행 요약

- **명령**: `sweep_eval_merge --num-pods 2`
- **진행**: sweep 등록(6iabx68i) → 라벨 볼륨 업로드 → **Pod 2개 생성 시도** → 둘 다 실패

## 원인: RunPod 인스턴스 부족

에러 메시지:

```text
'error': 'create pod: There are no instances currently available'
```

즉, **요청한 GPU 타입/리전에 지금 쓸 수 있는 인스턴스가 없다**는 뜻입니다.  
코드나 Prefect 쪽 문제가 아니라, RunPod 쪽 **용량/가용성** 이슈입니다.

## 참고

- 이전에 봤던 `Something went wrong. Please try again later` 와는 다르게, 이번에는 **“지금 인스턴스 없음”** 이라고 명시돼 있습니다.
- `num-pods 2`라서 **동시에 2개** 생성 요청이 나갔고, 둘 다 같은 이유로 500을 받은 상태입니다.

## 대응 제안

1. **시간을 두고 재시도**  
   - 잠시 후(몇 분~몇십 분) 같은 명령 다시 실행.
2. **RunPod 대시보드 확인**  
   - 요청한 GPU 타입/리전의 가용량, 대기/예약 옵션 확인.
3. **GPU 타입/리전 완화**  
   - `get_default_pod_payload(use="train")` 에서 쓰는 GPU가 너무 인기 있는 타입이면, 다른 타입이나 리전을 쓰도록 payload를 바꿔보기.
4. **우선 1개만 생성해 보기**  
   - `--num-pods 1` 로 한 번만 생성되는지 확인해 보면, “동시 2개” 부담을 줄일 수 있습니다.

원하면 `runpod_api_500.md` 같은 문서에 **“There are no instances currently available”** 케이스를 추가하는 문장도 같이 정리해 줄 수 있습니다.

---

09:44:41.127 | INFO    | prefect - Starting temporary server on http://127.0.0.1:8943
See https://docs.prefect.io/v3/concepts/server#how-to-guides for more information on running a dedicated Prefect server.
09:44:43.592 | INFO    | Flow run 'serious-quetzal' - Beginning flow run 'serious-quetzal' for flow 'sweep_eval_merge_flow'
09:44:43.641 | INFO    | Flow run 'fresh-wildebeest' - Beginning subflow run 'fresh-wildebeest' for flow 'run_sweep_and_evaluate_flow'
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: jin-soo (jin-soo-none) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.25.0
wandb: Run data is saved locally in /Users/js/tasteam-new-async/wandb/run-20260302_094444-myne9voc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ruby-capybara-31
wandb: ⭐️ View project at https://wandb.ai/jin-soo-none/tasteam-distill
wandb: 🚀 View run at https://wandb.ai/jin-soo-none/tasteam-distill/runs/myne9voc
wandb: 🚀 View run ruby-capybara-31 at: https://wandb.ai/jin-soo-none/tasteam-distill/runs/myne9voc
wandb: ⭐️ View project at: https://wandb.ai/jin-soo-none/tasteam-distill
wandb: Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20260302_094444-myne9voc/logs
09:44:47.600 | INFO    | Task run 'ensure-wandb-project-task-e58' - Finished in state Completed()
09:44:48.328 | INFO    | Task run 'register-sweep-task-288' - Create sweep with ID: wpwb65os
09:44:48.331 | INFO    | Task run 'register-sweep-task-288' - Sweep URL: https://wandb.ai/jin-soo-none/tasteam-distill/sweeps/wpwb65os
09:44:48.335 | INFO    | Task run 'register-sweep-task-288' - Finished in state Completed()
09:44:55.897 | INFO    | Task run 'upload-labeled-to-volume-for-sweep-task-ac7' - Finished in state Completed()
09:44:57.176 | ERROR   | Task run 'run-sweep-on-pod-task-d88' - Task run failed with exception: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}")
Traceback (most recent call last):
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1014, in run_context
    yield self
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1673, in run_task_sync
    engine.call_task_fn(txn)
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1031, in call_task_fn
    result = call_with_parameters(self.task.fn, parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/utilities/callables.py", line 210, in call_with_parameters
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 909, in run_sweep_on_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 207, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}
09:44:57.187 | ERROR   | Task run 'run-sweep-on-pod-task-d88' - Finished in state Failed("Task run encountered an exception HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: Something went wrong. Please try again later or contact support.', 'status': 500}")

---


코드 기준으로 보면 아래처럼 이해할 수 있습니다.

## 원인 후보 두 가지

### 1) **첫 번째 Pod(sweep-pod-0) 생성은 됐는데, 응답만 500으로 온 경우**

- `num_pods=2`일 때 `run_sweep_on_pod_task`가 **두 번** 동시에 실행됩니다(pod_index=0, 1).
- 그중 **sweep-pod-0**을 만드는 요청에서 RunPod 쪽에서는 Pod를 실제로 만들었는데, 응답을 주는 단계에서 오류가 나서 **HTTP 500**을 돌려줬을 수 있습니다.
- 그러면:
  - RunPod 콘솔에는 sweep-pod-0이 보이고,
  - 우리 코드는 `create_pod()`에서 500을 받아 `HTTPError`를 던져서 실패로 처리하고,
  - 해당 task에서는 `pod_id`를 못 받으니 `_track_pod_created`도 호출되지 않아, 나중에 Ctrl+C 해도 이 Pod는 자동 삭제 대상에 안 들어갑니다.

즉, **“생성은 성공 + 응답 실패”** 형태의 RunPod 쪽 일시적 오류로 보는 경우입니다.

### 2) **두 번째 Pod(sweep-pod-1) 생성 요청이 500으로 실패한 경우**

- 첫 번째 task(pod_index=0)는 성공해서 **sweep-pod-0**만 생성됐고,
- 두 번째 task(pod_index=1)의 `create_pod()`가 500을 반환해 그 task만 실패했을 수 있습니다.
- 로그에 찍힌 `run-sweep-on-pod-task-d88`이 **어느 쪽 task인지**는 로그만으로는 구분이 안 됩니다.

이 경우에는 **리소스/쿼터 부족**이나 **동시 생성 제한** 등으로 두 번째 Pod만 거절당한 상황일 수 있습니다.

---

## 확인해보면 좋은 것

1. **RunPod 콘솔에서 Pod 개수**
   - sweep-pod-0만 있는지, sweep-pod-1도 있는지 확인.
   - sweep-pod-0만 있으면 → 위 1) 또는 2) 중 “두 번째만 실패” 둘 다 가능.
   - sweep-pod-0과 sweep-pod-1 둘 다 있으면 → 1)처럼 “응답만 500” 가능성이 더 큼.

2. **RunPod 상태/이슈**
   - [RunPod Status](https://status.runpod.io/) 등에서 당시 500/일시 장애 공지가 있었는지 확인.

3. **재시도**
   - RunPod 500은 “나중에 다시 시도해 보라”는 식의 일시 오류인 경우가 많아서, 같은 명령을 다시 실행해 보는 것이 좋습니다.

---

## 정리

- **sweep-pod-0이 콘솔에 보이는데 우리 플로우만 실패한 이유**는:
  - RunPod가 **Pod 생성은 했지만** 그 요청에 대해 **500 응답**을 보냈거나,
  - **두 번째 Pod(sweep-pod-1)** 생성 요청이 500으로 실패했기 때문일 가능성이 큽니다.
- 로그만으로는 “첫 번째 요청이 500”인지 “두 번째 요청이 500”인지 구분은 안 되므로, 콘솔에서 sweep-pod-1 존재 여부와 RunPod 상태 페이지 확인이 가장 도움이 됩니다.

---


그럼 상황이 이렇게 정리됩니다.

- **sweep-pod-0**: 생성 요청이 성공해서 RunPod에만 보이는 상태.
- **sweep-pod-1**: 같은 타이밍에 보낸 두 번째 `create_pod()` 요청이 **500**을 받아서, Pod는 아예 생성되지 않음.

즉, **두 번째 Pod(sweep-pod-1) 생성 요청이 500으로 실패**했고, 그 task가 실패하면서 플로우 전체가 실패한 것입니다.  
pod0만 있다는 건 “첫 번째 요청은 성공, 두 번째 요청만 500”이라는 뜻입니다.

가능한 원인은 대략:

1. **동시 생성 제한**  
   같은 계정/리전에서 짧은 시간에 Pod 2개를 동시에 만들 때 RunPod 쪽에서 500을 반환하는 경우.
2. **일시적인 RunPod 서버/용량 문제**  
   두 번째 요청 처리 중에만 오류가 난 경우.
3. **GPU/템플릿 가용량**  
   첫 번째 Pod는 할당됐고, 두 번째 할당 시점에 리소스 부족으로 500이 난 경우.

**할 수 있는 대응:**

- **당장**: RunPod 콘솔에서 **sweep-pod-0을 수동 종료**해 두고,  
  `--num-pods 1` 로 다시 실행해 보기 (한 개만 생성하면 500이 안 나는지 확인).
- **나중에 2개 쓰고 싶을 때**:  
  두 번째 Pod를 첫 번째 생성 **성공 후 몇 초~수십 초 뒤**에 요청하도록, 코드에서 Pod 생성 순차화나 재시도(지수 백오프)를 넣는 방법을 고려할 수 있습니다.

이 내용은 `docs/runpod/troubleshooting/runpod_api_500.md`에 “num_pods=2일 때 두 번째 Pod 생성만 500으로 실패하고 pod0만 남는 경우”로 정리해 두면 다음에 참고하기 좋습니다.

---


적용했습니다. 변경 요약만 정리합니다.

### 1. **재시도** — `scripts/runpod_cli/pod_create_delete_cli.py`

- `create_pod()`에 **지수 백오프 재시도** 추가:
  - **재시도 대상**: HTTP 500, 502, 503
  - **최대 재시도**: 4번 (첫 시도 + 재시도 3번)
  - **대기**: 2초 → 4초 → 8초 (2^attempt 초)
  - 옵션: `max_retries`, `retry_statuses`, `base_delay_sec` (기본값 그대로 사용)

### 2. **순차화(스태거)** — `scripts/distill_flows.py`

- `num_pods > 1`일 때 **Pod 생성 시작 시점을 15초씩 간격** 두도록 변경:
  - 업로드는 그대로 한 번만 수행
  - `run_sweep_on_pod_task.submit()`을 **한 번에 다 보내지 않고**, 0번 제출 → 15초 대기 → 1번 제출 → … 순서로 실행
  - 그 다음 `f.result()`로 모두 대기 (sweep 자체는 여전히 멀티 Pod 병렬)

이제 두 번째 Pod 생성이 500이 나와도 클라이언트에서 자동 재시도하고, 첫 번째와 두 번째 생성 요청이 최소 15초 차이로 나가서 동시 요청으로 인한 500 가능성이 줄어듭니다.

---

10:00:43.495 | ERROR   | Task run 'run-sweep-on-pod-task-730' - Finished in state Failed("Task run encountered an exception HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}")

---


코드 기준으로 보면 이런 상황입니다.

## API로 쓰는 4090 sweep Pod 요청

`get_default_pod_payload(use="train")`로 sweep Pod를 만들 때 실제로 넘어가는 조건은 대략 다음과 같습니다.

- **GPU**: `gpuTypeIds`: `["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"]`, `gpuTypePriority`: `"availability"`
- **리전**: `dataCenterIds`에 EU-RO-1, CA-MTL-1, US-IL-1 등 여러 데이터센터 지정, `dataCenterPriority`: `"availability"`
- **기타**: `allowedCudaVersions`: `["13.0"]`, `minRAMPerGPU`: 8, `minVCPUPerGPU`: 2, `minDiskBandwidthMBps` 등

즉, “4090 또는 A5000 + 위 리전들 + CUDA 13.0 + 최소 스펙”을 **한꺼번에** 만족하는 인스턴스를 API로 찾는 상태입니다.

## “No instance”가 나오는 이유 (수동 생성은 되는 경우)

RunPod의 “no instance”는 **그 API 요청 조건을 동시에 만족하는 인스턴스가 지금 없다**는 뜻입니다. 수동으로 4090 Pod을 만들 수 있었던 건 보통 아래 중 하나(또는 조합) 때문입니다.

1. **요청 조건이 다름**  
   - API: 4090 **또는** A5000 + 여러 리전 + CUDA 13.0 + min 스펙을 한 번에 요구.  
   - 수동: 특정 리전 하나만 선택하거나, CUDA/스펙이 다르거나, 4090만 골랐을 수 있음.  
   → “4090만, 리전 하나, 다른 기본값”이면 수동 요청이 더 느슨해서 인스턴스가 잡힐 수 있음.

2. **시점 차이**  
   - API 호출 시점에는 전역으로 4090/리전/스펙 조합에 맞는 게 없었고,  
   - 나중에 수동으로 만들 때는 그 리전·GPU에 자리가 생겼을 수 있음.

3. **콘솔 vs API 동작 차이**  
   - 콘솔은 “지금 보이는 리전/GPU” 위주로 슬롯을 보여주고,  
   - API는 `dataCenterIds` + `gpuTypeIds` + CUDA/스펙을 **모두** 만족하는 인스턴스만 골라서, 더 빡빡할 수 있음.

그래서 “4090 sweep pod 0만 생성되고 no instance”가 나왔지만, 수동으로 4090 Pod은 만들 수 있었던 상황과 잘 맞습니다.

## 정리 및 대응 제안

- **원인**: API가 요청하는 **조건 조합**(4090/A5000 + 여러 리전 + CUDA 13.0 + min 스펙)에 맞는 인스턴스가 그 순간 없어서 “no instance”가 났고, 수동 생성은 (리전/옵션/시점이 달라서) 다른 요청으로 인스턴스가 잡힌 것으로 보는 것이 타당합니다.
- **대응**:  
  - 나중에 다시 sweep 돌리기(같은 API로 재시도).  
  - 또는 payload를 완화: `allowedCudaVersions`에 `"12.0"` 추가, `dataCenterIds` 확대, `gpuTypeIds`에 다른 4090 표기(있다면) 추가 등으로 “no instance”가 덜 나오게 조정할 수 있습니다.

이 내용을 `runpod_api_500.md`에 “no instance지만 수동 생성은 되는 경우”로 짧게 정리해 두면 나중에 참고하기 좋습니다.