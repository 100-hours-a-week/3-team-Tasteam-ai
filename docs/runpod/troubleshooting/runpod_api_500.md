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