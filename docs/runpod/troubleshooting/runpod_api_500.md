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