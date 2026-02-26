(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python scripts/distill_flows.py labeling_pod_only \
  --train-path distill_pipeline_output/datasets/20260226_051036/train.json \
  --gold-path distill_pipeline_output/labeled/20260226_051037/train_labeled_gold_only.json \
  --out-dir distill_pipeline_output
22:10:20.792 | INFO    | prefect - Starting temporary server on http://127.0.0.1:8015
See https://docs.prefect.io/v3/concepts/server#how-to-guides for more information on running a dedicated Prefect server.
22:10:23.248 | INFO    | Flow run 'pompous-dogfish' - Beginning flow run 'pompous-dogfish' for flow 'labeling_pod_only_flow'
22:10:24.414 | ERROR   | Task run 'labeling-pod-only-task-09b' - Task run failed with exception: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}")
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 349, in labeling_pod_only_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}
22:10:24.430 | ERROR   | Task run 'labeling-pod-only-task-09b' - Finished in state Failed("Task run encountered an exception HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}")
22:10:24.431 | ERROR   | Flow run 'pompous-dogfish' - Encountered exception during execution: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}")
Traceback (most recent call last):
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 989, in run_context
    yield self
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1634, in run_flow_sync
    engine.call_flow_fn()
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1009, in call_flow_fn
    result = call_with_parameters(self.flow.fn, self.parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/utilities/callables.py", line 210, in call_with_parameters
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 412, in labeling_pod_only_flow
    return labeling_pod_only_task(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/tasks.py", line 1209, in __call__
    return run_task(
           ^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1900, in run_task
    return run_task_sync(**kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1675, in run_task_sync
    return engine.state if return_type == "state" else engine.result()
                                                       ^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 611, in result
    raise self._raised
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 349, in labeling_pod_only_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}
22:10:24.454 | INFO    | Flow run 'pompous-dogfish' - Finished in state Failed("Flow run encountered an exception: HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}")
Traceback (most recent call last):
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1254, in <module>
    main()
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1174, in main
    result = labeling_pod_only_flow(
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flows.py", line 1850, in __call__
    return run_flow(
           ^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1791, in run_flow
    ret_val = run_flow_sync(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1636, in run_flow_sync
    return engine.state if return_type == "state" else engine.result()
                                                       ^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 539, in result
    raise self._raised
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 989, in run_context
    yield self
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1634, in run_flow_sync
    engine.call_flow_fn()
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/flow_engine.py", line 1009, in call_flow_fn
    result = call_with_parameters(self.flow.fn, self.parameters)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/utilities/callables.py", line 210, in call_with_parameters
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 412, in labeling_pod_only_flow
    return labeling_pod_only_task(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/tasks.py", line 1209, in __call__
    return run_task(
           ^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1900, in run_task
    return run_task_sync(**kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 1675, in run_task_sync
    return engine.state if return_type == "state" else engine.result()
                                                       ^^^^^^^^^^^^^^^
  File "/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/prefect/task_engine.py", line 611, in result
    raise self._raised
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 349, in labeling_pod_only_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: There are no instances currently available', 'status': 500}
22:10:24.467 | INFO    | prefect - Stopping temporary server on http://127.0.0.1:8015

---


에러 원인은 **코드가 아니라 RunPod 쪽**입니다.

- **메시지**: `'create pod: There are no instances currently available'`
- **의미**: 요청한 GPU/리전/스펙으로 **지금 사용 가능한 인스턴스가 없다**는 뜻입니다. RunPod API가 500을 반환한 상태입니다.

즉, `labeling_pod_only` flow는 정상적으로 Pod 생성 요청까지 갔고, RunPod가 “인스턴스 없음”으로 실패한 상황입니다.

---

**할 수 있는 것**

1. **나중에 다시 실행**  
   잠시 뒤(몇 분~몇 시간) 같은 명령을 다시 실행해 보기.  
   `python scripts/distill_flows.py labeling_pod_only --train-path ... --gold-path ... --out-dir ...`

2. **RunPod 대시보드 확인**  
   사용 중인 GPU 타입/리전이 인기 많으면 자주 발생합니다.  
   다른 GPU나 리전을 쓰도록 `runpod_cli`의 pod payload(예: `get_default_pod_payload(use="labeling")`)를 바꿀 수 있는지 확인해 보기.

3. **재시도 로직 추가(선택)**  
   같은 500 에러일 때 N회 재시도 + 지수 백오프를 넣고 싶다면, Agent 모드에서 `labeling_pod_only_task`의 `client.create_pod(payload)` 호출 부분에 재시도 루프를 넣는 식으로 구현할 수 있습니다.

지금 단계에서는 **코드 수정 없이, 시간을 두고 같은 명령을 다시 실행**해 보는 것이 우선입니다.