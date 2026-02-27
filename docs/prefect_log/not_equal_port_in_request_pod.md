(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python scripts/distill_flows.py labeling_pod_only \
  --train-path distill_pipeline_output/datasets/20260226_051036/train.json \
  --gold-path distill_pipeline_output/labeled/20260226_051037/train_labeled_gold_only.json \
  --out-dir distill_pipeline_output \
--public-ip-wait-timeout 600
11:22:52.747 | INFO    | prefect - Starting temporary server on http://127.0.0.1:8095
See https://docs.prefect.io/v3/concepts/server#how-to-guides for more information on running a dedicated Prefect server.
11:22:55.225 | INFO    | Flow run 'lovely-hippo' - Beginning flow run 'lovely-hippo' for flow 'labeling_pod_only_flow'
11:22:56.956 | INFO    | Task run 'labeling-pod-only-task-7a5' - Pod created: b7dz2gxy835mjt
11:24:29.447 | INFO    | Task run 'labeling-pod-only-task-7a5' - Pod ready: b7dz2gxy835mjt base_url: http://213.173.111.7:8000/v1
11:27:34.547 | INFO    | Task run 'labeling-pod-only-task-7a5' - Cleaning up pod: b7dz2gxy835mjt
11:27:35.255 | ERROR   | Task run 'labeling-pod-only-task-7a5' - Task run failed with exception: TimeoutError('vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s')
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 364, in labeling_pod_only_task
    _wait_for_vllm_ready(base_url, timeout_sec=vllm_ready_timeout_sec)
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 192, in _wait_for_vllm_ready
    raise TimeoutError(f"vLLM at {base_url} did not become ready within {timeout_sec}s")
TimeoutError: vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s
11:27:35.265 | ERROR   | Task run 'labeling-pod-only-task-7a5' - Task run failed due to timeout: TimeoutError('vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s')
11:27:35.270 | ERROR   | Task run 'labeling-pod-only-task-7a5' - Finished in state TimedOut("Task run failed due to timeout: TimeoutError('vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s')", type=FAILED)
11:27:35.272 | ERROR   | Flow run 'lovely-hippo' - Flow run failed due to timeout: TimeoutError('vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s')
11:27:35.300 | INFO    | Flow run 'lovely-hippo' - Finished in state TimedOut("Flow run failed due to timeout: TimeoutError('vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s')", type=FAILED)
Traceback (most recent call last):
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1267, in <module>
    main()
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1184, in main
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 415, in labeling_pod_only_flow
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 364, in labeling_pod_only_task
    _wait_for_vllm_ready(base_url, timeout_sec=vllm_ready_timeout_sec)
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 192, in _wait_for_vllm_ready
    raise TimeoutError(f"vLLM at {base_url} did not become ready within {timeout_sec}s")
TimeoutError: vLLM at http://213.173.111.7:8000/v1 did not become ready within 180s
11:27:35.314 | INFO    | prefect - Stopping temporary server on http://127.0.0.1:8095

---


RunPod 문서와 코드를 기준으로 정리하면 아래와 같습니다.

---

## 1. 왜 8000으로 보내고 있나?

코드에서 **base_url을 포트 8000으로 고정**하고 있습니다.

- `distill_flows.py` 254행, 361행:  
  `base_url = f"http://{public_ip}:8000/v1"`

RunPod 동작은 이렇게 됩니다.

- Pod 설정의 **`"8000/http"`** = **컨테이너 안에서** vLLM이 떠 있는 포트(8000).
- RunPod는 이 **내부 8000**을 **외부에서는 다른 포트**(예: 13979)로 매핑합니다.
- 그래서 **실제로 접속해야 하는 주소는 `publicIp:13979`** 같은 형태이고,  
  우리가 계속 `publicIp:8000`으로만 요청해서 실패하는 상황입니다.

즉, “8000번이 아니고 13979를 보내야 하는데 8000으로 보내고 있다”는 현상은,  
**할당된 외부 포트를 쓰지 않고 내부 포트(8000)를 하드코딩해서 그렇다**고 보면 됩니다.

---

## 2. TCP 포트를 8000으로 맞춰야 하나? / HTTP와의 관계

- **Pod payload의 `"8000/http"`**  
  - `8000`: 컨테이너 **내부** 포트 (vLLM이 listen하는 포트).  
  - `http`: “이 포트는 HTTP 트래픽”이라는 힌트. RunPod가 HTTP로 처리할 때 쓰는 값.
- **실제 외부 접속**  
  - RunPod가 **외부 포트**(13979 등)를 할당하고,  
    그 포트로 들어온 TCP(HTTP) 트래픽을 **컨테이너의 8000**으로 넘겨줍니다.
- 따라서:
  - **컨테이너 안**에서는 계속 8000으로 서비스해도 됩니다.
  - **우리 클라이언트**는 **RunPod가 준 외부 포트(13979)**로 접속해야 하고,  
    “TCP 포트를 8000으로 맞춰야 하냐”는 **아니요** – 외부에서는 할당된 포트(13979)를 써야 합니다.
- HTTP는 “TCP 위에서 쓰는 프로토콜”이라,  
  포트만 올바르게(13979) 쓰면 `http://publicIp:13979/v1` 같이 HTTP 요청을 보내는 것이 맞습니다.

---

## 3. 어떻게 고치면 좋은지

- **할당된 외부 포트를 쓰려면**  
  RunPod **GET pod** 응답에 “어떤 내부 포트가 어떤 외부 포트로 매핑됐는지”가 나와야 합니다.  
  (문서: *“Use the GET /pods/{podId} endpoint to retrieve Pod information, which includes the exposed port configuration and public mappings.”*)
- 응답 구조는 RunPod 버전/API에 따라 다를 수 있어서,  
  실제로 **한 번 Pod를 띄운 뒤 `GET /pods/{podId}` 응답**을 찍어서  
  `publicIp` 옆에 오는 **포트 매핑 필드**(예: `runtime.ports`, `exposedPorts` 등)를 확인하는 게 좋습니다.
- 그 필드에서 **내부 8000에 대응하는 외부 포트**(13979)를 읽어와서  
  `base_url = f"http://{public_ip}:{그_포트}/v1"` 처럼 만들면,  
  “8000이 아니라 213.173.111.7:13979를 보내야 하는데 8000으로 보내는” 문제는 해결됩니다.
- RunPod 문서에 따르면 **대칭 매핑**(내부=외부 같은 번호)을 쓰려면  
  TCP 설정에서 **70000 이상** 번호를 요청하는 방식도 있지만,  
  그 경우에도 “실제 할당된 포트”는 API 응답이나 Connect 메뉴에서 확인해야 합니다.

정리하면, **반드시 8000으로 맞출 필요는 없고**, RunPod가 준 **실제 외부 포트(13979)**를 GET pod 응답에서 파싱해 써야 하며, 그 포트로 HTTP 요청을 보내면 됩니다.  
원하면 Agent 모드에서 GET pod 응답 구조를 가정해, `public_ip`와 함께 포트를 읽어 `base_url`을 만드는 코드 위치와 수정 예시를 짚어줄 수 있습니다.