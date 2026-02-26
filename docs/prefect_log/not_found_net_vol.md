{"labeled_path": "/Users/js/tasteam-new-async/distill_pipeline_output/labeled/20260226_051037/train_labeled_gold_only.json", "meta": {"openai_count": 500, "self_hosted_count": 0, "filter_drop_count": 0, "llm_fail_count": 0}, "val_labeled_path": "/Users/js/tasteam-new-async/distill_pipeline_output/labeled/20260226_051037/val_labeled.json", "test_labeled_path": "/Users/js/tasteam-new-async/distill_pipeline_output/labeled/20260226_051037/test_labeled.json"}
21:19:18.115 | ERROR   | Task run 'labeling-with-pod-task-13a' - Task run failed with exception: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 242, in labeling_with_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}
21:19:18.138 | ERROR   | Task run 'labeling-with-pod-task-13a' - Finished in state Failed("Task run encountered an exception HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
21:19:18.139 | ERROR   | Flow run 'enigmatic-mastiff' - Encountered exception during execution: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 310, in labeling_with_pod_flow
    return labeling_with_pod_task(
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 242, in labeling_with_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}
21:19:18.175 | INFO    | Flow run 'enigmatic-mastiff' - Finished in state Failed("Flow run encountered an exception: HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
21:19:18.176 | ERROR   | Flow run 'colossal-panther' - Encountered exception during execution: HTTPError("HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 984, in distill_pipeline_all_sweep
    lb = labeling_with_pod_flow(
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 310, in labeling_with_pod_flow
    return labeling_with_pod_task(
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 242, in labeling_with_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}
21:19:18.185 | INFO    | Flow run 'colossal-panther' - Finished in state Failed("Flow run encountered an exception: HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}")
Traceback (most recent call last):
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1142, in <module>
    main()
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 1130, in main
    result = distill_pipeline_all_sweep(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 984, in distill_pipeline_all_sweep
    lb = labeling_with_pod_flow(
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 310, in labeling_with_pod_flow
    return labeling_with_pod_task(
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
  File "/Users/js/tasteam-new-async/scripts/distill_flows.py", line 242, in labeling_with_pod_task
    pod = client.create_pod(payload)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 20, in create_pod
    return self._handle_json_response(resp)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/js/tasteam-new-async/scripts/runpod_cli/pod_create_delete_cli.py", line 162, in _handle_json_response
    raise requests.HTTPError(
requests.exceptions.HTTPError: HTTP 500 for POST https://rest.runpod.io/v1/pods: {'error': 'create pod: get attached volume: network volume not found', 'status': 500}
21:19:18.195 | INFO    | prefect - Stopping temporary server on http://127.0.0.1:8648
(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % 