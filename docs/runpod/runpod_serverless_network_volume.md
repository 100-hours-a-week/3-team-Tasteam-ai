
Network volumes for Serverless
When attached to a Serverless endpoint, a network volume is mounted at /runpod-volume within the worker environment.
​
Benefits for Serverless
Using network volumes with Serverless provides several advantages:
Reduced cold starts: Store large models or datasets on a network volume so workers can access them quickly without downloading on each cold start.
Cost efficiency: Network volume storage costs less than frequently re-downloading large files.
Simplified data management: Centralize your datasets and models for easier updates and management across multiple workers and endpoints.
​
Attach to an endpoint
To enable workers on an endpoint to use network volumes:
Navigate to the Serverless section of the Runpod console.
Select an existing endpoint and click Manage, then select Edit Endpoint.
In the endpoint configuration menu, scroll down and expand the Advanced section.
Click Network Volumes and select one or more network volumes you want to attach to the endpoint.
Configure any other fields as needed, then select Save Endpoint.
Data from the attached network volume(s) will be accessible to workers from the /runpod-volume directory. Use this path to read and write shared data in your handler function.
When you attach multiple network volumes to an endpoint, you can only select one network volume per datacenter.
Writing to the same network volume from multiple endpoints or workers simultaneously may result in conflicts or data corruption. Ensure your application logic handles concurrent access appropriately for write operations.
​
Attach multiple volumes
If you attach a single network volume to your Serverless endpoint, worker deployments will be constrained to the datacenter where the volume is located. This may impact GPU availability and failover options.
To improve GPU availability and reduce downtime during datacenter maintenance, you can attach multiple network volumes to your endpoint. Workers will be distributed across the datacenters where the volumes are located, with each worker receiving exactly one network volume based on its assigned datacenter.
Data does not sync automatically between multiple network volumes even if they are attached to the same endpoint. You’ll need to manually copy data (using the S3-compatible API or runpodctl) if you need the same data to be available to all workers on the endpoint (regardless of which volume they’re attached to).

---

## vLLM 워커에서 네트워크 볼륨으로 모델 로드

이 프로젝트의 **Dockerfile.runpod-serverless-vllm** 이미지는 vLLM이 **네트워크 볼륨 경로**에서 모델을 읽도록 설정되어 있습니다. **최초 1회 다운로드**: 해당 경로에 모델이 없으면 워커 기동 시 `huggingface_hub.snapshot_download`로 자동 다운로드한 뒤 vLLM을 띄웁니다. 이미 있으면 다운로드 없이 바로 vLLM만 기동합니다.

### 1. 네트워크 볼륨에 모델 배치

워커가 기대하는 기본 경로는 다음과 같습니다.

- **경로**: `/runpod-volume/llm-models/Qwen/Qwen2.5-7B-Instruct`
- RunPod가 엔드포인트에 볼륨을 연결하면 워커 컨테이너 내부에서 위 경로가 해당 볼륨의 디렉터리와 일치합니다.

모델을 이 경로에 넣는 방법 예시:

- **runpodctl**로 볼륨에 접속한 뒤, `huggingface-cli download` 또는 `git clone`으로 해당 경로에 모델 파일을 받기.
- 또는 RunPod에서 **일회성 Pod/Job**을 만들어 같은 네트워크 볼륨을 마운트하고, 그 안에서 모델을 다운로드해 `/runpod-volume/llm-models/Qwen/Qwen2.5-7B-Instruct` 구조로 저장.

다른 모델을 쓰려면 볼륨 안에 해당 이름으로 디렉터리를 만들고, 엔드포인트 환경 변수 `MODEL_NAME`을 그 경로로 설정하면 됩니다 (예: `MODEL_NAME=/runpod-volume/llm-models/Mistral-7B-Instruct-v0.2`).

### 2. 엔드포인트 설정

1. **Serverless** → 엔드포인트 생성 또는 기존 엔드포인트 **Edit**.
2. **이미지**: `Dockerfile.runpod-serverless-vllm`으로 빌드·푸시한 이미지 선택.
3. **Advanced** → **Network Volumes**: 위에서 모델을 넣어 둔 네트워크 볼륨을 선택해 연결.
4. (선택) **Environment Variables**에서 `MODEL_NAME`을 오버라이드할 수 있음. 기본값은 `/runpod-volume/llm-models/Qwen/Qwen2.5-7B-Instruct`.

저장 후 워커가 기동하면 vLLM이 `/runpod-volume`(연결된 네트워크 볼륨)에서 모델을 로드합니다.

### 3. 요약

| 항목 | 내용 |
|------|------|
| 볼륨 마운트 경로 | `/runpod-volume` (RunPod가 자동 마운트) |
| 이 이미지 기본 모델 경로 | `/runpod-volume/llm-models/Qwen/Qwen2.5-7B-Instruct` |
| 모델 배치 | (선택) 미리 넣거나, 비워 두면 워커 최초 기동 시 자동 다운로드(snapshot_download). 네트워크 볼륨 연결 필수. |
| 효과 | 2회차부터는 HF 다운로드 없이 기동, 기동 시간 단축 및 비용 절감 |

### 4. Pod 버전 (네트워크 볼륨 /workspace)

**Dockerfile.runpod-pod-vllm** 이미지는 RunPod **Pod** 전용이며, 네트워크 볼륨을 **/workspace** 에 마운트하는 구성을 전제로 합니다. 모델 경로는 `/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct` 이고, 해당 경로에 모델이 없으면 최초 기동 시 `snapshot_download`로 자동 다운로드한 뒤 vLLM을 띄웁니다. Pod 생성 시 네트워크 볼륨 마운트 경로를 `/workspace`로 설정하면 됩니다.

---

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.runpod.io/llms.txt
> Use this file to discover all available pages before exploring further.

# Overview

> Write custom handler functions to process incoming requests to your queue-based endpoints.

export const LoadBalancingEndpointTooltip = () => {
  return <Tooltip headline="Load balancing endpoint" tip="A Serverless endpoint that routes requests directly to worker HTTP servers without queuing, ideal for real-time applications and streaming. Supports custom HTTP frameworks like FastAPI or Flask." cta="Learn more about load balancing endpoints" href="/serverless/load-balancing/overview">load balancing endpoint</Tooltip>;
};

export const QueueBasedEndpointsTooltip = () => {
  return <Tooltip headline="Queue-based endpoint" tip="A Serverless endpoint that processes requests sequentially through a managed queue, providing guaranteed execution and automatic retries. Uses handler functions and standard operations like /run and /runsync." cta="Learn more about queue-based endpoints" href="/serverless/endpoints/overview#queue-based-endpoints">queue-based endpoints</Tooltip>;
};

export const WorkersTooltip = () => {
  return <Tooltip headline="Worker" tip="A container that runs your application code and processes requests to your Serverless endpoint. Workers are automatically started and stopped by Runpod to handle traffic spikes and ensure optimal resource utilization." cta="Learn more about workers" href="/serverless/workers/overview">worker</Tooltip>;
};

export const RequestsTooltip = () => {
  return <Tooltip headline="Requests" tip="HTTP requests that you send to an endpoint, which can include parameters, payloads, and headers that define what the endpoint should process." cta="Learn more about requests" href="/serverless/endpoints/send-requests">requests</Tooltip>;
};

export const JobTooltip = () => {
  return <Tooltip headline="Job" tip="A unit of work submitted to a queue-based Serverless endpoint. Jobs progress through states like IN_QUEUE, RUNNING, and COMPLETED as they are processed by workers." cta="Learn more about job states" href="/serverless/endpoints/job-states">job</Tooltip>;
};

Handler functions form the core of your Runpod Serverless applications. They define how your <WorkersTooltip /> process <RequestsTooltip /> and return results. This section covers everything you need to know about creating effective handler functions.

<Warning>
  Handler functions are only required for <QueueBasedEndpointsTooltip />. If you're building a <LoadBalancingEndpointTooltip />, you can define your own custom API endpoints using any HTTP framework of your choice (like FastAPI or Flask).
</Warning>

## Understanding job input

Before writing a handler function, make sure you understand the structure of the input. When your endpoint receives a request, it sends a JSON object to your handler function in this general format:

```json  theme={"theme":{"light":"github-light","dark":"github-dark"}}
{
    "id": "eaebd6e7-6a92-4bb8-a911-f996ac5ea99d",
    "input": { 
        "key": "value" 
    }
}
```

`id` is a unique identifier for the <JobTooltip /> randomly generated by Runpod, while `input` contains data sent by the client for your handler function to process.

To learn how to structure requests to your endpoint, see [Send API requests](/serverless/endpoints/send-requests).

## Basic handler implementation

Here's a simple handler function that processes an endpoint request:

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod

def handler(job):
    job_input = job["input"]  # Access the input from the request

    # Add your custom code here to process the input

    return "Your job results"

runpod.serverless.start({"handler": handler})  # Required
```

The handler takes extracts the input from the job request, processes it, and returns a result. The `runpod.serverless.start()` function launches your serverless application with the specified handler.

## Local testing

To test your handler locally, you can create a `test_input.json` file with the input data you want to test:

```json test_input.json theme={"theme":{"light":"github-light","dark":"github-dark"}}
{
    "input": {
        "prompt": "Hey there!"
    }
}
```

Then run your handler function using your local terminal:

```sh  theme={"theme":{"light":"github-light","dark":"github-dark"}}
python handler.py
```

Instead of creating a `test_input.json` file, you can also provide test input directly in the command line prompt:

```sh  theme={"theme":{"light":"github-light","dark":"github-dark"}}
python handler.py --test_input '{"input": {"prompt": "Test prompt"}}'
```

For more information on local testing, including command-line flags and starting a local API server, see [Local testing](/serverless/development/local-testing).

## Handler types

You can create several types of handler functions depending on the needs of your application.

### Standard handlers

The simplest handler type, standard handlers process inputs synchronously and return them when the job is complete.

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod
import time

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt")
    seconds = job_input.get("seconds", 0)
    
    # Simulate processing time
    time.sleep(seconds)
    
    return prompt

runpod.serverless.start({"handler": handler})
```

### Streaming handlers

Streaming handlers stream results incrementally as they become available. Use these when your application requires real-time updates, for example when streaming results from a language model.

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod

def streaming_handler(job):
    for count in range(3):
        result = f"This is the {count} generated output."
        yield result

runpod.serverless.start({
    "handler": streaming_handler,
    "return_aggregate_stream": True  # Optional, makes results available via /run
})
```

By default, outputs from streaming handlers are only available using the `/stream` operation. Set `return_aggregate_stream` to `True` to make outputs available from the `/run` and `/runsync` operations as well.

### Asynchronous handlers

Asynchronous handlers process operations concurrently for improved efficiency. Use these for tasks involving I/O operations, API calls, or processing large datasets.

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod
import asyncio

async def async_handler(job):
    for i in range(5):
        # Generate an asynchronous output token
        output = f"Generated async token output {i}"
        yield output
        
        # Simulate an asynchronous task
        await asyncio.sleep(1)
        
runpod.serverless.start({
    "handler": async_handler,
    "return_aggregate_stream": True
})
```

Async handlers allow your code to handle multiple tasks concurrently without waiting for each operation to complete. This approach offers excellent scalability for applications that deal with high-frequency requests, allowing your workers to remain responsive even under heavy load. Async handlers are also useful for streaming data scenarios and long-running tasks that produce incremental outputs.

<Tip>
  When implementing async handlers, ensure proper use of `async` and `await` keywords throughout your code to maintain truly non-blocking operations and prevent performance bottlenecks, and consider leveraging the `yield` statement to generate outputs progressively over time.

  Always test your async code thoroughly to properly handle asynchronous exceptions and edge cases, as async error patterns can be more complex than in synchronous code.
</Tip>

### Concurrent handlers

Concurrent handlers process multiple requests simultaneously with a single worker. Use these for small, rapid operations that don't fully utlize the worker's GPU.

When increasing concurrency, it's crucial to monitor memory usage carefully and test thoroughly to determine the optimal concurrency levels for your specific workload. Implement proper error handling to prevent one failing request from affecting others, and continuously monitor and adjust concurrency parameters based on real-world performance.

Learn how to build a concurrent handler by [following this guide](/serverless/workers/concurrent-handler).

## Error handling

When an exception occurs in your handler function, the Runpod SDK automatically captures it, marks the [job status](/serverless/endpoints/job-states) as `FAILED` and returns the exception details in the job results.

For custom error responses:

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod

def handler(job):
    job_input = job["input"]
    
    # Validate the presence of required inputs
    if not job_input.get("seed", False):
        return {
            "error": "Input is missing the 'seed' key. Please include a seed."
        }
    
    # Proceed if the input is valid
    return "Input validation successful."

runpod.serverless.start({"handler": handler})
```

Exercise caution when using `try/except` blocks to avoid unintentionally suppressing errors. Either return the error for a graceful failure or raise it to flag the job as `FAILED`.

## Advanced handler controls

Use these features to fine-tune your Serverless applications for specific use cases.

### Progress updates

Send progress updates during job execution to inform clients about the current state of processing:

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
import runpod

def handler(job):
    for update_number in range(0, 3):
        runpod.serverless.progress_update(job, f"Update {update_number}/3")
    
    return "done"

runpod.serverless.start({"handler": handler})
```

Progress updates will be available when the job status is polled.

### Worker refresh

For long-running or complex jobs, you may want to refresh the worker after completion to start with a clean state for the next job. Enabling worker refresh clears all logs and wipes the worker state after a job is completed.

For example:

```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
# Requires runpod python version 0.9.0+
import runpod
import time

def handler(job):
    job_input = job["input"]  # Access the input from the request

    results = []
    
    # Compute results
    ...

    # Return the results and indicate the worker should be refreshed
    return {"refresh_worker": True, "job_results": results}


# Configure and start the Runpod serverless function
runpod.serverless.start(
    {
        "handler": handler,  # Required: Specify the sync handler
        "return_aggregate_stream": True,  # Optional: Aggregate results are accessible via /run operation
    }
)
```

Your handler must return a dictionary that contains the `refresh_worker` flag. This flag will be removed before the remaining job output is returned.

## Handler function best practices

A short list of best practices to keep in mind as you build your handler function:

1. **Initialize outside the handler**: Load models and other heavy resources outside your handler function to avoid repeated initialization.

   ```python handler.py theme={"theme":{"light":"github-light","dark":"github-dark"}}
   import runpod
   import torch
   from transformers import AutoModelForSequenceClassification, AutoTokenizer

   # Load model and tokenizer outside the handler
   model_name = "distilbert-base-uncased-finetuned-sst-2-english"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name)

   # Move model to GPU if available
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)

   def handler(job):
       # ...

   runpod.serverless.start({"handler": handler})
   ```

2. **Input validation**: [Validate inputs](#error-handling) before processing to avoid errors during execution.

3. **Local testing**: [Test your handlers locally](/serverless/development/local-testing) before deployment.

## Payload limits

Be aware of payload size limits when designing your handler:

* `/run` operation: 10 MB
* `/runsync` operation: 20 MB

If your results exceed these limits, consider stashing them in cloud storage and returning links instead.

## Next steps

Once you've created your handler function, you can:

* [Explore flags for local testing.](/serverless/development/local-testing)
* [Create a Dockerfile for your worker.](/serverless/workers/create-dockerfile)
* [Deploy your worker image to a Serverless endpoint.](/serverless/workers/deploy)
