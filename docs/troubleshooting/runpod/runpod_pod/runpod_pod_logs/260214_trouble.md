runpod_pod_logs 디렉터리의 로그 분석 중입니다.


[2 tools called]


`runpod_pod_logs` 안의 두 파일을 기준으로 정리한 로그 분석입니다.

---

# RunPod Pod 로그 분석 요약

## 1. 파일 구분

| 파일 | 출처 | 내용 |
|------|------|------|
| `260214_runpod_pod_llm_container_error_logs.log` | **RunPod 컨테이너** (vLLM 서버) | vLLM 기동·모델 로드·API 요청 처리 로그 |
| `260214_runpod_pod_app_server_logs.log` | **로컬 앱 서버** (tasteam API) | Spark/비교 파이프라인 + RunPod LLM 호출 로그 |

---

## 2. RunPod 컨테이너 로그 (`container_error_logs`)

### 정상 동작 구간 (06:53:55 ~ 06:54:49 UTC)

- 엔트리포인트: `/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct` 에서 vLLM 기동.
- vLLM 0.11.0, CUDA, Flash Attention, `max_model_len=4096`.
- 모델 로드: 4개 shard, 약 15초, 약 14.25 GiB.
- KV cache: 6.35 GiB, 118,832 tokens, max concurrency 29x.
- CUDA graph 캡처 후 **API 서버가 `http://0.0.0.0:8000` 에서 기동 완료**.
- 라우트: `/v1/chat/completions`, `/v1/completions`, `/health` 등 정상 등록.

### 문제 구간 (07:04:22 UTC ~)

- **동일 에러 반복**:  
  `Error with model error=ErrorInfo(message='The model \`qwen/qwen2.5-7B-Instruct\` does not exist.', type='NotFoundError', ...)`  
  → `POST /v1/chat/completions` 가 **404 Not Found**.
- 요청 클라이언트 IP: `59.6.138.20` (외부에서 RunPod로 들어오는 요청).

**원인**:  
vLLM은 실제로 **경로 기준**으로 모델을 로드했고, 서버가 내부적으로 알고 있는 **모델 ID(또는 served model name)** 가 `qwen/qwen2.5-7B-Instruct` 가 아님.  
그런데 클라이언트(tasteam)는 요청 body에 `model: "qwen/qwen2.5-7B-Instruct"` 를 넣어 보내고 있어서, vLLM이 “그 이름의 모델은 없다”고 404를 반환하는 상황입니다.

---

## 3. 로컬 앱 서버 로그 (`app_server_logs`)

### Spark / PySpark

- **PYTHON_VERSION_MISMATCH**: Driver 3.11 vs Worker 3.13 → Spark Job 실패.
- `comparison_pipeline` 에서 Spark 실패 시 **Python 폴백**으로 진행해, 비교·리뷰 수 등은 정상 처리됨.

### RunPod LLM 호출

- `POST http://213.173.108.70:17517/v1/chat/completions` → **404 Not Found**.
- 에러 메시지: `The model \`qwen/qwen2.5-7B-Instruct\` does not exist.`
- 재시도 3회 후 “비교 해석 LLM 실패” 로 남고, batch API는 200 OK로 응답 (LLM 결과만 비어 있거나 폴백 처리된 상태로 보임).

즉, **앱 서버가 RunPod vLLM에 `model: "qwen/qwen2.5-7B-Instruct"` 로 요청하고 있고, RunPod 쪽에서는 그 이름의 모델이 없다고 응답**하고 있습니다.

---

## 4. 결론 및 조치 제안

| 구분 | 내용 |
|------|------|
| **RunPod vLLM** | 모델 로드·기동은 성공. `/v1/chat/completions` 는 열려 있으나, **요청에 넘어오는 `model` 값과 서버가 인식하는 모델 ID가 불일치**해서 404 발생. |
| **tasteam 앱** | Spark 버전 불일치는 Python 폴백으로 우회됨. **RunPod 호출 시 사용하는 모델 ID가 vLLM의 실제 served model name과 다름.** |

**권장 조치**

1. **RunPod vLLM 실제 모델 ID 확인**  
   - Pod에서 `GET /v1/models` (또는 RunPod/vLLM 문서의 “models list” API) 호출해, 응답에 나오는 `id` / `name` 확인.
```bash
# Runpod Terminal에서 curl http://localhost:8000/v1/models한 결과
   {"object":"list","data":[{"id":"/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct","object":"model","created":1771053302,"owned_by":"vllm","root":"/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-b720f80bea384c949a14944eb086451d","object":"model_permission","created":1771053302,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}root@91f949920462:/vllm-workspace#
```
2. **tasteam 설정 수정**  
   - `src/config.py` 또는 RunPod base URL과 함께 쓰는 **모델 이름 설정**을, 위에서 확인한 **vLLM이 반환하는 모델 ID와 동일하게** 변경 (예: 경로 기반이면 `/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct` 같은 값일 수 있음).
3. (선택) Spark 로그/버전 통일  
   - Driver/Worker Python 버전을 3.11로 맞추거나, `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` 로 3.11 지정해 PySpark 정상 동작으로 맞추면, Spark 관련 에러 로그는 사라짐.

정리하면, **RunPod Pod 로그에서 보이는 문제의 핵심은 “모델 ID 불일치로 인한 404”**이고, RunPod vLLM의 실제 모델 ID를 확인한 뒤 클라이언트의 `model` 값을 그에 맞추면 해결됩니다.