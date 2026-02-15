좋아. 지금 네 `new_async`(FastAPI) 안에 **LLM 요약/비교 + Sentiment(HF 1차 + LLM 재판정)**가 같이 들어있는 구조에서, “GPU가 필요한 부분(특히 LLM)”만 **별도 GPU inference 서버로 분리**하는 방향이 딱 자연스럽다. (현재 파이프라인/의존성 구조는 네 아키텍처 문서 기준으로 이해했어.) 

아래는 **실제로 분리할 때 제일 덜 아프고**, 운영/확장/비용 관점에서 깔끔한 설계안이야.

### 현재 LLM 아키텍처 (인프로세스 vLLM 제거됨)

앱은 **OpenAI 1차 사용**, **429 시 RunPod Serverless(vLLM) 폴백** 또는 **USE_POD_VLLM 시 RunPod만** 사용. 앱 내부에 `pip install vllm` 인프로세스 vLLM은 제거되어, GPU 추론은 RunPod HTTP 엔드포인트로만 수행.

![LLM 아키텍처: OpenAI primary + RunPod 429 fallback](ARCHITECTURE/llm_architecture_runpod_fallback.png)

---

## 1) 목표 구조 (권장: API 서버는 오케스트레이터, GPU 서버는 “추론 전용”)

### 지금(단일 프로세스)

* FastAPI가

  * Vector(Qdrant), Redis 락/캐시
  * Summary/Comparison에서 LLM 호출
  * Sentiment에서 HF + (일부) LLM 재판정
  * 메트릭/스킵/락 등 “업무 로직”까지 전부 담당 

### 바꾼 뒤(분리)

* **CPU/API 서버 (기존 new_async)**: “업무 로직/락/캐시/스킵/집계/응답 스키마” 유지
* **GPU Inference 서버(신규)**:

  * LLM 추론 (vLLM 같은 엔진)
  * (선택) Sentiment 모델도 GPU로 올려서 배치 추론
  * “추론만” 담당 (DB/Redis 락 같은 상태 로직은 여기 넣지 않는 걸 추천)

이렇게 하면, 너 문서에 있는 **락(SKIP/409), 배치 병렬, evidence 구성, price gate 같은 비즈 로직**은 그대로 두고 
**모델 호출 부분만 교체**하면 된다.

---

## 2) 서비스 분리 옵션 2가지

### 옵션 A) GPU 서버 하나에 “LLM + Sentiment” 같이 (구현 제일 쉬움)

* GPU pod/인스턴스 1개
* 내부에

  * `llm-service` (예: 8000)
  * `sentiment-service` (예: 8002)
* 장점: 운영 단순, 네가 원하는 “GPU 서버로 분리”를 가장 빠르게 달성
* 단점: 둘이 트래픽 패턴이 달라서 **스케일을 따로 못 함**

### 옵션 B) LLM GPU / Sentiment CPU(or 소형 GPU)로 분리 (운영 최적)

* 현실적으로 **LLM이 GPU 비용의 대부분**이라,

  * LLM만 GPU로 빼고
  * Sentiment HF는 CPU 유지(또는 아주 작은 GPU)
* 장점: 비용/스케일 최적, 장애 격리도 더 잘 됨
* 단점: 서비스가 2개가 됨

> 네가 “LLM, Sentiment 둘 다 GPU 서버로 분리”가 목표면 옵션 A로 시작해서, 나중에 옵션 B로 가도 돼.

---

## 3) API 서버 ↔ GPU 서버 계약(인터페이스)만 딱 정하면, 코드 변경이 작아짐

### (1) LLM inference API (권장 스펙)

* `POST /v1/chat/completions` (OpenAI 호환 형태로 맞추면 교체 쉬움)
* 스트리밍 필요하면 SSE 지원 (요약/비교는 스트리밍 없어도 되지만, TTFT/체감 개선엔 도움)

### (2) Sentiment inference API

* `POST /v1/sentiment:batch`
* 입력: `[{id, text}]` 배열
* 출력: `[{id, label, score}]`

### (3) Health/Readiness

* `GET /healthz` (프로세스 살아있나)
* `GET /readyz` (모델 로딩 끝났나, vLLM 엔진 준비됐나)

API 서버는 네 문서에 이미 있는 `/ready` 개념이 있으니, GPU 쪽도 동일하게 가져가면 운영이 편해져. 

---

## 4) “무엇을 GPU로 옮길지”를 파이프라인 기준으로 딱 찍으면

네 파이프라인 기준으로 옮길 포인트는 여기야: 

### Summary

* 카테고리별 검색/증거 구성/price gate → **API 서버에 남김**
* 최종 요약 생성(LLM 호출) → **GPU 서버**

### Comparison

* Kiwi/Spark 비율, lift 계산 → **API 서버**
* 해석 문장 생성(LLM 호출) → **GPU 서버**

### Sentiment

* HF 1차 분류 → (선택)

  * **CPU 유지**(지금도 `SENTIMENT_FORCE_CPU=true`로 안정성을 챙기고 있음) 
  * or **GPU로 이동**(리뷰가 많아질수록 배치 추론 이점)
* “부정 일부 LLM 재판정” → **GPU 서버(LLM)**

---

## 5) 운영에서 꼭 챙겨야 하는 6가지 (여기서 사고가 많이 남)

1. **타임아웃/서킷브레이커**

* GPU 서버 콜이 느려지면 API 서버 요청이 줄줄이 쌓여서 tail latency 폭발함
* `timeout(예: 20~60s) + 실패 시 fallback` 규칙 필요

  * fallback: (a) OpenAI로 우회, (b) 템플릿 문장/빈 요약(지금 스킵 응답처럼) 

2. **동시성 제한(세마포어)**

* 너는 이미 배치에서 `BATCH_LLM_CONCURRENCY` 같은 개념을 갖고 있어. 
* GPU 서버 호출도 **API 서버 쪽에서 동시성 제한**을 걸어야 GPU가 안정적으로 돌아.

3. **요청 ID 전파**

* `X-Request-Id`를 GPU 서버로 그대로 전달하고, GPU 로그/메트릭에도 찍히게.
* 디버깅 난이도가 확 내려감. 

4. **캐시/락의 위치**

* Redis 락/스킵 판단은 **API 서버가 계속 갖는 게 맞아.** 
* GPU 서버가 락까지 잡기 시작하면 “상태/업무 로직”이 분산돼서 운영이 어려워짐.

5. **Warm-up 위치 이동**

* 지금은 FastAPI lifespan에서 임베딩/감성 warm-up을 하고 있지? 
* 분리하면:

  * API 서버 warm-up: Qdrant/Redis/라우터 준비
  * GPU 서버 warm-up: 모델 로드 + 첫 토큰/첫 배치까지 완료
* API 서버의 `/ready`는 “GPU 서버 ready 확인”까지 포함할지 결정해야 함(보통 포함시키는 게 안전).

6. **모델/프롬프트 버전 관리**

* GPU 서버 응답에 `model_version`, `prompt_version` 같은 메타를 넣어두면,
  “왜 오늘 요약이 달라졌지?”가 바로 추적됨.

---

## 6) 추천하는 “최소 변경” 마이그레이션 순서

1. **LLM만 먼저 GPU로 분리** (요약/비교/재판정 LLM 호출만)
2. 안정화되면

   * Sentiment HF도 GPU로 옮길지 판단 (리뷰량/비용 보고)
3. 마지막으로

   * GPU on/off(온디맨드)까지 붙이기 (네가 전에 얘기한 last_request_at 기반 stop/terminate 같은 것)

---

원하면, 내가 네 `ARCHITECTURE_OVERVIEW.md` 기준으로 **“어느 모듈/함수에서 LLM 호출이 일어나는지”**(summary/comparison/sentiment 재판정) 지점을 기준으로,

* `LLMClient` 인터페이스 설계(로컬 GPU / OpenAI 스위치)
* 엔드포인트 스펙(요청/응답 JSON)
* 장애 시 fallback 정책(스킵/템플릿/우회)
  까지 한 번에 “바로 구현 가능한 작업 목록”으로 쪼개줄게. 

---

Runpod serverless endpoint

Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]
🚀 Deploy Guide: Follow our step-by-step deployment guide to deploy using the RunPod Console.
developer guide: https://docs.runpod.io/serverless/vllm/get-started


📦 Docker Image: runpod/worker-v1-vllm:<version>

Available Versions: See GitHub Releases

github releases: https://github.com/runpod-workers/worker-vllm/releases


CUDA Compatibility: Requires CUDA >= 12.1
Configuration
Configure worker-vllm using environment variables:

Environment Variable	Description	Default	Options
MODEL_NAME	Path of the model weights	"facebook/opt-125m"	Local folder or Hugging Face repo ID
HF_TOKEN	HuggingFace access token for gated/private models		Your HuggingFace access token
MAX_MODEL_LEN	Model's maximum context length		Integer (e.g., 4096)
QUANTIZATION	Quantization method		"awq", "gptq", "squeezellm", "bitsandbytes"
TENSOR_PARALLEL_SIZE	Number of GPUs	1	Integer
GPU_MEMORY_UTILIZATION	Fraction of GPU memory to use	0.95	Float between 0.0 and 1.0
MAX_NUM_SEQS	Maximum number of sequences per iteration	256	Integer
CUSTOM_CHAT_TEMPLATE	Custom chat template override		Jinja2 template string
ENABLE_AUTO_TOOL_CHOICE	Enable automatic tool selection	false	boolean (true or false)
TOOL_CALL_PARSER	Parser for tool calls		"mistral", "hermes", "llama3_json", "granite", "deepseek_v3", etc.
OPENAI_SERVED_MODEL_NAME_OVERRIDE	Override served model name in API		String
MAX_CONCURRENCY	Maximum concurrent requests	30	Integer
For the complete list of all available environment variables, examples, and detailed descriptions: Configuration

Configuration: https://github.com/runpod-workers/worker-vllm/blob/main/docs/configuration.md
