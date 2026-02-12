# OpenAI 429 Rate Limit 시 vLLM 폴백

OpenAI API에서 **429 Too Many Requests**(rate limit)가 발생하면 **vLLM(Qwen2.5-7B-Instruct 등)** 으로 자동 폴백하는 기능 설명입니다.

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| **목적** | OpenAI 일일/분당 한도(RPD/TPM) 소진 시 요약·비교 등 LLM 호출이 실패하지 않고, 동일 요청을 vLLM으로 재시도 |
| **적용 경로** | `LLMUtils._generate_response_async()` 를 사용하는 모든 호출 (Summary, Comparison 등). Sentiment 재판정은 별도 AsyncOpenAI 직접 호출이라 본 폴백 미적용 |
| **기본 동작** | `ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT=false` → 429 시 기존처럼 예외 발생 |
| **폴백 동작** | `ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT=true` + vLLM 설치 시: 429 발생 → vLLM 인스턴스 지연 초기화 후 동일 messages로 vLLM 생성 → 반환 |

---

## 2. 설정

### 2.1 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT` | `false` | `true` 로 설정 시 OpenAI 429 발생 시 vLLM 폴백 시도 |
| `VLLM_USE_RUNPOD_GPU` | `false` | `true` 면 폴백·1차 vLLM 모두 **RunPod Serverless** 엔드포인트 사용. 요청 시 GPU 기동, 유휴 시 스케일다운(별도 Watchdog 불필요). `RUNPOD_API_KEY` 필요. |
| `RUNPOD_VLLM_ENDPOINT_ID` | (없음) | RunPod vLLM 전용 엔드포인트 ID. 미설정 시 `RUNPOD_ENDPOINT_ID` 사용. |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | 폴백 시 사용할 vLLM 모델 (인프로세스 폴백 시만; RunPod 사용 시 엔드포인트 쪽 모델) |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel (인프로세스만) |
| `VLLM_MAX_MODEL_LEN` | (없음) | vLLM max_model_len (선택, 인프로세스만) |

### 2.2 RunPod GPU 사용 시 (권장: 429 시 GPU 기동·유휴 시 종료)

- **`VLLM_USE_RUNPOD_GPU=true`** 로 두면, 429 폴백과 기동 시 1차 vLLM 모두 **RunPod Serverless** 엔드포인트로 요청합니다.
- **동작**: 요청 시 RunPod가 GPU 워커를 기동 → Qwen 추론 → 유휴 시 RunPod가 자동 스케일다운. 별도 Watchdog 스크립트 없이 비용 절감.
- **필수**: `RUNPOD_API_KEY` 설정. vLLM(Qwen2.5-7B 등)을 서빙하는 RunPod Serverless 엔드포인트를 만들고, `RUNPOD_VLLM_ENDPOINT_ID`(또는 `RUNPOD_ENDPOINT_ID`)에 해당 엔드포인트 ID를 넣습니다.

### 2.3 의존성 (인프로세스 폴백만 해당)

- **vLLM 패키지**: 인프로세스 폴백(`VLLM_USE_RUNPOD_GPU=false`)일 때만 `pip install vllm>=0.3.3` 필요.
- **GPU**: 인프로세스 vLLM은 GPU 서버(RunPod Pod, 로컬 GPU 등) 전제. RunPod Serverless 사용 시 앱 서버는 GPU 불필요.

---

## 3. 동작 흐름

1. **정상**: `LLM_PROVIDER=openai`(또는 기본) → `_generate_response_async()` 에서 `AsyncOpenAI` 로 요청 → 성공 시 그대로 반환.
2. **429 발생**: OpenAI 가 429(Rate limit) 반환 → 예외 catch.
3. **폴백 조건**: `Config.ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT is True` 이면 vLLM 폴백 시도.
4. **폴백 경로 (둘 중 하나)**  
   - **RunPod GPU** (`VLLM_USE_RUNPOD_GPU=true` + `RUNPOD_API_KEY`): `messages` → 프롬프트 변환 후 RunPod Serverless 엔드포인트 호출(`_call_runpod_async`). 요청 시 GPU 기동, 유휴 시 스케일다운.  
   - **인프로세스 vLLM**: 첫 429 시점에 `_init_vllm_fallback()` 호출 → vLLM `LLM` 인스턴스 생성 후 `_vllm_fallback_llm` 에 캐시. `messages` 를 프롬프트로 변환 후 `_vllm_fallback_llm.generate(...)` 를 `asyncio.to_thread` 로 실행.
5. **폴백 실패**: RunPod 요청 실패·vLLM 미설치·초기화 실패·생성 예외 시 로그 후 **원래 429 예외**를 다시 발생시켜 호출자는 OpenAI rate limit 실패로 처리 가능.

---

## 4. 코드 위치

| 구분 | 파일 | 내용 |
|------|------|------|
| 설정 | `src/config.py` | `ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT` |
| 429 판별 | `src/llm_utils.py` | `_is_openai_rate_limit(e)` |
| 프롬프트 변환 | `src/llm_utils.py` | `_messages_to_prompt_for_vllm(messages)` |
| vLLM 지연 초기화 | `src/llm_utils.py` | `_init_vllm_fallback()` |
| 동기/비동기 폴백 생성 | `src/llm_utils.py` | `_generate_with_vllm_fallback_sync`, `_generate_with_vllm_fallback_async` |
| OpenAI 호출부 | `src/llm_utils.py` | `_generate_response_async()` 내 use_openai 분기에서 except 후 429 시 폴백 호출 |

---

## 5. 적용 범위와 제한

- **적용됨**: `llm_utils._generate_response_async()` 를 쓰는 경로 (예: Summary 배치/단일, Comparison 해석 등).
- **적용 안 됨**: `src/sentiment_analysis.py` 의 감성 재판정은 `AsyncOpenAI` 를 직접 호출하므로 본 폴백 로직과 무관. 429 시 그대로 실패. (필요 시 해당 호출부에서 429 catch 후 `llm_utils` 의 폴백 가능 메서드를 호출하도록 확장 가능.)
- **모델**: 폴백 시에는 `Config.LLM_MODEL`(기본 Qwen2.5-7B-Instruct) 사용. OpenAI와 출력 스타일/품질 차이는 있을 수 있음.

---

## 6. 사용 예시

```bash
# .env 또는 환경 변수
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
ENABLE_VLLM_FALLBACK_ON_RATE_LIMIT=true   # 429 시 vLLM 폴백 활성화
# vLLM 설치
pip install vllm>=0.3.3
```

vLLM은 GPU가 있는 환경에서 실행. RunPod Pod 등에서 vLLM 서버를 띄우고 같은 노드에서 FastAPI를 실행하는 구성이면, 429 시 동일 Pod 내 vLLM으로 폴백되어 요청이 유지됨.

---

## 7. 기동 시 vLLM 1차 사용 (폴백 아님)

OpenAI 대신 **처음부터 vLLM만** 쓰려면 다음처럼 설정하면 됩니다.

| 설정 | 값 | 설명 |
|------|-----|------|
| `LLM_PROVIDER` | `runpod` | `openai` 가 아니어야 함 (openai 이면 use_pod_vllm 이 꺼짐) |
| `USE_POD_VLLM` | `true` | 기동 시 vLLM 1차 LLM으로 사용 |
| `VLLM_USE_RUNPOD_GPU` | `true`(권장) | RunPod Serverless 엔드포인트 사용 시 요청 시 GPU 기동·유휴 시 스케일다운. `false` 면 인프로세스 vLLM(GPU 서버 필요). |

- **RunPod GPU 사용 시**: `VLLM_USE_RUNPOD_GPU=true` + `RUNPOD_API_KEY` + `RUNPOD_VLLM_ENDPOINT_ID`(또는 `RUNPOD_ENDPOINT_ID`) 설정 시, 인프로세스 vLLM을 로드하지 않고 RunPod 엔드포인트로만 요청합니다.
- **인프로세스 vLLM**: `VLLM_USE_RUNPOD_GPU=false` 이면 기동 시 vLLM 초기화 및 `_generate_with_vllm_primary_async()` 호출. vLLM 패키지 설치 및 GPU 환경 필요.

---

## 8. RunPod GPU vs Pod + Watchdog

| 방식 | 설정 | GPU 기동/종료 | 비고 |
|------|------|----------------|------|
| **RunPod Serverless** | `VLLM_USE_RUNPOD_GPU=true` | 요청 시 기동, 유휴 시 RunPod가 자동 스케일다운 | 별도 Watchdog 불필요. 429 폴백·기동 시 1차 vLLM 모두 적용. |
| **Pod + Watchdog** | `USE_POD_VLLM=true`, `VLLM_USE_RUNPOD_GPU=false` | Pod 상시 가동, 외부 Watchdog가 GPU 유휴 시 RunPod API로 Pod 종료 | 상시 지연 최소화가 필요할 때. `etc_md/RUNPOD_POD_VLLM_GUIDE.md` 참고. |

---

*이 문서는 OpenAI 429 폴백 기능, RunPod GPU 폴백/1차 사용, 기동 시 vLLM 1차 사용 구현 기준으로 작성되었습니다.*
