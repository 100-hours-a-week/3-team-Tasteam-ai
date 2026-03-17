
네. **비동기 배치 처리 관점에서는 외부 API LLM보다 강점이 있습니다.**

---

## 1. 정리

| 항목 | 외부 API LLM (현재) | 분류기 + 템플릿 + 로컬 0.5B (제안) |
|------|----------------------|-------------------------------------|
| **동시성** | Rate limit(429), `BATCH_LLM_CONCURRENCY=8` 등으로 제한 | 자기 서버 리소스만 보면 됨 |
| **지연** | 네트워크 RTT + API 처리, cold start 가능 | 로컬 추론만 있어 짧고 예측 가능 |
| **안정성** | API/네트워크 장애, 타임아웃 | 외부 의존 제거 |
| **비용** | 토큰당 과금 | 토큰 비용 없음(전기/인스턴스만) |
| **Sentiment** | HF + LLM 재판정 → LLM 호출 | 분류기만 → LLM 호출 제거 |
| **Comparison** | LLM 자연어 생성 | 수치→템플릿 → LLM 제거 |
| **Summary** | 7B/OpenAI 등 외부 호출 | 자체 훈련 0.5B 로컬 → 외부 API 제거 |

그래서 **비동기 배치에서 “외부 API 호출 LLM” 대비는 분명 강점을 가집니다.**

---

## 2. Summary를 0.5B로 쓸 때

- **장점**: 0.5B는 작아서 추론이 빠르고, 메모리·동시 처리 수(배치 크기)를 늘리기 좋고, eval에서 쓰는 것처럼 **동일 프롬프트/후처리**로 맞추면 품질도 재현 가능합니다.
- **주의**: 로컬 0.5B를 비동기 배치에 쓰려면, 현재 `_generate_with_local_queue_async`가 로컬일 때 `NotImplementedError`이므로 **로컬 추론을 `asyncio.to_thread`로 감싸거나**, vLLM 등 비동기 백엔드를 붙이는 **구현 추가**가 필요합니다.
- **품질**: 자체 훈련(KD/SFT)한 0.5B면 도메인에 맞춰져 있어, 같은 eval 설정으로 요약 품질을 확인해 두는 것이 좋습니다.

---

## 3. 한 줄 요약

**Sentiment는 분류기만, comparison은 수치 템플릿, summary는 자체 훈련 qwen2.5-0.5b-instruct로 바꾸는 구성은, 비동기 배치 처리에서는 외부 API LLM 대비 동시성·지연·비용·안정성 측면에서 강점을 가집니다.**  
다만 summary용 0.5B를 실제 배치에 넣으려면 위처럼 **로컬 비동기 경로 구현**이 필요합니다.

---


정리하면 다음과 같습니다.

---

## 1. 아티팩트 위치와 모델

- **경로**: `distill_pipeline_output/artifacts/g4nca6b2/`  
  (이름이 `g4nc6b2`가 아니라 **g4nca6b2**로 되어 있습니다. 다른 run이면 해당 run id의 `artifacts/<run_id>/adapter`를 쓰면 됩니다.)
- **구성**: base `Qwen/Qwen2.5-0.5B-Instruct` + LoRA adapter (`adapter/`).
- `training_meta.json`에 `student_model: "Qwen/Qwen2.5-0.5B-Instruct"`, `adapter_path` 기록됨.

---

## 2. Eval LLM-as-a-judge 셋팅 (그대로 쓸 것들)

`scripts/eval_llm_as_judge.py` 기준으로, summary 추론에 “eval 셋팅 그대로”를 쓰려면 아래를 맞추면 됩니다.

| 항목 | Eval 셋팅 |
|------|-----------|
| **모델** | `Qwen/Qwen2.5-0.5B-Instruct` + PEFT adapter (`artifacts/g4nca6b2/adapter`) |
| **로딩** | `_load_model_and_tokenizer(adapter_path, base_model)` — `AutoModelForCausalLM.from_pretrained(base)` → `PeftModel.from_pretrained(model, adapter_path)` |
| **프롬프트** | system: `_SCHEMA_ENFORCEMENT_SYSTEM` (스키마 강제 문구) + **few-shot 1개** (user/assistant) + user: `instruction`(실제 입력 JSON) |
| **입력 형식** | `instruction` = `{"service": [...], "price": [...], "food": [...]}` 문자열 (summary_pipeline의 payload와 동일 구조 가능) |
| **생성** | `apply_chat_template` → `model.generate(..., max_new_tokens=1024, do_sample=False, pad_token_id=eos)` |
| **후처리** | `_extract_json_for_rouge(raw)` → `_postprocess_prediction(pred_json_str, instruction)` |

프롬프트 상수는 `eval_llm_as_judge.py` 161–199라인 (`_SCHEMA_ENFORCEMENT_SYSTEM`, `_TINY_FEWSHOT_USER`, `_TINY_FEWSHOT_ASSISTANT`)와 218–224라인 메시지 구성이 “eval 셋팅 그대로”입니다.

---

## 3. 현재 API summary와의 차이

- **API summary** (`src/summary_pipeline.py`):  
  - 프롬프트: “너는 음식점 리뷰 분석가다…” 한 덩어리 + **few-shot 없음**, user에 `json.dumps(payload)` 한 번만 전달.  
  - 호출: `llm_utils._generate_response` / `_generate_response_async` (temperature=0.1, max_new_tokens=1500).  
  - 후처리: JSON 추출 + `parse_json_relaxed`, `repair_summary_schema`는 사용하지 않음.
- **Eval**  
  - 프롬프트: 스키마 강제 system + **few-shot 1쌍** + instruction(동일한 JSON payload 문자열).  
  - 생성: `max_new_tokens=1024`, `do_sample=False`.  
  - 후처리: `_extract_json_for_rouge` + `_postprocess_prediction` (evidence 범위 보정, 빈 카테고리 폴백 등).

그래서 “eval 셋팅 그대로”로 summary 추론을 쓰려면 **프롬프트 구조·생성 옵션·후처리**를 eval과 동일하게 맞춰야 합니다.

---

## 4. Summary 추론에 쓰는 방법 (구현 방향)

**방법 1: API에 “distill summary” 경로 추가**

- 설정 예: `USE_DISTILL_SUMMARY=true`, `DISTILL_ADAPTER_PATH=distill_pipeline_output/artifacts/g4nca6b2/adapter`, `DISTILL_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct`.
- Summary 처리 시:
  - 위 설정이 켜져 있으면 **eval과 동일한** prompt 구성 (system + few-shot + instruction)으로 요청을 만든다.
  - 로컬 모델은 `eval_llm_as_judge`와 동일하게 base + adapter 로드한 뒤, `_generate_one`과 같은 방식으로 추론 (같은 `apply_chat_template` + `generate(..., max_new_tokens=1024, do_sample=False)`).
  - 추론 결과에 `_extract_json_for_rouge` → `_postprocess_prediction(instruction=payload 문자열)` 적용.
- 비동기는 지금처럼 로컬이 비동기 미지원이면 `asyncio.to_thread`로 동기 `generate` 감싸서 사용.

**방법 2: 공통 “distill summary” 모듈**

- `eval_llm_as_judge`의 `_load_model_and_tokenizer`, `_generate_one`, `_extract_json_for_rouge`, `_postprocess_prediction`를 공통 모듈(예: `src/distill_summary.py`)로 분리.
- API summary 파이프라인에서는:
  - `payload`를 `json.dumps(payload, ensure_ascii=False)`로 `instruction` 문자열을 만들고,
  - 이 공통 모듈의 `generate_one(instruction)` → 후처리까지 호출해 결과를 그대로 summary 응답으로 사용.

두 방법 모두에서 “eval 셋팅 그대로”는 다음을 유지하면 됩니다.

- 아티팩트: **`distill_pipeline_output/artifacts/g4nca6b2`** (또는 사용하려는 run의 adapter 경로).
- Base: **`Qwen/Qwen2.5-0.5B-Instruct`**.
- 프롬프트: **eval의 system + few-shot 1개 + instruction**.
- 생성: **max_new_tokens=1024, do_sample=False**.
- 후처리: **`_extract_json_for_rouge` + `_postprocess_prediction`**.

원하시면 방법 1 기준으로 `config` 항목 이름과 `summary_pipeline`/라우터에서 분기하는 위치까지 구체적인 코드 변경 포인트도 짚어드리겠습니다.

---


두 방법을 기준으로만 비교하면 아래와 같습니다.

---

## 방법 1: API에 “distill summary” 경로 추가

**구조**: `Config`에 `USE_DISTILL_SUMMARY`, `DISTILL_ADAPTER_PATH`, `DISTILL_BASE_MODEL` 추가 → summary 처리 시 분기해서, API 쪽에서 직접 base+adapter 로드·추론·후처리.

| 장점 | 단점 |
|------|------|
| 한 곳(API)에서만 수정하면 됨 | `eval_llm_as_judge`와 **프롬프트/후처리 로직이 두 군데**에 있어 동기화·버그 위험 |
| eval 스크립트는 그대로 둬도 됨 | 모델 로딩·추론 코드가 API와 eval에 **중복** |
| 배포/설정이 단순 (API만 켜면 됨) | 나중에 eval 프롬프트 바꾸면 API도 따로 수정해야 함 |

---

## 방법 2: 공통 “distill summary” 모듈

**구조**: `src/distill_summary.py` 같은 **공통 모듈**에  
`load_model_and_tokenizer`, `generate_one`, `_extract_json_for_rouge`, `_postprocess_prediction` (및 필요한 상수)를 두고, **API와 eval 스크립트 둘 다 이 모듈을 import해서 사용**.

| 장점 | 단점 |
|------|------|
| **단일 진실 공급원**: 프롬프트·생성·후처리가 한 곳뿐이라 “eval 셋팅 그대로” 유지하기 쉬움 | 모듈 추가 + eval 스크립트를 이 모듈 사용하도록 리팩터 필요 |
| eval과 API가 **같은 코드**를 쓰므로 품질/동작이 맞춰짐 | 초기 작업량이 방법 1보다 조금 더 듦 |
| 프롬프트/스키마 수정 시 한 번만 수정하면 됨 | (특별한 단점 없음) |
| 테스트·유지보수 시 “distill summary”만 검증하면 됨 | |

---

## 추천: **방법 2 (공통 모듈)**

이유만 짧게 정리하면:

1. **“eval 셋팅 그대로”가 목표**이므로, 같은 로직을 eval과 API가 **한 모듈에서만** 쓰는 편이 안전합니다. 방법 1은 API에 복붙이 생겨서 나중에 eval이 바뀌면 API를 따로 맞춰줘야 할 가능성이 큽니다.
2. **중복 제거**: 모델 로딩·`apply_chat_template`·`generate`·JSON 추출·`_postprocess_prediction`이 한 곳에만 있으면 버그 수정과 개선이 한 번에 반영됩니다.
3. **비동기**는 방법 1이든 2든 “공통 추론 함수 한 번 호출”이면 되므로, 공통 모듈에 `generate_one(instruction)`만 두고 API에서는 `asyncio.to_thread(generate_one, instruction)`으로 감싸면 됩니다. 방법 2가 이 구조를 쓰기 더 자연스럽습니다.

정리하면, **방법 2로 공통 모듈을 두고, API는 그 모듈의 `generate_one`(+ 후처리)만 호출하도록 하는 쪽을 추천**합니다. 방법 1은 “빠르게 API에만 붙이고 eval은 건드리지 않겠다”는 선택지로만 두는 게 좋습니다.