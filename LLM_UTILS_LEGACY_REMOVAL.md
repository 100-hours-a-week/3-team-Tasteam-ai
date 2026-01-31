# src/llm_utils.py 레거시 제거 기록

`test_all_task.py`에서 사용하는 API 요청/응답 흐름에만 필요한 코드를 남기고, 그 외 레거시 메서드·함수를 제거한 내역입니다.

**기준**: test_all_task.py가 호출하는 API  
- `/api/v1/sentiment/analyze`, `/api/v1/sentiment/analyze/batch`  
- `/api/v1/llm/summarize`, `/api/v1/llm/summarize/batch`  
- `/api/v1/llm/extract/strengths`  
- `/api/v1/vector/upload`, `/api/v1/vector/search/similar`  

이 중 LLM 자연어 생성은 **요약(summary_pipeline)**과 **강점 설명(strength_pipeline.generate_strength_descriptions)**에서만 사용되며, 둘 다 `_generate_response` / `_generate_response_async`만 호출합니다.

---

## 남긴 것 (유지)

### 클래스 `LLMUtils`

| 항목 | 설명 |
|------|------|
| `__init__` | 모델·OpenAI/RunPod/로컬/vLLM 초기화 |
| `_init_vllm` | vLLM 사용 시 초기화 (use_pod_vllm) |
| `_call_runpod` | RunPod 서버리스 동기 호출 |
| `_call_runpod_async` | RunPod 서버리스 비동기 호출 |
| `_generate_response` | 동기 생성 (OpenAI / 로컬 큐). summary_pipeline, strength_pipeline에서 사용 |
| `_generate_response_async` | 비동기 생성. summary_pipeline 비동기 배치에서 사용 |
| `_generate_with_local_queue_async` | 로컬 큐 비동기 (RunPod 등) |
| `_generate_with_local_queue` | 로컬 큐 동기 (RunPod / 로컬 Transformers model.generate) |

### import 유지

- `json`, `logging`, `os`, `re`, `time`, `asyncio`, `requests`, `torch`, `httpx`
- `typing`, `concurrent.futures`
- `transformers` (AutoModelForCausalLM, AutoTokenizer)
- `.config.Config`

---

## 지운 것 (제거)

### 메서드 (클래스 내부)

| 메서드 | 설명 |
|--------|------|
| `_fix_truncated_json` | 잘린 JSON 복구. 호출처 없음 |
| `classify_reviews` | LLM으로 텍스트 배치 분류. 라우터/파이프라인 미사용 |
| `_generate_with_vllm` | vLLM 비동기 생성 + 메트릭. test_all_task 경로에서 미호출 |
| `_estimate_prefill_cost` | 프리필 비용(토큰 수) 추정. 미사용 |
| `_calculate_dynamic_batch_size` | 동적 배치 크기. 미사용 |
| `_extract_aspects_from_reviews` | 리뷰에서 aspect 추출. 구 요약 파이프라인용 |
| `_create_final_summary_from_aspects` | aspect로 최종 요약 생성. 구 파이프라인용 |
| `format_overall_summary_from_aspects` | aspect 기반 전체 요약 포맷. 구 파이프라인용 |
| `format_overall_summary_hybrid` | 템플릿 + LLM 개선. `summarize_reviews` 전용 |
| `_calculate_dynamic_similarity_threshold` | 동적 유사도 임계값. 미사용 |
| `validate_aspects_by_cosine_similarity` | cosine 유사도로 aspect 검증. `summarize_reviews` 전용 |
| `summarize_reviews` (인스턴스 메서드) | 긍정/부정 리뷰 요약(구 파이프라인). 현재 API는 `summary_pipeline.summarize_aspects_new`만 사용 |
| `extract_absolute_strengths` | 절대 강점 추출. 현재는 Kiwi+lift 강점만 사용 |
| `_parse_strengths_fallback` | 강점 파싱 폴백. `extract_absolute_strengths` 등에서만 사용 |
| `_create_summary_prompt` | 요약 프롬프트 생성. `summarize_reviews` 전용 |
| `_parse_summary_response` | 요약 응답 파싱. `summarize_reviews` 전용 |
| `summarize_multiple_restaurants_vllm` | vLLM 다중 레스토랑 요약. 미사용 |

### 모듈 최상위 함수

| 함수 | 설명 |
|------|------|
| `summarize_reviews(llm_utils, positive_reviews, negative_reviews)` | `llm_utils.summarize_reviews` 래퍼. 어떤 라우터/파이프라인에서도 호출 안 함 |

### import 제거

- `numpy` (제거된 `validate_aspects_by_cosine_similarity` 등에서만 사용)
- `.review_utils`의 `estimate_reviews_tokens`, `estimate_tokens` (제거된 메서드에서만 사용)

---

## 패키지 노출 변경 (src/__init__.py)

- **제거**: `summarize_reviews` export 및 `__all__`에서 제거  
- **유지**: `LLMUtils` export

---

## 요약

- **유지**: test_all_task → API → summary_pipeline / strength_pipeline → `_generate_response`, `_generate_response_async` 및 그 내부 호출(`_generate_with_local_queue`, `_call_runpod` 등).
- **제거**: 위 경로에 쓰이지 않는 구 요약 파이프라인(`summarize_reviews`, aspect 검증/포맷), vLLM 전용 생성, 분류/강점 추출 레거시, 미사용 유틸·import·모듈 레벨 함수.

이 문서는 제거 시점(레거시 제거 작업) 기준으로 작성되었습니다.
