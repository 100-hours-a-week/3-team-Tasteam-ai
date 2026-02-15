[3 tools called]


0533 로그 기준으로만 정리했습니다.

---

# RunPod Pod 로그 분석 (0533)

## 1. 파일

| 파일 | 내용 |
|------|------|
| **0533_app_server_log.log** | 로컬 앱 (2026-02-14 17:33~17:35) |
| **0533_pod_container_log.log** | RunPod vLLM 컨테이너 (400/4731 에러 포함) |

---

## 2. 앱 서버 로그 요약

### 정상

- 기동, 벡터/임베딩/Sentiment warm-up, RunPod Pod 연결
- 벡터 업로드 142개, 레스토랑 5개
- Sentiment: 단일 요청 200 OK, 배치 중 일부 200 OK
- 요약: **단일** `/api/v1/llm/summarize` 200 OK
- Comparison: 200 OK
- `/api/v1/llm/comparison`, `/api/v1/llm/comparison/batch` 200 OK

### 문제 1: Sentiment 400 (max_tokens 1024도 초과)

- **에러**: `max_tokens too large: 1024. ... your request has 3076 input tokens (1024 > 4096 - 3076)` / `3205 input tokens (1024 > 4096 - 3205)`.
- **의미**: 입력 3076이면 출력은 최대 1020, 입력 3205면 최대 891인데, 1024를 보내서 400.
- **원인**: 절대 상한 1024를 적용했어도, **실제 입력이 크면** 허용 출력(4096−input)이 1024보다 작아짐. 우리 추정이 실제보다 작아서 cap이 1024로만 잘리고, 서버 기준으로는 여전히 “너무 큰 max_tokens”가 감.

### 문제 2: Summary 배치 – 입력 4731 + KeyError

- **에러 1**: `your request has 4731 input tokens` → 입력만으로 4096 초과.
- **에러 2**:  
  `KeyError: 'service'`  
  `summary_pipeline.py` 346행: `out[cat]["evidence"] = [...]` 에서 `out["service"]`가 없음.
- **의미**:  
  - 어떤 레스토랑은 요약 payload가 4096을 넘어서 4731 토큰으로 들어갔고,  
  - 그 요청이 실패하거나, LLM이 기대와 다른 구조(예: `service` 키 없음)를 줘서 `out`에 `"service"`가 없는데 `out[cat]["evidence"]`를 쓰다가 KeyError → **배치 요약 500**.

---

## 3. Pod 컨테이너 로그

- vLLM이 400 반환:  
  - `max_tokens ... too large: 1024` (input 3076 / 3205)  
  - `your request has 4731 input tokens`
- 200 OK도 섞여 있음 (일부 요청만 실패).

---

## 4. 결론

| 구분 | 원인 | 필요한 조치 |
|------|------|-------------|
| **Sentiment 400** | 입력 3076/3205일 때 허용 출력(1020/891)보다 1024를 보냄. 추정이 낮아서 cap이 1024에서 멈춤. | **max_tokens를 (4096 − 추정_input − 256) 결과와 비교해 더 작은 값**으로 보내기. 절대 상한 1024는 그 **위에** 두되, **context 기반 cap이 더 작으면 그걸 써야** 함. 추정은 더 보수적으로(예: chars/2). |
| **Summary 4731** | payload가 4096 넘음. | `_payload_within_context_limit`가 해당 레스토랑에서도 적용되도록 하거나, 추정/한도 더 보수적으로. |
| **KeyError 'service'** | LLM 응답에 `service` 키가 없는데 `out[cat]["evidence"]` 접근. | `out.get(cat, {})` 등으로 **키 없어도 안 죽도록** 처리하고, `evidence`는 키 있을 때만 설정. |

정리하면, 0533에서 보인 문제는  
1) **max_tokens**: 1024 상한만으로는 부족하고, (4096−실제 입력)보다 작은 값으로 보내야 하며,  
2) **요약**: 입력 4731 넘지 않게 자르기 강화,  
3) **배치 500**: `out["service"]` 등이 없을 때를 가정한 방어 코드  
이 세 가지를 반영하면 됩니다.

---

## 5. 반영한 조치

| 이슈 | 반영 내용 |
|------|-----------|
| **Sentiment 400** | `src/llm_utils.py`: `_estimate_input_tokens`를 `(total+2)//3` → `total//2`로 보수화. `cap < 1`일 때 절대 1024 대신 **최대 512**로 제한해 서버 400 방지. |
| **Summary 4731** | `src/summary_pipeline.py`: `_estimate_tokens`를 `(len+3)//4` → `(len+2)//3`으로 보수화해 payload가 4096을 넘지 않도록 더 일찍 잘라냄. |
| **KeyError 'service'** | `src/summary_pipeline.py`: sync/async 모두 evidence 매핑 전에 `out.setdefault(cat, {"summary": "", "bullets": [], "evidence": []})` 추가해 LLM이 service/price/food 키를 주지 않아도 안전하게 처리. |

---

프롬프트에서 요구하는 구조 확인 중입니다.


[2 tools called]


두 가지로 나눠서 보는 게 좋습니다.

---

### 1. LLM이 항상 `service` 키를 반환하게 할 수 있나?

**완전히 보장하는 것은 어렵습니다.**  
지금도 프롬프트에 스키마를 적어 두었지만, LLM은 가끔 키를 빼거나 구조를 바꿔서 응답할 수 있습니다.

다만 **될 가능성을 높이는** 방법은 있습니다.

- 프롬프트에 **“반드시”** 강조  
  - 예: `"반드시 service, price, food, overall_summary 네 키를 모두 포함한 JSON만 출력하라."`
- **예시 JSON**을 스키마 아래에 하나 넣기  
  - 예: `{"service": {"summary": "", "bullets": [], "evidence": []}, "price": {...}, ...}`  
  - 이렇게 하면 모델이 형식을 따라갈 확률이 올라갑니다.
- 가능하다면 **structured output / JSON mode**를 쓰기  
  - 사용 중인 RunPod/vLLM/API가 스키마를 강제하는 옵션을 지원하면, 그걸 쓰는 편이 가장 안정적입니다.

그래서 “LLM이 100% 항상 반환”은 못 하고, “프롬프트/설정으로 최대한 맞추기 + 아래처럼 코드에서 보정”이 현실적인 선입니다.

---

### 2. 우리 쪽에서는 항상 `service` 키가 있게 할 수 있나? → **가능합니다**

LLM 응답과 상관없이, **파싱 직후 한 번만** 기본 구조를 채우면 됩니다.

지금은 `for cat in ("service", "price", "food"):` 안에서 `out.setdefault(cat, {...})`로 각 키를 채우고 있어서, 이미 “키가 없어도 안 죽도록” 처리한 상태입니다.  
이걸 **파싱 직후 한 번에** 하면 “항상 service(와 price, food)가 있다”는 걸 코드에서 보장할 수 있습니다.

예시는 다음과 같습니다.

- `out = parse_json_relaxed(text)` 직후, `out`이 dict이면:
  - 기본 구조:
    - `default = {"service": {"summary": "", "bullets": [], "evidence": []}, "price": {...}, "food": {...}, "overall_summary": {"summary": ""}}`
  - 정규화:
    - `for k in default: out.setdefault(k, default[k])`
    - 각 `out["service"]`, `out["price"]`, `out["food"]`가 dict이 아니면 `out[k] = default[k]`로 덮어쓰기
  - 그 다음부터는 항상 `out["service"]`, `out["price"]`, `out["food"]`가 존재한다고 가정하고 evidence 등만 채우면 됨.

이렇게 하면:

- **LLM이** `service`를 안 줘도,
- **우리 코드 기준**으로는 항상 `service`(와 price, food) 키가 있는 것처럼 다룰 수 있습니다.

정리하면:

- **“LLM이 항상 service 키를 반환하게”** → 완전 보장은 어렵고, 프롬프트/스키마/structured output으로 줄일 수만 있음.
- **“우리 코드에서는 항상 service 키가 있게”** → 파싱 직후 기본 구조로 `out`을 채우면 가능하고, 이미 쓰는 `setdefault`를 그 시점으로 모으면 “항상 service 키를 반환한 것처럼” 다루는 것과 동일한 효과를 낼 수 있습니다.

원하시면 Agent 모드에서 파싱 직후 정규화하는 코드 위치와 수정 예시까지 구체적으로 적어 드리겠습니다.

---

요청하신 대로 적용합니다. (1) 카테고리별 디폴트 문구 정의 및 setdefault 적용 (2) 서치 결과가 없을 때 최근 N개 채우기 제거하고 해당 카테고리는 디폴트 문구 사용


[3 tools called]


수정 적용 중입니다.


[13 tools called]


요청하신 내용 반영했습니다.

---

### 1. **카테고리별 디폴트 문구** (`src/summary_pipeline.py`)

- **상수 추가**: `CATEGORY_EMPTY_DEFAULT`
  - `service`: `"서비스 관련 언급이 적어 요약할 내용이 없어요."`
  - `price`: `"가격 관련 언급이 적어 요약할 내용이 없어요."`
  - `food`: `"음식 관련 언급이 적어 요약할 내용이 없어요."`

- **키가 없을 때**: `out.setdefault(cat, dict(CATEGORY_EMPTY_DEFAULT[cat]))` 로 위 디폴트 사용 (sync/async 동일).

- **서치 결과가 없을 때**: `len(payload[cat]) == 0` 이면 해당 카테고리는 `out[cat] = dict(CATEGORY_EMPTY_DEFAULT[cat])` 로 덮어써서 동일 문구가 나가도록 함.

- **예외/파싱 실패 시**: 반환값도 `CATEGORY_EMPTY_DEFAULT` 기반으로 맞춤 (실패 메시지는 `overall_summary`만 `"요약 생성에 실패했어요."` 유지).

---

### 2. **최근 리뷰 N개 fallback 제거** (`src/api/routers/llm.py`)

- `_retrieve_category_hits_accuracy_first` 에서 **4차 단계**(`get_recent_restaurant_reviews`로 최근 리뷰 N개 채우기) 제거.
- 3차(넓은 쿼리 재검색)까지 수행 후, 결과가 부족하면 그대로 반환.  
  → 해당 카테고리는 빈 리스트로 summary 파이프라인에 넘어가고, 위 디폴트 문구가 사용됨.
- 함수 docstring을 이 동작에 맞게 수정함.

---

이제 서치 결과가 없거나 적은 카테고리는 최근 리뷰로 채우지 않고, 항상  
`"서비스/가격/음식 관련 언급이 적어 요약할 내용이 없어요."` 가 나가도록 되어 있습니다.