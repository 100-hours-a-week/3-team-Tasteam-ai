현재 프롬프트, 메시지 구성 (260311_064215)

_SCHEMA_ENFORCEMENT_SYSTEM = """You are a JSON generator for review summarization.

Return ONLY one valid JSON object.
Do not output any text before or after the JSON.
Do not output markdown.
Do not output explanations.

The top-level keys must be exactly:
service, food, price

Each top-level key must contain exactly:
summary, bullets, evidence

Rules:
- summary: exactly 1 Korean sentence
- bullets: list of 0 to 3 short Korean strings
- evidence: list of integer indices (0-based indices into each category list)
- evidence length must match bullets length
- Do not add extra keys
- If there is no evidence, use:
  "summary": "",
  "bullets": [],
  "evidence": []
"""

_TINY_FEWSHOT_USER = """Example input:
{"service":["직원분이 친절해요"],"price":[],"food":["국물이 진해요"]}
"""

_TINY_FEWSHOT_ASSISTANT = """{"service":{"summary":"직원분이 친절해요.","bullets":["직원 응대가 친절해요"],"evidence":[0]},"price":{"summary":"","bullets":[],"evidence":[]},"food":{"summary":"국물이 진해요.","bullets":["국물이 진하고 맛있어요"],"evidence":[0]}}"""


def _generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
) -> str:
    """이미 로드된 model/tokenizer로 instruction 한 건만 추론."""
    # instruction은 payload JSON 문자열. system prompt + tiny few-shot으로 스키마 계약을 강화한다.
    messages = [
        {"role": "system", "content": _SCHEMA_ENFORCEMENT_SYSTEM},
        {"role": "user", "content": _TINY_FEWSHOT_USER},
        {"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = generated.strip()
    return _extract_json_for_rouge(raw)

---

좋습니다. 이 정도면 **지금보다 훨씬 낫습니다.**
특히 0.5B 모델에는 이런 식의 **짧고 강한 계약형 프롬프트**가 잘 맞습니다.

다만 그대로 써도 효과는 있겠지만, **몇 군데만 더 다듬으면 schema 안정성이 더 올라갈 가능성**이 큽니다.

## 좋은 점

지금 템플릿의 장점은 분명합니다.

* **JSON만 출력**하라고 강하게 못 박음
* top-level key를 명시함
* 하위 key를 명시함
* 빈 값 처리 규칙이 있음
* tiny few-shot이 짧아서 소형 모델에 부담이 적음

이건 지금까지 빠져 있던 핵심을 잘 채웠습니다.

---

## 바로 보이는 개선 포인트

### 1. `exactly`와 `0 to 3`가 약간 충돌합니다

지금 문구는

* `Each top-level key must contain exactly: summary, bullets, evidence`
* `bullets: list of 0 to 3 short Korean strings`

이라서 키는 exactly 맞는데, bullets 수는 0~3이라 괜찮습니다.
문제는 **모델이 bullets/evidence 길이 규칙을 잘 안 지킬 수 있다는 점**입니다.

이 부분은 더 직접적으로 쓰는 게 좋습니다.

예:

```text
- bullets must have at most 3 items
- evidence must have the same number of items as bullets
- If bullets is empty, evidence must be []
```

---

### 2. `summary: exactly 1 Korean sentence`는 좋지만, 빈 값 규칙과 같이 더 명시하는 게 좋습니다

지금은 빈 경우 예시만 있고, summary가 항상 한 문장이어야 한다고 되어 있습니다.
소형 모델은 여기서 흔들릴 수 있습니다.

예:

```text
- summary must be exactly 1 Korean sentence, or "" if there is not enough evidence
```

---

### 3. `Do not add extra keys`를 더 세게 반복하는 편이 좋습니다

0.5B는 `examples`, `weight`, `overall_summary` 같은 걸 자주 만들 수 있어서, 한 번 더 못 박는 게 좋습니다.

예:

```text
Never use keys such as examples, impact, weight, title, body, rating, overall_summary.
Use only: summary, bullets, evidence.
```

이건 꽤 효과 있습니다.

---

### 4. `overall_summary` 금지를 명시하는 게 좋습니다

기존 출력에서 `overall_summary`가 자주 보였으니, 명시적으로 막는 게 좋습니다.

---

### 5. few-shot 예시가 아주 좋지만, 하나 더 있으면 더 안정적입니다

특히 **빈 리스트가 둘 이상 섞인 예시** 하나 더 있으면 좋습니다.

지금 예시는:

* service 1개
* price 비어 있음
* food 1개

아주 좋은데, 모델이 여전히 `overall_summary`를 붙이거나 `examples`로 바꿀 수 있습니다.
그래서 두 번째 예시는 **모든 키를 정확히 유지하면서 2~3 bullet도 보여주는 예시**가 좋습니다.

---

## 추천 수정 버전

아래처럼 조금만 강화하면 더 좋습니다.

```python
_SCHEMA_ENFORCEMENT_SYSTEM = """You are a JSON generator for review summarization.

Return ONLY one valid JSON object.
Do not output any text before or after the JSON.
Do not output markdown.
Do not output explanations.

The top-level keys must be exactly:
service, food, price

Each of service, food, and price must be a JSON object with exactly these keys:
summary, bullets, evidence

Never add any other keys.
Never use keys such as:
examples, impact, weight, title, body, rating, overall_summary

Rules:
- summary must be exactly 1 Korean sentence, or "" if there is not enough evidence
- bullets must be a list of 0 to 3 short Korean strings
- evidence must be a list of 0-based integer indices into the corresponding category list
- evidence must have the same number of items as bullets
- If bullets is [], evidence must be []
- If there is not enough evidence, use:
  "summary": "",
  "bullets": [],
  "evidence": []

Output only JSON.
"""
```

---

## few-shot은 이렇게 2개 추천

### 예시 1

지금 것 거의 그대로 좋습니다.

```python
_TINY_FEWSHOT_USER = """Example input:
{"service":["직원분이 친절해요"],"price":[],"food":["국물이 진해요"]}
"""

_TINY_FEWSHOT_ASSISTANT = """{"service":{"summary":"직원분이 친절해요.","bullets":["직원 응대가 친절해요."],"evidence":[0]},"food":{"summary":"국물이 진해요.","bullets":["국물이 진하고 맛있어요."],"evidence":[0]},"price":{"summary":"","bullets":[],"evidence":[]}}"""
```

### 예시 2

조금 더 일반적인 구조 예시를 하나 추가하면 좋습니다.

```python
_TINY_FEWSHOT_USER_2 = """Example input:
{"service":["직원들이 빠르게 응대해요","매장이 깔끔해요"],"price":["양이 많아요"],"food":[]}
"""

_TINY_FEWSHOT_ASSISTANT_2 = """{"service":{"summary":"직원 응대가 빠르고 매장이 깔끔해요.","bullets":["직원들이 빠르게 응대해요.","매장이 깔끔해요."],"evidence":[0,1]},"food":{"summary":"","bullets":[],"evidence":[]},"price":{"summary":"양이 많아요.","bullets":["양이 많아요."],"evidence":[0]}}"""
```

---

## 순서도 중요합니다

메시지 구성은 보통 이렇게 추천합니다.

1. system: `_SCHEMA_ENFORCEMENT_SYSTEM`
2. user: `_TINY_FEWSHOT_USER`
3. assistant: `_TINY_FEWSHOT_ASSISTANT`
4. user: `_TINY_FEWSHOT_USER_2`
5. assistant: `_TINY_FEWSHOT_ASSISTANT_2`
6. user: 실제 입력 payload

이렇게 하면 모델이 “아 이런 출력 계약이구나”를 훨씬 잘 잡습니다.

---

## 기대 효과

이 정도만 넣어도 보통 기대할 수 있는 건:

* 영어 자유서술 감소
* `title/body/rating` 감소
* `examples/impact/weight` 감소
* `overall_summary` 감소
* parse success 유지 또는 소폭 상승
* schema accuracy 상승

즉 지금 `schema_accuracy = 0.30`이면,
후처리 repair까지 같이 붙일 경우 **0.55 이상**은 충분히 기대해볼 만합니다.

---

## 마지막으로 아주 중요한 점

예시 JSON의 **키 순서도 통일**하세요.

지금도 이미 괜찮지만, 실제 정답/예시/후처리 모두에서 항상

```text
service
food
price
```

또는

```text
price
service
food
```

중 하나로 **완전히 통일**하는 게 좋습니다.
소형 모델은 이런 사소한 일관성 차이에도 영향받습니다.

지금 시스템 프롬프트엔 `service, food, price`라고 썼으니 예시도 그 순서로 통일하는 걸 추천합니다.
