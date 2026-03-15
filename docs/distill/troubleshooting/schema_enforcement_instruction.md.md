현재 상황:


문서 권장 항목별로 코드 기준 충족 여부만 정리했습니다.

---

## schema_enforcement.md 대비 현재 충족도

| 항목 | 문서 권장 | 현재 구현 | 충족도 |
|------|-----------|-----------|--------|
| **1. 추론 프롬프트 강화** | "Return ONLY a valid JSON...", "top-level keys must be exactly: service, food, price", "Do not write any text before or after the JSON" 등 **강한 문구** | Distill 추론 시 **instruction = `json.dumps(payload)`** 만 사용. payload는 service/price/food 리뷰 목록. **스키마/출력 형식을 강제하는 문장이 없음** (`eval_distill.py` 91라인, `label_for_distill.py` 219·272·333라인) | ❌ **미충족** |
| **2. Few-shot 예시 1~2개** | 완벽한 정답 포맷 + 빈 리스트 + bullets/evidence 대응 예시 | 추론 시 **단일 메시지(instruction만)** 사용. 예시를 앞에 붙이는 코드 없음 (`eval_distill.py` 90–91라인) | ❌ **미충족** |
| **3. 후처리 validation / repair** | JSON parse, 최상위 키 채우기, 허용 키 외 제거, `examples`→`bullets` 보정, summary/bullets/evidence 기본값, evidence 숫자 리스트 정리, bullets·evidence 개수 맞추기 | **Parse**: `_extract_json_for_rouge`에서 `extract_json_block` + `parse_json_relaxed`로 JSON 블록 추출·파싱만 수행 (`eval_distill.py` 67–81라인). **Repair**: 키 채우기/제거/alias/기본값/evidence 정리/bullets–evidence 맞추기 등 **어디에도 없음**. `comparison_lib/json_parse_utils.py`는 문법만 보정(trailing comma, 따옴표). `kd_sft_analysis.py`는 스키마 **검사**만 하고 수정하지 않음 | ⚠️ **부분 충족** (parse만, repair 없음) |
| **4. Length control** | summary 1문장, bullets 최대 3개, evidence=bullets 길이, bullet 25자 이하 등 | 라벨링용 `SUMMARY_INSTRUCTIONS`에 "bullets: 3~5개", "summary: 1문장" 있음 (`label_for_distill.py` 45–66라인). **추론 경로**에는 길이 제한/잘라내기 없음 | ❌ **미충족** (inference 쪽) |

---

## 요약

- **추론 프롬프트 강화**: 구현 안 됨. 지금은 “입력 payload JSON”만 넣고, “JSON만 출력하라 / 키는 service·food·price만” 같은 문구 없음.
- **Few-shot**: 구현 안 됨. 추론 시 예시 1~2개를 붙이는 부분 없음.
- **후처리 repair**: **JSON 파싱(블록 추출 + 완화 파싱)만 있고**, 문서에서 말한 **스키마 보정(키 채우기, alias, 기본값, evidence 정리, bullets–evidence 맞추기)** 는 없음.
- **Length control**: 라벨링 가이드에는 일부 있으나, **추론 시점**에는 길이 제한·자르기 없음.

문서에서 말한 “추론 프롬프트 + few-shot + 후처리 repair” 3가지 중, **실제로 잘 갖춰진 것은 “parse만 있는 후처리” 한 부분**이고, 나머지는 아직 미구현에 가깝습니다.

---

맞습니다. 지금 설명한 상태면 사실상

**“모델이 알아서 잘 해주길 기대하는 추론”**에 가깝습니다.

그래서 현재 결과가 딱 자연스럽습니다.

* parse는 어느 정도 됨
* schema는 자주 틀림
* 형식 drift 발생
* 길이도 조금씩 늘어짐

즉, 지금은 **막을 수 있었던 실패를 그대로 허용하는 구조**입니다.

## 지금 상태 진단

네 파이프라인은 현재 추론 단계에서 아래가 빠져 있습니다.

* **출력 계약 명시**
* **정답 포맷 시연**
* **출력 길이 제어**
* **출력 보정 장치**

그래서 0.5B 모델 입장에서는 그냥:

1. 입력 JSON을 보고
2. “대충 요약해달라는 거구나” 이해하고
3. 자기식으로 그럴듯하게 생성

을 하고 있는 겁니다.

이러면 small instruct 모델은 거의 반드시 흔들립니다.

## 그래서 결론

**재학습 안 해도 되지만, 추론 파이프라인은 반드시 손봐야 합니다.**

지금 네가 적은 4개가 전부 미구현이면, 우선순위는 아주 명확합니다.

### 1순위: 추론 프롬프트 강화

이건 제일 먼저 해야 합니다.

지금 가장 큰 문제는 모델이
“무슨 내용을 써야 하는지”보다
“어떤 형식으로 반드시 써야 하는지”를 안내받지 못하고 있다는 점입니다.

0.5B에서는 이게 치명적입니다.

---

### 2순위: 후처리 repair

이건 실용성 면에서 거의 필수입니다.

모델이 100% 정확하게 schema를 맞출 거라고 기대하는 것보다,
**살릴 수 있는 출력은 살리는 게 훨씬 현실적**입니다.

특히 지금처럼 parse success가 0.88이면,
repair만 넣어도 schema accuracy가 꽤 뛸 가능성이 큽니다.

---

### 3순위: few-shot

이것도 효과 큽니다.

다만 프롬프트 강화와 repair보다 우선은 아닐 수 있습니다.
왜냐하면 예시를 넣어도, 끝에 “반드시 이렇게만 출력하라”는 제약이 약하면 또 튈 수 있기 때문입니다.

---

### 4순위: length control

이건 보조 장치입니다.

지금 문제의 본질은 길이 자체보다 **구조 일탈**이라서,
length control만으로 해결되진 않습니다.
하지만 drift 완화에는 확실히 도움 됩니다.

## 현실적으로는 이렇게 하면 됩니다

가장 효율적인 최소 수정안은 이겁니다.

### A. system/instruction prompt 추가

지금 입력 payload JSON만 넣고 있다면,
그 앞에 아주 짧고 강한 instruction block을 붙이세요.

예를 들면:

```text
You are a JSON generator for review summarization.

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
- evidence: list of integer indices
- evidence length must match bullets length
- Do not add extra keys
- If there is no evidence, use:
  "summary": "",
  "bullets": [],
  "evidence": []
```

이것만 넣어도 지금보다 훨씬 낫습니다.

---

### B. 입력 앞에 tiny few-shot 1개

아주 짧은 예시 하나만 붙여도 됩니다.

핵심은

* top-level 키
* 내부 키
* 빈 리스트 처리
* evidence가 정수 인덱스라는 것

을 보여주는 것입니다.

---

### C. 후처리 repair 함수 추가

이건 제일 실용적입니다.

후처리에서 최소한 다음은 해야 합니다.

* top-level에 `service`, `food`, `price` 없으면 생성
* 각 항목에 `summary`, `bullets`, `evidence` 없으면 기본값 채움
* `examples` → `bullets`
* `impact` 같은 허용되지 않은 키 제거
* bullet이 문자열 아니면 문자열화
* evidence가 숫자 아니면 제거
* bullets와 evidence 길이 맞추기
* bullets 3개 초과면 자르기

이 정도만 해도 많이 안정됩니다.

## 네 상황에서 가장 중요한 포인트

지금은 “모델 성능이 낮다”가 아니라,
**“모델 성능을 시스템이 받아주는 장치가 없다”**에 더 가깝습니다.

예를 들어 지금 모델이 내는 출력 중 상당수는
완전히 쓸모없는 게 아니라,
조금만 고치면 usable한 것들입니다.

그런데 현재 파이프라인은

* parse만 시도
* parse 안 되면 실패
* schema 틀리면 그대로 실패

라서 손실이 큽니다.

## 그래서 추천하는 실제 작업 순서

딱 이렇게 가세요.

1. **강한 instruction prompt 추가**
2. **후처리 repair 추가**
3. **few-shot 1개 추가**
4. **length control 추가**
5. 다시 eval

이 순서가 가장 효율적입니다.

## 기대 효과

재학습 없이도 보통 기대할 수 있는 건:

* parse success: 소폭 상승
* schema accuracy: 꽤 상승
* LLM judge: 소폭 상승
* 체감 안정성: 크게 상승

특히 네 케이스는 현재 schema accuracy가 너무 낮아서,
추론 제어와 repair만 넣어도 개선폭이 눈에 띌 가능성이 큽니다.

## 아주 솔직히 말하면

지금 상태는 아직 “모델 평가”를 한 것이라기보다
**“모델 + 미완성 추론 파이프라인”을 평가한 것**에 가깝습니다.

그래서 추론 단만 제대로 보강해도 결과 해석이 훨씬 공정해집니다.

원하면 다음 답변에서 바로 쓸 수 있게
**추론 프롬프트 템플릿**이랑 **repair용 파이썬 함수 골격**을 바로 적어드리겠습니다.
