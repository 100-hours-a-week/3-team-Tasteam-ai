
두 프롬프트를 비교하면 다음과 같은 차이가 있습니다.

---

## 1. overall_summary

| | label_for_distill (teacher) | eval_distill |
|---|---|---|
| 스키마 | `service`, `price`, `food`, **overall_summary** | `service`, `food`, `price`만 허용 |
| 규칙 | overall_summary 2~3문장 | "Never use keys such as: overall_summary" |

**충돌**: 학습 데이터에는 `overall_summary`가 있지만, eval에서는 금지되어 있음.

---

## 2. bullets 개수

| | label_for_distill | eval_distill |
|---|---|---|
| 규칙 | bullets: **3~5개** | bullets: **0~3개** |

**충돌**: ref는 3~5개를 배우지만, eval은 최대 3개까지만 허용.

---

## 3. 근거 없을 때 처리 (이미 언급한 부분)

| | label_for_distill | eval_distill |
|---|---|---|
| 규칙 | "언급이 적어요" 같은 폴백 표현 | `summary=""`, `bullets=[]`, `evidence=[]` |

**충돌**: teacher는 폴백 문구, eval은 빈 값.

---

## 4. price 처리

| | label_for_distill | eval_distill |
|---|---|---|
| 규칙 | "가격 숫자 없으면 가성비/양/구성/만족감 같은 **우회표현으로 요약**하라" | "price: **allow empty bullets** when there is no direct mention of price" |

**충돌**: teacher는 가격 언급 없어도 우회 표현으로 채우라고 하고, eval은 빈 bullets를 허용.

---

## 5. 기타

- Teacher: "각 카테고리 summary: 1문장, 과장 금지" — eval과 일치
- evidence 규칙: 둘 다 0-based 인덱스, bullets와 개수 맞추기 — 대체로 일치
- eval의 "service: 친절/응대/대기/분위기·좌석", "food: 메뉴/맛/식감" — teacher에는 없고 eval에서만 명시

---

## 요약

상충되는 부분:

1. **overall_summary** — teacher: 있음, eval: 금지  
2. **bullets 개수** — teacher: 3~5개, eval: 0~3개  
3. **근거 없을 때** — teacher: 폴백 문구, eval: 빈 값  
4. **price** — teacher: 우회 표현 유도, eval: 빈 bullets 허용  

수정 시 라벨링 스키마/규칙과 eval 스키마/규칙을 어느 쪽에 맞출지 먼저 정해야 합니다.

---

llm as a judge의 평가 기준이 

1️⃣ teacher schema를 따르는가
2️⃣ teacher fallback 정책을 따르는가
3️⃣ teacher 스타일 요약인가
4️⃣ evidence가 input에 기반하는가

이거인지 봐줘

---

지금 Judge 기준과 네 가지 평가 축을 이렇게 매핑할 수 있습니다.

---

## 1) teacher schema를 따르는가

**v2 `schema_adherence`**에서 일부만 다룹니다.

- v2: "required key 존재, 허용된 구조 유지, **price 직접 언급 없으면 빈 bullets 허용**"
- teacher 스키마: `service`, `price`, `food`, **`overall_summary`**, bullets 3~5개
- v2 프롬프트에는 `overall_summary`가 없고, "price 직접 언급 없으면 빈 bullets 허용"이라 **teacher의 폴백 정책과는 다름**

→ teacher 스키마 전체를 명시적으로 요구하지는 않음 (특히 `overall_summary`, bullets 개수 등).

---

## 2) teacher fallback 정책을 따르는가

**명시적으로는 없음.**

- teacher: 근거 없을 때 `"언급이 적어요"` 같은 폴백 문구 사용
- v2: "price 직접 언급 없으면 **빈 bullets 허용**" (빈 값 사용)
- v2에는 “폴백 문구 사용”이라는 평가 축이 없음

→ teacher fallback 정책을 직접 평가하지 않음.

---

## 3) teacher 스타일 요약인가

**간접적으로만 다룹니다.**

- v2 `naturalness`: "문장이 깨지거나 반복되지 않는가"
- teacher 스타일: "~해요" 체, bullets 3~5개, summary 1문장 등
- Judge 프롬프트에 "해요"체, bullets 개수 같은 teacher 스타일 지침은 없음

→ teacher 스타일을 별도로 명시적으로 평가하지는 않음.

---

## 4) evidence가 input에 기반하는가

**잘 반영되어 있습니다.**

- v2 `evidence_validity`: "evidence index가 실제 문장 인덱스인가, bullet-support 관계가 맞는가"
- v2 `faithfulness`: "입력에 없는 내용 추론 시 큰 감점"
- v2 규칙: "evidence index 오류 시 큰 감점"

→ 4번 항목은 충분히 평가됨.

---

## 요약

| 네 가지 기준 | 현재 Judge에서의 반영 |
|-------------|----------------------|
| 1. teacher schema | schema_adherence로 일부만 (overall_summary 등 누락, price는 빈 값 쪽 규칙) |
| 2. teacher fallback 정책 | 없음 (빈 bullets vs "언급이 적어요" 폴백 차이) |
| 3. teacher 스타일 요약 | naturalness로 간접적으로만 |
| 4. evidence가 input 기반 | evidence_validity + faithfulness로 잘 반영됨 |

즉, 4번은 잘 되어 있고, 1~3번은 현재 Judge가 teacher 기준과 완전히 맞추어져 있지는 않습니다.