지금 상황을 보면 **세 가지 문제를 각각 다르게 처리해야 해.**
전부 같은 방법으로 해결하려고 하면 오히려 더 꼬여.
정리해서 말해줄게.

---

# 1️⃣ category 혼입 (service ↔ food ↔ price)

> service에 food/price 내용 들어가고
> food에 service 내용 들어가고

이건 **프롬프트 문제 + 데이터 분포 문제 둘 다**야.

하지만 **0.5B 모델에서는 프롬프트 효과가 꽤 큼**.

그래서 **프롬프트 강화는 하는 게 맞다.**

예를 들어 지금 프롬프트에 이런 **boundary rule**을 명확하게 넣는 게 좋다.

```
Category definitions:

service:
- 직원 친절도
- 응대
- 대기시간
- 좌석 편의
- 매장 분위기

food:
- 음식 맛
- 메뉴
- 식감
- 양
- 조리 상태

price:
- 가격
- 가성비
- 할인
- 가격 언급

Do not place food information in service.
Do not place service information in food.
Do not infer price if the input does not mention price.
```

이건 **작은 모델에서 특히 효과가 있는 규칙**이야.

---

# 2️⃣ generic bullet

> generic bullet은 학습 데이터 문제

맞아.

이건 **프롬프트로 거의 못 고친다.**

왜냐면 모델이 학습 중에

```
직원 응대가 친절해요
매장이 깔끔해요
가격에 대한 언급이 적어요
```

같은 **safe template**을 배웠기 때문이야.

이걸 고치려면

* teacher 데이터 다양화
* label 규칙 변경
* 또는 teacher temperature 올리기

같은 **데이터 측면 개선**이 필요해.

하지만 포트폴리오 단계에서는 보통 **그냥 인정하고 넘어간다.**

---

# 3️⃣ evidence index 문제

이건 사실 **가장 중요한 포인트**야.

지금 구조를 보면 모델은 사실

```
요약 생성
+
근거 index 생성
```

두 task를 동시에 하고 있어.

문제는

> **0.5B 모델은 index grounding을 매우 어려워한다.**

특히 이런 형태:

```
review list → summary → evidence index
```

이건 **reasoning + retrieval**이 같이 필요한 구조라서
0.5B에게는 꽤 어렵다.

그래서 실제 서비스에서는 보통 **두 가지 방식** 중 하나를 쓴다.

---

## 방법 A (추천) evidence 유지 + 프롬프트 강화

evidence 규칙을 더 강하게 만든다.

예:

```
evidence must reference the exact review sentence that supports the bullet.
Do not invent evidence.
Each evidence index must correspond to a real sentence in the input list.
```

그리고 **few-shot을 evidence 중심으로 만든다.**

예:

```
input:
["직원이 친절했어요", "국물이 진해요"]

output:
service:
bullets ["직원이 친절해요"]
evidence [0]
```

이 방식은 **grounding을 조금 개선**시킨다.

---

## 방법 B (실무에서 많이 쓰는 방식)

**LLM은 요약만 하고 evidence는 retrieval로 찾는다.**

즉

```
LLM → bullet 생성
retriever → bullet과 유사한 리뷰 찾기
→ evidence index
```

이 방식이 실제 서비스에서 **훨씬 안정적**이다.

---

# 4️⃣ evidence를 포기해야 하냐?

완전히 포기할 필요는 없지만
**0.5B 모델에게 evidence를 100% 맡기면 안정성이 떨어진다.**

그래서 현실적인 옵션은 세 가지야.

### 옵션 1 (포트폴리오용 추천)

LLM이 evidence 생성

* validator

지금 구조 유지.

---

### 옵션 2 (실서비스 방식)

```
LLM → bullet
retriever → evidence
```

가장 안정적.

---

### 옵션 3

evidence 제거

```
summary
bullets
```

만 남김.

하지만 이러면

> **explainability가 사라짐**

---

# 5️⃣ 지금 네 프로젝트 기준 추천

내가 보면 **이게 제일 좋다.**

### 1️⃣ 프롬프트 강화

category boundary 추가

---

### 2️⃣ evidence 규칙 강화

```
Each bullet must reference a real review sentence.
Evidence must be an index of that sentence.
```

---

### 3️⃣ few-shot을 evidence 중심으로 수정

지금 few-shot은 너무 **schema 중심**이야.

**grounding example**을 넣어야 한다.

---

# 6️⃣ 중요한 것 하나

지금 성능은 사실 나쁘지 않아.

```
ROUGE ≈ 0.73
BERTScore ≈ 0.83
schema_accuracy ≈ 0.88
```

이 정도면

> **0.5B distillation 프로젝트 기준 꽤 괜찮은 결과**

야.

문제는 **semantic grounding**이야.

---

# 7️⃣ 결론

네 질문에 딱 답하면

* 프롬프트 강화 → **해야 한다**
* generic bullet → **데이터 문제라 크게 못 고친다**
* evidence index → **강화는 가능하지만 완벽 해결은 어려움**
* evidence 포기 → **굳이 할 필요 없음**

---

그리고 솔직히 말하면
지금 프로젝트에서 **가장 큰 문제는 모델 성능이 아니라 misalignment였어.**

그걸 발견한 것 자체가 **포트폴리오에서 꽤 큰 포인트**야.

---

원하면 내가 **지금 프로젝트 기준으로 "마지막 한 번만 하면 되는 개선"**을 딱 하나만 알려줄게.
이거 하면 **judge 점수가 가장 크게 올라갈 가능성이 높다.**
