네, **재학습 없이도 그 정도만 해도 꽤 의미 있게 좋아질 가능성이 높습니다.**

다만 우선순위를 나누면 더 정확합니다.

## 결론

**지금 당장 효과 큰 순서**

1. **추론 프롬프트 강화**
2. **few-shot 예시 1~2개 추가**
3. **후처리 validation / repair**
4. **length control**
5. 재학습은 나중

즉, 네가 적은 방향으로 가도 됩니다.
특히 지금처럼 **parse는 88% 되는데 schema가 30%인 상태**에서는, 학습보다 **추론 제어 + 후처리**가 훨씬 가성비 좋습니다.

---

## 왜 이걸로 충분할 가능성이 크냐

현재 문제는 성능 자체가 완전 낮은 게 아니라

* JSON 비슷하게는 출력함
* 내용도 대충 맞음
* 근데 **정확한 key contract를 자주 어김**

이 상태라서, 모델의 “지식”을 더 넣는 것보다
**출력 행동을 강하게 묶는 것**이 더 중요합니다.

즉 지금은:

```text
모델이 뭘 써야 하는지 모르는 문제
```

보다

```text
모델이 아는 걸 정해진 형식으로 안 내는 문제
```

에 가깝습니다.

---

## 각 항목별로 보면

### 1. 추론 프롬프트 강화

이건 거의 필수입니다.

특히 0.5B는 프롬프트를 두루뭉술하게 쓰면 바로 자기 습관대로 갑니다.

좋습니다. 꼭 넣는 게 좋습니다.

예시는 이런 식으로 더 단단하게 써도 됩니다.

```text
Return ONLY a valid JSON object.
Do not write any text before or after the JSON.
The top-level keys must be exactly: service, food, price.
Each of service, food, and price must contain exactly these keys:
summary, bullets, evidence
Do not add any other keys.
summary must be a single sentence in Korean.
bullets must be a list of 0 to 3 short Korean strings.
evidence must be a list of integer indices.
The number of evidence items must match the number of bullets.
If there is not enough evidence, use an empty list.
Do not output markdown.
Do not output explanations.
Do not output English unless the input is English.
```

---

### 2. few-shot 예시 1~2개

이것도 효과 큽니다.

특히 중요한 건:

* **완벽한 정답 포맷**
* **빈 리스트 처리 예시**
* **bullets/evidence 대응 예시**

이 3개입니다.

소형 모델은 “설명”보다 “예시”를 더 잘 따라가는 경우가 많습니다.

---

### 3. length control

이것도 넣는 게 좋습니다.

네 경우에는 특히 **장황해지다가 schema에서 이탈**하는 경향이 보였으니까 효과 있습니다.

예:

* `summary`: 1문장
* `bullets`: 최대 3개
* `evidence`: bullets와 같은 길이
* 각 bullet: 25자 이하 또는 짧은 문장

이렇게 짧게 제한하면 drift가 줄어듭니다.

---

### 4. 후처리 validation / repair

실서비스라면 거의 반드시 추천입니다.

이건 “모델이 못해서”가 아니라,
원래 structured generation은 **후처리까지 포함해서 시스템으로 완성**하는 경우가 많습니다.

후처리에서 해줄 것:

* JSON parse
* 최상위 키 없으면 채우기
* 허용되지 않은 키 제거
* `examples -> bullets` 같은 alias 보정
* `summary/bullets/evidence` 없으면 기본값 채우기
* evidence가 숫자 리스트 아니면 정리
* bullets 수와 evidence 수 맞추기

이렇게만 해도 체감 안정성이 꽤 올라갑니다.

---

## 현실적으로 어디까지 기대할 수 있냐

재학습 없이도 보통 기대 가능한 건:

* **parse success**: 조금 상승
* **schema accuracy**: 꽤 상승
* **LLM judge score**: 소폭~중간 상승
* **서비스 안정성**: 크게 상승

특히 네 상태는 already “완전 망한 모델”이 아니라서
**schema accuracy 0.30 → 0.55~0.70 근처**까지는 충분히 노려볼 만합니다.
물론 데이터와 프롬프트 품질에 따라 다르지만, 지금 상태에서는 가능성 있습니다.

---

## 다만 하나는 알아야 함

이걸로 **완벽한 품질**까지 가는 건 아닐 수 있습니다.

왜냐하면:

* 내용 hallucination
* evidence index 부정확성
* 미묘한 요약 품질 부족

이런 건 결국 학습 품질 영향도 받기 때문입니다.

그래서 지금 방식은 정확히 말하면:

**“모델 성능 자체를 크게 끌어올린다”기보다
“현재 성능을 더 안정적이고 usable하게 만든다”**

에 가깝습니다.

근데 지금 단계에서는 이게 맞습니다.

---

## 추천하는 최소 구성

재학습 안 할 거면 최소한 이 3개는 꼭 하세요.

1. **강한 추론 프롬프트**
2. **few-shot 1개 + 빈 리스트 예시 1개**
3. **후처리 repair**

length control은 그 다음입니다.

---

## 한 줄로 정리

**응, 그렇게만 해도 된다.**
그리고 지금 네 상황에서는 그게 **가장 실용적인 선택**입니다.

원하면 다음 답변에서 내가 바로
**붙여 넣어 쓸 수 있는 추론 프롬프트 템플릿**이랑
**파이썬 후처리 repair 함수 예시**를 같이 써줄게.
