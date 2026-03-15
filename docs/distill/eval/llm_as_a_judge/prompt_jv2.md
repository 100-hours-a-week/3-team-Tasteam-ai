응, **바꾸는 게 좋을 가능성이 높아.**
지금 네 task에서는 judge 기준이 너무 일반 요약 평가면, 네가 진짜 개선하려는 포인트를 잘 못 잡을 수 있어.

핵심은 **judge 기준을 “자유 요약 품질”이 아니라 “구조화 요약 품질” 중심으로 재정의**하는 거야.

지금 네가 중요하게 보는 건 이런 거잖아.

* category 분리 정확성
* 입력에 없는 내용 금지
* evidence 정합성
* price 언급 없을 때 빈 값 허용
* schema 준수
* 메뉴/맛/식감 중심의 food 요약

그런데 judge가 그냥
“자연스러운가, 전반적으로 잘 요약했는가”
위주로 보면,

* 근거 없는 추론을 해도 말만 자연스러우면 점수 잘 나올 수 있고
* 반대로 보수적으로 빈값 처리한 output은 점수가 낮게 나올 수도 있어

즉, **현재 judge 기준이 네 목표와 안 맞을 가능성**이 큼.

## 어떻게 바꾸면 좋나

judge rubric을 이런 식으로 쪼개는 게 좋아.

### 1. Faithfulness

* 입력에 없는 내용이 들어갔는가
* category별 내용이 실제 리뷰 근거에 기반하는가

### 2. Category correctness

* service / price / food가 올바르게 분리되었는가
* food 내용을 service에 넣거나 그 반대가 없는가

### 3. Schema adherence

* required key 존재
* 허용된 구조 유지
* price 무언급 시 빈 bullets 허용이 잘 지켜졌는가

### 4. Evidence validity

* evidence index가 실제 문장 인덱스인가
* bullet/support 관계가 맞는가

### 5. Completeness

* 입력에 존재하는 중요한 포인트를 과도하게 놓치지 않았는가

### 6. Naturalness

* 문장이 너무 깨지거나 반복되지 않는가

이렇게 두는 게 좋아.

---

## 점수 방식도 바꾸는 게 좋음

지금처럼 단일 총점만 보면 원인 분석이 잘 안 돼.
가능하면

* faithfulness: 1~5
* category correctness: 1~5
* schema adherence: 1~5
* evidence validity: 1~5
* completeness: 1~5
* naturalness: 1~5

이런 식으로 **축별 점수**를 받는 게 훨씬 좋아.

그러면 예를 들어

* naturalness는 4인데
* faithfulness는 2
* evidence validity는 2

이런 식으로 나와서,
“겉보기는 그럴듯하지만 서비스용으론 위험하다”를 바로 볼 수 있어.

---

## 특히 네 경우엔 이걸 꼭 넣는 게 좋다

judge prompt에 명시적으로:

* 입력에 없는 내용 추론 시 큰 감점
* evidence index 오류 시 큰 감점
* price 직접 언급이 없으면 빈 bullets 허용, 감점 금지
* service는 친절/응대/대기/분위기·좌석 편의 범위만
* food는 메뉴/맛/식감 중심만
* category contamination 감점

이걸 넣는 게 좋다.

이걸 안 넣으면 judge가 자기 일반 상식으로
“price도 좀 써주는 게 더 좋은 요약 아닌가?”
이렇게 판단할 수 있어.

---

## 다만 주의할 점

judge 기준을 바꾸면 **이전 점수와 직접 비교가 어려워져.**

그래서 가장 좋은 방식은:

* 기존 judge score는 “구 기준”
* 새 judge score는 “신 기준”

이라고 분리해서 기록하는 거야.

포트폴리오에서도 오히려 좋아.
왜냐하면

“초기에는 일반 요약 judge를 썼지만, 구조화 요약 task와 평가 기준 불일치를 발견했고, 이후 task-specific rubric으로 judge를 재설계했다”

라고 말할 수 있거든.
이건 되게 좋은 실험 설계 감각으로 보인다.

---

## 결론

* **응, judge 기준 바꾸는 게 좋다.**
* 특히 네 과제는 **자유 요약**이 아니라 **근거 기반 구조화 요약**이라서,
  일반 요약 judge로는 한계가 크다.
* 단일 총점보다 **faithfulness / schema / evidence / category correctness / completeness / naturalness**로 쪼개는 게 가장 좋다.

원하면 내가 바로 **네 task에 맞는 LLM-as-a-Judge rubric 프롬프트 초안**을 짜줄게.
