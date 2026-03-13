근데 프롬프트에서 리뷰 내용을 반영해서 요약을 하는 few-shot 예제를 넣는 방향으로 개선을 하기는 어려울까?

---

가능해. 그리고 **꽤 현실적인 개선 방향**이야.

다만 핵심은 **“few-shot을 넣을 수 있느냐”가 아니라, 어디에 넣느냐**야.
지금 네 상황에서는 **학습 데이터 생성 단계**에 few-shot을 넣는 건 괜찮고, **추론 프롬프트에 few-shot을 길게 넣는 건 0.5B에서는 부담**이 커.

## 왜 괜찮은 방향이냐

네 모델 문제를 보면 대충 이렇지.

* JSON 틀은 어느 정도 맞춤
* 그런데 리뷰 내용을 충실히 반영하지 못함
* generic한 표현으로 뭉개짐
* evidence나 세부 bullet에서 hallucination/왜곡이 생김

이럴 때 teacher가 더 좋은 예시를 보여주면, student가 배우는 target 자체가 좋아질 수 있어.
즉 **few-shot은 student를 직접 똑똑하게 만든다기보다, teacher labeling 품질을 올리는 용도**로 더 유효해.

## 어디에 넣는 게 좋냐

### 1. 가장 추천: teacher labeling 프롬프트에 few-shot 추가

이게 제일 좋다.

흐름은:

* raw reviews 입력
* teacher에게 few-shot 포함한 프롬프트로 정답 생성
* 그 결과를 student SFT target으로 사용

이 방식의 장점:

* 추론 시 0.5B에 긴 few-shot 안 넣어도 됨
* teacher output 품질이 더 안정화될 가능성 높음
* 데이터셋 전체 스타일 일관성이 올라감

즉, **few-shot을 “데이터 생성 품질 개선 장치”로 쓰는 것**.

### 2. 덜 추천: student 추론 프롬프트에 few-shot 직접 넣기

할 수는 있는데, 0.5B라면 좀 조심해야 해.

이유:

* 컨텍스트 길이 부담
* 지시 + few-shot + 실제 리뷰까지 들어가면 토큰이 커짐
* 작은 모델은 예시 패턴을 따라하다가 오히려 내용 복사/형식 과적합이 날 수 있음
* 실제 서비스 비용/지연도 증가

그래서 서비스 목적이면 이건 보통 덜 예쁘다.

## 어떻게 넣는 게 좋냐

few-shot도 아무 예제나 넣으면 안 되고, **실패 패턴을 보정하는 예제**를 넣어야 해.

지금 너한테 필요한 예제는 이런 유형이야.

* 분위기/서비스/음식이 섞여 있을 때 카테고리 분리하는 예제
* price 언급이 거의 없을 때 억지 생성하지 않는 예제
* evidence index를 정확히 고르는 예제
* generic summary 대신 구체적인 리뷰 표현을 반영하는 예제
* 부정/혼합 리뷰를 균형 있게 요약하는 예제

즉, “잘된 예시”보다
**“작은 모델이 헷갈리는 케이스를 teacher가 어떻게 정리하는지 보여주는 예시”**가 더 중요해.

## 실전적으로는 이렇게 하는 게 좋다

### A안: teacher prompt에 1~2 shot만 추가

가장 무난함.

구성:

* system: 역할/출력 규칙
* example 1: 입력 리뷰 묶음 → 이상적인 JSON 출력
* example 2: price 언급 적은 케이스
* 실제 입력

1~2개면 충분한 경우가 많아.
너무 많이 넣으면 오히려 스타일 고정이 심해질 수 있어.

### B안: 카테고리별 규칙을 더 명시

few-shot보다 먼저 먹히는 경우도 많아.

예:

* service에는 친절, 응대, 대기, 분위기/좌석 편의 포함
* price는 가격 직접 언급 없으면 빈 bullets 허용
* food는 메뉴/맛/식감 중심
* evidence는 실제 문장 인덱스만 허용
* 입력에 없는 내용 추론 금지

작은 모델 문제는 종종 “학습 부족”보다 **label target의 일관성 부족**이라, 규칙 강화만으로도 좋아질 수 있어.

### C안: evidence 없는 버전으로 먼저 teacher data 재생성

이건 특히 추천.

지금 병목이 evidence라면,
먼저

* summary
* bullets
* overall_summary

까지만 few-shot으로 teacher data를 다시 만들고
그걸로 student를 재학습하는 게 효율적이야.

그 다음 필요하면 evidence를 별도 단계로 분리.

## 어려운 경우도 있다

few-shot이 만능은 아니야. 특히 이런 경우는 한계가 있어.

* 0.5B가 너무 작은데 task가 복합적일 때
* teacher output 구조가 지나치게 길 때
* training data 자체가 noisy할 때
* 같은 입력에도 label 스타일이 많이 흔들릴 때

즉 few-shot 넣어도 좋아질 수는 있지만,
**“요약 + 카테고리 분류 + bullet 생성 + evidence index 선택”**를 한 번에 다 잘하게 만들지는 못할 수 있어.

## 그래서 추천 우선순위

내 추천은 이 순서야.

1. **evidence 제거한 간소화 schema 정의**
2. **teacher labeling prompt에 1~2 shot 추가**
3. **price 언급 없을 때 빈 값 허용 규칙 명시**
4. **teacher 데이터 일부 샘플링해서 품질 직접 확인**
5. **재학습**
6. 필요하면 나중에 evidence task 분리

이렇게 가면 훨씬 현실적이야.

## 한 줄로 정리하면

**few-shot 개선은 가능하고, 꽤 좋은 방향이다.**
하지만 **student 추론 프롬프트보다 teacher labeling 단계에 few-shot을 넣는 방식**이 훨씬 낫고,
가능하면 **evidence 없는 단순 schema로 먼저 재학습**하는 게 성공 확률이 높아.

원하면 내가 바로
**네 현재 작업에 맞는 teacher few-shot 프롬프트 템플릿**을 짜줄게.
