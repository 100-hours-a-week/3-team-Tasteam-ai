# 트러블슈팅

## comparison_display LLM 변동성

### 현상

`comparison_display` 해석 문장이 10번 중 약 9번은 일관되게 나오나, 1번 정도 예외적으로 다른 표현이 나올 수 있음.

### 재현 방법

재현 절차:
1) 동일 restaurant_id, 동일 lift_pct/tone로 API를 20회 호출한다.
2) 출력에 아래 키워드가 포함되면 변형으로 분류한다:
   - 리뷰 수 / 표본 수 / 신뢰 / 상당히 / 매우 / 굉장히
3) 출력 문자열을 키워드 매칭으로 검사하여 변형 여부를 자동 분류한다.
4) 변형 발생률(%)을 기록한다.


### 예시

**일반 (대부분)**:
- 서비스 만족도는 평균보다 약 3% 높아요. 서비스에 대한 고객 만족도가 평균보다 약간 높은 편입니다.
- 가격 만족도는 평균보다 약 175% 높아요. 고객들은 가격에 대한 만족도가 평균보다 높은 편이며, 많은 리뷰가 이를 뒷받침하고 있습니다.

**가끔 나타나는 변형**:
- 서비스 만족도는 평균보다 약 3% 높아요. 서비스에 대한 고객 만족도가 평균보다 약간 높은 편이며, **리뷰 수가 적지 않아 신뢰할 수 있는 결과입니다.**
- 가격 만족도는 평균보다 약 175% 높아요. 가격에 대한 고객 만족도가 평균보다 높은 **비율이 상당히** 높으며, 많은 리뷰가 이를 뒷받침하고 있습니다.

### 원인

| 원인 | 설명 |
|------|------|
| Temperature 0.3 | `llm_utils.generate_comparison_interpretation_async`에서 `temperature=0.3` 사용. 0이 아니면 확률적 샘플링으로 동일 입력에도 출력이 가끔 달라짐. |
| n_reviews 프롬프트 포함 | user_content에 `리뷰 수(표본 수): {n_reviews}`를 넣어 LLM이 이 값을 활용해 "리뷰 수가 적지 않아 신뢰할 수 있는 결과입니다" 등 추가 해석을 할 수 있음. |
| 금지 표현 목록 미포함 | "최고", "압도적", "완벽"만 금지. "상당히", "신뢰할 수 있는" 등은 금지 목록에 없어 사용될 수 있음. |
| 출력 검증 없음 | LLM 출력에 대한 사후 검증/교정 로직이 없어 변형 문장이 그대로 반환됨. |

### 현재 코드 (`src/llm_utils.py`)

```python
async def generate_comparison_interpretation_async(
    self,
    category: str,
    lift_pct: float,
    tone: str,
    n_reviews: int,
) -> Optional[str]:
    system_content = (
        "당신은 음식점 비교 해석 문장을 만드는 도우미입니다. "
        "반드시 다음을 지킵니다: (1) 주어진 숫자를 그대로 사용하고, 숫자 계산이나 새 숫자 생성 금지. "
        "(2) '최고', '압도적', '완벽' 등 과장 표현 금지. "
        "(3) 표본 톤에 맞춰 자연스럽게 한 문장만 출력. "
        "(4) lift는 '만족도가 평균보다 높은 비율'입니다. 가격 lift가 높다 = 가성비/가격에 대한 고객 만족도가 평균보다 높음. "
        "'가격 상승', '가격 인상', '가격이 올랐다' 등과 혼동 금지. "
        "반드시 JSON 형식으로만 답하세요: {\"interpretation\": \"한 문장\"}."
    )
    user_content = (
        f"카테고리: {category}. lift 퍼센트: {round(lift_pct)}% (만족도가 평균보다 이만큼 높음). "
        f"표본 톤: {tone}. 리뷰 수(표본 수): {n_reviews}. "  # ← n_reviews 포함 (원인 2)
        "위 톤을 반영해 해석 문장 한 문장만 만들어 주세요. 해석 문장 안에 숫자를 넣지 마세요."
    )
    # ...
    raw = await self._generate_response_async(
        messages,
        temperature=0.3,  # ← 원인 1
        max_new_tokens=80,
    )
```

### 해결 방안

1. **temperature, top_p 낮추기** (700행 근처):
```python
raw = await self._generate_response_async(
    messages,
    temperature=0.0, # 또는 0.1
    top_p=0.2,
    max_new_tokens=80,
)
```
참고: top_p는 클라이언트/모델 API가 지원할 경우에만 적용한다.


2. **n_reviews 제거** (user_content):
```python
user_content = (
    f"카테고리: {category}. lift 퍼센트: {round(lift_pct)}% (만족도가 평균보다 이만큼 높음). "
    f"표본 톤: {tone}. "
    "위 톤을 반영해 해석 문장 한 문장만 만들어 주세요. 해석 문장 안에 숫자를 넣지 마세요."
)
```

3. **금지 표현 확대** (system_content):
지금은 system은 금지가 약함. “금지 단어”만 나열하면 모델이 변형어로 피해갈 수 있음.
따라서, 금지 카테고리 + 예시로 넣는 게 효과가 훨씬 좋음.
```python
system_content = (
    "당신은 음식점 비교 해석 문장을 만드는 도우미입니다. "
    "반드시 다음을 지킵니다: "
    "(1) 주어진 숫자를 그대로 사용하고, 숫자 계산이나 새 숫자 생성 금지. "
    "(2) 과장/강조/확신 표현 금지. 예: '최고', '압도적', '완벽', '상당히', '매우', '굉장히', '확실히', '단연'. "
    "(3) 리뷰수/표본수/신뢰도 언급 금지. 예: '리뷰 수가', '표본 수가', '신뢰할 수', '충분한 리뷰', '데이터가 많아'. "
    "(4) 한 문장만 출력. "
    "(5) lift는 '만족도가 평균보다 높은 비율'이며 가격 lift는 '가격/가성비 만족' 의미. "
    "반드시 JSON 형식으로만 답하세요: {\"interpretation\": \"한 문장\"}."
    "(6) 동일 의미 반복 금지. 예: '높은 비율이 높다', '평균보다 높다'를 두 번 말하지 말 것. "
)
```

4. **출력 필터** (interp 반환 전):

본 필터는 comparison_display 전용이며, 다른 LLM 출력에는 적용하지 않는다.

```python
# 반환 직전에 검증
FORBIDDEN_PHRASES = (
    # comparison_display 전용 금지어
    # 메타 신뢰도/표본 설명은 톤으로만 반영하고, 문장으로는 금지
    "리뷰 수", "표본 수", "신뢰", "믿을 수",  # 메타 언급 차단
    # 강조 부사는 출력 일관성 훼손 가능성 있어 금지
    "상당히", "매우", "굉장히", "확실히", "단연", # 강조 차단
)

def is_valid_interp(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if any(p in s for p in FORBIDDEN_PHRASES):
        return False
    # 중복 강조 패턴(대표)
    INTENSIFIERS = ("상당히", "매우", "굉장히", "확실히", "단연")
    if any(w in s for w in INTENSIFIERS) and ("높은 비율" in s or "높" in s):
        return False
    return True

interp = parsed.get("interpretation")
if isinstance(interp, str) and is_valid_interp(interp):
    return interp.strip()
return None  # 폴백(format_comparison_display)
```

### 성공 기준


성공 기준:
- 의미 일관성: 100% (카테고리 오해/수치 오해 없어야 함)
- 금지 표현 포함률: 0% (후처리 필터로 보장)
- 변형 발생률: <= 1% (생성 단계에서 최대한 억제)
- 폴백율: <= 5% (정상 범위, 초과 시 프롬프트 재조정)



### 리스크/부작용

부작용/리스크:
- temperature=0.0 적용 시 문장 다양성 감소(거의 템플릿화) 가능.
- 금지어 필터 강화 시 과도한 폴백 발생 가능 → 폴백율 모니터링 필요.

- 출력 다양성 감소는 의도된 결과이며, LLM 출력의 일관성을 우선한다.

### 추가적인 문제

1. 논리 오류

system_content에선 "숫자를 그대로 사용하라"로 되어 있지만,
user_content에선 "해석 문장 안에 숫자를 사용하지 말라"는 서술로 LLM이 논리적 충돌을 겪어 답변이 일관적이지 않을 가능성 존재.
 
```python
system_content = ("... (1) 주어진 숫자를 그대로 사용하고, ...")
user_content = ("... 해석 문장 안에 숫자를 넣지 마세요.")
```

### 해결 방안

user content에 **퍼센트 숫자**는 포함해도 된다고 명확히 명시.

```python
user_content = ("... lift 퍼센트 숫자는 문장에 포함해도 됩니다.")
```

