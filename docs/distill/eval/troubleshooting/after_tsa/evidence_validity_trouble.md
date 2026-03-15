
v18 결과에서 **evidence_validity**로 드러나는 문제를 유형별로 정리하면 아래와 같습니다.

---

## 1. evidence 인덱스 오류 (가장 많음)

**패턴**
- **전부 0으로 채움**: `evidence: [0, 0, 0, 0]`, `[0, 0, 0]`처럼 bullet 개수만큼 같은 인덱스(대부분 0)만 반복.
- bullet마다 **서로 다른 리뷰**를 가리켜야 하는데, 한 리뷰만 반복 참조.
- 예: sample 839 – service 8개 리뷰인데 pred는 `evidence [0,0,0,0]` → 4개 bullet이 모두 0번 리뷰만 가리킴.

**원인 추정**: 모델이 “evidence = 해당 카테고리 배열 인덱스”를 이해하지만, **bullet별로 어떤 리뷰가 맞는지**를 잘 못 골라서 디폴트로 0을 반복 사용.

---

## 2. evidence와 bullet의 불일치

- evidence 개수와 bullet 개수는 맞는데, **evidence[i]가 bullet[i]를 실제로 지지하지 않음**.
- 예: bullet “직원들이 친절해요”에 evidence 3 → 3번 리뷰에는 친절 언급이 없음.
- judge 입장에서는 “인덱스는 유효해도, 그 인덱스의 리뷰가 bullet 내용을 뒷받침하지 않음” → evidence_validity 감점.

---

## 3. 카테고리 혼입 + 잘못된 evidence

- **해당 카테고리 리뷰에 없는 내용**을 bullet으로 씀 (예: service에 “쌀국수 맛있어요”, “카페라벨”, 루프탑 바에 “직원 친절/음식 빨리 나옴” 등).
- 이때 evidence는 유효 인덱스(0,1,2,…)를 쓰지만, 그 리뷰에는 그 내용이 없음 → **faithfulness + evidence_validity** 둘 다 낮아짐.
- “입력에 없는 내용을 지지하는 evidence”로 쓰는 셈이라 evidence_validity에서도 문제로 잡힘.

---

## 4. price evidence 비거나 전부 0

- price 리뷰가 여러 개인데 `evidence: []` 또는 `evidence: [0, 0, 0]`.
- 가격 관련 bullet은 있는데, 그에 대응하는 **다양한 리뷰 인덱스**를 안 쓰고 비우거나 0만 반복.

---

## 5. 스키마/생성 오류 (일부 샘플)

- sample 508: food bullets 안에 `.format("포스트")` 같은 **코드 조각**이 들어가고, evidence에 **0~100** 같은 비정상 인덱스 나열 (후처리로 잘렸을 수 있음).
- 생성이 깨지면서 evidence도 함께 invalid해져 evidence_validity 1로 감점.

---

## 요약 표

| 유형 | 설명 | 빈도(judge reason 기준) |
|------|------|-------------------------|
| evidence 인덱스 오류 | 전부 0, 또는 bullet과 안 맞는 인덱스 | 거의 모든 샘플 |
| evidence–bullet 불일치 | 인덱스는 유효하나 해당 리뷰가 bullet을 지지하지 않음 | 다수 |
| 카테고리 혼입 | 다른 카테고리/도메인 내용 + 그걸 “지지”하는 evidence | 다수 |
| price evidence 부재/단조로움 | [] 또는 [0,0,…]만 사용 | 일부 |

**정리**: evidence_validity가 낮은 이유는  
(1) **인덱스 사용 오류**(0만 쓰기, 범위 초과, 코드 오류),  
(2) **인덱스와 bullet의 의미적 불일치**(그 리뷰가 bullet을 지지하지 않음),  
(3) **카테고리/도메인 혼입**으로 “그 리뷰에 없는 내용”을 bullet으로 쓰면서 evidence를 붙이는 경우가 겹치기 때문입니다.