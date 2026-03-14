
정리하면 아래와 같습니다.

---

## avg_score 2.62 → 3.0+ 로 올리기

`avg_score`는 6개 축의 평균이라, **가장 낮은 축을 올리는 것**이 효과가 큽니다.

### 현재 6축

| 축 | 평균 | 상태 |
|----|------|------|
| style_adherence | 3.26 | 양호 |
| schema_adherence | 3.08 | 양호 |
| category_correctness | 2.58 | 개선 여지 |
| fallback_adherence | 2.5 | 개선 여지 |
| faithfulness | 2.28 | 낮음 |
| **evidence_validity** | **2.02** | **가장 낮음** |

---

### 1. evidence_validity (2.02) 올리기 — 최우선

결과를 보면 **evidence 인덱스 오류**가 반복됩니다.

- `"evidence": [0, 0, 0]` (price 등), `[0, 6, 8]` (리뷰 개수 초과), `[0, 1, 3, 0]` (중복)
- `overall_summary`에 `evidence` 배열이 들어가거나, 비정상적으로 긴 evidence

**추천:**

1. **추론 직후 후처리로 인덱스 보정**  
   `eval_llm_as_judge.py`의 `_extract_json_for_rouge`(또는 JSON 추출 직후)에서, **instruction으로부터 카테고리별 리뷰 개수**를 구한 뒤:
   - 각 카테고리별로 `evidence`를 `0 <= i < len(해당_카테고리_리뷰)` 범위로 클램핑
   - 범위 밖/중복 인덱스 제거 후 `bullets` 개수와 맞추기  
   → 인덱스 오류가 줄어들어 **evidence_validity**와 **faithfulness**가 함께 올라갑니다.

2. **학습/프롬프트 강화**  
   - distill용 시스템 프롬프트·few-shot에  
     “evidence는 **해당 카테고리 리뷰 배열의 0-based 인덱스**이며, **배열 길이를 넘으면 안 된다**”를 더 명시
   - 예: “bullet 1 → 리뷰 인덱스 0, bullet 2 → 리뷰 인덱스 1” 형태의 예제 추가

3. **`schema_repair`에 인덱스 검증 추가(선택)**  
   `repair_summary_schema`는 현재 evidence 길이만 bullets에 맞추고 **범위 검증은 하지 않습니다**.  
   카테고리별 리뷰 개수를 넘겨줄 수 있다면, 여기서 `evidence`를 유효 인덱스만 남기고 bullets와 길이 맞추면, 다른 파이프라인에서도 재사용할 수 있습니다.

---

### 2. fallback_adherence (2.5) 올리기

- **완전 빈 출력**(sample 3624처럼 모든 카테고리 `summary=""`, `bullets=[]`)이 있으면 **fallback_adherence**가 1로 깎입니다.
- 추론 직후에:
  - 어떤 카테고리든 `summary`가 빈 문자열이고 `bullets`/`evidence`가 비어 있으면,
    - price → `"가격 관련 언급이 적어요."` 등
    - service/food → `"언급이 적어요."` 등  
  으로 채우는 **폴백 치환**을 한 번 넣어주면, 빈 문자열만 쓰는 사례가 줄어 fallback_adherence가 올라갑니다.
- 추가로 teacher/distill 프롬프트에 “근거 없을 때는 빈 문자열 대신 ‘언급이 적어요’ 등 해요체 폴백 사용”을 더 강조해 두면 좋습니다.

---

### 3. faithfulness / category_correctness

- **faithfulness**: evidence를 위처럼 “유효 인덱스만 남기는 후처리”를 하면, 잘못된 인덱스로 인한 “리뷰와 무관한 내용” 감점이 줄어들어 일부 개선됩니다.
- **category_correctness**: “service=서비스·직원·분위기, price=가격·가성비·양, food=음식·메뉴·맛”을 프롬프트/ few-shot에서 반복하고, 학습 데이터에 카테고리 경계가 분명한 예제를 더 넣는 것이 좋습니다.

---

### 4. 구현 우선순위 제안

1. **추론 직후 evidence 인덱스 검증·보정**  
   instruction 파싱 → 카테고리별 리뷰 수 계산 → evidence 클램핑 + bullets와 길이 맞춤.  
   (코드 위치: `eval_llm_as_judge.py`의 `_extract_json_for_rouge` 호출 직후 또는 그 안의 `repair_summary_schema` 호출 부근.)
2. **빈 summary/bullets에 대한 폴백 문구 자동 치환**  
   위와 같은 후처리 단계에서 한 번에 처리 가능.
3. **프롬프트/ few-shot 정리**  
   evidence 규칙·폴백 규칙을 distill용 프롬프트와 `eval_llm_as_judge`용 프롬프트에 맞춰 강화.

이 순서로 적용하면 **evidence_validity**와 **fallback_adherence**가 가장 크게 올라가고, 그다음 **faithfulness**·**category_correctness**가 따라오면서 **avg_score를 3.0 이상**으로 끌어올리기 쉬워집니다.