
v11 결과만 요약하면 아래와 같습니다.

**v11 메타 (v10 대비)**  
- **avg_score**: 2.70 (v10: 2.62) → +0.08  
- **avg_schema_adherence**: 3.14 (3.08) → 소폭 상승  
- **avg_fallback_adherence**: 2.72 (2.50) → 상승  
- **avg_style_adherence**: 3.38 (3.26) → 상승  
- **avg_evidence_validity**: 1.98 (2.02) → 소폭 하락  
- **avg_faithfulness**: 2.34 (2.28) → 소폭 상승  
- **avg_category_correctness**: 2.66 (2.58) → 소폭 상승  

**해석**  
- 후처리(폴백 치환·evidence 정리) 덕분에 **fallback_adherence**와 **avg_score**가 올랐고, 스키마/스타일/카테고리 정확도도 조금 나아졌습니다.  
- **evidence_validity**는 오히려 약간 떨어졌습니다. 인덱스만 보정하고 bullet–evidence 대응이 어긋난 경우 judge가 더 엄하게 줄 수 있어서일 가능성이 있습니다.  

**다음으로 할 수 있는 것**  
- 3.0을 넘기려면 **evidence_validity**를 올리는 게 중요합니다.  
  - 후처리에서 “범위 밖 인덱스만 제거”가 아니라, **bullet과 evidence를 1:1로 맞추거나**, 유효하지 않은 evidence가 있는 bullet은 제거하는 식으로 정교하게 맞추는 방안을 검토해 볼 수 있습니다.  
  - 또는 judge 쪽에서 “인덱스 범위만 맞으면 부분 점수”를 주도록 루브릭/프롬프트를 조정하는 것도 한 방법입니다.  

원하시면 v11 결과 JSON에서 evidence_validity가 낮게 나온 샘플 몇 개만 골라서 패턴을 같이 보겠습니다.

---


v11 결과에서 **evidence_validity**가 낮게 나온 패턴을 정리하면 아래와 같습니다.

---

## 1. evidence_validity = 1 (5건)

| sample_id | 이유(reason) | 패턴 |
|-----------|--------------|------|
| **3624** | "모든 카테고리에 폴백 문구만 사용하고, evidence가 전혀 없어요." | instruction에 service 3개, price 4개, food 8개 리뷰가 있는데 **모델이 모든 카테고리를 빈 bullets/evidence로 냄** → 후처리에서 폴백 문구만 채움. judge는 "evidence 없음"으로 1점. |
| **936** | "스키마와 evidence 기반이 부족하고, 폴백 정책이 부분적으로만 적용됨." | service=[](빈 배열), price 4개, food 8개인데 **pred가 전부 폴백만** (언급이 적어요 / 가격 관련 언급이 적어요). 리뷰가 있는 price/food도 bullet 없이 폴백만 사용. |
| **508** | "여러 카테고리에서 근거와 불일치하거나 잘못된 정보" | price evidence **[0,1,0,0,0]** 반복, food에 **환각/오타** (colesanders, 부위기, avicke 등) → bullet–evidence 대응이 깨짐. |
| (나머지 2건) | 비슷하게 evidence 부재 또는 bullet–evidence 불일치 | |

**공통**:  
- **완전 폴백만 출력** (리뷰가 있는데도 bullet/evidence를 안 냄)  
- 또는 **evidence를 전부 0으로 두거나 잘못된 인덱스 + 환각**으로 judge가 “근거 없음/근거 불일치”로 1점 처리.

---

## 2. evidence_validity = 2 (다수)

judge가 “evidence 인덱스 오류”, “evidence와 bullet 불일치”, “카테고리 혼입” 등을 이유로 2점을 준 경우입니다.

| 패턴 | 예시 (sample_id) | 설명 |
|------|------------------|------|
| **price evidence [0,0,0] 또는 [0,0,0,0]** | 839, 2950, 504, 2896, 2961 | bullet은 3~4개인데 evidence가 전부 0. “각 bullet이 어떤 리뷰를 참조하는지 불명확” 또는 “의미 없는 인덱스 반복”으로 감점. |
| **service evidence [0,0,0,0]** | 856 | 서비스 bullet 4개에 evidence가 모두 0 → bullet–evidence 대응 없음. |
| **인덱스 범위 초과** | 853 | food evidence에 **8** 포함 (food 리뷰 8개 → 유효 인덱스 0~7). 후처리에서 8 제거 후 0 패딩하면 “나머지 bullet이 리뷰 0에 대응”으로 보여 bullet–support 불일치로 감점. |
| **bullet–evidence 의미적 불일치** | 2950, 3562 | 인덱스는 범위 안이지만 “이 bullet을 그 인덱스 리뷰가 실제로 지원하지 않음” (예: price bullet에 직원 친절 언급, food에 쌀국수인데 중식당 등). |
| **overall_summary에 evidence 배열** | 853 | 스키마에 없는 `overall_summary.evidence` [0..167] 등 → 스키마/정합성 문제로 인한 감점 가능. |
| **카테고리 혼입** | 504, 2896, 3028, 2884 | food에 “쌀국수 맛집” (실제는 라디오베이/팔복), service에 “점심 대기” 등 **다른 카테고리/다른 가게 템플릿** 반복 → faithfulness·category_correctness와 함께 evidence_validity도 낮게 나옴. |

---

## 3. 요약: 낮은 evidence_validity의 세 가지 축

1. **모델이 아예 bullet/evidence를 안 냄**  
   → 리뷰가 있어도 전부 폴백만 사용 (3624, 936 등). **학습/추론 안정성** 또는 **폴백 조건** 재검토 필요.
2. **evidence를 전부 0으로 두거나 반복**  
   → bullet 수만 채우고 인덱스를 0으로 패딩. judge는 “bullet별로 다른 리뷰를 가리켜야 한다”고 봐서 감점. **후처리에서 “0만 있는 evidence”는 폴백으로 넘기거나**, **모델이 bullet당 실제 인덱스를 쓰도록** 유도하는 게 좋음.
3. **인덱스 범위·bullet–support 불일치 + 카테고리 혼입**  
   → 범위 초과는 후처리로 잡았지만, **패딩으로 넣은 0**이 “해당 bullet을 지원하지 않는 리뷰”로 해석되어 감점.  
   → 카테고리/가게가 다른 문구 반복은 **faithfulness·category_correctness**와 같이 **evidence_validity**까지 끌어내림.

---

## 4. 개선 시도 시 제안

- **완전 폴백만 나오는 샘플**:  
  - 해당 instruction/길이·구성에 대한 학습 데이터 보강, 또는  
  - “리뷰가 있으면 반드시 bullet/evidence를 채울 것”을 프롬프트/강화 대상에 명시.
- **evidence [0,0,0] 다수**:  
  - 후처리에서 “유효한(서로 다른) 인덱스가 하나도 없으면” 해당 카테고리는 bullets를 비우고 폴백만 두는 정책 검토.  
  - 또는 judge 루브릭에서 “evidence가 전부 0이면 bullet–evidence 미준수로 감점”을 명시해 두고, 모델/데이터는 “0만 쓰지 않도록” 학습.
- **bullet–evidence 의미 일치**:  
  - “evidence는 해당 bullet을 **실제로 지원하는** 리뷰 인덱스만 사용”을 teacher/학습 데이터와 judge 기준에 맞춰 정리.

이렇게 패턴을 나눠두면, evidence_validity를 올리기 위한 수정 포인트(데이터, 후처리, 루브릭)를 골라서 적용하기 쉬울 거예요.