좋은 질문이고, 핵심은 **모델 용량이 작을수록(특히 0.5B)** 거의 맞는 가정입니다.

결론부터 말하면:

- **같은 모델 크기/데이터 품질/학습 예산**이라면  
  **evidence 제외 스키마로 학습한 모델이**, evidence 포함 스키마 모델보다  
  **evidence를 제외한 항목(summary, style, category, fallback) 성능이 올라갈 가능성이 높아요.**
- 이유는 단순히 태스크 난도가 내려가서, 제한된 capacity를 “요약 품질”에 더 쓸 수 있기 때문입니다.

## 왜 올라가기 쉬운가
- evidence 포함 태스크는 사실상  
  1) 요약 생성 + 2) 카테고리 분리 + 3) bullet 생성 + 4) 인덱스 정렬(근거 매핑)  
  까지 한 번에 요구합니다.
- 0.5B에서는 4번이 특히 어려워서, 여기서 생긴 오류가 다른 축(faithfulness 등)까지 같이 깎는 경우가 많습니다.
- evidence를 빼면 학습 objective가 “텍스트 품질”에 집중되어 평균 점수가 올라가기 쉽습니다.

## 예외/주의
- **큰 모델(예: 7B+)**로 가면 evidence 포함해도 나머지 성능이 크게 안 떨어질 수 있습니다.
- judge를 evidence 포함 루브릭으로 그대로 두면, evidence 제외 모델은 schema 쪽에서 불리해져 “상승”이 가려질 수 있습니다.
- 데이터/프롬프트 품질이 나쁘면 스키마 단순화 효과가 제한적일 수 있습니다.

## 실무적으로 보는 기준
- 지금 질문의 조건(0.5B, capacity 병목)에서는  
  **“evidence 제외 학습 모델 > 기존 evidence 포함 학습 모델”**  
  (단, 비교 축은 evidence 제외 항목)일 확률이 높습니다.
- 다만 최종 목표가 evidence까지 수행 가능한 모델이라면,  
  지금 방식은 **단기 성능 최적화(요약 품질)**,  
  이후 큰 모델에서 **전체 태스크 복귀**가 맞는 전략입니다.

---

재학습을 하지 않는다면,

학생 모델이 evidence를 ‘정답 형식’으로 이미 학습했을 가능성이 커서, 프롬프트만 바꿔도 습관처럼 evidence를 계속 찍거나(혹은 judge/후처리에서 계속 충돌) 품질이 흔들릴 수 있어요.

---

## 재학습 계획

좋아요. 지금 목표라면 **“0.5B 전용 트랙”**을 따로 두는 게 가장 깔끔합니다.  
핵심은 **학습 라벨 + 추론 프롬프트 + 후처리 + judge**를 한 세트로 맞추는 거예요.

## 가장 안전한 방법 (추천)

### 1) 0.5B 전용 라벨셋 만들기 (evidence 제거)
기존 `train_labeled.json`/`val_labeled.json`/`test_labeled.json`의 `output` JSON에서
- `service.evidence`
- `price.evidence`
- `food.evidence`
를 제거한 **새 라벨셋**을 만듭니다.

즉, `instruction`은 그대로 두고 `output`만 아래 형태:
```json
{
  "service": {"summary": "...", "bullets": ["..."]},
  "price": {"summary": "...", "bullets": ["..."]},
  "food": {"summary": "...", "bullets": ["..."]},
  "overall_summary": {"summary": "..."}
}
```

> 장점: 라벨 재생성 비용 없이 빠르게 실험 가능.

---

### 2) 0.5B 학습 프롬프트/템플릿도 no-evidence로 맞추기
현재 코드에서 evidence를 강하게 요구하는 지점이 여러 군데 있습니다:
- `src/distill_summary.py`의 `SCHEMA_ENFORCEMENT_SYSTEM`, few-shot 예시
- `scripts/eval_distill.py`의 `_SCHEMA_ENFORCEMENT_SYSTEM`
- `scripts/eval_llm_as_judge.py`의 v2 루브릭(증거축 포함)

0.5B 트랙에서는 이들을 **“summary+bullets only”**로 바꾼 버전을 써야 합니다.  
(대형 모델 트랙은 기존 유지)

---

### 3) 후처리에서 evidence를 다시 붙이지 않게 분리
현재 `src/schema_repair.py`와 `src/distill_summary.py`의 후처리는 evidence를 기본 필드로 다룹니다.
- `repair_summary_schema()`가 evidence를 패딩/정리
- `postprocess_prediction()`이 evidence를 강제 정렬

0.5B 트랙에서는
- no-evidence용 repair/postprocess를 따로 두거나
- 최소한 no-evidence 모드에서 evidence 로직을 건너뛰게 해야
학습/추론 출력이 일관됩니다.

---

### 4) Judge도 0.5B 전용으로 분리
`scripts/eval_llm_as_judge.py`의 현재 v2는 evidence를 필수 축으로 봅니다.  
0.5B(no-evidence) 평가용으로는:
- `evidence_validity` 제거
- `schema_adherence` 정의에서 evidence 필수 제거
- 평균 점수 계산도 5축 기준으로 변경  
해야 공정 비교가 됩니다.

---

### 5) 실험 설계 (필수)
같은 split으로 A/B 비교:
- A: 기존( evidence 포함 라벨 학습)
- B: no-evidence 라벨 학습 (0.5B 전용)

비교 지표:
- 공통 5축(schema/fallback/style/faithfulness/category)
- JSON parse/schema pass rate
- 샘플 기반 수동검수 20~30개

이렇게 해야 “정말 no-evidence 학습이 0.5B에 이득인지”를 확실히 판단할 수 있어요.

---

## 실무 팁
- **처음엔 라벨 변환 방식(기존 라벨에서 evidence 제거)**으로 빠르게 1차 검증
- 효과 확인되면 그때 `scripts/label_for_distill.py`의 `SUMMARY_INSTRUCTIONS`까지 no-evidence 버전으로 분기해서 본격 운영

원하면 다음 답변에서 **“파일별로 정확히 어떤 문자열/블록을 no-evidence로 바꿔야 하는지”** 체크리스트 형태로 바로 써드릴게요.