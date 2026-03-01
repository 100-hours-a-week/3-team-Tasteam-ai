
아니요. 둘 다 학습 루프에 넣을 필요는 없습니다.

---

## val_labeled.json → 학습 루프에 사용 ✅

- **목적**: 학습 중 generalization 모니터링
- **역할**: eval_loss, early stopping, load_best_model_at_end, sweep best 선택
- 학습 중에 사용하는 것이 맞습니다.

---

## test_labeled.json → 학습 루프에 넣지 않음 ❌

- **목적**: 학습 완료 후 최종 성능 보고용
- **역할**: ROUGE 등 최종 평가 메트릭
- 학습 과정에는 사용하지 않고, 학습·하이퍼파라미터 튜닝이 끝난 뒤에만 한 번 사용합니다.
- 학습 중에 사용하면 test leakage가 되어, test 평가 결과가 신뢰할 수 없어집니다.

---

## 정리

| 파일               | 학습 루프 사용 여부 |
|--------------------|---------------------|
| val_labeled.json   | 사용 (eval_dataset) |
| test_labeled.json  | 사용하지 않음       |

---


아니요, **중복이 아니라 용도가 다릅니다.** 둘 다 해도 됩니다.

---

## val_labeled를 쓰는 두 가지 용도

| 용도 | 시점 | 메트릭 | 역할 |
|------|------|--------|------|
| 학습 루프 | 학습 중 | **eval_loss** (cross-entropy) | early stopping, best checkpoint, sweep best run 선택 |
| 품질 평가 | 학습 후 | **ROUGE** | adapter 품질 보고, merge 결정 |

같은 `val_labeled.json`을 쓰지만, **메트릭이 다릅니다.**

- **eval_loss**: 다음 토큰 예측 손실 → 학습이 얼마나 잘 되고 있는지 모니터링
- **ROUGE**: 생성 문장 vs 정답 문장의 유사도 → 실제로 필요한 “품질” 지표

---

## 둘 다 수행해야 하나?

둘 다 수행하는 쪽이 더 낫습니다.

1. **학습 루프**: `val_labeled`를 eval_dataset으로 넣어  
   → eval_loss 기반 early stopping, best checkpoint 저장, sweep best run 선택

2. **품질 평가**: 학습이 끝난 후 `eval_distill`로 `val_labeled`에 대해  
   → ROUGE 계산해 adapter 품질 보고, merge 여부 결정

---

## 요약

- val_labeled는 **학습 중**: eval_loss  
- val_labeled는 **학습 후**: ROUGE  
- 같은 파일을 두 번 쓰는 것이지만, 목적과 메트릭이 달라서 둘 다 필요합니다.

---


아니요. **val을 학습 루프의 eval에 쓰는 것은 leakage가 아닙니다.**

---

## Leakage vs 정상적인 val 사용

| 구분 | leakage | 정상적인 val 사용 |
|------|---------|-------------------|
| val로 **그래디언트 계산** | ✅ leakage | val로는 gradient를 계산하지 않음 |
| val로 **가중치 업데이트** | ✅ leakage | val로는 업데이트하지 않음 |
| val로 **평가만** (loss, 메트릭 계산) | ❌ | 평가만 함 |
| val 기반 **early stopping** / best checkpoint 선택 | ❌ | 모델 선택에 사용하는 것이 val의 용도 |

학습 루프에서는 val 데이터로 **loss/메트릭만 계산**하고, **가중치는 업데이트하지 않습니다.**  
그래서 일반적으로 말하는 “데이터 leakage”에는 해당하지 않습니다.

---

## val의 본래 역할

- 학습 중 generalization을 **모니터링**
- early stopping / best checkpoint 선택
- 하이퍼파라미터 튜닝 시 **모델 선택**에 사용

즉, val은 처음부터 “학습 루프 안에서 평가·선택용으로 쓰기 위해” 두는 데이터입니다.

---

## 유의할 점

- val으로 best model을 고르면, **val 성능은 어느 정도 과대평가될 수 있습니다** (우리가 val에서 가장 좋은 모델을 골랐기 때문).
- 그 때문에 **최종 성능 보고는 test set**에서 합니다.
- 정리하면: val = 모니터링·선택, test = 최종 보고용.

---

**정리**: val_labeled를 학습 루프의 eval_dataset으로 사용하는 것은 표준적인 방법이고 leakage가 아닙니다.