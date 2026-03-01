좋아. **SFT 기준으로만**(소규모 팀 + 자동화 + 비용 민감) sweep run 수 설계 기준 정리해줄게.

## SFT sweep run 수 설계 기준

### 1) SFT에서 “run 수”는 보통 이렇게 잡음

* **코스 탐색(Random)**: **20~50 runs**
* **파인 탐색(Bayes)**: **10~30 runs**
* **재현성 체크**: Top-1~2 설정을 **seed 2~3회 반복** (**2~6 runs**)

➡️ 실무적으로 많이 쓰는 총합: **35~80 runs**
자동화/예산 민감이면 보통 **40~60**에서 끊는 편이 가장 흔함.

---

## 2) 1 run 비용(시간) 기준 “현실적” 추천

* **1 run ≤ 15분**: **50~100 runs**도 가능 (그래도 자동화면 60~80 권장)
* **15~45분**: **30~60 runs**
* **45~120분**: **15~35 runs**
* **2시간+**: **8~20 runs** (대신 탐색 범위 매우 좁게)

---

## 3) SFT에서 sweep에 넣을 “핵심 하이퍼” (6개만)

run 수를 늘리기보다 **여기만** 잘 보는 게 이득이 큼:

1. **learning_rate** (log-uniform)
2. **effective batch** = batch_size × grad_accum
3. **num_train_steps(or epochs)** + **early stopping**
4. **warmup_ratio**
5. **weight_decay**
6. **max_seq_len / truncation 전략** (잘리면 성능 급락)

> 이 6개 외엔 가급적 고정(optimizer, scheduler 타입 등)해서 탐색 공간을 줄이는 게 보통 더 성능/비용 효율 좋음.

---

## 4) 자동화 파이프라인에서 “안전한 기본값”

너 상황 같은 팀 프로젝트 자동화면 이렇게 박아두면 안전해:

* **run_cap: 50** (SFT 기본 추천)
* **Top-2**만 seed 3회 반복 → +6 runs
* 총 **~56 runs**로 마감

---

## 5) 바로 쓸 수 있는 sweep config 예시 (SFT)

```yaml
method: bayes
metric:
  name: eval_loss      # 또는 eval_rougeL / eval_accuracy 등
  goal: minimize

run_cap: 50

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 5e-6
    max: 3e-4

  grad_accum:
    values: [1, 2, 4, 8]

  warmup_ratio:
    distribution: uniform
    min: 0.0
    max: 0.1

  weight_decay:
    distribution: log_uniform_values
    min: 1e-7
    max: 1e-3

  max_seq_len:
    values: [512, 1024, 2048]
```

---

### 결론

SFT sweep은 소규모 팀 자동화 기준으로 **run_cap 40~60**이 “가장 흔하고 안전한” 구간이고,
그 안에서 **learning_rate / effective batch / steps / warmup / wd / seq_len**만 집중해서 보면 충분히 좋은 결과가 나오는 편이야.

원하면 네 태스크가 **요약/분류/생성** 중 뭐인지 기준으로 metric이랑 early stopping 기준(예: patience, min_delta)도 SFT에 맞춰서 추천해줄게.
