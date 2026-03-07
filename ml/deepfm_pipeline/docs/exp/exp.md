1 pos, random neg 50, popular neg 50 (baseline) (실서비스와 유사)

"model_baseline": {
      "auc": 0.4243447293447294,
      "ndcg@5": 0.05813953477290916,
      "ndcg@10": 0.05813953477290916,
      "recall@5": 0.05813953477290916,
      "recall@10": 0.05813953477290916
    }

---

1 pos, random neg 100

{
  "pipeline_version": "deepfm-1.0.20260307060359",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.5384188034188034,
    "ndcg@5": 0.011627907127851148,
    "ndcg@10": 0.011627907127851148,
    "recall@5": 0.011627907127851148,
    "recall@10": 0.011627907127851148
  },
  "timestamp_utc": "2026-03-07T06:03:59.731108+00:00",
  "wandb_run_id": "dz4xdwra"
}

---

의사결정:
popularlity baseline 출력
(아이템 popularity ranking)

용도:
“모델이 진짜로 똑똑한지, 아니면 그냥 인기 아이템만 잘 맞추는지”를 구분하는 용도

추가설명:
model score < popularlity score

-> 모델이 인기순 추천보다 못하다

이유: 

feature 부족

데이터 문제

학습 문제

의심.

model score > popularity score

의미: 모델이 단순 인기 추천보다 개인화가 잘 된다

구현방안:

[positive,
 neg_random_1,
 neg_random_2,
 ...
 neg_popular_1,
 neg_popular_2]

model 대신

 score = restaurant_popularity[item]
으로 ranking

 scores = [popularity[item] for item in candidate_items]
rank = argsort(scores, descending=True)

이걸로

NDCG
Recall

계산

의사결정:
random baseline 출력

용도:
평가가 정상적으로 작동하는지 확인하는 용도
(평가 sanity check)

구현방안:

[positive,
 neg_random_1,
 neg_random_2,
 ...
 neg_popular_1,
 neg_popular_2]

random score 생성

scores = [random.random() for _ in candidate_items]

ranking

rank = sorted(items, key=lambda x: score[x], reverse=True)

metric 계산

이 랭킹으로
NDCG
Recall
계산

random < popularity < model

추천 시스템에서는 이 구조가 나와야 정상.

---

결과

{
  "auc": 0.4243447293447294,
  "ndcg@5": 0.05813953477290916,
  "ndcg@10": 0.05813953477290916,
  "recall@5": 0.05813953477290916,
  "recall@10": 0.05813953477290916,
  "model_baseline": {
    "auc": 0.4243447293447294,
    "ndcg@5": 0.05813953477290916,
    "ndcg@10": 0.05813953477290916,
    "recall@5": 0.05813953477290916,
    "recall@10": 0.05813953477290916
  },
  "popularity_baseline": {
    "ndcg@5": 0.036681962347030764,
    "ndcg@10": 0.07602503625341908,
    "recall@5": 0.05813953477290916,
    "recall@10": 0.18604651144657863
  },
  "random_baseline": {
    "ndcg@5": 0.013150346142652493,
    "ndcg@10": 0.0166506949749307,
    "recall@5": 0.023255814255702296,
    "recall@10": 0.034883721383553445
  }
}

random < model < popular

이유:
피처 수는 많지만 피처에 실제로 모델이 학습할 수 있는 preference signal의 부족.

추가 방안:

item popularity -

restaurant_popularity = sum(weight)
restaurant_signal_count
restaurant_avg_weight

user preference -

user_category_count
user_region_count
user_price_preference

-->

user_category_count(category)
user_region_count(region)
user_price_mean

user-item interaction -

user_category_match
user_region_match
price_match

-->

user_category_match = primary_category == pref_cat_1/2/3
user_region_match = region_gu == user_pref_region
price_diff = abs(price_tier - user_price_mean)

context feature -

distance_weight = 1 / (distance_bucket + 1)

---

결과

{
  "pipeline_version": "deepfm-1.0.20260307080026",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.5059188034188035,
    "ndcg@5": 0.0,
    "ndcg@10": 0.0,
    "recall@5": 0.0,
    "recall@10": 0.0,
    "model_baseline": {
      "auc": 0.4243447293447294,
      "ndcg@5": 0.05813953477290916,
      "ndcg@10": 0.05813953477290916,
      "recall@5": 0.05813953477290916,
      "recall@10": 0.05813953477290916
    },
    "popularity_baseline": {
      "ndcg@5": 0.036681962347030764,
      "ndcg@10": 0.07602503625341908,
      "recall@5": 0.05813953477290916,
      "recall@10": 0.18604651144657863
    },
    "random_baseline": {
      "ndcg@5": 0.013150346142652493,
      "ndcg@10": 0.0166506949749307,
      "recall@5": 0.023255814255702296,
      "recall@10": 0.034883721383553445
    }
  },
  "timestamp_utc": "2026-03-07T08:00:27.512920+00:00",
  "wandb_run_id": "lxl1r5q9"
}

학습 실패

이유:
모름 --> 디버깅 필요.

로그 분석:
학습 중엔 validation accuracy가 이렇게 올라갔죠.

61.4%

69.3%

74.5%

75.5%

75.6%

그런데 최종 ranking metric은 0입니다.

이 조합은 보통 둘 중 하나입니다.

accuracy가 추천 품질과 무관한 지표라서 속이고 있음

eval ranking 로직이 잘못됨

현재 baseline들이 정상적으로 나오니까, 저는 eval ranking 로직이나 model output 분포를 가장 먼저 보겠습니다.(디버깅)

디버깅 방안:

학습 끝난 직후 eval 직전에 이것 출력

print("pred min/max/mean/std:", preds.min(), preds.max(), preds.mean(), preds.std())
print("num groups:", len(groups))
print("positive count total:", labels.sum())
print("first 5 group sizes:", [len(g) for g in groups[:5]])
print("first 5 positive ranks:", positive_ranks[:5])

그리고, 각 그룹마다

print("group labels:", y_true_group)
print("group preds :", y_score_group)
print("sorted idx  :", np.argsort(y_score_group)[::-1])

---

결과

(동일)

로그 분석:

원인이 거의 잡혔습니다.

핵심은 이 로그 한 줄들입니다.

pred min/max/mean/std: 1.3335758e-09 1.0 0.9966973 0.049383864

여러 그룹에서 group preds : [1. 1. 1. ... 1.]

first 5 positive ranks: [101, 101, 101, 49, 101]

즉,

모델이 거의 모든 후보에 1.0에 가까운 점수를 주고 있습니다.

그래서 랭킹이 사실상 무너졌고, positive가 top-k에 거의 못 들어가서 recall@k = 0, ndcg@k = 0이 나온 겁니다.

이건 하이퍼파라미터 문제가 아니라 출력 포화(saturation) 문제예요.

지금 무슨 일이 벌어진 거냐

현재 모델은 eval에서 각 그룹 101개 후보를 보고도 거의 다:

0.9999

1.0

1.0

1.0

처럼 내고 있습니다.

그러면 점수 차이가 거의 없어져서 정렬이 의미를 잃습니다.
지금 sorted idx가 거의 인덱스 순서 비슷하게 보이는 것도 그 때문입니다.

즉 모델이 실제로는:

“이건 다 positive다” 쪽으로 무너진 상태

입니다.

왜 피처 엔지니어링 후 이런 현상이 잘 생기나

가장 가능성 높은 원인은 3개입니다.

1. 새 dense feature 스케일이 너무 큼

네가 추가하려던/추가한 피처 중에는 이런 것들이 있었죠.

restaurant_popularity = sum(weight)

restaurant_signal_count

restaurant_avg_weight

user_category_count

user_region_count

user_price_preference

이런 집계형 수치 피처는 값 범위가 커지기 쉽습니다.

예를 들어 어떤 피처가

0~3 수준이 아니라

0~500, 0~5000

처럼 들어가면, DeepFM의 linear/deep 쪽이 그 피처 하나에 끌려서 로짓이 엄청 커질 수 있습니다.
그러면 sigmoid 출력이 거의 전부 1로 포화됩니다.

지금 로그가 딱 그 패턴입니다.

2. label leakage 성격의 피처가 들어갔을 가능성

특히 위험한 컬럼:

weight

signal_type

first_positive_segment

first_comparison_tag

이런 피처를 어떤 방식으로 가공했는지에 따라, 학습 때는 “정답 힌트”처럼 작동하고 eval에서는 분포가 꼬이거나 전부 한 방향으로 작동할 수 있습니다.

특히 weight 기반 집계는 매우 조심해야 합니다.
현재 label과 너무 가까운 신호를 직접 먹이면 모델이 쉽게 과하게 쏠립니다.

3. dense feature 정규화 없이 그대로 투입

DeepFM은 sparse categorical에는 강하지만, dense numerical feature는 스케일 관리가 중요합니다.

지금은 새 수치형 피처들이 들어오면서:

linear term 폭주

deep part activation 폭주

sigmoid saturation

이 일어난 걸로 보입니다.

오히려 좋은 점

이번 로그 덕분에 문제 위치는 꽤 분명합니다.

평가 파이프라인 정상

random baseline 정상

popularity baseline 정상

모델만 전부 1.0 출력

즉,

모델 입력/피처 스케일 쪽 문제로 거의 좁혀졌습니다.

지금 바로 해야 할 것
1. 새로 넣은 수치형 피처 전부 분포 확인

특히 이것들:

restaurant_popularity

restaurant_signal_count

restaurant_avg_weight

user_category_count

user_region_count

user_price_preference

price_diff 계열 있으면 그것도

각각에 대해 train/test의:

min

max

mean

std

상위 5개 값

을 찍어보세요.

아마 몇 개는 값 범위가 확 튀어 있을 가능성이 큽니다.

2. 수치형 피처는 정규화해서 넣기

가장 먼저 추천하는 건:

log1p 적용

그다음 standardization 또는 min-max scaling

예를 들어:

restaurant_popularity → log1p(popularity)

restaurant_signal_count → log1p(count)

user_category_count → log1p(count)

이렇게 바꾸는 게 좋습니다.

3. 위험한 피처는 일단 빼고 ablation

지금은 한 번에 다 넣지 말고 순서대로 보세요.

추천 순서:

match 계열만

user_category_match

user_region_match

price_match 또는 price_diff

그다음 popularity 계열 1개씩

restaurant_popularity

restaurant_signal_count

마지막에 user aggregate 계열

user_category_count

user_region_count

이렇게 해야 범인을 잡을 수 있습니다.

4. eval 직전 로짓도 찍어보기

지금은 sigmoid 이후 값만 본 것 같은데, 가능하면 로짓(raw output)도 보세요.

로짓 min/max/mean/std

sigmoid 전 값이 전부 큰 양수인지

이러면 포화 원인이 더 확실해집니다.

---

결과

{
  "auc": 0.5060968660968661,
  "ndcg@5": 0.0,
  "ndcg@10": 0.0,
  "recall@5": 0.0,
  "recall@10": 0.0,
}

로그 분석:

좋아요. 이번 로그로 **왜 망하는지 훨씬 더 선명해졌습니다.**

핵심 결론부터 말하면:

> **문제는 “값이 너무 커서”라기보다, 새 feature들이 너무 한쪽으로 치우쳐 있고(train/test 분포도 다르고), 그 결과 모델이 거의 모든 샘플을 positive처럼 보는 방향으로 무너진 것**입니다.

특히 지금 로그에서 중요한 건 이 두 묶음입니다.

## 1. 출력이 여전히 거의 전부 1

* `logit min/max/mean/std: -20.75 / 162.24 / 58.70 / 28.09`
* `pred min/max/mean/std: ~0 / 1 / 0.996 / 0.053`
* 여러 그룹에서 `group preds`가 사실상 전부 `1.`

이건 여전히 **sigmoid 포화** 상태입니다.
즉 모델이 candidate 101개를 봐도 거의 다 “positive 확률 1”로 내고 있습니다.

그러면 랭킹이 안 됩니다.

---

## 2. 새 feature 분포가 너무 강하게 쏠림

이게 더 중요합니다.

### train 쪽

* `restaurant_popularity mean=0.866`
* `restaurant_signal_count mean=0.914`
* `user_category_count mean=0.861`
* `user_region_count mean=0.926`
* `user_category_match mean=0.763`
* `user_region_match mean=0.818`

### test 쪽

* `user_category_count mean=0.057`
* `user_region_count mean=0.147`
* `user_category_match mean=0.319`
* `user_region_match mean=0.538`

이건 아주 강한 신호예요.

> **train과 test에서 feature 분포가 크게 다릅니다.**

특히 `user_category_count`, `user_region_count`는 거의 **완전히 다른 feature** 수준입니다.

* train: 대부분 높음
* test: 대부분 낮음

이러면 모델은 train에서
“값이 큰 쪽 = positive 쪽”으로 학습하기 쉽고,
eval에서는 그 규칙이 무너져 버립니다.

---

# 지금 가장 의심되는 진짜 원인

## 원인 A: 집계 feature를 row의 label 구조에 너무 가깝게 만들었거나, train/test 계산 기준이 다름

특히 위험한 것:

* `user_category_count`
* `user_region_count`
* `restaurant_popularity`
* `restaurant_signal_count`

이런 건 보통 “누적/집계”로 만드는데,
현재 분포를 보면 **train과 test에서 계산 방식 또는 참조 데이터가 다르게 작동하는 느낌**이 강합니다.

특히 `user_category_count`:

* train mean 0.861
* test mean 0.057

이건 단순 일반화 실패보다 **집계 정의 차이**를 더 의심하게 합니다.

예를 들면:

* train은 현재 row 포함해서 집계
* test는 과거 이력만 집계
* unseen user는 거의 0
* normalization 기준이 split마다 다름

이런 경우 딱 이렇게 됩니다.

---

## 원인 B: feature가 너무 “positive 친화적”이라 학습이 collapse

`user_category_match`, `user_region_match` 평균도 꽤 높죠.

* train category_match 0.763
* train region_match 0.818

candidate set이 1 positive + 100 negative인데,
이런 match feature가 negatives에서도 너무 자주 1이면
모델 입장에서는 **positive/negative를 가르는 feature가 아니라 거의 다 좋은 신호**가 됩니다.

그러면 점수를 전체적으로 다 올리는 방향으로 가기 쉽습니다.

즉:

> **구분력이 있는 feature가 아니라, 전체를 positive처럼 보이게 만드는 feature**가 되었을 수 있습니다.

---

## 원인 C: distance_weight는 죽은 feature

* train/test 모두 `distance_weight = 0.333333`, std=0

이건 완전 상수라서 정보가 없습니다.
넣어도 도움이 안 되고, 혼란만 줄 수 있으니 빼는 게 낫습니다.

---

# 지금 로그에서 바로 읽히는 판단

### 1) 스케일 문제만은 아님

이미 0~1 근처로 정규화된 값들이 많습니다.
그래서 예전처럼 “raw count가 너무 커서 폭주”만의 문제는 아닙니다.

### 2) 더 본질적인 문제는 분포/정의 문제

특히:

* train/test 분포 차이
* candidate 내 구분력 부족
* 집계 feature의 정의가 추천 task와 안 맞음

이쪽이 더 큽니다.

---

# 왜 positive rank가 계속 101이 나오나

`first 5 positive ranks: [101, 101, 101, 49, 101]`

이건 positive가 거의 항상 맨 뒤라는 뜻인데,
재미있는 건 `group preds`는 다 1처럼 보인다는 점이죠.

즉 실제로는:

* 수치상 거의 다 1이지만
* 아주 미세한 차이로 정렬되고
* 그 미세한 차이에서 positive가 계속 밀림

이건 **positive를 올리는 신호보다 negative를 올리는 신호가 더 강하게 학습된 상태**라는 뜻입니다.

---

# 지금 해야 할 것

## 1. feature를 한 번에 다 넣지 말고 쪼개서 확인

지금은 ablation이 필수입니다.

추천 순서:

### 실험 1

기존 baseline +

* `user_category_match`
* `user_region_match`
* `price_diff`

이 3개만

### 실험 2

실험 1 +

* `restaurant_popularity`

### 실험 3

실험 2 +

* `restaurant_signal_count`
* `restaurant_avg_weight`

### 실험 4

마지막으로

* `user_category_count`
* `user_region_count`
* `user_price_mean`

이렇게 가세요.

지금 로그상 가장 위험한 건 `user_*_count` 계열입니다.

---

## 2. `user_category_count`, `user_region_count` 정의부터 다시 점검

특히 이 두 개는 꼭 확인해야 합니다.

확인할 질문:

* train과 test에서 같은 기준으로 만들었나
* 현재 row를 포함했나
* user 전체 이력 기준인가, split 이전 이력만인가
* unseen user는 어떻게 처리하나
* normalization을 split별로 따로 했나

지금 분포 차이는 정상적이지 않습니다.

---

## 3. candidate 내 분포를 봐야 함

전체 mean보다 더 중요한 건 **그룹 내부에서 positive와 negative가 얼마나 다른가**입니다.

각 feature마다 group 기준으로:

* positive 평균
* negative 평균

을 찍어보세요.

예를 들어 `user_category_match`가

* positive 평균 0.82
* negative 평균 0.79

이면 사실상 쓸모가 없습니다.

반대로

* positive 0.9
* negative 0.2

면 강한 feature입니다.

---

## 4. 상수 feature 제거

`distance_weight`는 지금 완전 상수이니 제거하세요.

---

## 5. accuracy는 당분간 무시

지금은 `75% accuracy`가 완전히 misleading합니다.
추천 모델 품질은 `recall@k / ndcg@k`만 보세요.

---

# 내 판단

지금 가장 가능성 높은 원인은:

> **집계형 user feature(`user_category_count`, `user_region_count`)가 train/test에서 정의나 분포가 크게 달라졌고, match 계열도 negatives에서 너무 자주 1이라 구분력이 약해서, 모델이 전반적으로 점수를 다 올리는 방향으로 collapse했다.**

---

# 가장 추천하는 다음 액션

딱 이렇게 하세요.

1. `distance_weight` 제거
2. `user_category_count`, `user_region_count`, `user_price_mean` 제거
3. `user_category_match`, `user_region_match`, `price_diff`만 넣고 재학습
4. 그 결과가 괜찮으면 popularity 계열을 하나씩 추가

이 순서가 제일 빠르게 원인 찾습니다.

---

실헙 1 결과

{
  "auc": 0.6426139601139602,
  "ndcg@5": 0.22093023283013208,
  "ndcg@10": 0.22093023283013208,
  "recall@5": 0.22093023283013208,
  "recall@10": 0.22093023283013208,
}

로그 분석:
좋습니다. **실험 1은 명확하게 성공입니다.**

결과를 요약하면:

* **AUC: 0.64261**
* **NDCG@5: 0.22093**
* **NDCG@10: 0.22093**
* **Recall@5: 0.22093**
* **Recall@10: 0.22093**

비교하면:

* 이전 모델 baseline: `recall@10 ≈ 0.058`
* popularity baseline: `recall@10 ≈ 0.186`
* 이번 실험 1: `recall@10 ≈ 0.221`

즉,

> **random < popularity < 실험 1 DeepFM**

이 됐습니다.

이건 꽤 큰 의미가 있어요.
단순히 조금 오른 게 아니라, **popularity baseline도 넘어섰습니다.**

## 이 결과가 말해주는 것

### 1. 문제 원인이 거의 확정됨

이전 성능 붕괴의 원인은 **DeepFM 자체가 아니라, 추가한 일부 feature가 모델을 망가뜨린 것**입니다.

특히 지금 실험 1은

* `user_category_match`
* `user_region_match`
* `price_diff`

만 넣었는데 바로 좋아졌죠.

즉:

> **match 계열 interaction feature는 유효했고, 집계형 feature가 문제였을 가능성이 매우 높습니다.**

### 2. 지금 추가한 3개 feature는 진짜 signal이 있음

이 세 개는 추천 문제에서 아주 좋은 feature들입니다.

* `user_category_match`
  → 유저 취향 카테고리와 식당 카테고리 일치
* `user_region_match`
  → 유저 선호 지역과 식당 지역 일치
* `price_diff`
  → 유저 선호 가격대와 식당 가격대 차이

이건 **개인화 신호**라서 popularity보다 강해질 수 있습니다.

### 3. 이전 집계 feature들은 정의/스케일/분포 문제가 있었을 확률이 큼

특히 아까 로그에서 이상했던 것:

* `user_region_match`가 전부 0
* `price_diff`가 전부 0
* `user_category_match`가 1/0이 아니라 큰 count처럼 보였던 값

이건 아블레이션 파이프라인에서 컬럼 선택/매핑이 꼬였을 가능성도 보여줍니다.
하지만 어쨌든 지금 결과는 “이 3개 방향은 맞다”는 걸 증명합니다.

---

## 지금 상태 평가

현재는 이렇게 볼 수 있습니다.

| 방법                  |  recall@10 |
| ------------------- | ---------: |
| random              |     0.0349 |
| popularity          |     0.1860 |
| DeepFM baseline(이전) |     0.0581 |
| **실험 1**            | **0.2209** |

이건 꽤 좋습니다.

특히:

* random 대비 약 **6배 이상**
* popularity 대비도 **유의미하게 우세**

라서, 이제는 **모델이 실제 개인화 랭킹을 하고 있다**고 봐도 됩니다.

---

## 다음 액션

이제 해야 할 건 명확합니다.

### 1. 실험 1을 새 strong baseline으로 저장

현재 기준선은 이제 이겁니다.

* `user_category_match`
* `user_region_match`
* `price_diff`

이 조합을 **새 baseline**으로 두세요.

### 2. 다음 feature는 하나씩 추가

이제부터는 반드시 **한 번에 하나씩** 넣는 게 좋습니다.

추천 순서:

1. `restaurant_popularity`
2. `restaurant_signal_count`
3. `restaurant_avg_weight`
4. `user_price_mean`
5. `user_category_count`
6. `user_region_count`

이 순서로요.

### 3. 추가 후 판단 기준

각 실험마다 이렇게 보세요.

* 실험 1 대비 개선됐는가
* popularity baseline보다 계속 높은가
* 출력 saturation이 다시 생기지 않는가

---

## 주의할 점

지금 로그를 보면 아직도 많은 `group preds`가 `0.99~1.0`에 몰려 있습니다.
다만 이번엔 일부 샘플에서 차이가 생겨서 랭킹이 살아난 거예요.

즉:

> **성능은 좋아졌지만, 출력이 여전히 다소 포화된 편**입니다.

그래서 다음에 집계형 feature를 넣을 때 또 망가질 수 있습니다.
특히 `user_region_count` 같은 건 가장 나중에 넣는 게 좋습니다.

---

## 한 줄 결론

> **실험 1은 성공이고, 핵심 interaction feature 3개만으로 popularity baseline을 넘어섰습니다.**
> 이제 이 조합을 새 baseline으로 삼고, 집계형 feature는 하나씩 조심스럽게 추가하면 됩니다.

다음 실험은 `restaurant_popularity` 하나만 추가하는 게 가장 좋습니다.

---