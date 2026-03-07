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