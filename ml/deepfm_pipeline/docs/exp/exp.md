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