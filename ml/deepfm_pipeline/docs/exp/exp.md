1 pos, random neg 50, popular neg 50 (baseline) (실서비스와 유사)

{
  "pipeline_version": "deepfm-1.0.20260306065402",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.5011680911680912,
    "ndcg@5": 0.0697674419007603,
    "ndcg@10": 0.0697674419007603,
    "recall@5": 0.0697674419007603,
    "recall@10": 0.0697674419007603
  },
  "timestamp_utc": "2026-03-06T06:54:02.935354+00:00",
  "wandb_run_id": "v88jv97s"
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
