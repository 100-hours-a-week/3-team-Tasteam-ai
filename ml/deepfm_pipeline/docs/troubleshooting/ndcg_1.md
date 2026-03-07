{
  "pipeline_version": "deepfm-1.0.20260306055912",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.6168522835189503,
    "ndcg@5": 0.9497540438503487,
    "ndcg@10": 0.9673938125217519,
    "recall@5": 0.25925925925925924,
    "recall@10": 0.25925925925925924
  },
  "timestamp_utc": "2026-03-06T05:59:13.544033+00:00",
  "wandb_run_id": "zfrx8bsn"
}

---

ndcg가 1에 가까운 원인

지금 NDCG는 label이 아니라 weight를 gain으로 쓰고 있습니다. 그리고 전처리에서 test.txt는 ... + label + (sample_weight) 순서로 저장됩니다.

그래서 다음 상황이면 NDCG가 1.0에 가깝게 나올 수 있습니다.
그룹 내 weights가 거의 다 동일(예: 전부 1.0) → 어떤 순서로 정렬해도 DCG=IDCG라서 NDCG=1
negative의 weight가 1.0으로 들어가고, positive의 weight가 그보다 작거나 비슷 → “이상한 기준”으로는 완벽 정렬처럼 보일 수도 있음

---

수정 방안

좋아요. 지금 평가 로직을 “binary relevance = label”로 바꾸고, sample_weight는 말씀하신 대로 (1) loss 쪽은 그대로 두고 (2) metric 평균을 그룹 단위 가중평균으로만 쓰도록 evaluate.py를 수정하겠습니다

---

새로운 baseline (ndcg 정상화)

{
  "auc": 0.5011680911680912,
  "ndcg@5": 0.0697674419007603,
  "ndcg@10": 0.0697674419007603,
  "recall@5": 0.0697674419007603,
  "recall@10": 0.0697674419007603
}