문제

candidate set은 1개이고, 그 안에 후보는 2,283개입니다.
recommendation_id가 비어 있어서 evaluate에서 rec_id가 전부 같은 값(nan 또는 fallback 시 "single")이 되고, 그래서 한 덩어리로만 평가되는 상태입니다.

--> 그래서 recall 값이 낮음.

run_menifest.json

{
  "pipeline_version": "deepfm-1.0.20260306022948",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.5182146432146433,
    "ndcg@5": 0.5423762067744511,
    "ndcg@10": 0.4692176943526925,
    "recall@5": 0.005005005005005005,
    "recall@10": 0.01001001001001001
  },
  "timestamp_utc": "2026-03-06T02:29:49.074453+00:00",
  "wandb_run_id": "5uopxs7q"
}

---

해결

전처리에서 recommendation_id를 u_{user_id} / a_{anon_id} 로 채우는 수정 이후로 다시 전처리·학습을 돌리면, 유저별로 여러 개의 candidate set이 생기고 그룹 개수와 그룹당 후보 개수가 나뉘게 됩니다.