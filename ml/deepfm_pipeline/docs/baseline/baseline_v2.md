feature:
python /Users/js/tasteam-new-async/ml/deepfm_pipeline/training_flow.py --exp-ablation "user_category_match,user_region_match,price_diff"


{
  "pipeline_version": "deepfm-1.0.20260307084041",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.6426139601139602,
    "ndcg@5": 0.22093023283013208,
    "ndcg@10": 0.22093023283013208,
    "recall@5": 0.22093023283013208,
    "recall@10": 0.22093023283013208,
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
  "timestamp_utc": "2026-03-07T08:40:42.757629+00:00",
  "wandb_run_id": "7vlg2u5p"
}