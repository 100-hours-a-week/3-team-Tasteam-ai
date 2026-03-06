eval 결과 (default config)

{
  "pipeline_version": "deepfm-1.0.20260306052211",
  "model_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/model.pt",
  "feature_sizes_path": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/output/manual/feature_sizes.txt",
  "processed_data_dir": "/Users/js/tasteam-new-async/ml/deepfm_pipeline/data",
  "metrics": {
    "auc": 0.5657924536848303,
    "ndcg@5": 0.962276715924885,
    "ndcg@10": 0.963531299100964,
    "recall@5": 0.9962351147751664,
    "recall@10": 0.9994257823715187
  },
  "timestamp_utc": "2026-03-06T05:22:12.266233+00:00",
  "wandb_run_id": "7et41c3l"
}

---

--> condidate 전체(모든 condidate set을 합친 전체) 기준, positive 97%, negative 3%, 클래스 매우 불균형