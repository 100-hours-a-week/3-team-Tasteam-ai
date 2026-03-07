baseline eval 결과 (default config)

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

현재 eval용 candidate set의 positive/negative 구성을 정리하면 아래와 같습니다.

---

## 1. 전체 test (총 candidate 행)

| 구분 | 개수 | 비율 |
|------|------|------|
| **Positive (라벨 1)** | 2,008 | **약 97.3%** |
| **Negative (라벨 0)** | 55 | **약 2.7%** |
| **합계** | 2,063 | 100% |

→ negative는 negative sampling으로 추가된 것이고, 전체적으로 positive가 압도적으로 많음.

---

## 2. recommendation_id 그룹별 (리스트 단위)

- **그룹 수**: 141개 (유저/anon별로 `u_*`, `a_syn_*` 등).
- **그룹당 후보 수**: 5~18개 정도.

**예시 (상위 몇 그룹)**  
- `u_52.0`: 18개 중 **positive 11, negative 7**  
- `u_4.0`: 12개 중 **positive 6, negative 6**  
- `u_3.0`: 8개 중 **positive 5, negative 3**  
- `a_syn_0431c72c-...`: 8개 중 **positive 2, negative 6**  
- 많은 `a_syn_*` 그룹: **positive 0, negative 5~7** (전부 음성 샘플만 있는 리스트)

---

## 3. 정리

- **전체**: 한 번에 보는 candidate set은 positive 97%, negative 3%로 매우 불균형.
- **그룹별**:  
  - 일부 그룹은 positive/negative가 둘 다 있음 (예: u_52, u_4, u_3).  
  - 일부 그룹은 negative만 있음 (positive 0, negative 5~7).  
- NDCG/Recall이 높은 이유: 그룹당 후보 수가 작고(5~18), positive가 있는 그룹에서는 상위 5~10개만 잘 맞춰도 recall이 쉽게 1에 가깝게 나옴.  
- AUC가 0.56인 이유: 전체 2,063개 중 negative가 55개뿐이라, 이진 분류 구분 난이도는 여전히 높은 편임.