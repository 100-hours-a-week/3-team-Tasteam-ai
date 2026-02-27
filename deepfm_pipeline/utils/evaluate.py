"""
data_designe/evaluation_strategy.md 반영 평가.

- Primary: Weighted NDCG@K (gain=weight), 모델 점수로 재정렬 후 계산.
- Secondary: Recall@K.
- Monitoring: AUC.
- Warm / User-Cold / Item-Cold 구간별 리포트.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
from data.dataset import _get_continuous_feature_count, _get_total_field_count
from model.DeepFM import DeepFM


def dcg_at_k(rel: list[float], k: int) -> float:
    """rel: relevance (gain) list, top-k. rel[0] = highest ranked."""
    rel = np.asarray(rel[:k])
    if len(rel) == 0:
        return 0.0
    denom = np.log2(np.arange(2, len(rel) + 2))
    return np.sum(rel / denom)


def ndcg_at_k(rel: list[float], k: int) -> float:
    """NDCG@K. rel = relevance (gain) in predicted order. Ideal = sorted desc."""
    dcg = dcg_at_k(rel, k)
    ideal = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def recall_at_k(rel_binary: list[int], k: int) -> float:
    """rel_binary: 1 if relevant. Recall@K = (relevant in top-k) / total relevant."""
    rel_binary = np.asarray(rel_binary[:k])
    total_relevant = sum(rel_binary)
    if total_relevant == 0:
        return 0.0
    return float(np.sum(rel_binary) / total_relevant)


def run_evaluation(
    processed_data_dir: str,
    model_path: str,
    feature_sizes: list[int],
    k_list: list[int] | None = None,
    use_cuda: bool = False,
    batch_size: int = 256,
) -> dict:
    """
    test.txt + test_meta.csv + split_meta.json 기준으로
    모델 점수 재정렬 후 NDCG@K, Recall@K, AUC 계산.
    recommendation_id로 그룹 지어 리스트 단위 평가.
    """
    k_list = k_list or [5, 10]
    root = Path(processed_data_dir)
    test_path = root / "test.txt"
    meta_path = root / "test_meta.csv"
    split_meta_path = root / "split_meta.json"

    if not test_path.exists():
        return {"error": "test.txt not found"}

    data = pd.read_csv(test_path)
    n_fields = len(feature_sizes)
    if data.shape[1] <= n_fields:
        return {"error": "test.txt has no label column"}

    X = data.iloc[:, :n_fields].values
    y_true = data.iloc[:, n_fields].values.astype(np.float32)
    weights = data.iloc[:, n_fields + 1].values.astype(np.float32) if data.shape[1] > n_fields + 1 else np.ones_like(y_true)

    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    split_meta = {}
    if split_meta_path.exists():
        with open(split_meta_path, encoding="utf-8") as f:
            split_meta = json.load(f)
    train_user_ids = set(split_meta.get("train_user_ids", []))
    train_restaurant_ids = set(split_meta.get("train_restaurant_ids", []))

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = DeepFM(feature_sizes, use_cuda=use_cuda)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_cont = _get_continuous_feature_count(str(root))
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            Xi_cont = np.zeros((len(batch), n_cont))
            Xi_cat = batch[:, n_cont:]
            Xi = torch.from_numpy(
                np.concatenate([Xi_cont, Xi_cat], axis=1).astype(np.int32)
            ).unsqueeze(-1).to(device)
            Xv_cont = batch[:, :n_cont].astype(np.float32)
            Xv_cat = np.ones_like(Xi_cat, dtype=np.float32)
            Xv = torch.from_numpy(
                np.concatenate([Xv_cont, Xv_cat], axis=1)
            ).to(device)
            out = model(Xi, Xv)
            preds.append(torch.sigmoid(out).cpu().numpy().ravel())
    preds = np.concatenate(preds)

    if meta is not None and "recommendation_id" in meta.columns and meta["recommendation_id"].astype(str).str.strip().ne("").any():
        rec_id = meta["recommendation_id"].astype(str)
        groups = rec_id.groupby(rec_id)
    else:
        rec_id = pd.Series(["single"] * len(y_true))
        groups = rec_id.groupby(rec_id)

    ndcg_results = {f"ndcg@{k}": [] for k in k_list}
    recall_results = {f"recall@{k}": [] for k in k_list}
    warm_ndcg = {f"ndcg@{k}": [] for k in k_list}
    warm_recall = {f"recall@{k}": [] for k in k_list}
    user_cold_ndcg = {f"ndcg@{k}": [] for k in k_list}
    item_cold_ndcg = {f"ndcg@{k}": [] for k in k_list}

    for name, idx in groups.indices.items():
        if name == "" or len(idx) == 0:
            continue
        order = np.argsort(-preds[idx])
        rel = weights[idx][order].tolist()
        rel_bin = (y_true[idx][order] >= 0.5).astype(int).tolist()
        for k in k_list:
            ndcg_results[f"ndcg@{k}"].append(ndcg_at_k(rel, k))
            n_rel = max(1, int(np.sum(y_true[idx] >= 0.5)))
            recall_results[f"recall@{k}"].append(min(1.0, np.sum((y_true[idx][order] >= 0.5)[:k]) / n_rel))

        if meta is not None and len(meta) == len(y_true):
            u = str(meta.iloc[idx[0]].get("user_id", ""))
            r = str(meta.iloc[idx[0]].get("restaurant_id", ""))
            in_train_user = u in train_user_ids
            in_train_item = r in train_restaurant_ids
            for k in k_list:
                n = ndcg_at_k(rel, k)
                rec = recall_results[f"recall@{k}"][-1]
                if in_train_user and in_train_item:
                    warm_ndcg[f"ndcg@{k}"].append(n)
                    warm_recall[f"recall@{k}"].append(rec)
                elif not in_train_user:
                    user_cold_ndcg[f"ndcg@{k}"].append(n)
                elif not in_train_item:
                    item_cold_ndcg[f"ndcg@{k}"].append(n)

    auc = _auc(y_true, preds)
    result = {
        "auc": float(auc),
        **{k: float(np.mean(v)) if v else 0.0 for k, v in ndcg_results.items()},
        **{k: float(np.mean(v)) if v else 0.0 for k, v in recall_results.items()},
    }
    if warm_ndcg["ndcg@5"]:
        result["warm"] = {
            **{k: float(np.mean(v)) for k, v in warm_ndcg.items()},
            **{k: float(np.mean(v)) for k, v in warm_recall.items()},
        }
    if user_cold_ndcg["ndcg@5"]:
        result["user_cold"] = {k: float(np.mean(v)) for k, v in user_cold_ndcg.items()}
    if item_cold_ndcg["ndcg@5"]:
        result["item_cold"] = {k: float(np.mean(v)) for k, v in item_cold_ndcg.items()}
    return result


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if np.unique(y_true).size < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))
