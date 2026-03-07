"""
data_designe/evaluation_strategy.md 반영 평가.

- Primary: NDCG@K (binary relevance=label), 모델 점수로 재정렬 후 계산.
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

    - NDCG/Recall의 relevance는 binary(label) 사용 (positive=1, negative=0)
    - sample_weight는 metric 평균을 그룹 단위 가중평균할 때만 사용 (gain으로 사용하지 않음)
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
    group_weights = {f"ndcg@{k}": [] for k in k_list}
    group_weights.update({f"recall@{k}": [] for k in k_list})

    warm_ndcg = {f"ndcg@{k}": [] for k in k_list}
    warm_recall = {f"recall@{k}": [] for k in k_list}
    warm_w = {f"ndcg@{k}": [] for k in k_list}
    warm_w.update({f"recall@{k}": [] for k in k_list})

    user_cold_ndcg = {f"ndcg@{k}": [] for k in k_list}
    user_cold_w = {f"ndcg@{k}": [] for k in k_list}
    item_cold_ndcg = {f"ndcg@{k}": [] for k in k_list}
    item_cold_w = {f"ndcg@{k}": [] for k in k_list}

    n_preds = len(preds)
    for name, idx in groups.indices.items():
        if name == "" or len(idx) == 0:
            continue
        idx = np.asarray(idx)
        idx = idx[idx < n_preds]
        if len(idx) == 0:
            continue
        order = np.argsort(-preds[idx])
        rel_bin = (y_true[idx][order] >= 0.5).astype(int).tolist()

        # 그룹 가중치: 해당 그룹의 positive 샘플 weight 중 최대값(없으면 1.0)
        pos_mask = y_true[idx] >= 0.5
        if np.any(pos_mask):
            gw = float(np.max(weights[idx][pos_mask]))
        else:
            # positive가 없으면 NDCG/Recall 의미가 없어 skip (eval list는 보통 1 pos 보장)
            continue
        if not np.isfinite(gw) or gw <= 0:
            gw = 1.0

        for k in k_list:
            n = ndcg_at_k(rel_bin, k)
            n_rel = int(np.sum(y_true[idx] >= 0.5))
            r = 0.0 if n_rel <= 0 else float(np.sum(np.asarray(rel_bin)[:k]) / n_rel)
            ndcg_results[f"ndcg@{k}"].append(n)
            recall_results[f"recall@{k}"].append(min(1.0, r))
            group_weights[f"ndcg@{k}"].append(gw)
            group_weights[f"recall@{k}"].append(gw)

        if meta is not None and len(meta) == len(y_true):
            u = str(meta.iloc[idx[0]].get("user_id", ""))
            r = str(meta.iloc[idx[0]].get("restaurant_id", ""))
            in_train_user = u in train_user_ids
            in_train_item = r in train_restaurant_ids
            for k in k_list:
                n = ndcg_results[f"ndcg@{k}"][-1]
                rec = recall_results[f"recall@{k}"][-1]
                if in_train_user and in_train_item:
                    warm_ndcg[f"ndcg@{k}"].append(n)
                    warm_recall[f"recall@{k}"].append(rec)
                    warm_w[f"ndcg@{k}"].append(gw)
                    warm_w[f"recall@{k}"].append(gw)
                elif not in_train_user:
                    user_cold_ndcg[f"ndcg@{k}"].append(n)
                    user_cold_w[f"ndcg@{k}"].append(gw)
                elif not in_train_item:
                    item_cold_ndcg[f"ndcg@{k}"].append(n)
                    item_cold_w[f"ndcg@{k}"].append(gw)

    auc = _auc(y_true, preds)

    def _wmean(vals: list[float], wts: list[float]) -> float:
        if not vals:
            return 0.0
        w = np.asarray(wts, dtype=np.float64)
        v = np.asarray(vals, dtype=np.float64)
        if len(w) != len(v):
            return float(np.mean(v))
        s = float(np.sum(w))
        if s <= 0:
            return float(np.mean(v))
        return float(np.sum(v * w) / s)

    result = {
        "auc": float(auc),
        **{k: _wmean(v, group_weights[k]) for k, v in ndcg_results.items()},
        **{k: _wmean(v, group_weights[k]) for k, v in recall_results.items()},
    }
    if warm_ndcg["ndcg@5"]:
        result["warm"] = {
            **{k: _wmean(v, warm_w[k]) for k, v in warm_ndcg.items()},
            **{k: _wmean(v, warm_w[k]) for k, v in warm_recall.items()},
        }
    if user_cold_ndcg["ndcg@5"]:
        result["user_cold"] = {k: _wmean(v, user_cold_w[k]) for k, v in user_cold_ndcg.items()}
    if item_cold_ndcg["ndcg@5"]:
        result["item_cold"] = {k: _wmean(v, item_cold_w[k]) for k, v in item_cold_ndcg.items()}
    return result


def run_popularity_baseline(
    processed_data_dir: str,
    k_list: list[int] | None = None,
) -> dict:
    """
    같은 평가셋에서 popularity baseline NDCG@K, Recall@K 계산.
    score = restaurant_popularity[item] 로 랭킹 후 binary relevance로 NDCG/Recall 산출.
    split_meta.json에 restaurant_positive_counts가 있어야 함 (전처리 시 저장).
    """
    k_list = k_list or [5, 10]
    root = Path(processed_data_dir)
    test_path = root / "test.txt"
    meta_path = root / "test_meta.csv"
    split_meta_path = root / "split_meta.json"
    sizes_path = root / "feature_sizes.txt"

    if not test_path.exists():
        return {"error": "test.txt not found"}
    if not split_meta_path.exists():
        return {"error": "split_meta.json not found"}

    with open(split_meta_path, encoding="utf-8") as f:
        split_meta = json.load(f)
    popularity: dict[str, int] = split_meta.get("restaurant_positive_counts") or {}
    if not popularity:
        return {"error": "restaurant_positive_counts not in split_meta (re-run preprocess)"}

    feature_sizes = [int(x) for x in np.loadtxt(str(sizes_path), delimiter=",").ravel()] if sizes_path.exists() else []
    if not feature_sizes:
        return {"error": "feature_sizes.txt not found"}
    n_fields = len(feature_sizes)

    data = pd.read_csv(test_path)
    if data.shape[1] <= n_fields:
        return {"error": "test.txt has no label column"}
    y_true = data.iloc[:, n_fields].values.astype(np.float32)
    weights = data.iloc[:, n_fields + 1].values.astype(np.float32) if data.shape[1] > n_fields + 1 else np.ones_like(y_true)

    meta = pd.read_csv(meta_path) if meta_path.exists() else None
    if meta is None or "recommendation_id" not in meta.columns or "restaurant_id" not in meta.columns:
        return {"error": "test_meta.csv with recommendation_id and restaurant_id required"}

    rec_id = meta["recommendation_id"].astype(str)
    groups = rec_id.groupby(rec_id)

    ndcg_results = {f"ndcg@{k}": [] for k in k_list}
    recall_results = {f"recall@{k}": [] for k in k_list}
    group_weights = {f"ndcg@{k}": [] for k in k_list}
    group_weights.update({f"recall@{k}": [] for k in k_list})

    n_rows = len(y_true)
    for name, idx in groups.indices.items():
        if name == "" or len(idx) == 0:
            continue
        idx = np.asarray(idx)
        idx = idx[idx < n_rows]
        if len(idx) == 0:
            continue
        pos_mask = y_true[idx] >= 0.5
        if not np.any(pos_mask):
            continue
        gw = float(np.max(weights[idx][pos_mask]))
        if not np.isfinite(gw) or gw <= 0:
            gw = 1.0

        restaurant_ids = [str(meta.iloc[i].get("restaurant_id", "") or "").strip() for i in idx]
        scores = np.array([float(popularity.get(rid, 0)) for rid in restaurant_ids])
        order = np.argsort(-scores)
        rel_bin = (y_true[idx][order] >= 0.5).astype(int).tolist()
        n_rel = int(np.sum(y_true[idx] >= 0.5))

        for k in k_list:
            n = ndcg_at_k(rel_bin, k)
            r = 0.0 if n_rel <= 0 else float(np.sum(np.asarray(rel_bin)[:k]) / n_rel)
            ndcg_results[f"ndcg@{k}"].append(n)
            recall_results[f"recall@{k}"].append(min(1.0, r))
            group_weights[f"ndcg@{k}"].append(gw)
            group_weights[f"recall@{k}"].append(gw)

    def _wmean(vals: list[float], wts: list[float]) -> float:
        if not vals:
            return 0.0
        w = np.asarray(wts, dtype=np.float64)
        v = np.asarray(vals, dtype=np.float64)
        if len(w) != len(v):
            return float(np.mean(v))
        s = float(np.sum(w))
        if s <= 0:
            return float(np.mean(v))
        return float(np.sum(v * w) / s)

    return {
        **{k: _wmean(v, group_weights[k]) for k, v in ndcg_results.items()},
        **{k: _wmean(v, group_weights[k]) for k, v in recall_results.items()},
    }


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if np.unique(y_true).size < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))
