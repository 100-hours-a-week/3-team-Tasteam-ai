"""
Tasteam DeepFM용 데이터 전처리.

tasteam_deepfm_data.md + docs/data_designe 반영:
- 연속형: taste_preferences(4), visit_time_distribution(4), is_anonymous(1), pref_w_1~3(3) = 12개
- 범주형: user_id, anon_cohort_id, avg_price_tier, restaurant_id, primary_category,
  pref_cat_1~3, price_tier, region_gu, region_dong, geohash, day_of_week, time_slot,
  admin_dong, distance_bucket, weather_bucket, dining_type, first_positive_segment,
  first_comparison_tag = 20개
- 시간 기준 split, recommendation 단위 보존, sample_weight(옵션 C), warm/cold 메타 출력
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# --- 연속형 피처 ---
CONTINUOUS_FEATURE_KEYS = [
    "taste_spicy", "taste_sweet", "taste_savory", "taste_light",
]
VISIT_TIME_KEYS = ["breakfast", "lunch", "afternoon", "dinner"]
PREF_WEIGHT_KEYS = ["pref_w_1", "pref_w_2", "pref_w_3"]
CONTINUOUS_FEATURES = (
    CONTINUOUS_FEATURE_KEYS + VISIT_TIME_KEYS + ["is_anonymous"] + PREF_WEIGHT_KEYS
)  # 12개

# --- 범주형 피처 (user_id / anon_cohort_id 분리, preferred_categories Top-K) ---
CATEGORICAL_FEATURES = [
    "user_id",
    "anon_cohort_id",
    "avg_price_tier",
    "restaurant_id",
    "primary_category",
    "pref_cat_1",
    "pref_cat_2",
    "pref_cat_3",
    "price_tier",
    "region_gu",
    "region_dong",
    "geohash",
    "day_of_week",
    "time_slot",
    "admin_dong",
    "distance_bucket",
    "weather_bucket",
    "dining_type",
    "first_positive_segment",
    "first_comparison_tag",
]
PREF_CAT_K = 3

SIGNAL_WEIGHTS = {
    "REVIEW": 1.0, "CALL": 0.8, "ROUTE": 0.7,
    "SAVE": 0.6, "SHARE": 0.4, "CLICK": 0.2,
}


def _safe_json(s: Any) -> Any:
    if s is None or (isinstance(s, str) and s.strip() == ""):
        return None
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if isinstance(s, str) else s
    except (json.JSONDecodeError, TypeError):
        return None


def _get_taste_preferences(row: dict) -> dict[str, float]:
    raw = _safe_json(row.get("taste_preferences"))
    if not isinstance(raw, dict):
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _get_visit_time_distribution(row: dict) -> dict[str, float]:
    raw = _safe_json(row.get("visit_time_distribution"))
    if not isinstance(raw, dict):
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def _get_preferred_categories_topk(
    row: dict, k: int = PREF_CAT_K
) -> tuple[list[str], list[float]]:
    """preferred_categories JSONB → 상위 K개 (카테고리 리스트, 가중치 리스트). 가중치 합=1 정규화."""
    raw = _safe_json(row.get("preferred_categories"))
    if not isinstance(raw, list) or len(raw) == 0:
        return [""] * k, [0.0] * k
    items = []
    for x in raw[: 2 * k]:
        if isinstance(x, dict):
            cat = x.get("category") or x.get("cat")
            w = x.get("weight") or x.get("w")
            if cat is not None and w is not None:
                try:
                    items.append((str(cat).strip(), float(w)))
                except (TypeError, ValueError):
                    pass
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            try:
                items.append((str(x[0]).strip(), float(x[1])))
            except (TypeError, ValueError, IndexError):
                pass
    items.sort(key=lambda t: -t[1])
    items = items[:k]
    if not items:
        return [""] * k, [0.0] * k
    cats = [t[0] or "" for t in items]
    weights = [max(0.0, t[1]) for t in items]
    total = sum(weights)
    if total <= 0:
        return cats + [""] * (k - len(cats)), [0.0] * k
    weights = [w / total for w in weights]
    while len(cats) < k:
        cats.append("")
        weights.append(0.0)
    return cats[:k], weights[:k]


def _get_primary_category(row: dict) -> str:
    raw = _safe_json(row.get("categories"))
    if isinstance(raw, list) and len(raw) > 0:
        return str(raw[0]) if raw[0] is not None else ""
    if isinstance(raw, str):
        raw = _safe_json(raw)
        if isinstance(raw, list) and len(raw) > 0:
            return str(raw[0])
    return ""


def _get_first_positive_segment(row: dict) -> str:
    raw = _safe_json(row.get("positive_segments"))
    if isinstance(raw, list) and len(raw) > 0:
        return str(raw[0]) if raw[0] is not None else ""
    return ""


def _get_first_comparison_tag(row: dict) -> str:
    raw = _safe_json(row.get("comparison_tags"))
    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, dict) and "tag" in first:
            return str(first["tag"])
        return str(first) if first is not None else ""
    return ""


def _user_id_str(row: dict) -> str:
    uid = row.get("user_id")
    if uid is not None and str(uid).strip() != "":
        return f"u_{uid}"
    return ""


def _anon_cohort_str(row: dict) -> str:
    aid = row.get("anonymous_cohort_id") or row.get("anonymous_id")
    if aid is not None and str(aid).strip() != "":
        return f"a_{aid}"
    return ""


def _is_anonymous(row: dict) -> float:
    uid = row.get("user_id")
    if uid is not None and str(uid).strip() != "":
        return 0.0
    return 1.0


def _extract_continuous(row: dict) -> list[float]:
    taste = _get_taste_preferences(row)
    visit = _get_visit_time_distribution(row)
    out = []
    for key in CONTINUOUS_FEATURE_KEYS:
        k = key.replace("taste_", "")
        v = taste.get(k, 0.0)
        out.append(max(0.0, min(1.0, float(v))))
    for k in VISIT_TIME_KEYS:
        v = visit.get(k, 0.0)
        out.append(max(0.0, min(1.0, float(v))))
    out.append(_is_anonymous(row))
    _, pref_weights = _get_preferred_categories_topk(row)
    out.extend(pref_weights)
    return out


def _extract_categorical(row: dict) -> list[str]:
    pref_cats, _ = _get_preferred_categories_topk(row)
    return [
        _user_id_str(row),
        _anon_cohort_str(row),
        str(row.get("avg_price_tier") or "").strip(),
        str(row.get("restaurant_id") or "").strip(),
        _get_primary_category(row),
        pref_cats[0],
        pref_cats[1],
        pref_cats[2],
        str(row.get("price_tier") or "").strip(),
        str(row.get("region_gu") or "").strip(),
        str(row.get("region_dong") or "").strip(),
        str(row.get("geohash") or "").strip(),
        str(row.get("day_of_week") or "").strip(),
        str(row.get("time_slot") or "").strip(),
        str(row.get("admin_dong") or "").strip(),
        str(row.get("distance_bucket") or "").strip(),
        str(row.get("weather_bucket") or "").strip(),
        str(row.get("dining_type") or "").strip(),
        _get_first_positive_segment(row),
        _get_first_comparison_tag(row),
    ]


class CategoryDictGenerator:
    def __init__(self, num_feature: int):
        self.dicts: list[dict[str, int]] = []
        self.num_feature = num_feature
        for _ in range(num_feature):
            self.dicts.append(defaultdict(int))

    def build(self, rows: list[dict], cutoff: int = 0) -> None:
        for row in rows:
            vals = _extract_categorical(row)
            for i in range(self.num_feature):
                if i < len(vals):
                    v = vals[i].strip() if vals[i] else ""
                    if v:
                        self.dicts[i][v] += 1
        for i in range(self.num_feature):
            filtered = [(k, c) for k, c in self.dicts[i].items() if c >= cutoff]
            filtered.sort(key=lambda x: (-x[1], x[0]))
            vocabs = [x[0] for x in filtered]
            self.dicts[i] = {v: idx for idx, v in enumerate(vocabs, start=1)}
            self.dicts[i]["<unk>"] = 0

    def gen(self, idx: int, key: str) -> int:
        return self.dicts[idx].get(key or "", self.dicts[idx]["<unk>"])

    def dicts_sizes(self) -> list[int]:
        return [len(self.dicts[i]) for i in range(self.num_feature)]


def _label_from_row(row: dict) -> str:
    w = row.get("weight")
    if w is not None:
        try:
            return "1" if float(w) >= 0.5 else "0"
        except (TypeError, ValueError):
            pass
    if row.get("label") is not None:
        return str(int(row["label"]))
    return "1"


def _sample_weight_from_row(row: dict) -> float:
    """옵션 C: 이진 라벨 + sample_weight. 없으면 1.0 (negative도 1.0)."""
    w = row.get("weight")
    if w is not None:
        try:
            return max(0.0, min(1.0, float(w)))
        except (TypeError, ValueError):
            pass
    return 1.0


def _parse_ts(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(v))
        except (OSError, ValueError):
            return None
    try:
        return pd.Timestamp(v).to_pydatetime()
    except Exception:
        return None


def _time_based_split(
    df: pd.DataFrame,
    time_column: str,
    train_end: str,
    valid_end: str,
    test_end: str,
    group_column: str | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """시간 구간으로 train/val/test 분할. group_column이 있으면 같은 그룹은 같은 구간으로."""
    train_end_dt = _parse_ts(train_end) or datetime.min
    valid_end_dt = _parse_ts(valid_end) or datetime.max
    test_end_dt = _parse_ts(test_end) or datetime.max
    if time_column not in df.columns:
        return df.to_dict("records"), [], []

    def bucket(ts: datetime | None) -> str:
        if ts is None:
            return "train"
        if ts < train_end_dt:
            return "train"
        if ts < valid_end_dt:
            return "valid"
        if ts < test_end_dt:
            return "test"
        return "test"

    if group_column and group_column in df.columns:
        df = df.copy()
        df["_ts"] = pd.to_datetime(df[time_column], errors="coerce")
        df["_bucket"] = df["_ts"].apply(bucket)
        g = df.groupby(group_column, sort=False)
        group_bucket = g["_ts"].max().apply(bucket)
        df["_assign"] = df[group_column].map(group_bucket)
        train_rows = df[df["_assign"] == "train"].drop(columns=["_ts", "_bucket", "_assign"]).to_dict("records")
        val_rows = df[df["_assign"] == "valid"].drop(columns=["_ts", "_bucket", "_assign"]).to_dict("records")
        test_rows = df[df["_assign"] == "test"].drop(columns=["_ts", "_bucket", "_assign"]).to_dict("records")
    else:
        df = df.copy()
        df["_ts"] = pd.to_datetime(df[time_column], errors="coerce")
        df["_bucket"] = df["_ts"].apply(bucket)
        train_rows = df[df["_bucket"] == "train"].drop(columns=["_ts", "_bucket"]).to_dict("records")
        val_rows = df[df["_bucket"] == "valid"].drop(columns=["_ts", "_bucket"]).to_dict("records")
        test_rows = df[df["_bucket"] == "test"].drop(columns=["_ts", "_bucket"]).to_dict("records")
    return train_rows, val_rows, test_rows


def _load_all_rows(datadir: str, filenames: list[str], limit: int | None) -> list[dict]:
    out = []
    for name in filenames:
        path = os.path.join(datadir, name)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, nrows=limit)
        out.extend(df.to_dict("records"))
    return out


def preprocess(
    datadir: str,
    outdir: str,
    num_train_sample: int | None = 10000,
    num_test_sample: int | None = 10000,
    categorical_cutoff: int = 2,
    use_sample_weight: bool = True,
    time_column: str | None = None,
    train_end: str | None = None,
    valid_end: str | None = None,
    test_end: str | None = None,
    group_column: str | None = None,
) -> None:
    """
    Tasteam + data_designe 반영 전처리.

    - use_sample_weight: True면 train/val/test 마지막에 sample_weight 컬럼 추가 (옵션 C).
    - time_column: 있으면 시간 기준 split (train_end, valid_end, test_end 구간 사용).
    - group_column: recommendation_id 또는 generated_at 등, 같은 값은 같은 구간으로.
    - 시간 split 시 train.csv만 있어도 됨. 없으면 train.csv / test.csv 행 그대로 사용.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if time_column and train_end and valid_end and test_end:
        df = pd.read_csv(os.path.join(datadir, "train.csv"))
        if os.path.exists(os.path.join(datadir, "test.csv")):
            df_test = pd.read_csv(os.path.join(datadir, "test.csv"))
            df = pd.concat([df, df_test], ignore_index=True)
        train_rows, val_rows, test_rows = _time_based_split(
            df, time_column, train_end, valid_end, test_end, group_column
        )
        if num_train_sample is not None:
            train_rows = train_rows[:num_train_sample]
        if num_test_sample is not None:
            test_rows = test_rows[:num_test_sample]
    else:
        train_rows = _load_all_rows(datadir, ["train.csv"], num_train_sample)
        test_rows = _load_all_rows(datadir, ["test.csv"], num_test_sample)
        val_rows = []
        if not train_rows and test_rows:
            train_rows, test_rows = test_rows, []
        if not train_rows:
            raise FileNotFoundError(f"데이터 없음: {datadir}")

    dicts = CategoryDictGenerator(len(CATEGORICAL_FEATURES))
    dicts.build(train_rows, cutoff=categorical_cutoff)
    dict_sizes = dicts.dicts_sizes()
    sizes = [1] * len(CONTINUOUS_FEATURES) + dict_sizes
    with open(os.path.join(outdir, "feature_sizes.txt"), "w") as f:
        f.write(",".join(map(str, sizes)))

    def write_rows(rows: list[dict], path: str, include_sw: bool) -> None:
        with open(os.path.join(outdir, path), "w") as f:
            for row in rows:
                cont = _extract_continuous(row)
                cont_str = ",".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in cont)
                cat_vals = _extract_categorical(row)
                cat_str = ",".join(str(dicts.gen(i, cat_vals[i])) for i in range(len(cat_vals)))
                label = _label_from_row(row)
                parts = [cont_str, cat_str, label]
                if include_sw:
                    parts.append(f"{_sample_weight_from_row(row):.6f}")
                f.write(",".join(parts) + "\n")

    write_rows(train_rows, "train.txt", use_sample_weight)
    if val_rows:
        write_rows(val_rows, "val.txt", use_sample_weight)
    write_rows(test_rows, "test.txt", use_sample_weight)

    if not test_rows and train_rows:
        n = min(num_test_sample or 1000, len(train_rows))
        write_rows(train_rows[:n], "test.txt", use_sample_weight)

    train_user_ids = set()
    train_restaurant_ids = set()
    for row in train_rows:
        uid = row.get("user_id")
        if uid is not None and str(uid).strip():
            train_user_ids.add(str(uid))
        rid = row.get("restaurant_id")
        if rid is not None and str(rid).strip():
            train_restaurant_ids.add(str(rid))
    split_meta = {
        "train_end": train_end,
        "valid_end": valid_end,
        "test_end": test_end,
        "time_column": time_column,
        "group_column": group_column,
        "train_user_ids": list(train_user_ids),
        "train_restaurant_ids": list(train_restaurant_ids),
        "use_sample_weight": use_sample_weight,
    }
    with open(os.path.join(outdir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    if test_rows:
        meta_rows = []
        for row in test_rows:
            meta_rows.append({
                "user_id": row.get("user_id") or "",
                "restaurant_id": row.get("restaurant_id") or "",
                "recommendation_id": row.get("recommendation_id") or row.get("generated_at") or "",
            })
        pd.DataFrame(meta_rows).to_csv(
            os.path.join(outdir, "test_meta.csv"), index=False
        )


if __name__ == "__main__":
    preprocess("../data/raw", "../data")
