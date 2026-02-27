"""
Tasteam DeepFM용 데이터 전처리.

tasteam_deepfm_data.md의 피처 정의에 맞춰
user_feature, restaurant_feature, implicit_feedback, context_snapshot 기반
train/test CSV를 DeepFM 입력 형식(train.txt, test.txt, feature_sizes.txt)으로 변환한다.

- 연속형: taste_preferences(4), visit_time_distribution(4) → 8개
- 범주형: user_identifier, avg_price_tier, restaurant_id, primary_category, price_tier,
  region_gu, region_dong, geohash, day_of_week, time_slot, admin_dong,
  distance_bucket, weather_bucket, dining_type, first_positive_segment, first_comparison_tag → 16개
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# --- tasteam_deepfm_data.md 기반 피처 정의 ---

# 연속형 피처 (JSONB에서 추출한 수치, 0~1 클리핑)
CONTINUOUS_FEATURE_KEYS = [
    "taste_spicy",
    "taste_sweet",
    "taste_savory",
    "taste_light",
]
VISIT_TIME_KEYS = [
    "breakfast",
    "lunch",
    "afternoon",
    "dinner",
]
CONTINUOUS_FEATURES = CONTINUOUS_FEATURE_KEYS + VISIT_TIME_KEYS  # 8개

# 범주형 피처 (순서 = 출력 CSV 컬럼 순서)
CATEGORICAL_FEATURES = [
    "user_identifier",       # user_id 또는 anonymous_cohort_id
    "avg_price_tier",       # LOW/MID/HIGH/PREMIUM
    "restaurant_id",
    "primary_category",     # categories[0] 또는 첫 번째
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

# implicit_feedback 가중치 (라벨 이진화 시 threshold 참고)
SIGNAL_WEIGHTS = {
    "REVIEW": 1.0,
    "CALL": 0.8,
    "ROUTE": 0.7,
    "SAVE": 0.6,
    "SHARE": 0.4,
    "CLICK": 0.2,
}


def _safe_json(s: Any) -> Any:
    if s is None or (isinstance(s, str) and s.strip()) == "":
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


def _user_identifier(row: dict) -> str:
    uid = row.get("user_id")
    if uid is not None and str(uid).strip() != "":
        return f"u_{uid}"
    aid = row.get("anonymous_cohort_id") or row.get("anonymous_id")
    if aid is not None and str(aid).strip() != "":
        return f"a_{aid}"
    return ""


def _extract_continuous(row: dict) -> list[float]:
    taste = _get_taste_preferences(row)
    visit = _get_visit_time_distribution(row)
    out = []
    for k in CONTINUOUS_FEATURE_KEYS:
        key = k.replace("taste_", "")
        v = taste.get(key, 0.0)
        out.append(max(0.0, min(1.0, float(v))))
    for k in VISIT_TIME_KEYS:
        v = visit.get(k, 0.0)
        out.append(max(0.0, min(1.0, float(v))))
    return out


def _extract_categorical(row: dict) -> list[str]:
    return [
        _user_identifier(row),
        str(row.get("avg_price_tier") or "").strip(),
        str(row.get("restaurant_id") or "").strip(),
        _get_primary_category(row),
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
    """범주형 피처별 vocab 생성 및 인덱스 부여."""

    def __init__(self, num_feature: int):
        self.dicts: list[dict[str, int]] = []
        self.num_feature = num_feature
        for _ in range(num_feature):
            self.dicts.append(defaultdict(int))

    def build(self, rows: list[dict], cutoff: int = 0) -> None:
        for row in rows:
            vals = _extract_categorical(row)
            for i in range(self.num_feature):
                if i < len(vals) and vals[i]:
                    self.dicts[i][vals[i]] += 1
        for i in range(self.num_feature):
            filtered = [(k, c) for k, c in self.dicts[i].items() if c >= cutoff]
            filtered.sort(key=lambda x: (-x[1], x[0]))
            vocabs = [x[0] for x in filtered]
            self.dicts[i] = {v: idx for idx, v in enumerate(vocabs, start=1)}
            self.dicts[i]["<unk>"] = 0

    def gen(self, idx: int, key: str) -> int:
        return self.dicts[idx].get(key, self.dicts[idx]["<unk>"])

    def dicts_sizes(self) -> list[int]:
        return [len(self.dicts[i]) for i in range(self.num_feature)]


def _label_from_row(row: dict) -> str:
    """라벨: weight 컬럼이 있으면 0.5 기준 이진화, 없으면 1."""
    w = row.get("weight")
    if w is not None:
        try:
            return "1" if float(w) >= 0.5 else "0"
        except (TypeError, ValueError):
            pass
    if row.get("label") is not None:
        return str(int(row["label"]))
    return "1"


def _load_csv_rows(datadir: str, filename: str, limit: int | None = None) -> list[dict]:
    import pandas as pd

    path = os.path.join(datadir, filename)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, nrows=limit)
    return df.to_dict("records")


def preprocess(
    datadir: str,
    outdir: str,
    num_train_sample: int = 10000,
    num_test_sample: int = 10000,
    categorical_cutoff: int = 2,
    seed: int = 0,
) -> None:
    """
    Tasteam 스키마 기반 CSV를 DeepFM용 train.txt, test.txt, feature_sizes.txt로 변환.

    - datadir: train.csv, test.csv 위치 (또는 train.csv만 있으면 train에서 train/val 분리)
    - outdir: train.txt, test.txt, feature_sizes.txt 저장 경로
    - CSV 컬럼: user_id, anonymous_cohort_id, preferred_categories, avg_price_tier,
      taste_preferences, visit_time_distribution, restaurant_id, categories, price_tier,
      region_gu, region_dong, geohash, positive_segments, comparison_tags,
      day_of_week, time_slot, admin_dong, distance_bucket, weather_bucket, dining_type,
      weight 또는 label
    """
    random.seed(seed)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    train_rows = _load_csv_rows(datadir, "train.csv", limit=num_train_sample)
    test_rows = _load_csv_rows(datadir, "test.csv", limit=num_test_sample)

    if not train_rows and not test_rows:
        # Criteo 호환: train.txt / test.txt (tab) 있으면 그대로 사용하고 feature_sizes만 생성하지 않음
        train_path = os.path.join(datadir, "train.txt")
        if os.path.exists(train_path):
            raise FileNotFoundError(
                "Tasteam 전처리를 사용하려면 datadir에 train.csv(또는 test.csv)를 두세요. "
                "Criteo raw train.txt만 있는 경우 기존 Criteo 전처리 스크립트를 사용하세요."
            )
        raise FileNotFoundError(f"데이터 없음: {datadir} 에 train.csv 또는 test.csv 필요")

    if not train_rows:
        train_rows = test_rows
        test_rows = []

    # 범주형 vocab 구축 (train 기준)
    dicts = CategoryDictGenerator(len(CATEGORICAL_FEATURES))
    dicts.build(train_rows, cutoff=categorical_cutoff)
    dict_sizes = dicts.dicts_sizes()

    # feature_sizes: 연속 8개는 1, 범주형은 vocab 크기
    sizes = [1] * len(CONTINUOUS_FEATURES) + dict_sizes
    with open(os.path.join(outdir, "feature_sizes.txt"), "w") as f:
        f.write(",".join(map(str, sizes)))

    # train.txt
    with open(os.path.join(outdir, "train.txt"), "w") as out_train:
        for row in train_rows:
            cont = _extract_continuous(row)
            cont_str = ",".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in cont)
            cat_vals = _extract_categorical(row)
            cat_str = ",".join(str(dicts.gen(i, cat_vals[i])) for i in range(len(cat_vals)))
            label = _label_from_row(row)
            out_train.write(",".join([cont_str, cat_str, label]) + "\n")

    # test.txt (라벨 없음 또는 있으면 포함)
    with open(os.path.join(outdir, "test.txt"), "w") as out_test:
        for row in test_rows:
            cont = _extract_continuous(row)
            cont_str = ",".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in cont)
            cat_vals = _extract_categorical(row)
            cat_str = ",".join(str(dicts.gen(i, cat_vals[i])) for i in range(len(cat_vals)))
            label = _label_from_row(row)
            out_test.write(",".join([cont_str, cat_str, label]) + "\n")

    # test가 없고 train만 있으면 test.txt는 train에서 일부 복사 (기존 플로우 호환)
    if not test_rows and train_rows:
        with open(os.path.join(outdir, "test.txt"), "w") as out_test:
            for row in train_rows[: min(num_test_sample, len(train_rows))]:
                cont = _extract_continuous(row)
                cont_str = ",".join(f"{x:.6f}".rstrip("0").rstrip(".") for x in cont)
                cat_vals = _extract_categorical(row)
                cat_str = ",".join(str(dicts.gen(i, cat_vals[i])) for i in range(len(cat_vals)))
                label = _label_from_row(row)
                out_test.write(",".join([cont_str, cat_str, label]) + "\n")


if __name__ == "__main__":
    preprocess("../data/raw", "../data")
