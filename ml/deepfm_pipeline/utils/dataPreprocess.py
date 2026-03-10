"""
Tasteam DeepFM용 데이터 전처리.

tasteam_deepfm_data.md + docs/data_designe + recommendation_techspec §스코어 계산 기준 + exp.md 반영:
- 연속형: taste(4), visit_time(4), is_anonymous(1), pref_w_1~3(3) = 12개
  + 스코어 피처 6개(선호 카테고리, 가격대, 맛×긍정구간, 시간대×컨텍스트, 거리, 날씨; 암묵적 피드백 제외·라벨 누수 방지)
  + exp 10개: restaurant_popularity, restaurant_signal_count, restaurant_avg_weight, user_category_count, user_region_count, user_price_mean, user_category_match, user_region_match, price_diff, distance_weight = 28개
- 범주형: member_id, anon_cohort_id, avg_price_tier, restaurant_id, primary_category,
  pref_cat_1~3, price_tier, region_gu, region_dong, geohash, day_of_week, time_slot,
  admin_dong, distance_bucket, weather_bucket, dining_type, first_positive_segment,
  first_comparison_tag = 20개
- 시간 기준 split, recommendation 단위 보존, sample_weight(옵션 C), warm/cold 메타 출력
"""
from __future__ import annotations

import json
import math
import os
import random
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

# --- 범주형 피처 (member_id / anon_cohort_id 분리, preferred_categories Top-K) ---
CATEGORICAL_FEATURES = [
    "member_id",
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

# recommendation_techspec §스코어 계산 기준 (548-556) 6개 피처 → 연속형 0~1 (암묵적 피드백 이력 제외: 현재 행 signal_type 사용 시 라벨 누수)
NUM_SCORING_FEATURES = 6
# exp.md (156-188): item popularity / user preference / user-item / context
NUM_EXP_FEATURES = 10  # restaurant_popularity(sum weight), restaurant_signal_count, restaurant_avg_weight, user_category_count, user_region_count, user_price_mean, user_category_match, user_region_match, price_diff, distance_weight
EXP_FEATURE_NAMES = [
    "restaurant_popularity", "restaurant_signal_count", "restaurant_avg_weight",
    "user_category_count", "user_region_count", "user_price_mean",
    "user_category_match", "user_region_match", "price_diff", "distance_weight",
]
DISTANCE_BUCKET_SCORE = {"NEAR": 1.0, "CLOSE": 2.0 / 3.0, "MID": 1.0 / 3.0, "FAR": 0.0}
DISTANCE_BUCKET_INDEX = {"NEAR": 0, "CLOSE": 1, "MID": 2, "FAR": 3}  # distance_weight = 1/(idx+1)


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


def _member_id_value(row: dict) -> Any:
    """로그인 사용자 식별자. 우선 member_id, 없으면 기존 user_id도 허용(호환)."""
    mid = row.get("member_id")
    if mid is None or (isinstance(mid, float) and pd.isna(mid)):
        mid = row.get("user_id")
    return mid


def _user_id_str(row: dict) -> str:
    """호환을 위해 함수명은 유지. 내부적으로 member_id(또는 user_id)를 사용."""
    mid = _member_id_value(row)
    if mid is not None and str(mid).strip() != "":
        return f"u_{mid}"
    return ""


def _anon_cohort_str(row: dict) -> str:
    aid = row.get("anonymous_cohort_id") or row.get("anonymous_id")
    if aid is not None and str(aid).strip() != "":
        return f"a_{aid}"
    return ""


def _is_anonymous(row: dict) -> float:
    mid = _member_id_value(row)
    if mid is not None and str(mid).strip() != "":
        return 0.0
    return 1.0


def _user_key_for_lookup(row: dict) -> str:
    """member_id(또는 user_id 호환) 또는 anonymous_id 기반 단일 키 (exp lookups용)."""
    mid = _member_id_value(row)
    if mid is not None and str(mid).strip():
        return f"u_{str(mid).strip()}"
    aid = row.get("anonymous_id") or row.get("anonymous_cohort_id")
    if aid is not None and str(aid).strip():
        return f"a_{str(aid).strip()}"
    return ""


def _price_tier_to_num(s: str) -> float:
    """가격대 문자열을 0~1 수치로. 빈 값은 0.5."""
    s = (s or "").strip()
    if not s:
        return 0.5
    try:
        x = float(s)
        return max(0.0, min(1.0, x / 5.0)) if x > 1 else max(0.0, min(1.0, x))
    except ValueError:
        s_lower = s.lower()
        if s_lower in ("low", "l", "1"): return 0.2
        if s_lower in ("mid", "m", "2"): return 0.5
        if s_lower in ("high", "h", "3"): return 0.8
        return 0.5


def build_exp_lookups(train_rows: list[dict]) -> dict[str, Any]:
    """
    exp.md (156-188): item popularity = sum(weight), signal_count, avg_weight;
    user preference = user_category_count, user_region_count, user_price_mean;
    user_top_region for user_region_match.
    """
    # Item popularity: 모든 train 행 기준 (weight 합, positive 개수, 평균 weight)
    rid_sum_weight: dict[str, float] = defaultdict(float)
    rid_count: dict[str, int] = defaultdict(int)
    rid_signal_count: dict[str, int] = defaultdict(int)

    user_cat: dict[tuple[str, str], int] = defaultdict(int)
    user_region: dict[tuple[str, str], int] = defaultdict(int)
    user_price_list: dict[str, list[float]] = defaultdict(list)

    for row in train_rows:
        w = _sample_weight_from_row(row)
        rid = str(row.get("restaurant_id") or "").strip()
        if rid:
            rid_sum_weight[rid] += w
            rid_count[rid] += 1
            if _is_positive_row(row):
                rid_signal_count[rid] += 1

        uk = _user_key_for_lookup(row)
        if not uk or not _is_positive_row(row):
            continue
        primary = str(row.get("primary_category") or "").strip() or _get_primary_category(row)
        if primary:
            user_cat[(uk, primary)] += 1
        rgu = str(row.get("region_gu") or "").strip()
        if rgu:
            user_region[(uk, rgu)] += 1
        pt = _price_tier_to_num(str(row.get("price_tier") or "").strip())
        user_price_list[uk].append(pt)

    # exp.md (429-436): log1p 적용 후 정규화
    log1p_pop = [math.log1p(s) for s in rid_sum_weight.values()]
    max_log1p_pop = max(log1p_pop, default=1.0)
    restaurant_popularity = {rid: math.log1p(s) / max_log1p_pop for rid, s in rid_sum_weight.items()}
    log1p_signal = [math.log1p(c) for c in rid_signal_count.values()]
    max_log1p_signal = max(log1p_signal, default=1.0)
    restaurant_signal_count_norm = {rid: math.log1p(c) / max_log1p_signal for rid, c in rid_signal_count.items()}
    restaurant_avg_weight = {}
    for rid in rid_sum_weight:
        n = rid_count[rid]
        restaurant_avg_weight[rid] = rid_sum_weight[rid] / n if n else 0.0

    user_price_mean: dict[str, float] = {}
    for uk, vals in user_price_list.items():
        user_price_mean[uk] = sum(vals) / len(vals) if vals else 0.5

    user_top_region: dict[str, str] = {}
    for (uk, rgu), c in user_region.items():
        if uk not in user_top_region or user_region[(uk, user_top_region[uk])] < c:
            user_top_region[uk] = rgu

    return {
        "restaurant_popularity": restaurant_popularity,
        "restaurant_signal_count": restaurant_signal_count_norm,
        "restaurant_avg_weight": restaurant_avg_weight,
        "user_category_count": dict(user_cat),
        "user_region_count": dict(user_region),
        "user_price_mean": user_price_mean,
        "user_top_region": user_top_region,
        "max_user_cat_count": max(user_cat.values(), default=1),
        "max_user_region_count": max(user_region.values(), default=1),
    }


def _extract_exp_features(row: dict, lookups: dict[str, Any], include_names: list[str] | None = None) -> list[float]:
    """
    exp.md (156-188): 10개 - item popularity(3), user preference(3), user-item(3), context(1).
    include_names가 있으면 해당 피처만 해당 순서로 반환 (exp.md 444-458 ablation).
    """
    rid = str(row.get("restaurant_id") or "").strip()
    uk = _user_key_for_lookup(row)
    primary = str(row.get("primary_category") or "").strip() or _get_primary_category(row)
    rgu = str(row.get("region_gu") or "").strip()
    price_tier_num = _price_tier_to_num(str(row.get("price_tier") or "").strip())
    pref_cats = [str(row.get("pref_cat_1") or "").strip(), str(row.get("pref_cat_2") or "").strip(), str(row.get("pref_cat_3") or "").strip()]

    restaurant_popularity = lookups["restaurant_popularity"].get(rid, 0.0)
    restaurant_signal_count = lookups["restaurant_signal_count"].get(rid, 0.0)
    restaurant_avg_weight = lookups["restaurant_avg_weight"].get(rid, 0.0)

    max_cat = lookups["max_user_cat_count"]
    max_reg = lookups["max_user_region_count"]
    cat_count = lookups["user_category_count"].get((uk, primary), 0)
    reg_count = lookups["user_region_count"].get((uk, rgu), 0)
    # exp.md (433): user_category_count → log1p(count)
    user_category_count = math.log1p(cat_count) / math.log1p(max_cat) if max_cat else 0.0
    user_region_count = math.log1p(reg_count) / math.log1p(max_reg) if max_reg else 0.0
    user_price_mean = lookups["user_price_mean"].get(uk, 0.5)

    user_category_match = 1.0 if primary and primary in [c for c in pref_cats if c] else 0.0
    user_region_match = 1.0 if (uk and lookups["user_top_region"].get(uk) == rgu) else 0.0
    price_diff = min(1.0, abs(price_tier_num - user_price_mean) / 3.0)

    dist_bucket = str(row.get("distance_bucket") or "").strip().upper()
    idx = DISTANCE_BUCKET_INDEX.get(dist_bucket, 2)
    distance_weight = 1.0 / (idx + 1)

    all_vals = [
        restaurant_popularity,
        restaurant_signal_count,
        restaurant_avg_weight,
        user_category_count,
        user_region_count,
        user_price_mean,
        user_category_match,
        user_region_match,
        price_diff,
        distance_weight,
    ]
    if include_names is None:
        return all_vals
    return [all_vals[EXP_FEATURE_NAMES.index(n)] for n in include_names if n in EXP_FEATURE_NAMES]


def print_exp_feature_distribution(outdir: str) -> None:
    """
    exp.md (403-421): 수치형 exp 피처 전부 train/test에 대해
    min, max, mean, std, 상위 5개 값 출력.
    """
    sizes_path = os.path.join(outdir, "feature_sizes.txt")
    if not os.path.exists(sizes_path):
        return
    with open(sizes_path) as f:
        sizes = [int(x) for x in f.read().strip().split(",")]
    n_cont = sum(1 for s in sizes if s == 1)
    base_cont = len(CONTINUOUS_FEATURES) + NUM_SCORING_FEATURES  # 18
    num_exp = n_cont - base_cont
    if num_exp <= 0:
        return
    exp_start = base_cont
    names = EXP_FEATURE_NAMES[:num_exp]

    for fname in ("train.txt", "test.txt"):
        path = os.path.join(outdir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, header=None)
        if df.shape[1] <= n_cont:
            continue
        # 연속형은 앞 n_cont 컬럼 (인덱스 0..n_cont-1)
        block = df.iloc[:, exp_start:n_cont].astype(float)
        print(f"[exp feature distribution] {fname}")
        for j, name in enumerate(names):
            col = block.iloc[:, j]
            top5 = col.nlargest(5).tolist()
            print(f"  {name}: min={col.min():.6f} max={col.max():.6f} mean={col.mean():.6f} std={col.std():.6f} top5={top5}")
        print()


def _extract_scoring_features(row: dict) -> list[float]:
    """
    recommendation_techspec §스코어 계산 기준 6개 피처를 0~1 연속값으로.
    (1) 사용자 선호 카테고리 매칭 (2) 가격대 매칭 (3) 맛×긍정구간 (4) 시간대분포×컨텍스트
    (5) 거리 bucket (6) 날씨 bucket 적합도.
    암묵적 피드백 이력은 현재 행 signal_type과 라벨이 동일 정보라 누수되므로 제외.
    """
    pref_cats, _ = _get_preferred_categories_topk(row)
    if not any(pref_cats):
        pref_cats = [str(row.get("pref_cat_1") or "").strip(), str(row.get("pref_cat_2") or "").strip(), str(row.get("pref_cat_3") or "").strip()]
    primary = str(row.get("primary_category") or "").strip()
    cat_match = 1.0 if primary and primary in [c for c in pref_cats if c] else 0.0

    avg_pt = str(row.get("avg_price_tier") or "").strip()
    pt = str(row.get("price_tier") or "").strip()
    price_match = 1.0 if avg_pt and pt and avg_pt == pt else 0.0

    taste = _get_taste_preferences(row)
    pos_seg = str(_get_first_positive_segment(row) or "").strip()
    taste_pos = max(0.0, min(1.0, float(taste.get(pos_seg, 0.0)))) if pos_seg else 0.0

    visit = _get_visit_time_distribution(row)
    time_slot = str(row.get("time_slot") or "").strip()
    time_match = max(0.0, min(1.0, float(visit.get(time_slot, 0.0) or visit.get("other", 0.0))))

    dist_bucket = str(row.get("distance_bucket") or "").strip().upper()
    distance_score = DISTANCE_BUCKET_SCORE.get(dist_bucket, 0.5)

    weather_bucket = str(row.get("weather_bucket") or "").strip()
    weather_score = 0.5 if not weather_bucket else 1.0

    return [
        cat_match,
        price_match,
        taste_pos,
        time_match,
        distance_score,
        weather_score,
    ]


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
    out.extend(_extract_scoring_features(row))
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


def _save_categorical_dicts(dicts: list[dict[str, int]], path: str) -> None:
    """범주형 필드별 vocab(문자열→인덱스)를 JSON으로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([dict(d) for d in dicts], f, ensure_ascii=False)


def _serialize_exp_lookups(lookups: dict[str, Any]) -> dict[str, Any]:
    """tuple 키를 JSON 호환 키로 변환 (리스트를 JSON 문자열로)."""
    out = {}
    for k, v in lookups.items():
        if k == "user_category_count":
            out[k] = {json.dumps([a, b], ensure_ascii=False): c for (a, b), c in v.items()}
        elif k == "user_region_count":
            out[k] = {json.dumps([a, b], ensure_ascii=False): c for (a, b), c in v.items()}
        else:
            out[k] = v
    return out


def _deserialize_exp_lookups(data: dict[str, Any]) -> dict[str, Any]:
    """저장된 키를 tuple 키로 복원."""
    out = {}
    for k, v in data.items():
        if k == "user_category_count" and isinstance(v, dict):
            out[k] = {tuple(json.loads(s)): c for s, c in v.items()}
        elif k == "user_region_count" and isinstance(v, dict):
            out[k] = {tuple(json.loads(s)): c for s, c in v.items()}
        else:
            out[k] = v
    return out


def _save_exp_lookups(
    lookups: dict[str, Any],
    exp_ablation: list[str] | None,
    path: str,
) -> None:
    """exp lookups + exp_ablation을 JSON으로 저장."""
    payload = {
        "exp_ablation": exp_ablation,
        "lookups": _serialize_exp_lookups(lookups),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_exp_lookups(run_dir: str | Path) -> tuple[dict[str, Any], list[str] | None] | None:
    """
    run_dir에서 exp_lookups.json 로드.
    반환: (lookups, exp_ablation) 또는 없으면 None.
    """
    path = Path(run_dir) / "exp_lookups.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    lookups = _deserialize_exp_lookups(payload.get("lookups", {}))
    exp_ablation = payload.get("exp_ablation")
    return lookups, exp_ablation


def load_categorical_dicts(run_dir: str | Path) -> CategoryDictGenerator | None:
    """
    run 디렉터리(또는 processed_data_dir)에서 categorical_dicts.json 로드.
    추론 시 raw CSV를 같은 인코딩으로 변환할 때 사용. 없으면 None.
    """
    path = Path(run_dir) / "categorical_dicts.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    num_feature = len(raw)
    gen = CategoryDictGenerator(num_feature)
    gen.dicts = [dict(x) for x in raw]
    return gen


def _n_continuous_from_run_dir(run_dir: str | Path) -> int:
    """feature_sizes.txt에서 연속형 필드 개수(선두 1의 개수) 반환."""
    path = Path(run_dir) / "feature_sizes.txt"
    if not path.exists():
        return len(CONTINUOUS_FEATURES) + NUM_SCORING_FEATURES + NUM_EXP_FEATURES
    line = path.read_text(encoding="utf-8").strip()
    sizes = [int(x.strip()) for x in line.split(",") if x.strip()]
    n = 0
    for s in sizes:
        if s == 1:
            n += 1
        else:
            break
    return n if n > 0 else (len(CONTINUOUS_FEATURES) + NUM_SCORING_FEATURES + NUM_EXP_FEATURES)


def raw_rows_to_feature_matrix(
    raw_rows: list[dict],
    run_dir: str | Path,
) -> list[list[float]]:
    """
    raw 행 리스트를 run_dir의 vocab·exp_lookups로 학습 시와 동일한 feature 벡터로 변환.
    - 연속형: _extract_continuous(row) (18) + exp는 exp_lookups.json 있으면 _extract_exp_features, 없으면 0.
    - 범주형: load_categorical_dicts(run_dir)로 인코딩.
    categorical_dicts.json이 없으면 ValueError.
    """
    dicts = load_categorical_dicts(run_dir)
    if dicts is None:
        raise ValueError(f"categorical_dicts.json not found in run_dir: {run_dir}")
    n_cont = _n_continuous_from_run_dir(run_dir)
    n_cont_from_row = len(CONTINUOUS_FEATURES) + NUM_SCORING_FEATURES  # 18
    n_exp = max(0, n_cont - n_cont_from_row)
    n_cat = len(CATEGORICAL_FEATURES)
    exp_loaded = load_exp_lookups(run_dir)
    out = []
    for row in raw_rows:
        cont_base = _extract_continuous(row)
        if exp_loaded is not None:
            lookups, exp_ablation = exp_loaded
            exp_vals = _extract_exp_features(row, lookups, include_names=exp_ablation)
            cont = cont_base + exp_vals
        else:
            cont = cont_base + [0.0] * n_exp
        cat_vals = _extract_categorical(row)
        cat_idx = [float(dicts.gen(i, cat_vals[i])) for i in range(n_cat)]
        out.append(cont + cat_idx)
    return out


def _label_from_row(row: dict) -> str:
    """
    implicit_feedback 유무로 0/1 라벨.
    - label 컬럼이 있으면 사용.
    - 없으면: signal_type이 있고 비어있지 않으며 NO_FEEDBACK/IMPRESSION 등이 아니면 1, else 0.
    """
    if row.get("label") is not None:
        return str(int(row["label"]))
    st = row.get("signal_type")
    if st is None:
        return "0"
    st = str(st).strip().upper()
    if not st or st in ("NO_FEEDBACK", "NONE", "IMPRESSION"):
        return "0"
    return "1"


def _sample_weight_from_row(row: dict) -> float:
    """
    recommendation_techspec (385-392) signal 강도를 weight로 사용.
    REVIEW=1.0, CALL=0.8, ROUTE=0.7, SAVE=0.6, SHARE=0.4, CLICK=0.2.
    signal_type 없음(음성 샘플)이면 1.0.
    """
    st = row.get("signal_type")
    if st is None or not str(st).strip():
        return 1.0
    st = str(st).strip().upper()
    if st in ("NO_FEEDBACK", "NONE", "IMPRESSION"):
        return 1.0
    return float(SIGNAL_WEIGHTS.get(st, 0.2))


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


def _is_positive_row(row: dict) -> bool:
    return _label_from_row(row) == "1"


# 음식점별 피처(전처리 전 raw row 기준). 음성 샘플 생성 시 해당 음식점 정보로 덮어쓸 필드.
_RESTAURANT_KEYS = (
    "restaurant_id", "primary_category", "price_tier",
    "region_gu", "region_dong", "first_positive_segment", "first_comparison_tag",
)


def _add_negative_samples(
    rows: list[dict],
    all_rows: list[dict],
    ratio: float,
    seed: int = 42,
) -> list[dict]:
    """
    positive 행당 ratio개 음성 행 추가. 같은 유저/컨텍스트, 다른 음식점(미클릭)으로 라벨 0.
    """
    if ratio <= 0 or not rows:
        return rows
    rng = random.Random(seed)
    restaurant_lookup: dict[str, dict] = {}
    for r in all_rows:
        rid = r.get("restaurant_id")
        if rid is not None and str(rid).strip():
            rid = str(rid).strip()
            if rid not in restaurant_lookup:
                restaurant_lookup[rid] = {k: r.get(k) for k in _RESTAURANT_KEYS}

    all_rids = list(restaurant_lookup.keys())
    if len(all_rids) < 2:
        return rows

    user_positive: dict[str, set[str]] = defaultdict(set)
    for r in all_rows:
        if not _is_positive_row(r):
            continue
        uid = str(_member_id_value(r) or r.get("anonymous_id") or r.get("anonymous_cohort_id") or "").strip()
        if not uid:
            uid = "_anon"
        rid = r.get("restaurant_id")
        if rid is not None and str(rid).strip():
            user_positive[uid].add(str(rid).strip())

    out = list(rows)
    n_per_row = max(1, int(ratio)) if ratio >= 1 else 1
    for row in rows:
        if not _is_positive_row(row):
            continue
        uid = str(_member_id_value(row) or row.get("anonymous_id") or row.get("anonymous_cohort_id") or "").strip()
        if not uid:
            uid = "_anon"
        pos_set = user_positive.get(uid, set())
        pool = [rid for rid in all_rids if rid not in pos_set]
        if not pool:
            continue
        k = min(n_per_row, len(pool))
        chosen = rng.sample(pool, k)
        for neg_rid in chosen:
            feat = restaurant_lookup.get(neg_rid, {})
            neg_row = dict(row)
            for key in _RESTAURANT_KEYS:
                neg_row[key] = feat.get(key)
            neg_row["signal_type"] = "NO_FEEDBACK"
            neg_row["weight"] = 1.0
            if "label" in neg_row:
                del neg_row["label"]
            out.append(neg_row)
    return out


def _row_recommendation_id(row: dict) -> str:
    """그룹 키: recommendation_id 또는 generated_at 또는 u_{member_id} / a_{aid} / single."""
    rec_id = row.get("recommendation_id") or row.get("generated_at")
    if rec_id is not None and str(rec_id).strip().lower() not in ("", "nan", "none"):
        return str(rec_id).strip()
    uid = _member_id_value(row)
    aid = row.get("anonymous_id") or row.get("anonymous_cohort_id")
    if uid is not None and str(uid).strip().lower() not in ("", "nan", "none"):
        return f"u_{uid}"
    if aid is not None and str(aid).strip().lower() not in ("", "nan", "none"):
        return f"a_{aid}"
    return "single"


def _build_eval_lists(
    rows: list[dict],
    all_rows: list[dict],
    list_size: int = 101,
    num_neg: int = 100,
    num_popular_neg: int = 50,
    popular_top_k: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    리스트 단위 평가용: 그룹별로 1 pos + num_neg neg = list_size행, 동일 recommendation_id 부여.
    - positive가 하나 이상인 그룹만 처리.
    - negative는 (인기 아이템 중 num_popular_neg) + (랜덤 샘플링)으로 구성.
      인기 아이템은 all_rows에서 restaurant_id별 positive count 상위 popular_top_k.
    - 인기 풀/랜덤 풀에서 부족하면 가능한 만큼 채움(그래도 부족하면 해당 그룹 skip).
    """
    if list_size <= 0 or num_neg <= 0 or not rows:
        return rows
    rng = random.Random(seed)
    # 그룹 by recommendation_id
    rec_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        rec_id = _row_recommendation_id(row)
        rec_to_indices[rec_id].append(i)
    # restaurant 풀 & 유저별 positive set (all_rows 기준)
    restaurant_lookup: dict[str, dict] = {}
    for r in all_rows:
        rid = r.get("restaurant_id")
        if rid is not None and str(rid).strip():
            rid = str(rid).strip()
            if rid not in restaurant_lookup:
                restaurant_lookup[rid] = {k: r.get(k) for k in _RESTAURANT_KEYS}
    # 인기 아이템 풀 (restaurant_id별 positive count)
    pop_count: dict[str, int] = defaultdict(int)
    for r in all_rows:
        rid = r.get("restaurant_id")
        if rid is None or not str(rid).strip():
            continue
        rid = str(rid).strip()
        if _is_positive_row(r):
            pop_count[rid] += 1
    popular_rids = [rid for rid, _ in sorted(pop_count.items(), key=lambda x: (-x[1], x[0]))[: max(0, int(popular_top_k))]]
    user_positive: dict[str, set[str]] = defaultdict(set)
    for r in all_rows:
        if not _is_positive_row(r):
            continue
        uid = str(r.get("user_id") or r.get("anonymous_id") or r.get("anonymous_cohort_id") or "").strip()
        if not uid:
            uid = "_anon"
        rid = r.get("restaurant_id")
        if rid is not None and str(rid).strip():
            user_positive[uid].add(str(rid).strip())
    all_rids = list(restaurant_lookup.keys())
    if len(all_rids) < 2:
        return rows

    out: list[dict] = []
    for rec_id, indices in rec_to_indices.items():
        group_rows = [rows[i] for i in indices]
        positives = [r for r in group_rows if _is_positive_row(r)]
        if not positives:
            continue
        pos_row = rng.choice(positives)
        pos_rid = str(pos_row.get("restaurant_id") or "").strip()
        uid = str(pos_row.get("user_id") or pos_row.get("anonymous_id") or pos_row.get("anonymous_cohort_id") or "").strip()
        if not uid:
            uid = "_anon"
        user_pos = user_positive.get(uid, set()) | {pos_rid}
        # 1) 인기 negative
        num_pop = max(0, min(int(num_popular_neg), int(num_neg)))
        pop_pool = [rid for rid in popular_rids if rid not in user_pos]
        chosen_pop = rng.sample(pop_pool, min(num_pop, len(pop_pool))) if pop_pool and num_pop > 0 else []
        # 2) 랜덤 negative (인기에서 뽑은 것 제외)
        remain = num_neg - len(chosen_pop)
        rand_pool = [rid for rid in all_rids if rid not in user_pos and rid not in set(chosen_pop)]
        if remain > 0:
            if len(rand_pool) < remain:
                # deterministic fill (still excluding user_pos + chosen_pop)
                rand_pool = rand_pool + [r for r in all_rids if r not in user_pos and r not in set(chosen_pop) and r not in rand_pool][: remain - len(rand_pool)]
            if len(rand_pool) < remain:
                continue
            chosen_rand = rng.sample(rand_pool, remain)
        else:
            chosen_rand = []
        chosen_neg = chosen_pop + chosen_rand
        list_rec_id = rec_id
        pos_copy = dict(pos_row)
        pos_copy["recommendation_id"] = list_rec_id
        out.append(pos_copy)
        for neg_rid in chosen_neg:
            neg_row = dict(pos_row)
            for key in _RESTAURANT_KEYS:
                neg_row[key] = restaurant_lookup.get(neg_rid, {}).get(key)
            neg_row["signal_type"] = "NO_FEEDBACK"
            neg_row["weight"] = 1.0
            if "label" in neg_row:
                del neg_row["label"]
            neg_row["recommendation_id"] = list_rec_id
            out.append(neg_row)
    return out if out else rows


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
    negative_sampling_ratio: float = 1.0,
    negative_sampling_seed: int = 42,
    eval_list_size: int = 101,
    eval_num_neg: int = 100,
    eval_num_popular_neg: int = 50,
    eval_popular_top_k: int = 1000,
    eval_list_seed: int = 42,
    exp_ablation: list[str] | None = None,
) -> None:
    """
    Tasteam + data_designe 반영 전처리.

    - exp_ablation: None이면 exp 피처 10개 전부. 리스트면 해당 이름만 해당 순서로 포함 (exp.md 444-458).
      예: ["user_category_match", "user_region_match", "price_diff"] → match 계열만.
    - use_sample_weight: True면 train/val/test 마지막에 sample_weight 컬럼 추가 (옵션 C).
    - time_column: 있으면 시간 기준 split (train_end, valid_end, test_end 구간 사용).
    - group_column: recommendation_id 또는 generated_at 등, 같은 값은 같은 구간으로.
    - negative_sampling_ratio: positive 1건당 추가할 음성 샘플 수 (0이면 미적용). AUC/이진 분류용.
    - eval_list_size: >0이면 test/val을 리스트 단위로 재구성 (1 pos + eval_num_neg neg = eval_list_size행, 동일 recommendation_id).
      eval_num_popular_neg 만큼은 인기 아이템에서 negative를 뽑고, 나머지는 랜덤 negative.
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

    all_rows = train_rows + val_rows + test_rows
    if negative_sampling_ratio > 0:
        train_rows = _add_negative_samples(train_rows, all_rows, negative_sampling_ratio, negative_sampling_seed)
        if eval_list_size <= 0:
            test_rows = _add_negative_samples(test_rows, all_rows, negative_sampling_ratio, negative_sampling_seed + 1)
            if val_rows:
                val_rows = _add_negative_samples(val_rows, all_rows, negative_sampling_ratio, negative_sampling_seed + 2)
    if eval_list_size > 0 and eval_num_neg > 0:
        test_rows = _build_eval_lists(
            test_rows,
            all_rows,
            list_size=eval_list_size,
            num_neg=eval_num_neg,
            num_popular_neg=eval_num_popular_neg,
            popular_top_k=eval_popular_top_k,
            seed=eval_list_seed,
        )
        if val_rows:
            val_rows = _build_eval_lists(
                val_rows,
                all_rows,
                list_size=eval_list_size,
                num_neg=eval_num_neg,
                num_popular_neg=eval_num_popular_neg,
                popular_top_k=eval_popular_top_k,
                seed=eval_list_seed + 1,
            )

    dicts = CategoryDictGenerator(len(CATEGORICAL_FEATURES))
    dicts.build(train_rows, cutoff=categorical_cutoff)
    dict_sizes = dicts.dicts_sizes()
    num_exp = len(exp_ablation) if exp_ablation else NUM_EXP_FEATURES
    sizes = [1] * (len(CONTINUOUS_FEATURES) + NUM_SCORING_FEATURES + num_exp) + dict_sizes
    with open(os.path.join(outdir, "feature_sizes.txt"), "w") as f:
        f.write(",".join(map(str, sizes)))

    # 범주형 vocab(문자열→인덱스) 저장. 추론 시 raw→feature 인코딩 재사용용.
    _save_categorical_dicts(dicts.dicts, os.path.join(outdir, "categorical_dicts.json"))

    exp_lookups = build_exp_lookups(train_rows)
    _save_exp_lookups(exp_lookups, exp_ablation, os.path.join(outdir, "exp_lookups.json"))

    def write_rows(rows: list[dict], path: str, include_sw: bool) -> None:
        with open(os.path.join(outdir, path), "w") as f:
            for row in rows:
                cont = _extract_continuous(row) + _extract_exp_features(row, exp_lookups, include_names=exp_ablation)
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

    print_exp_feature_distribution(outdir)

    train_user_ids = set()
    train_restaurant_ids = set()
    restaurant_positive_counts: dict[str, int] = defaultdict(int)
    for row in train_rows:
        uid = _member_id_value(row)
        if uid is not None and str(uid).strip():
            train_user_ids.add(str(uid))
        rid = row.get("restaurant_id")
        if rid is not None and str(rid).strip():
            train_restaurant_ids.add(str(rid))
        if _is_positive_row(row) and rid is not None and str(rid).strip():
            restaurant_positive_counts[str(rid).strip()] += 1
    split_meta = {
        "train_end": train_end,
        "valid_end": valid_end,
        "test_end": test_end,
        "time_column": time_column,
        "group_column": group_column,
        "train_member_ids": list(train_user_ids),
        "train_restaurant_ids": list(train_restaurant_ids),
        "restaurant_positive_counts": dict(restaurant_positive_counts),
        "use_sample_weight": use_sample_weight,
    }
    with open(os.path.join(outdir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    if test_rows:
        meta_rows = []
        for row in test_rows:
            rec_id = row.get("recommendation_id") or row.get("generated_at")
            def _empty(v: Any) -> bool:
                if v is None:
                    return True
                if isinstance(v, float) and pd.isna(v):
                    return True
                s = str(v).strip().lower()
                return not s or s == "nan" or s == "none"
            if not _empty(rec_id):
                rec_id = str(rec_id).strip()
            else:
                uid = _member_id_value(row)
                aid = row.get("anonymous_id") or row.get("anonymous_cohort_id")
                if not _empty(uid):
                    rec_id = f"u_{uid}"
                elif not _empty(aid):
                    rec_id = f"a_{aid}"
                else:
                    rec_id = "single"
            meta_rows.append({
                "member_id": _member_id_value(row) or "",
                "restaurant_id": row.get("restaurant_id") or "",
                "recommendation_id": rec_id,
            })
        pd.DataFrame(meta_rows).to_csv(
            os.path.join(outdir, "test_meta.csv"), index=False
        )


if __name__ == "__main__":
    preprocess("../data/raw", "../data")
