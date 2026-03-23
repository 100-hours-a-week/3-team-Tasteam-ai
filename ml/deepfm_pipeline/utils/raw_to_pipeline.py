"""
S3 Raw(events / restaurants / menus) → 파이프라인용 CSV 변환.

service_constract.md §4 스키마를 dataPreprocess가 기대하는 행 스키마로 조인·매핑한다.
- events + restaurants(restaurant_id) + menus(restaurant_id)
- event_name → signal_type, label
- occurred_at → day_of_week, time_slot
- 음식점/메뉴 컬럼 → primary_category, region_gu, geohash, price_tier 등
- taste_preferences, preferred_categories 등 사용자 집계 필드는 비워 두고 파이프라인 기본값 사용
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _normalize_base_prefix(base_prefix: str) -> str:
    return base_prefix.strip("/")


def read_table(path: Path) -> pd.DataFrame:
    """
    CSV 또는 .json.gz 파일을 DataFrame으로 로드.
    - .csv: pd.read_csv
    - .json.gz: JSON 배열 또는 JSON Lines(NDJSON). gzip 해제 후 파싱.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".gz" and path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return pd.DataFrame()
        stripped = content.strip()
        if stripped.startswith("["):
            data = json.loads(content)
            return pd.DataFrame(data) if data else pd.DataFrame()
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    raise ValueError(f"Unsupported format: {path}. Use .csv or .json.gz")

# event_name → signal_type (대문자). 라벨 1: REVIEW, CALL, ROUTE, SAVE, SHARE, CLICK / 0: view, impression 등
EVENT_NAME_TO_SIGNAL = {
    "review": "REVIEW",
    "click": "CLICK",
    "view": "IMPRESSION",
    "impression": "IMPRESSION",
    "call": "CALL",
    "route": "ROUTE",
    "save": "SAVE",
    "share": "SHARE",
}
POSITIVE_SIGNALS = {"REVIEW", "CALL", "ROUTE", "SAVE", "SHARE", "CLICK"}


def _safe_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _load_partition_csvs(base_dir: Path, data_type: str, base_prefix: str = "raw") -> pd.DataFrame:
    """{base_prefix}/{data_type}/dt=*/part-*.csv 및 part-*.json.gz 를 모두 읽어 하나의 DataFrame으로."""
    normalized_prefix = _normalize_base_prefix(base_prefix)
    prefix = base_dir / normalized_prefix / data_type
    if not prefix.exists():
        return pd.DataFrame()
    frames = []
    for part_dir in sorted(prefix.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("dt="):
            continue
        for pattern in ("part-*.csv", "part-*.json.gz"):
            for f in part_dir.glob(pattern):
                try:
                    df = read_table(f)
                    if not df.empty:
                        frames.append(df)
                except Exception:
                    continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _restaurant_lookup(restaurants_df: pd.DataFrame) -> dict[str, dict]:
    """restaurant_id -> 행 dict (마지막 dt 기준)."""
    if restaurants_df.empty or "restaurant_id" not in restaurants_df.columns:
        return {}
    out = {}
    for _, row in restaurants_df.iterrows():
        rid = _safe_str(row.get("restaurant_id"))
        if rid:
            out[rid] = row.to_dict()
    return out


def _menu_lookup(menus_df: pd.DataFrame) -> dict[str, dict]:
    """restaurant_id -> 행 dict."""
    if menus_df.empty or "restaurant_id" not in menus_df.columns:
        return {}
    out = {}
    for _, row in menus_df.iterrows():
        rid = _safe_str(row.get("restaurant_id"))
        if rid:
            out[rid] = row.to_dict()
    return out


def _parse_occurred_at(ts: Any) -> tuple[str, str]:
    """occurred_at → (day_of_week, time_slot). day: 0-6 Mon-Sun, time_slot: breakfast/lunch/afternoon/dinner."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "", ""
    try:
        t = pd.to_datetime(ts)
        day = str(t.dayofweek)  # 0=Mon
        hour = t.hour
        if hour < 10:
            slot = "breakfast"
        elif hour < 14:
            slot = "lunch"
        elif hour < 18:
            slot = "afternoon"
        else:
            slot = "dinner"
        return day, slot
    except Exception:
        return "", ""


def _price_mean_to_tier(price_mean: Any) -> str:
    """메뉴 price_mean 또는 price_tier → avg_price_tier 문자열 (LOW/MID/HIGH)."""
    if price_mean is None or (isinstance(price_mean, float) and pd.isna(price_mean)):
        return ""
    if isinstance(price_mean, str):
        return price_mean.strip().upper() or ""
    try:
        x = float(price_mean)
        if x < 10000:
            return "LOW"
        if x < 20000:
            return "MID"
        return "HIGH"
    except (TypeError, ValueError):
        return ""


def transform_raw_to_pipeline_rows(
    events_df: pd.DataFrame,
    restaurants_df: pd.DataFrame,
    menus_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    events + restaurants + menus → 파이프라인 입력 행 리스트.
    각 행은 dataPreprocess._extract_continuous / _extract_categorical 가 기대하는 키를 가짐.
    """
    if events_df.empty:
        return []
    rest_lookup = _restaurant_lookup(restaurants_df)
    menu_lookup = _menu_lookup(menus_df)

    rows = []
    for _, ev in events_df.iterrows():
        rid = _safe_str(ev.get("restaurant_id"))
        rest = rest_lookup.get(rid, {}) if rid else {}
        menu = menu_lookup.get(rid, {}) if rid else {}

        event_name = _safe_str(ev.get("event_name")).lower()
        signal_type = EVENT_NAME_TO_SIGNAL.get(event_name, event_name.upper() if event_name else "IMPRESSION")
        if signal_type not in POSITIVE_SIGNALS:
            signal_type = "IMPRESSION"
        label = 1 if signal_type in POSITIVE_SIGNALS else 0

        day_of_week, time_slot = _parse_occurred_at(ev.get("occurred_at"))

        # primary_category: food_category_name. 파이프라인은 categories JSON도 사용
        primary_category = _safe_str(rest.get("food_category_name"))
        categories = json.dumps([primary_category]) if primary_category else "[]"

        price_tier = _safe_str(menu.get("price_tier") or rest.get("price_tier"))
        avg_pt = _price_mean_to_tier(menu.get("price_mean")) or price_tier or ""

        row = {
            "member_id": _safe_str(ev.get("member_id")),
            "user_id": _safe_str(ev.get("member_id")),
            "anonymous_id": _safe_str(ev.get("anonymous_id")),
            "restaurant_id": rid,
            "primary_category": primary_category,
            "categories": categories,
            "region_gu": _safe_str(rest.get("sigungu")),
            "region_dong": _safe_str(rest.get("eupmyeondong")),
            "admin_dong": _safe_str(rest.get("eupmyeondong")),
            "geohash": _safe_str(rest.get("geohash")),
            "price_tier": price_tier,
            "avg_price_tier": avg_pt,
            "dining_type": _safe_str(ev.get("dining_type")),
            "distance_bucket": _safe_str(ev.get("distance_bucket")),
            "weather_bucket": _safe_str(ev.get("weather_bucket")),
            "day_of_week": day_of_week,
            "time_slot": time_slot,
            "signal_type": signal_type,
            "label": label,
            "pref_cat_1": "",
            "pref_cat_2": "",
            "pref_cat_3": "",
            "taste_preferences": "{}",
            "visit_time_distribution": "{}",
            "preferred_categories": "[]",
            "positive_segments": "[]",
            "comparison_tags": "[]",
        }
        rows.append(row)
    return rows


def raw_dir_to_pipeline_csv(
    raw_download_dir: str | Path,
    output_path: str | Path,
    data_types: tuple[str, ...] = ("events", "restaurants", "menus"),
    base_prefix: str = "raw",
) -> int:
    """
    s3_raw_poll_download.py 로 받은 로컬 디렉터리({base_prefix}/events/, {base_prefix}/restaurants/, {base_prefix}/menus/)를
    읽어 파이프라인용 CSV 하나로 저장.

    - raw_download_dir: 다운로드 기준 디렉터리 (그 하위에 {base_prefix}/events/dt=.../ 등이 있음)
    - output_path: 출력 CSV 경로 (train.csv 또는 training_dataset.csv 등)
    - 반환: 저장된 행 수
    """
    base = Path(raw_download_dir)
    events_df = _load_partition_csvs(base, "events", base_prefix=base_prefix)
    restaurants_df = _load_partition_csvs(base, "restaurants", base_prefix=base_prefix) if "restaurants" in data_types else pd.DataFrame()
    menus_df = _load_partition_csvs(base, "menus", base_prefix=base_prefix) if "menus" in data_types else pd.DataFrame()

    rows = transform_raw_to_pipeline_rows(events_df, restaurants_df, menus_df)
    if not rows:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return 0
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)
