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


# Raw/메타 CSV·JSON에 camelCase가 섞일 때 snake_case로 통일 (snake가 이미 있으면 camel 열은 제거)
_COLUMN_RENAME_CAMEL_TO_SNAKE: dict[str, str] = {
    "memberId": "member_id",
    "userId": "user_id",
    "anonymousId": "anonymous_id",
    "restaurantId": "restaurant_id",
    "eventName": "event_name",
    "occurredAt": "occurred_at",
    "eventId": "event_id",
    "sessionId": "session_id",
    "recommendationId": "recommendation_id",
    "distanceBucket": "distance_bucket",
    "weatherBucket": "weather_bucket",
    "diningType": "dining_type",
    "createdAt": "created_at",
    "eventVersion": "event_version",
    "contextSnapshot": "context_snapshot",
    "foodCategoryName": "food_category_name",
    "foodCategoryId": "food_category_id",
    "restaurantName": "restaurant_name",
    "menuCount": "menu_count",
    "priceMin": "price_min",
    "priceMax": "price_max",
    "priceMean": "price_mean",
    "priceMedian": "price_median",
    "representativeMenuName": "representative_menu_name",
    "topMenus": "top_menus",
    "priceTier": "price_tier",
}

_ID_STRING_COLUMNS: frozenset[str] = frozenset(
    {
        "user_id",
        "member_id",
        "anonymous_id",
        "restaurant_id",
        "anonymous_cohort_id",
    }
)


def normalize_dataframe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """camelCase 컬럼을 snake_case로 바꾼다. 대상 snake 컬럼이 이미 있으면 camel 쪽 열은 삭제한다."""
    if df.empty or len(df.columns) == 0:
        return df
    drop_camel: list[str] = []
    rename_map: dict[str, str] = {}
    for old, new in _COLUMN_RENAME_CAMEL_TO_SNAKE.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            drop_camel.append(old)
        else:
            rename_map[old] = new
    out = df.drop(columns=drop_camel, errors="ignore")
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _apply_id_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """식별자 컬럼은 float/NaN 혼입을 줄이기 위해 pandas string dtype으로 통일."""
    df = df.copy()
    for c in _ID_STRING_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def read_table(
    path: Path,
    *,
    normalize_column_names: bool = False,
    string_id_columns: bool = True,
) -> pd.DataFrame:
    """
    CSV 또는 .json.gz 파일을 DataFrame으로 로드.
    - .csv: pd.read_csv(low_memory=False)
    - .json.gz: JSON 배열 또는 JSON Lines(NDJSON). gzip 해제 후 파싱.

    normalize_column_names: True면 camelCase → snake_case 통일.
    string_id_columns: True면 user_id/member_id/anonymous_id/restaurant_id 등을 StringDtype으로 읽음.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif suf == ".gz" and path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return pd.DataFrame()
        stripped = content.strip()
        if stripped.startswith("["):
            data = json.loads(content)
            df = pd.DataFrame(data) if data else pd.DataFrame()
        else:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            df = pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        raise ValueError(f"Unsupported format: {path}. Use .csv or .json.gz")

    if df.empty or len(df.columns) == 0:
        return df
    if normalize_column_names:
        df = normalize_dataframe_column_names(df)
    if string_id_columns:
        df = _apply_id_string_dtypes(df)
    return df

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


def _first_present(row: pd.Series, *keys: str) -> Any:
    """
    row에서 첫 번째로 존재하는 키의 값을 반환.
    snake_case / camelCase 스키마를 모두 수용하기 위한 유틸.
    """
    for k in keys:
        if k in row:
            return row.get(k)
    return None


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
        for pattern in ("*.csv", "*.json.gz"):
            for f in part_dir.glob(pattern):
                try:
                    df = read_table(f, normalize_column_names=True, string_id_columns=True)
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
        rid = _safe_str(_first_present(ev, "restaurant_id", "restaurantId"))
        rest = rest_lookup.get(rid, {}) if rid else {}
        menu = menu_lookup.get(rid, {}) if rid else {}

        event_name = _safe_str(_first_present(ev, "event_name", "eventName")).lower()
        signal_type = EVENT_NAME_TO_SIGNAL.get(event_name, event_name.upper() if event_name else "IMPRESSION")
        if signal_type not in POSITIVE_SIGNALS:
            signal_type = "IMPRESSION"
        label = 1 if signal_type in POSITIVE_SIGNALS else 0

        day_of_week, time_slot = _parse_occurred_at(_first_present(ev, "occurred_at", "occurredAt"))

        # primary_category: food_category_name. 파이프라인은 categories JSON도 사용
        primary_category = _safe_str(rest.get("food_category_name"))
        categories = json.dumps([primary_category]) if primary_category else "[]"

        price_tier = _safe_str(menu.get("price_tier") or rest.get("price_tier"))
        avg_pt = _price_mean_to_tier(menu.get("price_mean")) or price_tier or ""

        row = {
            "member_id": _safe_str(_first_present(ev, "member_id", "memberId")),
            "user_id": _safe_str(_first_present(ev, "member_id", "memberId", "user_id", "userId")),
            "anonymous_id": _safe_str(_first_present(ev, "anonymous_id", "anonymousId")),
            "restaurant_id": rid,
            "primary_category": primary_category,
            "categories": categories,
            "region_gu": _safe_str(rest.get("sigungu")),
            "region_dong": _safe_str(rest.get("eupmyeondong")),
            "admin_dong": _safe_str(rest.get("eupmyeondong")),
            "geohash": _safe_str(rest.get("geohash")),
            "price_tier": price_tier,
            "avg_price_tier": avg_pt,
            "dining_type": _safe_str(_first_present(ev, "dining_type", "diningType")),
            "distance_bucket": _safe_str(_first_present(ev, "distance_bucket", "distanceBucket")),
            "weather_bucket": _safe_str(_first_present(ev, "weather_bucket", "weatherBucket")),
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
