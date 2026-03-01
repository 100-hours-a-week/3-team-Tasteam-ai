#!/usr/bin/env python3
"""
데이터 증강: docs/data_augmentation/data_augmentation.md 에 따른
슬라이딩 윈도우 + 식당 단위 train/val/test 분할.(디폴트: 80/10/10)

입력: tasteam_app_all_review_data.json (reviews: [{ id, restaurant_id, content, created_at }])
출력: 식당 단위로 분할된 train/val/test 샘플 (각 샘플 = 한 윈도우의 리뷰 목록)

사용 예:
  python scripts/data_augmentation.py
  python scripts/data_augmentation.py --input data.json --out-dir out --window 30 15 --add-full
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 문서 권장: Train 37 / Val 5 / Test 5 (47개 식당)
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
# 문서 예시 조합 (W, S): (30,15), (50,25), (20,10)
DEFAULT_WINDOWS = [(30, 15), (50, 25), (20, 10)]


def load_reviews(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reviews = data.get("reviews", data) if isinstance(data, dict) else data
    if not isinstance(reviews, list):
        raise ValueError("JSON must have 'reviews' array or be an array of reviews")
    return reviews


def group_by_restaurant(reviews: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_rid: dict[int, list[dict[str, Any]]] = {}
    for r in reviews:
        rid = r.get("restaurant_id")
        if rid is None:
            continue
        by_rid.setdefault(rid, []).append(r)
    for rid in by_rid:
        by_rid[rid].sort(key=lambda x: (x.get("created_at") or ""))
    return by_rid


def sliding_windows(
    reviews: list[dict[str, Any]],
    window_size: int,
    stride: int,
) -> list[list[dict[str, Any]]]:
    """한 식당의 리뷰 리스트에 대해 슬라이딩 윈도우 적용. 문서 공식: floor((R-W)/S)+1 개."""
    if window_size <= 0 or stride <= 0 or len(reviews) < window_size:
        return []
    out = []
    for start in range(0, len(reviews) - window_size + 1, stride):
        out.append(reviews[start : start + window_size])
    return out


def build_samples(
    by_restaurant: dict[int, list[dict[str, Any]]],
    window_configs: list[tuple[int, int]],
    add_full_restaurant: bool,
    seed: int,
) -> list[dict[str, Any]]:
    """윈도우 설정 여러 개 적용 + (선택) 식당 전체 1개씩. 각 샘플에 restaurant_id 부여."""
    rng = random.Random(seed)
    samples = []
    sample_id = 0
    for restaurant_id, revs in by_restaurant.items():
        if len(revs) < 2:
            continue
        for w, s in window_configs:
            if w > len(revs):
                continue
            for window_revs in sliding_windows(revs, w, s):
                sample_id += 1
                samples.append({
                    "sample_id": sample_id,
                    "restaurant_id": restaurant_id,
                    "reviews": window_revs,
                    "window_size": w,
                    "stride": s,
                })
        if add_full_restaurant and len(revs) >= 2:
            sample_id += 1
            samples.append({
                "sample_id": sample_id,
                "restaurant_id": restaurant_id,
                "reviews": revs,
                "window_size": len(revs),
                "stride": None,
            })
    return samples


def stratified_restaurant_split(
    by_restaurant: dict[int, list[dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """식당을 리뷰 수 기준으로 정렬한 뒤, 순서를 유지하면서 비율만큼 train/val/test에 배치 (stratified).
    리뷰 수가 적은/많은 식당이 한쪽 split에 몰리지 않도록, 정렬된 리스트에서 구간별로 비슷한 비율로 뽑음."""
    rng = random.Random(seed)
    # 리뷰 수 오름차순 정렬
    ordered = sorted(
        by_restaurant.keys(),
        key=lambda rid: len(by_restaurant[rid]),
    )
    n = len(ordered)
    # 정렬된 순서를 유지한 채로 80/10/10 비율로 배치 (인덱스 구간별로 골고루 → stratified)
    train_ids, val_ids, test_ids = [], [], []
    for i, rid in enumerate(ordered):
        # i를 n으로 나눈 위치로 비율 결정 (리뷰 수 적은 식당~많은 식당에 걸쳐 균등)
        r = (i + 0.5) / n if n else 0
        if r < train_ratio:
            train_ids.append(rid)
        elif r < train_ratio + val_ratio:
            val_ids.append(rid)
        else:
            test_ids.append(rid)
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def assign_split_to_samples(
    samples: list[dict[str, Any]],
    train_rids: list[int],
    val_rids: list[int],
    test_rids: list[int],
) -> None:
    train_set = set(train_rids)
    val_set = set(val_rids)
    test_set = set(test_rids)
    for s in samples:
        rid = s["restaurant_id"]
        if rid in train_set:
            s["split"] = "train"
        elif rid in val_set:
            s["split"] = "val"
        elif rid in test_set:
            s["split"] = "test"
        else:
            s["split"] = "train"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sliding-window data augmentation with restaurant-wise train/val/test split (see docs/data_augmentation/data_augmentation.md)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tasteam_app_all_review_data.json"),
        help="Input JSON path (reviews array with restaurant_id, content, created_at)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_augmentation_output"),
        help="Output directory for train.json, val.json, test.json, stats.json",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs="+",
        default=None,
        metavar="W S [W2 S2 ...]",
        help="Window size and stride pairs. Default: 30 15  50 25  20 10",
    )
    parser.add_argument(
        "--add-full",
        action="store_true",
        help="Add one sample per restaurant with all reviews (full-restaurant summary)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of restaurants for train (default 0.80)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of restaurants for val (default 0.10)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of restaurants for test (default 0.10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split",
    )
    args = parser.parse_args()

    if args.window is not None:
        if len(args.window) % 2 != 0:
            parser.error("--window must have pairs of (window_size stride)")
        window_configs = [(args.window[i], args.window[i + 1]) for i in range(0, len(args.window), 2)]
    else:
        window_configs = DEFAULT_WINDOWS

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error("train_ratio + val_ratio + test_ratio must be 1.0")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    reviews = load_reviews(args.input)
    logger.info("Loaded %d reviews", len(reviews))

    by_restaurant = group_by_restaurant(reviews)
    n_restaurants = len(by_restaurant)
    logger.info("Grouped into %d restaurants", n_restaurants)

    samples = build_samples(
        by_restaurant,
        window_configs,
        add_full_restaurant=args.add_full,
        seed=args.seed,
    )
    logger.info("Built %d samples (windows + optional full-restaurant)", len(samples))

    train_rids, val_rids, test_rids = stratified_restaurant_split(
        by_restaurant,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    logger.info(
        "Restaurant split: train=%d val=%d test=%d",
        len(train_rids),
        len(val_rids),
        len(test_rids),
    )

    assign_split_to_samples(samples, train_rids, val_rids, test_rids)

    train_samples = [s for s in samples if s["split"] == "train"]
    val_samples = [s for s in samples if s["split"] == "val"]
    test_samples = [s for s in samples if s["split"] == "test"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        out_path = args.out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"samples": payload}, f, ensure_ascii=False, indent=2)
        logger.info("Wrote %s: %d samples -> %s", name, len(payload), out_path)

    stats = {
        "n_reviews": len(reviews),
        "n_restaurants": n_restaurants,
        "n_samples_total": len(samples),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "window_configs": [{"window_size": w, "stride": s} for w, s in window_configs],
        "add_full_restaurant": args.add_full,
        "train_restaurants": len(train_rids),
        "val_restaurants": len(val_rids),
        "test_restaurants": len(test_rids),
    }
    stats_path = args.out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Wrote stats -> %s", stats_path)


if __name__ == "__main__":
    main()
