"""
comparison_lib: Comparison 파이프라인 전용 패키지 (Spark 서비스·메인 앱 공통).
src.config·vector_search 미의존 — Spark 서비스는 이 패키지만 import.
"""
from .comparison_pipeline import (
    calculate_all_average_ratios_from_file,
    calculate_all_average_ratios_from_reviews,
    calculate_all_average_ratios_from_texts,
    calculate_single_restaurant_ratios,
    calculate_comparison_lift,
    compute_recall_seeds_from_file,
    compute_recall_seeds_from_reviews,
    compute_recall_seeds_from_texts,
    recall_seeds_to_seed_lists,
    load_reviews_from_aspect_data_file,
    format_comparison_display,
    generate_comparison_descriptions,
)
from .json_parse_utils import parse_json_relaxed, extract_json_block

__all__ = [
    "calculate_all_average_ratios_from_file",
    "calculate_all_average_ratios_from_reviews",
    "calculate_all_average_ratios_from_texts",
    "calculate_single_restaurant_ratios",
    "calculate_comparison_lift",
    "compute_recall_seeds_from_file",
    "compute_recall_seeds_from_reviews",
    "compute_recall_seeds_from_texts",
    "recall_seeds_to_seed_lists",
    "load_reviews_from_aspect_data_file",
    "format_comparison_display",
    "generate_comparison_descriptions",
    "parse_json_relaxed",
    "extract_json_block",
]
