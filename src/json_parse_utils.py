"""
Re-export from comparison_lib (기존 호환). src 내 summary_pipeline, llm_utils, sentiment_analysis 등은 이 경로로 import 유지.
"""
from comparison_lib.json_parse_utils import (
    extract_json_block,
    parse_json_relaxed,
)

__all__ = ["extract_json_block", "parse_json_relaxed"]
