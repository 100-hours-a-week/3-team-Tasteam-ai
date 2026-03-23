#!/usr/bin/env python3
"""
KD SFT 분석: docs/distill/troubleshooting/kd_sft_analysis.md 에서 제안한 3가지 확인을 수행.

1. JSON 파싱 성공률 (eval 샘플에서 %)
2. 스키마 정확도 (필수 키 존재 / 타입 맞는지; no-evidence 평가는 summary+bullets만)
3. 길이·포맷 drift (pred vs ref 평균 길이, 비율)

입력: llm_as_a_judge_results.json 또는 pred/ref 포함 JSON (results 배열 또는 samples).
출력: report JSON + 요약 stdout.

사용:
  python scripts/kd_sft_analysis.py --input .../llm_as_a_judge_results.json
  python scripts/kd_sft_analysis.py --input ... --output-dir .../eval/20260303_053420
  python scripts/kd_sft_analysis.py --input ... --output .../eval/kd_sft_analysis_report_v3_jv2.json
  # llm_as_a_judge_results.json 의 meta(judge_rubric_version v2_no_evidence 등)로 no-evidence 스키마 자동 선택
  python scripts/kd_sft_analysis.py --input ... --no-evidence-schema
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 참조 스키마: 최상위 필수 키(instruction에 항상 있는 축). overall_summary는 선택.
REQUIRED_TOP_KEYS = {"service", "price", "food"}
OPTIONAL_TOP_KEYS = {"overall_summary"}
# 각 축(service/price/food) 내부: summary(str), bullets(list), evidence(list of int)
ASPECT_KEYS = {"summary", "bullets", "evidence"}
# no-evidence 스키마: summary + bullets만 (evidence 불필요)
ASPECT_KEYS_NO_EVIDENCE = {"summary", "bullets"}
# overall_summary는 summary만 있으면 됨
OVERALL_KEYS = {"summary"}


def _load_input(path: Path) -> list[dict]:
    samples, _ = _load_input_and_meta(path)
    return samples


def _load_input_and_meta(path: Path) -> tuple[list[dict], dict[str, Any] | None]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta: dict[str, Any] | None = None
    if isinstance(data, dict) and "meta" in data and isinstance(data["meta"], dict):
        meta = data["meta"]
    if isinstance(data, dict) and "results" in data:
        return data["results"], meta
    if isinstance(data, dict) and "samples" in data:
        return data["samples"], meta
    if isinstance(data, list):
        return data, meta
    return [], meta


def _infer_no_evidence_schema_from_meta(meta: dict[str, Any] | None) -> bool:
    """llm_as_a_judge_results.json meta에서 no-evidence 스키마로 볼지 추론."""
    if not meta:
        return False
    if meta.get("judge_rubric_version") == "v2_no_evidence":
        return True
    if meta.get("inference_no_evidence_prompt") is True:
        return True
    return False


def _extract_json_from_text(text: str) -> str | None:
    """텍스트에서 JSON 블록 추출. 첫 '{' ~ 마지막 '}'."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # 직접 파싱 시도
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    return text[start : end + 1]


def _parse_pred(pred: str) -> dict | None:
    """pred 문자열을 파싱해 dict 또는 None 반환."""
    if not pred:
        return None
    raw = _extract_json_from_text(pred)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _infer_required_keys_from_ref(ref: str) -> set[str]:
    """ref(또는 output) JSON에서 최상위 키 추출. 없으면 기본 REQUIRED_TOP_KEYS."""
    parsed = _parse_pred(ref) if isinstance(ref, str) else (ref if isinstance(ref, dict) else None)
    if not parsed or not isinstance(parsed, dict):
        return REQUIRED_TOP_KEYS
    keys = set(k for k in parsed if k in (REQUIRED_TOP_KEYS | OPTIONAL_TOP_KEYS))
    if keys >= REQUIRED_TOP_KEYS:
        return REQUIRED_TOP_KEYS
    return keys or REQUIRED_TOP_KEYS


def _check_aspect_cell(v: Any) -> bool:
    """한 축(service/price/food) 값이 {summary, bullets, evidence} 구조인지."""
    if not isinstance(v, dict):
        return False
    if "summary" not in v or not isinstance(v["summary"], str):
        return False
    if "bullets" not in v or not isinstance(v["bullets"], list):
        return False
    if "evidence" not in v or not isinstance(v["evidence"], list):
        return False
    if not all(isinstance(x, int) for x in v["evidence"]):
        return False
    return True


def _check_aspect_cell_no_evidence(v: Any) -> bool:
    """한 축이 {summary, bullets}만 필수 (evidence 없음)."""
    if not isinstance(v, dict):
        return False
    if "summary" not in v or not isinstance(v["summary"], str):
        return False
    if "bullets" not in v or not isinstance(v["bullets"], list):
        return False
    if not all(isinstance(x, str) for x in v["bullets"]):
        return False
    return True


def _check_overall_cell(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    return "summary" in v and isinstance(v["summary"], str)


def _schema_ok(parsed: dict, required_top_keys: set[str], *, no_evidence_schema: bool = False) -> bool:
    """필수 최상위 키 존재 + 각 축 타입/구조 일치."""
    if not isinstance(parsed, dict):
        return False
    for k in required_top_keys:
        if k not in parsed:
            return False
        v = parsed[k]
        if k == "overall_summary":
            if not _check_overall_cell(v):
                return False
        else:
            if no_evidence_schema:
                if not _check_aspect_cell_no_evidence(v):
                    return False
            else:
                if not _check_aspect_cell(v):
                    return False
    return True


def run_analysis(
    samples: list[dict],
    ref_key: str = "ref",
    pred_key: str = "pred",
    *,
    no_evidence_schema: bool = False,
) -> dict:
    """3가지 지표 계산."""
    n = len(samples)
    if n == 0:
        return {
            "n_samples": 0,
            "schema_mode": "no_evidence" if no_evidence_schema else "evidence",
            "json_parse_success_rate": 0.0,
            "schema_accuracy": 0.0,
            "length_drift": {},
            "details": [],
        }

    # ref에서 필수 키 추출 (첫 번째 파싱 가능한 ref 기준)
    required_top_keys = REQUIRED_TOP_KEYS
    for s in samples:
        ref = s.get(ref_key) or s.get("output", "")
        rk = _infer_required_keys_from_ref(ref)
        if rk:
            required_top_keys = rk
            break

    parse_ok = 0
    schema_ok_count = 0
    pred_lens: list[int] = []
    ref_lens: list[int] = []
    details: list[dict] = []

    for s in samples:
        pred = s.get(pred_key, "")
        ref = s.get(ref_key) or s.get("output", "")
        pred_str = pred if isinstance(pred, str) else json.dumps(pred, ensure_ascii=False)
        ref_str = ref if isinstance(ref, str) else json.dumps(ref, ensure_ascii=False)
        pl, rl = len(pred_str), len(ref_str)
        pred_lens.append(pl)
        ref_lens.append(rl)

        parsed = _parse_pred(pred_str)
        p_ok = parsed is not None
        if p_ok:
            parse_ok += 1
        s_ok = _schema_ok(parsed, required_top_keys, no_evidence_schema=no_evidence_schema) if parsed else False
        if s_ok:
            schema_ok_count += 1

        details.append({
            "sample_id": s.get("sample_id"),
            "parse_ok": p_ok,
            "schema_ok": s_ok,
            "pred_len": pl,
            "ref_len": rl,
            "ratio": round(pl / rl, 4) if rl else None,
        })

    avg_pred = sum(pred_lens) / n if pred_lens else 0
    avg_ref = sum(ref_lens) / n if ref_lens else 0
    ratio_avg = avg_pred / avg_ref if avg_ref else None

    return {
        "n_samples": n,
        "required_top_keys": list(required_top_keys),
        "schema_mode": "no_evidence" if no_evidence_schema else "evidence",
        "json_parse_success_rate": round(parse_ok / n, 4) if n else 0.0,
        "json_parse_success_count": parse_ok,
        "schema_accuracy": round(schema_ok_count / n, 4) if n else 0.0,
        "schema_accuracy_among_parsed": round(schema_ok_count / parse_ok, 4) if parse_ok else 0.0,
        "length_drift": {
            "avg_pred_len": round(avg_pred, 2),
            "avg_ref_len": round(avg_ref, 2),
            "pred_ref_ratio": round(ratio_avg, 4) if ratio_avg is not None else None,
            "min_pred_len": min(pred_lens) if pred_lens else None,
            "max_pred_len": max(pred_lens) if pred_lens else None,
        },
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KD SFT 분석: JSON 파싱률, 스키마 정확도, 길이 drift")
    parser.add_argument("--input", "-i", type=Path, required=True, help="llm_as_a_judge_results.json 또는 pred/ref 포함 JSON")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="보고서 저장 디렉터리 (기본: 입력 파일과 동일)")
    parser.add_argument("--output", type=Path, default=None, help="보고서 저장 경로 (파일 경로 지정 시 output-dir 무시, 파일명 자유 지정)")
    parser.add_argument("--ref-key", type=str, default="ref", help="참조 필드 이름 (ref 또는 output)")
    parser.add_argument("--pred-key", type=str, default="pred", help="예측 필드 이름")
    parser.add_argument(
        "--no-evidence-schema",
        action="store_true",
        help="축별 스키마를 summary+bullets만으로 검사 (evidence 불필요).",
    )
    parser.add_argument(
        "--evidence-schema",
        action="store_true",
        help="축별 스키마에 evidence 필수 (기본값; meta 자동 감지와 함께 쓸 때 명시).",
    )
    args = parser.parse_args()

    if args.no_evidence_schema and args.evidence_schema:
        print("Error: --no-evidence-schema and --evidence-schema cannot be used together", file=sys.stderr)
        sys.exit(1)

    path = args.input
    if not path.exists():
        print(f"Error: not found: {path}", file=sys.stderr)
        sys.exit(1)

    samples, file_meta = _load_input_and_meta(path)
    if not samples:
        print("Error: no results/samples in input", file=sys.stderr)
        sys.exit(1)

    if args.no_evidence_schema:
        no_ev_schema = True
    elif args.evidence_schema:
        no_ev_schema = False
    else:
        no_ev_schema = _infer_no_evidence_schema_from_meta(file_meta)

    result = run_analysis(
        samples,
        ref_key=args.ref_key,
        pred_key=args.pred_key,
        no_evidence_schema=no_ev_schema,
    )

    if args.output is not None:
        report_path = Path(args.output)
    else:
        out_dir = args.output_dir or path.parent
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "kd_sft_analysis_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 요약 출력
    print("=" * 60)
    print("KD SFT 분석 결과 (docs/distill/troubleshooting/kd_sft_analysis.md)")
    print("=" * 60)
    print(f"스키마 모드: {result.get('schema_mode', 'evidence')}")
    print(f"샘플 수: {result['n_samples']}")
    print(f"1. JSON 파싱 성공률: {result['json_parse_success_rate']:.2%} ({result['json_parse_success_count']}/{result['n_samples']})")
    print(f"2. 스키마 정확도(전체): {result['schema_accuracy']:.2%}")
    print(f"   스키마 정확도(파싱 성공 중): {result.get('schema_accuracy_among_parsed', 0):.2%}")
    ld = result["length_drift"]
    print(f"3. 길이/포맷 drift:")
    print(f"   평균 pred 길이: {ld['avg_pred_len']:.0f} | 평균 ref 길이: {ld['avg_ref_len']:.0f} | 비율: {ld.get('pred_ref_ratio') or '-'}")
    print(f"   pred 길이 범위: {ld.get('min_pred_len')} ~ {ld.get('max_pred_len')}")
    print("=" * 60)
    print(f"보고서 저장: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
