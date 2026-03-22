#!/usr/bin/env python3
"""
라벨 파일(train_labeled/val_labeled/test_labeled)에서 output JSON의 evidence 키를 제거한다.

사용:
  python scripts/strip_evidence_from_labeled.py \
    --input distill_pipeline_output/labeled/20260313_040811/train_labeled.json \
    --output distill_pipeline_output/labeled/20260313_040811/train_labeled_no_evidence.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _drop_evidence(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    for cat in ("service", "price", "food"):
        cell = obj.get(cat)
        if isinstance(cell, dict):
            cell.pop("evidence", None)
    return obj


def _transform_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    raw = sample.get("output", "")
    if not isinstance(raw, str) or not raw.strip():
        return out
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return out
    parsed = _drop_evidence(parsed)
    out["output"] = json.dumps(parsed, ensure_ascii=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip evidence from labeled output JSON")
    parser.add_argument("--input", type=Path, required=True, help="입력 labeled json 경로")
    parser.add_argument("--output", type=Path, required=True, help="출력 json 경로")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        samples = data["samples"]
        data["samples"] = [_transform_sample(s) for s in samples if isinstance(s, dict)]
        data.setdefault("meta", {})
        data["meta"]["evidence_stripped"] = True
        data["meta"]["evidence_stripped_from"] = str(args.input)
        out_data = data
    elif isinstance(data, list):
        out_data = [_transform_sample(s) for s in data if isinstance(s, dict)]
    else:
        raise ValueError("지원하지 않는 입력 형식입니다. dict(samples) 또는 list 형식이 필요합니다.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(json.dumps({"output_path": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
