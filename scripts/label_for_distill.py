#!/usr/bin/env python3
"""
라벨링 스크립트: distill 파이프라인용 instruction + output 생성.

OpenAI 골드(소량) + self-hosted teacher(나머지) + 품질 필터 (distill_strategy.md).
출력: EasyDistill 형식 train_labeled.json (instruction, output per sample).

사용:
  python scripts/label_for_distill.py --train-path datasets/xxx/train.json --openai-cap 500 --output-dir labeled/
  python scripts/label_for_distill.py --phase openai_first --train-path ... --openai-cap 500 --output-dir labeled/  # 골드만(Pod 없이)
  python scripts/label_for_distill.py --phase teacher_rest --train-path ... --gold-labeled-path labeled/train_labeled_gold_only.json --output-dir labeled/  # 나머지 teacher + 병합
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Summary 파이프라인과 동일한 힌트
PRICE_HINTS = [
    "가격", "가성비", "저렴", "비싸", "비쌈", "가격대", "합리", "구성", "구성비",
    "양", "푸짐", "리필", "무한", "만족", "혜자",
]
SERVICE_HINTS = [
    "친절", "서비스", "직원", "매니저", "주방", "포장", "배달", "매장", "분위기",
    "예약", "주문", "대기", "설명", "위생",
]
FOOD_HINTS = [
    "맛", "음식", "요리", "메뉴", "맛있", "맛나", "맛집", "맛있다", "맛있어",
    "재료", "요리", "맛집", "맛요리",
]

SUMMARY_INSTRUCTIONS = """너는 음식점 리뷰 분석가다.
입력으로 카테고리별 '근거 리뷰 목록'이 주어진다.
아래 JSON 스키마로만 출력하라(추가 텍스트 금지).

스키마:
{
  "service": {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "price":   {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "food":    {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "overall_summary": {"summary": string}
}

규칙:
- 말투: 모든 summary, bullets, overall_summary는 반드시 "~해요" 체로 쓴다(예: 좋아요, 있어요, 없어요).
- bullets: 3~5개, 중복 제거, 구체적으로
- evidence: 근거로 쓴 리뷰의 인덱스(각 카테고리 리스트에서 0-based)
- price는 '가격 숫자'가 없으면 '가성비/양/구성/만족감' 같은 우회표현을 근거로 요약하라.
- overall_summary는 2~3문장으로 종합 요약하라.
- 근거(입력 리뷰)에 없는 내용은 추측하지 말고 "언급이 적어요"처럼 해요체로 표현하라.
"""

# 품질 필터: 금지 표현 (hallucination 의심)
FORBIDDEN_PHRASES = ["30년", "100년", "전통 있는", "역대급", "세계 최고"]


def _has_hint(text: str, hints: list[str]) -> bool:
    t = text.replace(" ", "")
    return any(k.replace(" ", "") in t for k in hints)


def _classify_review(content: str) -> str:
    """리뷰를 service/price/food 중 하나로 분류 (휴리스틱)."""
    if _has_hint(content, PRICE_HINTS):
        return "price"
    if _has_hint(content, SERVICE_HINTS):
        return "service"
    if _has_hint(content, FOOD_HINTS):
        return "food"
    return "food"  # 기본


def _build_payload(sample: dict[str, Any], per_category_max: int = 8) -> dict[str, list[str]]:
    """샘플의 리뷰를 service/price/food로 분류하여 LLM 입력 payload 생성."""
    service_reviews: list[str] = []
    price_reviews: list[str] = []
    food_reviews: list[str] = []
    for r in sample.get("reviews", []):
        content = (r.get("content") or "").strip()
        if not content:
            continue
        cat = _classify_review(content)
        if cat == "service":
            service_reviews.append(content)
        elif cat == "price":
            price_reviews.append(content)
        else:
            food_reviews.append(content)
    return {
        "service": service_reviews[:per_category_max],
        "price": price_reviews[:per_category_max],
        "food": food_reviews[:per_category_max],
    }


def _call_llm(
    payload: dict,
    use_openai: bool,
) -> str | None:
    """LLM 호출 (OpenAI 또는 self-hosted teacher)."""
    try:
        if use_openai:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        else:
            from src.llm_utils import LLMUtils
            llm = LLMUtils()
            messages = [
                {"role": "system", "content": SUMMARY_INSTRUCTIONS},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            return llm._generate_response(messages=messages, temperature=0.1, max_new_tokens=1500)
    except Exception as e:
        logger.warning("LLM 호출 실패: %s", e)
        return None


def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 추출."""
    from src.json_parse_utils import parse_json_relaxed, extract_json_block
    block = extract_json_block(text, want_object=True)
    if block:
        out = parse_json_relaxed(block)
        return out if isinstance(out, dict) else None
    return None


def _drop_evidence_fields(raw: dict) -> dict:
    """라벨 output에서 카테고리별 evidence 키 제거."""
    if not isinstance(raw, dict):
        return raw
    for cat in ("service", "price", "food"):
        cell = raw.get(cat)
        if isinstance(cell, dict):
            cell.pop("evidence", None)
    return raw


def _quality_filter(raw: dict, payload: dict) -> tuple[bool, str]:
    """
    품질 필터: JSON 구조, 길이, 금지 표현, 근거 휴리스틱.
    Returns (pass, reason).
    """
    if not isinstance(raw, dict):
        return False, "not_dict"
    for key in ("service", "price", "food", "overall_summary"):
        if key not in raw:
            return False, f"missing_{key}"
    total_len = 0
    input_words = set()
    for cat in ("service", "price", "food"):
        for s in payload.get(cat, []):
            input_words.update(s.split())
    for cat in ("service", "price", "food"):
        c = raw.get(cat, {})
        if isinstance(c, dict):
            total_len += len(c.get("summary", ""))
            total_len += sum(len(b) for b in c.get("bullets", []) if isinstance(b, str))
    os = raw.get("overall_summary", {})
    if isinstance(os, dict):
        total_len += len(os.get("summary", ""))
    if total_len < 50:
        return False, "too_short"
    if total_len > 8000:
        return False, "too_long"
    text_all = json.dumps(raw, ensure_ascii=False)
    for phr in FORBIDDEN_PHRASES:
        if phr in text_all:
            return False, f"forbidden:{phr}"
    return True, "ok"


def _label_samples(
    samples: list[dict],
    openai_cap: int,
    seed: int,
    teacher_for_rest: bool,
    drop_evidence_output: bool = False,
) -> tuple[list[dict], dict]:
    """한 split 라벨링. teacher_for_rest=False면 전부 OpenAI."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    labeled: list[dict] = []
    meta = {"openai_count": 0, "self_hosted_count": 0, "filter_drop_count": 0, "llm_fail_count": 0}
    for i, idx in enumerate(indices):
        sample = samples[idx]
        payload = _build_payload(sample)
        if not any(payload[k] for k in ("service", "price", "food")):
            continue
        use_openai = (i < openai_cap) if teacher_for_rest else True
        raw_text = _call_llm(payload, use_openai=use_openai)
        if raw_text is None:
            meta["llm_fail_count"] += 1
            continue
        parsed = _extract_json(raw_text)
        if parsed is None:
            meta["filter_drop_count"] += 1
            continue
        ok, _ = _quality_filter(parsed, payload)
        if not ok:
            meta["filter_drop_count"] += 1
            continue
        if drop_evidence_output:
            parsed = _drop_evidence_fields(parsed)
        instruction = json.dumps(payload, ensure_ascii=False)
        output = json.dumps(parsed, ensure_ascii=False)
        labeled.append({
            "sample_id": sample.get("sample_id"),
            "restaurant_id": sample.get("restaurant_id"),
            "instruction": instruction,
            "output": output,
            "label_source": "openai" if use_openai else "self_hosted",
        })
        if use_openai:
            meta["openai_count"] += 1
        else:
            meta["self_hosted_count"] += 1
        if (len(labeled) % 100) == 0:
            logger.info("Labeled %d samples", len(labeled))
    return labeled, meta


def _label_openai_first_only(
    samples: list[dict],
    openai_cap: int,
    seed: int,
    drop_evidence_output: bool = False,
) -> tuple[list[dict], list[int], dict]:
    """
    OpenAI 골드만 먼저 라벨링 (Pod 없이 호출용).
    반환: (labeled_gold_list, gold_train_indices, meta).
    labeled_gold_list[i]에 대응하는 train 인덱스는 gold_train_indices[i].
    """
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    labeled: list[dict] = []
    gold_indices: list[int] = []
    meta = {"openai_count": 0, "self_hosted_count": 0, "filter_drop_count": 0, "llm_fail_count": 0}
    for i, idx in enumerate(indices):
        if meta["openai_count"] >= openai_cap:
            break
        sample = samples[idx]
        payload = _build_payload(sample)
        if not any(payload[k] for k in ("service", "price", "food")):
            continue
        raw_text = _call_llm(payload, use_openai=True)
        if raw_text is None:
            meta["llm_fail_count"] += 1
            continue
        parsed = _extract_json(raw_text)
        if parsed is None:
            meta["filter_drop_count"] += 1
            continue
        ok, _ = _quality_filter(parsed, payload)
        if not ok:
            meta["filter_drop_count"] += 1
            continue
        if drop_evidence_output:
            parsed = _drop_evidence_fields(parsed)
        instruction = json.dumps(payload, ensure_ascii=False)
        output = json.dumps(parsed, ensure_ascii=False)
        labeled.append({
            "sample_id": sample.get("sample_id"),
            "restaurant_id": sample.get("restaurant_id"),
            "instruction": instruction,
            "output": output,
            "label_source": "openai",
            "train_index": idx,
        })
        gold_indices.append(idx)
        meta["openai_count"] += 1
        if (len(labeled) % 100) == 0:
            logger.info("OpenAI gold labeled %d samples", len(labeled))
    return labeled, gold_indices, meta


def _label_teacher_rest_and_merge(
    samples: list[dict],
    gold_labeled_path: Path,
    seed: int,
    drop_evidence_output: bool = False,
) -> tuple[list[dict], dict]:
    """
    골드 파일을 불러와 나머지만 teacher로 라벨링 후 병합.
    gold_labeled_path: train_labeled_gold_only.json (samples에 train_index 포함).
    """
    with open(gold_labeled_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_samples = gold_data.get("samples", [])
    gold_meta = gold_data.get("meta", {})
    gold_by_index: dict[int, dict] = {s["train_index"]: s for s in gold_samples if "train_index" in s}
    gold_indices_set = set(gold_by_index.keys())

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    meta = {"openai_count": len(gold_samples), "self_hosted_count": 0, "filter_drop_count": 0, "llm_fail_count": 0}
    result: list[dict] = []
    for idx in indices:
        if idx in gold_indices_set:
            s = gold_by_index[idx].copy()
            s.pop("train_index", None)
            result.append(s)
            continue
        sample = samples[idx]
        payload = _build_payload(sample)
        if not any(payload[k] for k in ("service", "price", "food")):
            continue
        raw_text = _call_llm(payload, use_openai=False)
        if raw_text is None:
            meta["llm_fail_count"] += 1
            continue
        parsed = _extract_json(raw_text)
        if parsed is None:
            meta["filter_drop_count"] += 1
            continue
        ok, _ = _quality_filter(parsed, payload)
        if not ok:
            meta["filter_drop_count"] += 1
            continue
        if drop_evidence_output:
            parsed = _drop_evidence_fields(parsed)
        instruction = json.dumps(payload, ensure_ascii=False)
        output = json.dumps(parsed, ensure_ascii=False)
        result.append({
            "sample_id": sample.get("sample_id"),
            "restaurant_id": sample.get("restaurant_id"),
            "instruction": instruction,
            "output": output,
            "label_source": "self_hosted",
        })
        meta["self_hosted_count"] += 1
        if (len(result) % 100) == 0:
            logger.info("Merged+teacher labeled %d samples", len(result))
    return result, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Label train samples for distill (OpenAI gold + teacher)")
    parser.add_argument("--phase", choices=["single", "openai_first", "teacher_rest"], default="single",
                        help="single: one pass. openai_first: gold only (no Pod). teacher_rest: merge gold + teacher rest")
    parser.add_argument("--train-path", type=Path, required=True, help="train.json path")
    parser.add_argument("--gold-labeled-path", type=Path, default=None, help="train_labeled_gold_only.json (phase=teacher_rest)")
    parser.add_argument("--val-path", type=Path, default=None, help="val.json (optional, OpenAI-only for eval)")
    parser.add_argument("--test-path", type=Path, default=None, help="test.json (optional, OpenAI-only for eval)")
    parser.add_argument("--openai-cap", type=int, default=500, help="OpenAI gold label cap for train")
    parser.add_argument("--openai-only", action="store_true", help="Teacher를 4o mini로 단일화: 전부 OpenAI(gpt-4o-mini)로만 라벨링, Pod/teacher 미사용")
    parser.add_argument("--drop-evidence-output", action="store_true", help="라벨 output JSON에서 evidence 키를 제거 (0.5B no-evidence 트랙)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    with open(args.train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_samples = train_data.get("samples", [])
    if not train_samples:
        logger.error("No samples in %s", args.train_path)
        sys.exit(1)

    if args.phase == "openai_first":
        labeled_gold, gold_indices, meta = _label_openai_first_only(
            train_samples,
            args.openai_cap,
            args.seed,
            drop_evidence_output=args.drop_evidence_output,
        )
        out_path = args.output_dir / "train_labeled_gold_only.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"samples": labeled_gold, "gold_indices": gold_indices, "meta": meta}, f, ensure_ascii=False, indent=2)
        results["labeled_path"] = str(out_path)
        logger.info("Wrote %s: %d gold samples", out_path, len(labeled_gold))
        for name, path, fn in [
            ("val_labeled_path", args.val_path, "val_labeled.json"),
            ("test_labeled_path", args.test_path, "test_labeled.json"),
        ]:
            if path and path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data.get("samples", [])
                lbl, _ = _label_samples(
                    samples,
                    openai_cap=999999,
                    seed=args.seed,
                    teacher_for_rest=False,
                    drop_evidence_output=args.drop_evidence_output,
                )
                p = args.output_dir / fn
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"samples": lbl, "meta": {"openai_count": len(lbl)}}, f, ensure_ascii=False, indent=2)
                results[name] = str(p)
                logger.info("Wrote %s: %d samples", p, len(lbl))
        out = {"labeled_path": results["labeled_path"], "meta": meta}
        if "val_labeled_path" in results:
            out["val_labeled_path"] = results["val_labeled_path"]
        if "test_labeled_path" in results:
            out["test_labeled_path"] = results["test_labeled_path"]
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.phase == "teacher_rest":
        if not args.gold_labeled_path or not args.gold_labeled_path.exists():
            logger.error("teacher_rest requires --gold-labeled-path")
            sys.exit(1)
        labeled, meta = _label_teacher_rest_and_merge(
            train_samples,
            args.gold_labeled_path,
            args.seed,
            drop_evidence_output=args.drop_evidence_output,
        )
        out_path = args.output_dir / "train_labeled.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"samples": labeled, "meta": meta}, f, ensure_ascii=False, indent=2)
        results["labeled_path"] = str(out_path)
        logger.info("Wrote %s: %d samples", out_path, len(labeled))
        out = {"labeled_path": results["labeled_path"], "meta": meta}
        print(json.dumps(out, ensure_ascii=False))
        return

    teacher_for_rest = not getattr(args, "openai_only", False)
    openai_cap = 999999 if getattr(args, "openai_only", False) else args.openai_cap
    labeled, meta = _label_samples(
        train_samples,
        openai_cap,
        args.seed,
        teacher_for_rest=teacher_for_rest,
        drop_evidence_output=args.drop_evidence_output,
    )
    out_path = args.output_dir / "train_labeled.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"samples": labeled, "meta": meta}, f, ensure_ascii=False, indent=2)
    results["labeled_path"] = str(out_path)
    logger.info("Wrote %s: %d samples (openai=%d, self_hosted=%d, filtered=%d, fail=%d)",
                out_path, len(labeled), meta["openai_count"], meta["self_hosted_count"],
                meta["filter_drop_count"], meta["llm_fail_count"])

    for name, path, fn in [
        ("val_labeled_path", args.val_path, "val_labeled.json"),
        ("test_labeled_path", args.test_path, "test_labeled.json"),
    ]:
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            samples = data.get("samples", [])
            lbl, _ = _label_samples(
                samples,
                openai_cap=999999,
                seed=args.seed,
                teacher_for_rest=False,
                drop_evidence_output=args.drop_evidence_output,
            )
            p = args.output_dir / fn
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"samples": lbl, "meta": {"openai_count": len(lbl)}}, f, ensure_ascii=False, indent=2)
            results[name] = str(p)
            logger.info("Wrote %s: %d samples", p, len(lbl))

    out = {"labeled_path": results["labeled_path"], "meta": meta}
    if "val_labeled_path" in results:
        out["val_labeled_path"] = results["val_labeled_path"]
    if "test_labeled_path" in results:
        out["test_labeled_path"] = results["test_labeled_path"]
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
