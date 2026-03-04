#!/usr/bin/env python3
"""
Distill 평가 스크립트: student adapter → val/test ROUGE/BERTScore.

val_labeled_path, test_labeled_path: instruction + output(ground truth) 형식.
student로 instruction에 대해 추론 후 output과 비교.

사용:
  python scripts/eval_distill.py --val-labeled labeled/val_labeled.json --test-labeled labeled/test_labeled.json --adapter-path runs/xxx/adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir eval/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def _load_labeled(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data
    return samples


def _load_model_and_tokenizer(adapter_path: str, base_model: str):
    """모델·토크나이저를 한 번만 로드. (model, tokenizer) 반환."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def _extract_json_for_rouge(raw: str) -> str:
    """JSON 스키마 강제(후처리): 출력에서 JSON 블록만 추출해 ROUGE용 문자열로 반환. 실패 시 원문."""
    if not raw or not raw.strip():
        return raw or ""
    try:
        from src.json_parse_utils import extract_json_block, parse_json_relaxed
        block = extract_json_block(raw.strip(), want_object=True)
        if not block:
            return raw.strip()
        parsed = parse_json_relaxed(block)
        if isinstance(parsed, dict) and any(k in parsed for k in ("service", "price", "food")):
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    return raw.strip()


def _generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
) -> str:
    """이미 로드된 model/tokenizer로 instruction 한 건만 추론."""
    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    raw = generated.strip()
    return _extract_json_for_rouge(raw)


def _rouge(pred: str, ref: str) -> dict[str, float]:
    if not ROUGE_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    s = scorer.score(ref, pred)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


def _bertscore(preds: list[str], refs: list[str], lang: str = "ko") -> dict[str, float]:
    if not BERTSCORE_AVAILABLE or not preds or not refs:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}
    P, R, F1 = bert_score_fn(preds, refs, lang=lang, verbose=False)
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval student adapter on val/test")
    parser.add_argument("--val-labeled", type=Path, help="val_labeled.json (instruction+output)")
    parser.add_argument("--test-labeled", type=Path, help="test_labeled.json")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-eval", type=int, default=0, help="Max samples per split (0=all)")
    args = parser.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / "eval" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"val": {}, "test": {}, "meta": {}}
    model, tokenizer = None, None

    for split, path in [("val", args.val_labeled), ("test", args.test_labeled)]:
        if not path or not path.exists():
            report[split] = {"skipped": True, "reason": "path not provided or missing"}
            continue
        samples = _load_labeled(path)
        if args.max_eval > 0:
            samples = samples[: args.max_eval]
        if not samples:
            report[split] = {"skipped": True, "reason": "no samples"}
            continue

        if model is None:
            logger.info("Loading model and tokenizer (once for all splits)")
            model, tokenizer = _load_model_and_tokenizer(str(args.adapter_path), args.base_model)

        preds: list[str] = []
        refs: list[str] = []
        for i, s in enumerate(samples):
            if (i + 1) % 50 == 0:
                logger.info("%s: %d/%d", split, i + 1, len(samples))
            ins = s.get("instruction", "")
            ref = s.get("output", "")
            pred = _generate_one(model, tokenizer, ins, max_new_tokens=1024)
            preds.append(pred)
            refs.append(ref)

        rouge_scores = [_rouge(p, r) for p, r in zip(preds, refs)]
        r1 = sum(x["rouge1"] for x in rouge_scores) / len(rouge_scores) if rouge_scores else 0
        r2 = sum(x["rouge2"] for x in rouge_scores) / len(rouge_scores) if rouge_scores else 0
        rl = sum(x["rougeL"] for x in rouge_scores) / len(rouge_scores) if rouge_scores else 0
        bs = _bertscore(preds, refs)
        report[split] = {
            "n_samples": len(samples),
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
            **bs,
        }
        report[split]["samples"] = [
            {"sample_id": s.get("sample_id"), "pred_len": len(p), "ref_len": len(r)}
            for s, p, r in zip(samples[:10], preds[:10], refs[:10])
        ]

    human_path = out_dir / "human_eval_samples.json"
    sample_ids = [s.get("sample_id") for s in (_load_labeled(args.val_labeled) if args.val_labeled and args.val_labeled.exists() else [])[:50]]
    human_path.write_text(json.dumps({"sample_ids": sample_ids}, ensure_ascii=False, indent=2))

    report_path = out_dir / "report.json"
    report["meta"] = {"adapter_path": str(args.adapter_path), "base_model": args.base_model}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Report: %s", report_path)
    print(json.dumps({"report_path": str(report_path), "human_eval_sample_path": str(human_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
