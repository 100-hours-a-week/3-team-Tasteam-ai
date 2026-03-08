#!/usr/bin/env python3
"""
LLM-as-a-Judge 평가: sample_ids에 대해 student 추론 → OpenAI GPT-4o 품질 평가(1–5점 + 이유).

sample_ids는 --report( eval_distill report.json ) 또는 --llm-judge-samples(별도 JSON)로 지정.

사용:
  python scripts/eval_llm_as_judge.py --report eval/YYYYMMDD_HHMMSS/report.json \\
    --val-labeled labeled/.../val_labeled.json --adapter-path .../adapter --output .../llm_as_a_judge_results.json
  python scripts/eval_llm_as_judge.py --llm-judge-samples .../llm_as_a_judge_samples.json \\
    --val-labeled ... --adapter-path ... --output .../llm_as_a_judge_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


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
    return generated.strip()


JUDGE_SYSTEM_PROMPT = """당신은 레스토랑 리뷰 요약 품질을 평가하는 심사위원입니다.
입력: (1) 원문 리뷰/지시문(instruction), (2) 참조 요약(reference), (3) 평가 대상 모델의 예측 요약(prediction)
출력: JSON 형식으로 {"score": 1~5, "reason": "한 줄 이유"}
- score: 1(매우 나쁨) ~ 5(매우 좋음). 참조와 비교해 정확성, 요약 완전성, 자연스러움을 종합 판단.
- reason: 한 줄로 핵심 이유를 한국어로 기술.
반드시 유효한 JSON만 출력하세요."""

JUDGE_USER_TEMPLATE = """## Instruction (원문)
{instruction}

## Reference (정답 요약)
{reference}

## Prediction (평가 대상 모델 출력)
{prediction}

위 prediction을 reference와 비교해 1~5점으로 평가하고 이유를 JSON으로 출력하세요."""


def _call_judge(
    instruction: str,
    reference: str,
    prediction: str,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> dict[str, Any]:
    """OpenAI GPT-4o로 LLM-as-a-judge 평가. {"score": int, "reason": str} 반환."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_content = JUDGE_USER_TEMPLATE.format(
        instruction=instruction[:4000] if len(instruction) > 4000 else instruction,
        reference=reference[:3000] if len(reference) > 3000 else reference,
        prediction=prediction[:3000] if len(prediction) > 3000 else prediction,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    # 마크다운 코드블록 제거
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```", text)
        if m:
            text = m.group(1)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'"score"\s*:\s*(\d+)', text)
        score = int(m.group(1)) if m else 0
        mr = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        reason = mr.group(1) if mr else f"parse_failed: {text[:150]}"
        out = {"score": score, "reason": reason}
    score_val = out.get("score")
    if isinstance(score_val, (int, float)):
        out["score"] = max(1, min(5, int(score_val)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge: student 추론 + GPT-4o 평가")
    parser.add_argument("--report", type=Path, default=None, help="eval_distill report.json (meta.llm_judge_sample_ids 사용)")
    parser.add_argument("--llm-judge-samples", type=Path, default=None, help="llm_as_a_judge_samples.json (sample_ids). --report 없을 때만 사용")
    parser.add_argument("--val-labeled", type=Path, required=True, help="val_labeled.json (instruction+output)")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True, help="llm_as_a_judge_results.json 저장 경로")
    parser.add_argument("--openai-model", type=str, default="gpt-4o", help="Judge 모델 (OpenAI)")
    parser.add_argument("--openai-api-key", type=str, default=None, help="또는 OPENAI_API_KEY 환경변수")
    parser.add_argument("--max-samples", type=int, default=0, help="평가할 최대 샘플 수 (0=전부)")
    args = parser.parse_args()

    if not args.report and not args.llm_judge_samples:
        raise ValueError("--report 또는 --llm-judge-samples 중 하나 필요")
    if args.report and args.llm_judge_samples:
        raise ValueError("--report와 --llm-judge-samples 동시 지정 불가")
    if args.report and not args.report.exists():
        raise FileNotFoundError(f"report not found: {args.report}")
    if args.llm_judge_samples and not args.llm_judge_samples.exists():
        raise FileNotFoundError(f"llm_as_a_judge_samples not found: {args.llm_judge_samples}")
    if not args.val_labeled.exists():
        raise FileNotFoundError(f"val_labeled not found: {args.val_labeled}")
    if not args.adapter_path.exists():
        raise FileNotFoundError(f"adapter_path not found: {args.adapter_path}")

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수 또는 --openai-api-key 필요")

    if args.report:
        with open(args.report, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        sample_ids_list = report_data.get("meta", {}).get("llm_judge_sample_ids", [])
    else:
        with open(args.llm_judge_samples, "r", encoding="utf-8") as f:
            judge_data = json.load(f)
        sample_ids_list = judge_data.get("sample_ids", [])
    sample_ids = set(sample_ids_list)

    with open(args.val_labeled, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_samples = data.get("samples", data) if isinstance(data, dict) else data
    samples = [s for s in all_samples if s.get("sample_id") in sample_ids]
    # sample_ids 순서 유지
    sid_order = {sid: i for i, sid in enumerate(sample_ids_list)}
    samples.sort(key=lambda s: sid_order.get(s.get("sample_id"), 999999))

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if not samples:
        raise ValueError("No samples found for given sample_ids")

    logger.info("Loading model and tokenizer")
    model, tokenizer = _load_model_and_tokenizer(str(args.adapter_path), args.base_model)

    results: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        logger.info("Inference+Judge: %d/%d sample_id=%s", i + 1, len(samples), s.get("sample_id"))
        ins = s.get("instruction", "")
        ref = s.get("output", "")
        pred = _generate_one(model, tokenizer, ins, max_new_tokens=1024)
        judge_out = _call_judge(ins, ref, pred, model=args.openai_model, api_key=api_key)
        results.append({
            "sample_id": s.get("sample_id"),
            "instruction": ins[:500] + "..." if len(ins) > 500 else ins,
            "ref": ref[:500] + "..." if len(ref) > 500 else ref,
            "pred": pred,
            "score": judge_out.get("score", 0),
            "reason": judge_out.get("reason", ""),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_samples": len(results),
        "avg_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "judge_model": args.openai_model,
        "adapter_path": str(args.adapter_path),
    }
    out = {"meta": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %s (avg_score=%.2f)", args.output, summary["avg_score"])
    print(json.dumps({"output_path": str(args.output), "avg_score": summary["avg_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
