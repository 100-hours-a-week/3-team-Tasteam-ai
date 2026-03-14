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


def _extract_json_for_rouge(raw: str) -> str:
    """eval_distill과 동일: 출력에서 JSON 블록만 추출해 반환. 실패 시 원문."""
    if not raw or not raw.strip():
        return raw or ""
    try:
        from src.json_parse_utils import extract_json_block, parse_json_relaxed
        from src.schema_repair import repair_summary_schema
        block = extract_json_block(raw.strip(), want_object=True)
        if not block:
            return raw.strip()
        parsed = parse_json_relaxed(block)
        if isinstance(parsed, dict) and any(k in parsed for k in ("service", "price", "food")):
            repaired = repair_summary_schema(parsed, bullet_max=3)
            if isinstance(repaired, dict):
                return json.dumps(repaired, ensure_ascii=False)
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    return raw.strip()


# eval_distill과 동일한 프롬프트: report.json 생성 시와 같은 조건으로 추론
_SCHEMA_ENFORCEMENT_SYSTEM = """You are a JSON generator for review summarization.

Return ONLY one valid JSON object.
Do not output any text before or after the JSON.
Do not output markdown.
Do not output explanations.

The top-level keys must be exactly:
service, food, price

Each of service, food, and price must be a JSON object with exactly these keys:
summary, bullets, evidence

Never add any other keys.
Never use keys such as:
examples, impact, weight, title, body, rating, overall_summary

Rules:
- summary must be exactly 1 Korean sentence, or "" if there is not enough evidence
- bullets must be a list of 0 to 3 short Korean strings
- evidence must be a list of 0-based integer indices into the corresponding category list
- evidence must have the same number of items as bullets
- If bullets is [], evidence must be []
- If there is not enough evidence, use:
  "summary": "",
  "bullets": [],
  "evidence": []
- service: include only kindness, service/waiting, atmosphere/seating convenience
- price: allow empty bullets when there is no direct mention of price
- food: focus on menu, taste, and texture only
- evidence: use only indices that refer to actual sentences in the input; no fabrication
- Do not infer or add content that is not present in the input

Output only JSON.
"""

_TINY_FEWSHOT_USER = """Example input:
{"service":["직원분이 친절해요"],"price":[],"food":["국물이 진해요"]}
"""

_TINY_FEWSHOT_ASSISTANT = """{"service":{"summary":"직원분이 친절해요.","bullets":["직원 응대가 친절해요."],"evidence":[0]},"food":{"summary":"국물이 진해요.","bullets":["국물이 진하고 맛있어요."],"evidence":[0]},"price":{"summary":"","bullets":[],"evidence":[]}}"""

_TINY_FEWSHOT_USER_2 = """Example input:
{"service":["직원들이 빠르게 응대해요","매장이 깔끔해요"],"price":["양이 많아요"],"food":[]}
"""

_TINY_FEWSHOT_ASSISTANT_2 = """{"service":{"summary":"직원 응대가 빠르고 매장이 깔끔해요.","bullets":["직원들이 빠르게 응대해요.","매장이 깔끔해요."],"evidence":[0,1]},"food":{"summary":"","bullets":[],"evidence":[]},"price":{"summary":"양이 많아요.","bullets":["양이 많아요."],"evidence":[0]}}"""


def _generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
) -> str:
    """eval_distill과 동일: system + 2개 few-shot + instruction으로 추론 후 JSON 추출."""
    messages = [
        {"role": "system", "content": _SCHEMA_ENFORCEMENT_SYSTEM},
        {"role": "user", "content": _TINY_FEWSHOT_USER},
        {"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT},
        {"role": "user", "content": _TINY_FEWSHOT_USER_2},
        {"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": instruction},
    ]
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


# v1: 단일 총점 (구 방식)
JUDGE_SYSTEM_PROMPT_V1 = """당신은 레스토랑 리뷰 요약 품질을 평가하는 심사위원입니다.
입력: (1) 원문 리뷰/지시문(instruction), (2) 참조 요약(reference), (3) 평가 대상 모델의 예측 요약(prediction)
출력: JSON 형식으로 {"score": 1~5, "reason": "한 줄 이유"}
- score: 1(매우 나쁨) ~ 5(매우 좋음). 참조와 비교해 정확성, 요약 완전성, 자연스러움을 종합 판단.
- reason: 한 줄로 핵심 이유를 한국어로 기술.
반드시 유효한 JSON만 출력하세요."""

JUDGE_USER_TEMPLATE_V1 = """## Instruction (원문)
{instruction}

## Reference (정답 요약)
{reference}

## Prediction (평가 대상 모델 출력)
{prediction}

위 prediction을 reference와 비교해 1~5점으로 평가하고 이유를 JSON으로 출력하세요."""

# v2: 6축 루브릭 (구조화 요약 품질용, prompt_jv2)
JUDGE_SYSTEM_PROMPT_V2 = """당신은 레스토랑 리뷰의 **구조화 요약** 품질을 평가하는 심사위원입니다.
평가 대상은 JSON 형식의 요약(service, food, price 각각 summary/bullets/evidence)입니다.
자유 요약이 아니라, 입력(instruction)에 근거한 구조 준수·근거 정합성이 중요합니다.

## 출력 형식 (반드시 유효한 JSON만)
{
  "faithfulness": 1~5,
  "category_correctness": 1~5,
  "schema_adherence": 1~5,
  "evidence_validity": 1~5,
  "completeness": 1~5,
  "naturalness": 1~5,
  "reason": "한 줄 요약 (선택)"
}
모든 점수는 1(매우 나쁨) ~ 5(매우 좋음) 정수입니다.

## 평가 축 정의
1. **faithfulness**: 입력에 없는 내용이 들어갔는가. category별 내용이 실제 리뷰 근거에 기반하는가.
2. **category_correctness**: service / price / food가 올바르게 분리되었는가. food 내용을 service에 넣거나 그 반대가 없는가.
3. **schema_adherence**: required key 존재, 허용된 구조 유지. **price에 직접 언급이 없으면 빈 bullets 허용**이 잘 지켜졌는가.
4. **evidence_validity**: evidence index가 실제 문장 인덱스인가. bullet과 support 관계가 맞는가.
5. **completeness**: 입력에 존재하는 중요한 포인트를 과도하게 누락하지 않았는가.
6. **naturalness**: 문장이 너무 깨지거나 반복되지 않는가.

## 반드시 적용할 규칙
- **입력에 없는 내용 추론 시 큰 감점** (faithfulness 낮게).
- **evidence index 오류 시 큰 감점** (evidence_validity 낮게).
- **price 직접 언급이 없으면 빈 bullets 허용, 감점 금지** (schema_adherence에서 유리하게).
- **service**는 친절/응대/대기/분위기·좌석 편의 범위만. 이외 내용이 있으면 category_correctness 감점.
- **food**는 메뉴/맛/식감 중심만. 이외 내용이 있으면 category_correctness 감점.
- **category 간 혼입**(예: food 내용을 service에 넣음) 시 category_correctness 감점."""

JUDGE_USER_TEMPLATE_V2 = """## Instruction (원문 리뷰/입력)
{instruction}

## Reference (정답 구조화 요약)
{reference}

## Prediction (평가 대상 모델 출력)
{prediction}

위 Prediction을 Reference 및 Instruction과 비교하여, 6개 축(faithfulness, category_correctness, schema_adherence, evidence_validity, completeness, naturalness) 각각 1~5점으로 평가한 JSON을 출력하세요."""


JUDGE_AXES = (
    "faithfulness",
    "category_correctness",
    "schema_adherence",
    "evidence_validity",
    "completeness",
    "naturalness",
)


def _clamp_score(v: Any) -> int:
    if isinstance(v, (int, float)):
        return max(1, min(5, int(v)))
    return 1


def _call_judge(
    instruction: str,
    reference: str,
    prediction: str,
    *,
    rubric_version: str = "v2",
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> dict[str, Any]:
    """OpenAI GPT-4o로 LLM-as-a-judge 평가. rubric_version: v1(단일 총점) | v2(6축)."""
    from openai import OpenAI

    if rubric_version == "v1":
        sys_prompt = JUDGE_SYSTEM_PROMPT_V1
        user_tpl = JUDGE_USER_TEMPLATE_V1
    else:
        sys_prompt = JUDGE_SYSTEM_PROMPT_V2
        user_tpl = JUDGE_USER_TEMPLATE_V2

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_content = user_tpl.format(
        instruction=instruction[:4000] if len(instruction) > 4000 else instruction,
        reference=reference[:3000] if len(reference) > 3000 else reference,
        prediction=prediction[:3000] if len(prediction) > 3000 else prediction,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(\{[\s\S]*?\})\s*```", text)
        if m:
            text = m.group(1)

    if rubric_version == "v1":
        return _parse_judge_v1(text)
    return _parse_judge_v2(text)


def _parse_judge_v1(text: str) -> dict[str, Any]:
    """v1: {"score": 1~5, "reason": "..."} 파싱."""
    out: dict[str, Any] = {"reason": "", "score": 1}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            out["reason"] = parsed.get("reason", "")
            out["score"] = _clamp_score(parsed.get("score", 1))
    except json.JSONDecodeError:
        m = re.search(r'"score"\s*:\s*(\d+)', text)
        out["score"] = _clamp_score(int(m.group(1))) if m else 1
        mr = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        out["reason"] = mr.group(1) if mr else f"parse_failed: {text[:150]}"
    return out


def _parse_judge_v2(text: str) -> dict[str, Any]:
    """v2: 6축 + reason 파싱."""
    out: dict[str, Any] = {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            out["reason"] = parsed.get("reason", "")
            scores: list[int] = []
            for ax in JUDGE_AXES:
                val = _clamp_score(parsed.get(ax, 1))
                out[ax] = val
                scores.append(val)
            out["score"] = round(sum(scores) / len(scores), 2) if scores else 1
        else:
            out = _fallback_parse_judge_v2(text)
    except json.JSONDecodeError:
        out = _fallback_parse_judge_v2(text)
    return out


def _fallback_parse_judge_v2(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"reason": f"parse_failed: {text[:200]}"}
    for ax in JUDGE_AXES:
        m = re.search(rf'"{re.escape(ax)}"\s*:\s*(\d+)', text)
        out[ax] = _clamp_score(int(m.group(1))) if m else 1
    scores = [out[ax] for ax in JUDGE_AXES]
    out["score"] = round(sum(scores) / len(scores), 2) if scores else 1
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
    parser.add_argument("--rubric-version", choices=["v1", "v2"], default="v2", help="v1: 단일 총점, v2: 6축 루브릭 (기본 v2)")
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
        judge_out = _call_judge(
            ins, ref, pred,
            rubric_version=args.rubric_version,
            model=args.openai_model,
            api_key=api_key,
        )
        row: dict[str, Any] = {
            "sample_id": s.get("sample_id"),
            "instruction": ins[:500] + "..." if len(ins) > 500 else ins,
            "ref": ref[:500] + "..." if len(ref) > 500 else ref,
            "pred": pred,
            "score": judge_out.get("score", 0),
            "reason": judge_out.get("reason", ""),
        }
        if args.rubric_version == "v2":
            for ax in JUDGE_AXES:
                row[ax] = judge_out.get(ax, 1)
        results.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    rubric = args.rubric_version
    summary: dict[str, Any] = {
        "n_samples": n,
        "avg_score": round(sum(r["score"] for r in results) / n, 2) if n else 0,
        "judge_model": args.openai_model,
        "adapter_path": str(args.adapter_path),
        "judge_rubric_version": rubric,
    }
    if rubric == "v2":
        for ax in JUDGE_AXES:
            summary[f"avg_{ax}"] = round(sum(r[ax] for r in results) / n, 2) if n else 0
    out = {"meta": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if rubric == "v2":
        logger.info(
            "Wrote %s (avg_score=%.2f, avg_faithfulness=%.2f, avg_evidence_validity=%.2f)",
            args.output, summary["avg_score"], summary.get("avg_faithfulness", 0), summary.get("avg_evidence_validity", 0),
        )
    else:
        logger.info("Wrote %s (avg_score=%.2f)", args.output, summary["avg_score"])
    print(json.dumps({"output_path": str(args.output), "avg_score": summary["avg_score"], "judge_rubric_version": rubric}, ensure_ascii=False))


if __name__ == "__main__":
    main()
