#!/usr/bin/env python3
"""
LLM-as-a-Judge 평가: sample_ids에 대해 student 추론 → OpenAI GPT-4o 품질 평가(1–5점 + 이유).

sample_ids는 --report( eval_distill report.json ) 또는 --llm-judge-samples(별도 JSON)로 지정.

사용:
  python scripts/eval_llm_as_judge.py --report eval/YYYYMMDD_HHMMSS/report.json \\
    --val-labeled labeled/.../val_labeled.json --adapter-path .../adapter --output .../llm_as_a_judge_results.json
  # --rubric-version v2_no_evidence 만 주면: 기본 no postprocess + no-evidence 프롬프트 (--postprocess / --evidence-prompt 로 전환)
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

from src.distill_summary import (
    load_model_and_tokenizer,
    generate_one as distill_generate_one,
    postprocess_prediction,
)


def _generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
    no_postprocess: bool = False,
    no_evidence_output: bool = False,

    no_evidence_prompt: bool = False,
) -> str:
    """eval_distill과 동일: system + few-shot + instruction으로 추론 후 JSON 추출·후처리. (distill_summary 공통 모듈 사용)"""
    return distill_generate_one(
        model, tokenizer, instruction,
        max_new_tokens=max_new_tokens,
        postprocess=not no_postprocess,
        no_evidence_output=no_evidence_output,

        no_evidence_prompt=no_evidence_prompt,
    )


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

# v2: teacher 기준 4요소 → 6축 매핑 (schema, fallback, style, evidence)
JUDGE_SYSTEM_PROMPT_V2 = """당신은 teacher(label_for_distill) 기준에 맞는 구조화 요약 품질을 평가하는 심사위원입니다.
평가 기준은 다음 4가지를 teacher와 동일하게 적용합니다.

## Teacher 기준 (4요소)
1. **teacher 스키마 준수**: service, price, food, overall_summary 필수. 각 카테고리는 summary, bullets, evidence. bullets 3~5개(근거 있을 때).
2. **teacher 폴백 정책**: 근거(입력 리뷰)가 없을 때 "언급이 적어요" 같은 해요체 폴백 사용. price 가격 숫자 없으면 가성비/양/구성/만족감 우회표현 허용. 빈 summary("") 대신 폴백 문구 사용이 정상.
3. **teacher 스타일**: "~해요" 체, summary 1문장, overall_summary 2~3문장, bullets 구체적·중복 제거.
4. **evidence input 기반**: evidence는 입력 리뷰의 0-based 인덱스만. bullet과 support 일치. 추측·허구 금지.

## 출력 형식 (반드시 유효한 JSON만)
{
  "schema_adherence": 1~5,
  "fallback_adherence": 1~5,
  "style_adherence": 1~5,
  "evidence_validity": 1~5,
  "faithfulness": 1~5,
  "category_correctness": 1~5,
  "reason": "한 줄 요약 (선택)"
}
모든 점수는 1(매우 나쁨) ~ 5(매우 좋음) 정수입니다.

## 6축 정의 (teacher 4요소에 매핑)
1. **schema_adherence**: teacher 스키마 준수(service, price, food, overall_summary, bullets 3~5 when evidence exists).
2. **fallback_adherence**: teacher 폴백 정책. 근거 없을 때 "언급이 적어요" 스타일 사용, price 우회표현 허용. 빈 문자열만 쓰면 감점.
3. **style_adherence**: teacher 스타일. "~해요" 체, 1문장 summary, overall 2~3문장, bullets 구체적.
4. **evidence_validity**: evidence가 입력 리뷰 인덱스 기반인가. bullet-support 관계 맞는가. 인덱스 오류 시 큰 감점.
5. **faithfulness**: 입력에 없는 내용 추론 금지. category별 실제 리뷰 근거 기반인가.
6. **category_correctness**: service/price/food 올바르게 분리. category 간 혼입(예: food→service) 감점.

## 반드시 적용할 규칙
- teacher 폴백: 근거 없으면 "언급이 적어요" 등 폴백 문구 사용이 정상. 빈 문자열만 있으면 fallback_adherence 감점.
- price: 가격 직접 언급 없으면 가성비/양/구성 우회표현 허용. 전혀 없으면 "가격 관련 언급이 적어요" 등 폴백 허용.
- evidence index 오류 시 evidence_validity·faithfulness 큰 감점."""

JUDGE_USER_TEMPLATE_V2 = """## Instruction (원문 리뷰/입력)
{instruction}

## Reference (teacher 정답 요약)
{reference}

## Prediction (평가 대상)
{prediction}

위 Prediction이 teacher 기준(스키마, 폴백 정책, 스타일, evidence 기반)을 따르는지 6축으로 1~5점 평가한 JSON을 출력하세요."""

JUDGE_SYSTEM_PROMPT_V2_NO_EVIDENCE = """당신은 teacher(label_for_distill) 기준의 요약 품질을 평가하는 심사위원입니다.
단, 이번 평가는 no-evidence 트랙으로 evidence 항목은 평가에서 제외합니다.

## 평가 기준
1. **스키마 준수**: service, price, food, overall_summary 필수. 각 카테고리는 summary, bullets 필수. overall_summary에는 summary만 허용.
2. **폴백 정책**: 근거가 부족할 때 "언급이 적어요" 같은 폴백 문구 사용. price는 가격 직접 언급이 없어도 가성비/양/구성 우회표현 허용.
3. **스타일**: "~해요" 체, summary 1문장, overall_summary 2~3문장, bullets 구체적·중복 제거.
4. **faithfulness**: 입력에 없는 내용 추론 금지.
5. **category_correctness**: service/price/food 분리 정확성.

## 출력 형식 (반드시 유효한 JSON만)
{
  "schema_adherence": 1~5,
  "fallback_adherence": 1~5,
  "style_adherence": 1~5,
  "faithfulness": 1~5,
  "category_correctness": 1~5,
  "reason": "한 줄 요약 (선택)"
}
모든 점수는 1(매우 나쁨) ~ 5(매우 좋음) 정수입니다.
"""

JUDGE_USER_TEMPLATE_V2_NO_EVIDENCE = """## Instruction (원문 리뷰/입력)
{instruction}

## Reference (teacher 정답 요약; evidence 필드는 무시)
{reference}

## Prediction (평가 대상; evidence 필드는 무시)
{prediction}

위 Prediction이 no-evidence 기준(스키마, 폴백, 스타일, faithfulness, 카테고리 분리)을 따르는지 5축으로 1~5점 평가한 JSON을 출력하세요."""

JUDGE_AXES = (
    "schema_adherence",
    "fallback_adherence",
    "style_adherence",
    "evidence_validity",
    "faithfulness",
    "category_correctness",
)

JUDGE_AXES_NO_EVIDENCE = (
    "schema_adherence",
    "fallback_adherence",
    "style_adherence",
    "faithfulness",
    "category_correctness",
)


def _strip_evidence_json_text(raw: str) -> str:
    if not raw:
        return raw
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(obj, dict):
        return raw
    for cat in ("service", "price", "food"):
        cell = obj.get(cat)
        if isinstance(cell, dict):
            cell.pop("evidence", None)
    return json.dumps(obj, ensure_ascii=False)


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
    """OpenAI GPT-4o로 LLM-as-a-judge 평가. rubric_version: v1(단일 총점) | v2(6축) | v2_no_evidence(5축)."""
    from openai import OpenAI

    if rubric_version == "v1":
        sys_prompt = JUDGE_SYSTEM_PROMPT_V1
        user_tpl = JUDGE_USER_TEMPLATE_V1
    elif rubric_version == "v2_no_evidence":
        sys_prompt = JUDGE_SYSTEM_PROMPT_V2_NO_EVIDENCE
        user_tpl = JUDGE_USER_TEMPLATE_V2_NO_EVIDENCE
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
    if rubric_version == "v2_no_evidence":
        return _parse_judge_v2_no_evidence(text)
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


def _parse_judge_v2_no_evidence(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            out["reason"] = parsed.get("reason", "")
            scores: list[int] = []
            for ax in JUDGE_AXES_NO_EVIDENCE:
                val = _clamp_score(parsed.get(ax, 1))
                out[ax] = val
                scores.append(val)
            out["score"] = round(sum(scores) / len(scores), 2) if scores else 1
        else:
            out = _fallback_parse_judge_v2_no_evidence(text)
    except json.JSONDecodeError:
        out = _fallback_parse_judge_v2_no_evidence(text)
    return out


def _fallback_parse_judge_v2_no_evidence(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"reason": f"parse_failed: {text[:200]}"}
    for ax in JUDGE_AXES_NO_EVIDENCE:
        m = re.search(rf'"{re.escape(ax)}"\s*:\s*(\d+)', text)
        out[ax] = _clamp_score(int(m.group(1))) if m else 1
    scores = [out[ax] for ax in JUDGE_AXES_NO_EVIDENCE]
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
    parser.add_argument("--rubric-version", choices=["v1", "v2", "v2_no_evidence"], default="v2", help="v1: 단일 총점, v2: 6축, v2_no_evidence: 5축")

    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="학생 추론 후 postprocess_prediction 적용. v2_no_evidence 기본은 끔(명시 시에만 켜짐). v1/v2는 기본 켜짐.",
    )
    parser.add_argument("--no-postprocess", action="store_true", help="후처리 끔 (v1/v2에서 명시 시)")
    parser.add_argument(
        "--evidence-prompt",
        action="store_true",
        help="evidence 포함 스키마 프롬프트. v2_no_evidence 기본은 no-evidence 프롬프트(이 플래그로 evidence 스키마로 전환).",
    )
    parser.add_argument(
        "--no-evidence-prompt",
        action="store_true",
        help="v1/v2에서도 no-evidence 스키마 프롬프트 사용(비교 실험용)",
    )
    parser.add_argument("--prediction-no-evidence", action="store_true", help="학생 예측에서 evidence 키 제거(evidence 스키마 추론일 때)")

    parser.add_argument("--judge-strip-evidence", action="store_true", help="judge 입력(ref/pred)에서 evidence 필드를 제거하고 평가")
    args = parser.parse_args()

    if args.postprocess and args.no_postprocess:
        raise ValueError("--postprocess와 --no-postprocess 동시 지정 불가")
    if args.evidence_prompt and args.no_evidence_prompt:
        raise ValueError("--evidence-prompt와 --no-evidence-prompt 동시 지정 불가")

    # v2_no_evidence: 기본 no postprocess + no-evidence 프롬프트 (--postprocess / --evidence-prompt로 전환)
    if args.rubric_version == "v2_no_evidence":
        use_postprocess = bool(args.postprocess)
        use_no_evidence_prompt = not bool(args.evidence_prompt)
    else:
        use_postprocess = not bool(args.no_postprocess)
        if args.postprocess:
            use_postprocess = True
        use_no_evidence_prompt = bool(args.no_evidence_prompt)

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
    model, tokenizer = load_model_and_tokenizer(str(args.adapter_path), args.base_model)

    results: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        logger.info("Inference+Judge: %d/%d sample_id=%s", i + 1, len(samples), s.get("sample_id"))
        ins = s.get("instruction", "")
        ref = s.get("output", "")
        pred = _generate_one(
            model,
            tokenizer,
            ins,
            max_new_tokens=1024,

            no_postprocess=not use_postprocess,
            no_evidence_output=args.prediction_no_evidence,
            no_evidence_prompt=use_no_evidence_prompt,

        )
        judge_ref = ref
        judge_pred = pred
        if args.judge_strip_evidence or args.rubric_version == "v2_no_evidence":
            judge_ref = _strip_evidence_json_text(ref)
            judge_pred = _strip_evidence_json_text(pred)
        judge_out = _call_judge(
            ins, judge_ref, judge_pred,
            rubric_version=args.rubric_version,
            model=args.openai_model,
            api_key=api_key,
        )
        row: dict[str, Any] = {
            "sample_id": s.get("sample_id"),
            "instruction": ins,
            "ref": ref,
            "pred": pred,
            "score": judge_out.get("score", 0),
            "reason": judge_out.get("reason", ""),
        }
        if args.rubric_version == "v2":
            for ax in JUDGE_AXES:
                row[ax] = judge_out.get(ax, 1)
        elif args.rubric_version == "v2_no_evidence":
            for ax in JUDGE_AXES_NO_EVIDENCE:
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
        "inference_postprocess": use_postprocess,
        "inference_no_evidence_prompt": use_no_evidence_prompt,
        "prediction_no_evidence_flag": bool(args.prediction_no_evidence),
    }
    if rubric == "v2":
        for ax in JUDGE_AXES:
            summary[f"avg_{ax}"] = round(sum(r[ax] for r in results) / n, 2) if n else 0
    elif rubric == "v2_no_evidence":
        for ax in JUDGE_AXES_NO_EVIDENCE:
            summary[f"avg_{ax}"] = round(sum(r[ax] for r in results) / n, 2) if n else 0
    out = {"meta": summary, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if rubric == "v2":
        logger.info(
            "Wrote %s (avg_score=%.2f, schema=%.2f, fallback=%.2f, evidence=%.2f)",
            args.output, summary["avg_score"], summary.get("avg_schema_adherence", 0), summary.get("avg_fallback_adherence", 0), summary.get("avg_evidence_validity", 0),
        )
    elif rubric == "v2_no_evidence":
        logger.info(
            "Wrote %s (avg_score=%.2f, schema=%.2f, fallback=%.2f, faithfulness=%.2f)",
            args.output, summary["avg_score"], summary.get("avg_schema_adherence", 0), summary.get("avg_fallback_adherence", 0), summary.get("avg_faithfulness", 0),
        )
    else:
        logger.info("Wrote %s (avg_score=%.2f)", args.output, summary["avg_score"])
    print(json.dumps({"output_path": str(args.output), "avg_score": summary["avg_score"], "judge_rubric_version": rubric}, ensure_ascii=False))


if __name__ == "__main__":
    main()
