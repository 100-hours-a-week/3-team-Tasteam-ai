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
            repaired = repair_summary_schema(parsed, bullet_max=5)
            if isinstance(repaired, dict):
                return json.dumps(repaired, ensure_ascii=False)
            return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass
    return raw.strip()


def _get_category_lengths(instruction: str) -> dict[str, int]:
    """instruction(JSON)에서 카테고리별 리뷰 개수 반환. 파싱 실패 시 0."""
    out: dict[str, int] = {"service": 0, "price": 0, "food": 0}
    if not instruction or not instruction.strip():
        return out
    try:
        payload = json.loads(instruction)
        if isinstance(payload, dict):
            for k in out:
                arr = payload.get(k)
                if isinstance(arr, list):
                    out[k] = len(arr)
    except json.JSONDecodeError:
        pass
    return out


def _postprocess_prediction(pred_json_str: str, instruction: str) -> str:
    """
    추론 결과 후처리: evidence 인덱스 검증·보정, 빈 카테고리 폴백 치환.
    - evidence를 해당 카테고리 리뷰 개수 범위로만 유지(범위 밖 제거).
    - evidence와 bullets 1:1 맞춤: evidence 부족 시 bullets를 자름(0 패딩 없음).
    - summary/bullets/evidence가 모두 비어 있으면 teacher 폴백 문구로 채움.
    - overall_summary에 evidence 키가 있으면 제거.
    """
    if not pred_json_str or not pred_json_str.strip():
        return pred_json_str
    try:
        pred = json.loads(pred_json_str)
    except json.JSONDecodeError:
        return pred_json_str
    if not isinstance(pred, dict) or not any(k in pred for k in ("service", "price", "food")):
        return pred_json_str

    lengths = _get_category_lengths(instruction)
    fallback_price = "가격 관련 언급이 적어요."
    fallback_other = "언급이 적어요."

    def _as_str(x: Any) -> str:
        return (x if isinstance(x, str) else str(x) if x is not None else "") or ""

    for cat in ("service", "price", "food"):
        n = lengths.get(cat, 0)
        cell = pred.get(cat)
        if not isinstance(cell, dict):
            pred[cat] = {"summary": "", "bullets": [], "evidence": []}
            cell = pred[cat]

        summary = cell.get("summary")
        if not isinstance(summary, str):
            summary = ""
        bullets = cell.get("bullets")
        if not isinstance(bullets, list):
            bullets = []
        bullets = [b for b in bullets if isinstance(b, str) and b.strip()][:5]

        ev_raw = cell.get("evidence", [])
        evidence: list[int] = []
        if isinstance(ev_raw, list):
            for x in ev_raw:
                try:
                    i = int(x)
                    if 0 <= i < n:
                        evidence.append(i)
                except (TypeError, ValueError):
                    continue
        if evidence and n == 0:
            evidence = []
        # evidence와 bullets 1:1 맞춤: evidence 초과분은 자르고, evidence 부족 시 bullets를 자름(0 패딩 없음)
        if len(evidence) > len(bullets):
            evidence = evidence[: len(bullets)]
        elif len(evidence) < len(bullets):
            bullets = bullets[: len(evidence)]

        # 빈 카테고리 폴백
        if not summary.strip() and not bullets:
            summary = fallback_price if cat == "price" else fallback_other
            bullets = []
            evidence = []

        pred[cat] = {"summary": summary.strip(), "bullets": bullets, "evidence": evidence}

    # overall_summary: summary만 유지. bullets, evidence 제거(스키마에 없음)
    ov = pred.get("overall_summary")
    if isinstance(ov, dict):
        ov = {"summary": _as_str(ov.get("summary", "")).strip()[:200]}
        pred["overall_summary"] = ov

    return json.dumps(pred, ensure_ascii=False)


# eval_distill·teacher와 동일한 프롬프트
_SCHEMA_ENFORCEMENT_SYSTEM = """당신은 리뷰 요약 어시스턴트입니다.
입력과 출력은 항상 JSON 형식이다.
다음은 입력과 출력의 JSON 스키마이다.

입력 JSON 스키마:
{
  "service": [string, ...],
  "price": [string, ...],
  "food": [string, ...]
}

출력 JSON 스키마:
{
  "service": {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "price":   {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "food":    {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "overall_summary": {"summary": string}
}

입력 JSON 스키마 설명
- service/price/food 각각 근거 리뷰 문자열 배열. 입력 리뷰 배열의 첫번째 인덱스는 0.

출력 JSON 스키마 설명
- summary: 해당 카테고리 입력 리뷰들의 총 요약문. bullets: 해당 카테고리 입력 리뷰들의 요소별 요약문.
- evidence: bullets를 지지하는 입력 리뷰의 인덱스 배열. 입력 리뷰의 첫번째 인덱스는 0.
- overall_summary에는 summary만 있고 bullets/evidence 없음.

출력 시 따라야 하는 규칙
- 가격 직접 언급이 없으면 "가격 언급이 적어요" 등 우회 표현. 말투는 "~해요" 체.
- 반드시 출력 JSON 스키마 형태의 JSON을 출력하세요. 출력 JSON 앞뒤에 다른 글자나 설명 넣지 말 것.
- evidence는 bullets를 지지하는 입력 리뷰 인덱스 배열이어야 한다.
"""

_TINY_FEWSHOT_USER = """
예시 입력:
{"service":["맨날 점심시간만되면 엄청 웨이팅 장난아니라서 점심시간 아닐 때 방문해봤어요! 직원분들도 너무 친절하고 좋습니다!","판교 베트남 음식 르 메콩\n\n수요일 평일 11시 50분 방문\n대기팀 5팀\n25분 기다림 후 입장\n\n음식 주문 후 빠르게 나옴\n음식이 따뜻하고 튀김은 뜨거워서 좋음\n에어컨 온도 아쉬움\n맛은 한국식으로 맛있게 나옴\n\n근처 쌀국수집 중에서는 개인적으로 제일 맛있엇으나 기다림과 안에 에어컨은 재방문 의사를 고민하게 됩니다.","분위기도 좋고 맛도 너무 좋네요!","매장이 쾌적하고 맛있게 잘 먹었어요. 직원분들도 친절하세요!","팀점심으로 왔어요~ 음식이 깔끔하고 맛있어요!\n그리고 직원분들도 진짜 친절하십니다\n자주올게요~!"],"price":["판교에서 베트남 쌀국수 원티어입니다!! 양도 많고 분위기도 좋고 짱이에요!!!"],"food":["회사 근처여서 매번 와보고 싶었는데,\n오늘 와보네요.\n음식도 맛있고, 노란색 인테리어가 인상적이예요^^","너무 맛있어요 2번째 방문임댜","쌀국수 먹으러 항상 오는 곳이에요.\n직장 근처이기도 하고 무엇보다 너무 맛있어서 항상 입이 즐겁습니다 :) 계속 오픈 해주세요!!! 🥰","맛있게 잘 먹었습니다!!","맛있어요!","쌀국수 맛집 인정!!! 너무 맛있어서 팀원분들이랑 자주오게 되네요!! 번창하세요","쌀국수는 판교에서 이집이 최고입니다 ~~~!\n넘맛나요 ><","점심으로 먹기 정말 좋아요~ 자주오고싶은 쌀국수집~"]}
"""

_TINY_FEWSHOT_ASSISTANT = """
예시 출력:
{"service":{"summary":"직원들이 친절하고 응대가 만족스러워요.","bullets":["점심시간에 대기가 있지만 직원들이 친절해요.","음식이 비교적 빨리 나와서 만족스러워요.","매장이 쾌적하다고 해요.","직원분들이 친절하다고 언급해요."],"evidence":[0,1,3,4]},"price":{"summary":"양이 많아서 만족스럽다는 의견이 있어요.","bullets":["양이 많다고 해요."],"evidence":[0]},"food":{"summary":"음식이 맛있고 자주 방문하고 싶어요.","bullets":["쌀국수가 특히 맛있다고 해요.","음식이 전반적으로 맛있다고 해요.","팀원들과 자주 방문하게 된다고 해요.","점심으로 먹기 좋다고 해요."],"evidence":[2,0,5,7]},"overall_summary":{"summary":"전반적으로 서비스가 친절하고 음식 만족도가 높아요."}}
"""

_TINY_FEWSHOT_USER_2 = """
예시 입력:
{\"service\": [\"루프탑 분위기도 너무 좋구 안주, 칵테일 다 너무 맛있어요!!\", \"굳귿귿굳 분위기 좋아요~~~~\", \"분위기 좋고 술 맛있습니다. 판교 살면 꼭 와보세요.\", \"경치가 너무 좋아서 분위기 좋게 술 마시기 좋아요!\", \"하이볼도 맛있고 분위기 너무 좋아요♡♡\", \"분위기가 너무 좋고 칵테일도 예쁘고 맛있어요\", \"경차좋고 다트좋고 분위기좋아요 ㅎㅍ\", \"짱이에요! 분위기 운치 대박 ㅎㅎㅎㅎ\"], \"price\": [\"칵테일 맛이 정말 좋아요. 특히 위스키랑 하이볼도 다양하게 준비되어 있어서 취향에 맞게 골라 마실 수 있어 좋네요.\", \"페퍼로니 피자가 정말 만족스러웠어요. 치즈가 듬뿍 들어가서 쫄깃하고 고소한 맛이 좋았네요.\", \"판교 밤하늘을 즐길 수있는 최고의 루프탑 바입니다!\\n안주도 맛있고 술 종류도 다양해요\\n감성터지는 테라스와 포켓볼 다트도 즐길 수있는 판교 유일 루프탑 바 루프11추천이요!\", \"분위기 좋고 맛있고 다양하고  테라스좋고 야경좋고 아무튼 다 좋아요 최고\", \"고층에 위치해있어서 뷰가 좋아요. 탁트인 석양과 함께 즐기기 좋네요.\", \"처음 방문했는데 분위기도 좋고 경치가 너무 좋아요 다양한 맥주 먹을수있어서 더 좋네요! 또 방문할께요!!\"], \"food\": [\"Good place nice food and drink! 😁\", \"굿굿! 추천드려요\", \"전망도 좋고 칵테일도  맛있습니다!\", \"맛있었습니다!\", \"이벤트도많고 다트, 포켓볼 즐길수있고 노래도좋고 너무좋아요~~0~~~\", \"맛있는 캌테일 멋진 뷰\", \"나초로 이행시 하겠습니다\\n나 이런 곳 처음 와봐 자기야\\n초음 맞아 진짜야\", \"킵해놓은 술을 마시러 왔습니다 :) 뷰가 미쳤습니다\"]}
"""

_TINY_FEWSHOT_ASSISTANT_2 = """
예시 출력:
{\"service\": {\"summary\": \"서비스가 전반적으로 좋다고 해요.\", \"bullets\": [\"루프탑 분위기가 좋고 안주와 칵테일이 맛있어요.\", \"분위기가 좋고 술이 맛있어요.\", \"경치가 좋아서 술 마시기 좋은 곳이에요.\", \"칵테일이 예쁘고 맛있어요.\", \"운치 있는 분위기가 대박이에요.\"], \"evidence\": [0, 1, 2, 3, 5, 7]}, \"price\": {\"summary\": \"가격에 대한 언급은 적지만 가성비가 좋다고 해요.\", \"bullets\": [\"안주와 술 종류가 다양해서 좋다고 해요.\", \"분위기와 맛이 모두 만족스럽다고 해요.\", \"고층에서 즐기는 뷰가 좋다고 해요.\", \"다양한 맥주를 즐길 수 있어서 좋다고 해요.\"], \"evidence\": [0, 2, 3, 4, 5]}, \"food\": {\"summary\": \"음식이 맛있다고 해요.\", \"bullets\": [\"칵테일과 안주가 맛있어요.\", \"전망이 좋고 음식이 맛있다고 해요.\", \"다트와 포켓볼을 즐길 수 있어요.\", \"뷰가 멋지다고 해요.\"], \"evidence\": [0, 2, 4, 5, 7]}, \"overall_summary\": {\"summary\": \"전반적으로 분위기와 서비스가 좋고 음식도 맛있어요. 가격에 대한 언급은 적지만 가성비가 좋다고 해요.\"}}
"""   

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
        #{"role": "user", "content": _TINY_FEWSHOT_USER_2},
        #{"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT_2},
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

JUDGE_AXES = (
    "schema_adherence",
    "fallback_adherence",
    "style_adherence",
    "evidence_validity",
    "faithfulness",
    "category_correctness",
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
        pred = _postprocess_prediction(pred, ins)
        judge_out = _call_judge(
            ins, ref, pred,
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
            "Wrote %s (avg_score=%.2f, schema=%.2f, fallback=%.2f, evidence=%.2f)",
            args.output, summary["avg_score"], summary.get("avg_schema_adherence", 0), summary.get("avg_fallback_adherence", 0), summary.get("avg_evidence_validity", 0),
        )
    else:
        logger.info("Wrote %s (avg_score=%.2f)", args.output, summary["avg_score"])
    print(json.dumps({"output_path": str(args.output), "avg_score": summary["avg_score"], "judge_rubric_version": rubric}, ensure_ascii=False))


if __name__ == "__main__":
    main()
