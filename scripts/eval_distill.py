#!/usr/bin/env python3
"""
Distill 평가 스크립트: student adapter → val/test ROUGE/BERTScore.

val_labeled_path, test_labeled_path: instruction + output(ground truth) 형식.
student로 instruction에 대해 추론 후 output과 비교.

사용:
  python scripts/eval_distill.py --val-labeled labeled/val_labeled.json --test-labeled labeled/test_labeled.json --adapter-path runs/xxx/adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir eval/
  python scripts/eval_distill.py ... --prediction-no-evidence   # eval_llm_as_judge와 동일: distill_summary 프롬프트·후처리 후 evidence 제거
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_lightweight_src_package() -> None:
    """
    src/__init__.py 가 sentiment_analysis·vector_search 등 전역 import를 하므로,
    Pod eval처럼 가벼운 환경에서는 import 실패·과도한 의존성이 난다.
    namespace 패키지만 sys.modules에 넣어 두면 src.distill_summary 등 서브모듈만 로드된다.
    (이미 src가 로드된 프로세스에서는 건드리지 않음)
    """
    if "src" in sys.modules:
        return
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(_PROJECT_ROOT / "src")]  # type: ignore[attr-defined]
    sys.modules["src"] = pkg


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


# teacher(label_for_distill)와 동일한 스키마·규칙
_SCHEMA_ENFORCEMENT_SYSTEM = """You are a JSON generator for review summarization.
입력은 카테고리별 근거 리뷰 목록(JSON)이다. teacher와 동일한 스키마로만 출력하라.

Return ONLY one valid JSON object. No text before or after JSON.

스키마 (teacher와 동일):
{
  "service": {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "price":   {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "food":    {"summary": string, "bullets": [string, ...], "evidence": [int, ...]},
  "overall_summary": {"summary": string}
}
overall_summary에는 summary만 넣을 것. bullets, evidence는 금지.

카테고리 정의: service=직원·서비스·대기·분위기·매장, price=가격·가성비·양·비싸다/저렴하다, food=음식·메뉴·맛·요리. 한 카테고리 내용을 다른 카테고리 필드에 넣지 말 것.

규칙 (teacher와 동일):
- 해당 카테고리에 리뷰가 1개 이상 있으면 반드시 summary, bullets, evidence를 채울 것. 빈 문자열·빈 배열만 내지 말 것. price도 리뷰가 있으면 bullets와 evidence를 채울 것.
- 리뷰에 나온 내용만 요약할 것. 입력 리뷰에 없는 메뉴·가게·직원 설명을 넣지 말 것.
- 말투: 모든 summary, bullets, overall_summary는 "~해요" 체
- 각 카테고리 summary: 1문장, 과장 금지
- bullets: 3~5개(근거 있을 때), 중복 제거, 구체적으로. 근거 없으면 []
- evidence: 해당 카테고리 리뷰 배열 길이 미만의 0-based 인덱스만 사용. 각 bullet당 정확히 하나의 인덱스. bullets 개수와 동일.
- price: 가격 숫자 없으면 가성비/양/구성/만족감 같은 우회표현으로 요약 가능. 전혀 없으면 "가격 관련 언급이 적어요." 등
- 근거 없을 때만: summary에 "언급이 적어요"처럼 해요체로 표현 (빈 문자열 대신)
- overall_summary: 2~3문장으로 종합 요약 (summary 키만 사용)
- evidence는 입력 인덱스만 사용, 추측 금지
- Evidence must reference only review indices that explicitly support each bullet.
- Do not guess evidence indices.
- If evidence is weak or ambiguous, omit the bullet instead of guessing.

Output only JSON.
"""

_TINY_FEWSHOT_USER = """Example input:
{"service":["맨날 점심시간만되면 엄청 웨이팅 장난아니라서 점심시간 아닐 때 방문해봤어요! 직원분들도 너무 친절하고 좋습니다!","판교 베트남 음식 르 메콩\n\n수요일 평일 11시 50분 방문\n대기팀 5팀\n25분 기다림 후 입장\n\n음식 주문 후 빠르게 나옴\n음식이 따뜻하고 튀김은 뜨거워서 좋음\n에어컨 온도 아쉬움\n맛은 한국식으로 맛있게 나옴\n\n근처 쌀국수집 중에서는 개인적으로 제일 맛있엇으나 기다림과 안에 에어컨은 재방문 의사를 고민하게 됩니다.","분위기도 좋고 맛도 너무 좋네요!","매장이 쾌적하고 맛있게 잘 먹었어요. 직원분들도 친절하세요!","팀점심으로 왔어요~ 음식이 깔끔하고 맛있어요!\n그리고 직원분들도 진짜 친절하십니다\n자주올게요~!"],"price":["판교에서 베트남 쌀국수 원티어입니다!! 양도 많고 분위기도 좋고 짱이에요!!!"],"food":["회사 근처여서 매번 와보고 싶었는데,\n오늘 와보네요.\n음식도 맛있고, 노란색 인테리어가 인상적이예요^^","너무 맛있어요 2번째 방문임댜","쌀국수 먹으러 항상 오는 곳이에요.\n직장 근처이기도 하고 무엇보다 너무 맛있어서 항상 입이 즐겁습니다 :) 계속 오픈 해주세요!!! 🥰","맛있게 잘 먹었습니다!!","맛있어요!","쌀국수 맛집 인정!!! 너무 맛있어서 팀원분들이랑 자주오게 되네요!! 번창하세요","쌀국수는 판교에서 이집이 최고입니다 ~~~!\n넘맛나요 ><","점심으로 먹기 정말 좋아요~ 자주오고싶은 쌀국수집~"]}
"""

_TINY_FEWSHOT_ASSISTANT = """Example output:
{"service":{"summary":"직원들이 친절하고 응대가 만족스러워요.","bullets":["점심시간에 대기가 있지만 직원들이 친절해요.","음식이 비교적 빨리 나와서 만족스러워요.","매장이 쾌적하다고 해요.","직원분들이 친절하다고 언급해요."],"evidence":[0,1,3,4]},"price":{"summary":"양이 많아서 만족스럽다는 의견이 있어요.","bullets":["양이 많다고 해요."],"evidence":[0]},"food":{"summary":"음식이 맛있고 자주 방문하고 싶어요.","bullets":["쌀국수가 특히 맛있다고 해요.","음식이 전반적으로 맛있다고 해요.","팀원들과 자주 방문하게 된다고 해요.","점심으로 먹기 좋다고 해요."],"evidence":[2,0,5,7]},"overall_summary":{"summary":"전반적으로 서비스가 친절하고 음식 만족도가 높아요."}}
"""

def _generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
) -> str:
    """이미 로드된 model/tokenizer로 instruction 한 건만 추론."""
    # instruction은 payload JSON 문자열. system prompt + 2개 few-shot으로 스키마 계약을 강화한다.
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
    parser.add_argument(
        "--prediction-no-evidence",
        action="store_true",
        help="eval_llm_as_judge/API와 동일: distill_summary 프롬프트·후처리 후 evidence 필드 제거",
    )
    args = parser.parse_args()

    _install_lightweight_src_package()

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
            if args.prediction_no_evidence:
                from src.distill_summary import load_model_and_tokenizer as ds_load

                model, tokenizer = ds_load(str(args.adapter_path), args.base_model)
            else:
                model, tokenizer = _load_model_and_tokenizer(str(args.adapter_path), args.base_model)

        preds: list[str] = []
        refs: list[str] = []
        for i, s in enumerate(samples):
            if (i + 1) % 50 == 0:
                logger.info("%s: %d/%d", split, i + 1, len(samples))
            ins = s.get("instruction", "")
            ref = s.get("output", "")
            if args.prediction_no_evidence:
                from src.distill_summary import generate_one as ds_generate_one

                pred = ds_generate_one(
                    model,
                    tokenizer,
                    ins,
                    max_new_tokens=1024,
                    postprocess=True,
                    no_evidence_output=True,
                )
            else:
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

    llm_judge_sample_ids: list[int | str] = []
    if args.val_labeled and args.val_labeled.exists():
        val_samples = _load_labeled(args.val_labeled)[:50]
        llm_judge_sample_ids = [s.get("sample_id") for s in val_samples]

    report_path = out_dir / "report.json"
    report["meta"] = {
        "adapter_path": str(args.adapter_path),
        "base_model": args.base_model,
        "llm_judge_sample_ids": llm_judge_sample_ids,
        "prediction_no_evidence": bool(args.prediction_no_evidence),
        "inference_stack": "distill_summary_no_evidence" if args.prediction_no_evidence else "eval_distill_legacy",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Report: %s", report_path)
    print(json.dumps({"report_path": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
