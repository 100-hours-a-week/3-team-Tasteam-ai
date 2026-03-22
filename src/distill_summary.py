"""
공통 distill summary 모듈 (eval_llm_as_judge + API summary 파이프라인 공유).

- 프롬프트: eval과 동일 (system + few-shot + instruction).
- 추론: base + LoRA adapter, apply_chat_template + generate(do_sample=False, max_new_tokens=1024).
- 후처리: extract_json_for_rouge → postprocess_prediction.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ----- 프롬프트 상수 (eval_llm_as_judge와 동일) -----
SCHEMA_ENFORCEMENT_SYSTEM = """당신은 리뷰 요약 어시스턴트입니다.
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
- evidence에 넣는 숫자는 해당 카테고리 리뷰 배열의 인덱스만. 예를 들어 service 리뷰가 5개면 0,1,2,3,4만 사용하고 5 이상은 쓰지 말 것
- 해당 카테고리 배열 길이를 넘는 인덱스, 리뷰에 없는 내용을 지지하는 인덱스는 넣지 말 것.

예시에서 evidence는 해당 카테고리 배열 인덱스만 사용했음을 참고하세요.
"""

TINY_FEWSHOT_USER = """
예시 입력:
{"service":["맨날 점심시간만되면 엄청 웨이팅 장난아니라서 점심시간 아닐 때 방문해봤어요! 직원분들도 너무 친절하고 좋습니다!","판교 베트남 음식 르 메콩\n\n수요일 평일 11시 50분 방문\n대기팀 5팀\n25분 기다림 후 입장\n\n음식 주문 후 빠르게 나옴\n음식이 따뜻하고 튀김은 뜨거워서 좋음\n에어컨 온도 아쉬움\n맛은 한국식으로 맛있게 나옴\n\n근처 쌀국수집 중에서는 개인적으로 제일 맛있엇으나 기다림과 안에 에어컨은 재방문 의사를 고민하게 됩니다.","분위기도 좋고 맛도 너무 좋네요!","매장이 쾌적하고 맛있게 잘 먹었어요. 직원분들도 친절하세요!","팀점심으로 왔어요~ 음식이 깔끔하고 맛있어요!\n그리고 직원분들도 진짜 친절하십니다\n자주올게요~!"],"price":["판교에서 베트남 쌀국수 원티어입니다!! 양도 많고 분위기도 좋고 짱이에요!!!"],"food":["회사 근처여서 매번 와보고 싶었는데,\n오늘 와보네요.\n음식도 맛있고, 노란색 인테리어가 인상적이예요^^","너무 맛있어요 2번째 방문임댜","쌀국수 먹으러 항상 오는 곳이에요.\n직장 근처이기도 하고 무엇보다 너무 맛있어서 항상 입이 즐겁습니다 :) 계속 오픈 해주세요!!! 🥰","맛있게 잘 먹었습니다!!","맛있어요!","쌀국수 맛집 인정!!! 너무 맛있어서 팀원분들이랑 자주오게 되네요!! 번창하세요","쌀국수는 판교에서 이집이 최고입니다 ~~~!\n넘맛나요 ><","점심으로 먹기 정말 좋아요~ 자주오고싶은 쌀국수집~"]}
"""

TINY_FEWSHOT_ASSISTANT = """
예시 출력:
{"service":{"summary":"직원들이 친절하고 응대가 만족스러워요.","bullets":["점심시간에 대기가 있지만 직원들이 친절해요.","음식이 비교적 빨리 나와서 만족스러워요.","매장이 쾌적하다고 해요.","직원분들이 친절하다고 언급해요."],"evidence":[0,1,3,4]},"price":{"summary":"양이 많아서 만족스럽다는 의견이 있어요.","bullets":["양이 많다고 해요."],"evidence":[0]},"food":{"summary":"음식이 맛있고 자주 방문하고 싶어요.","bullets":["쌀국수가 특히 맛있다고 해요.","음식이 전반적으로 맛있다고 해요.","팀원들과 자주 방문하게 된다고 해요.","점심으로 먹기 좋다고 해요."],"evidence":[2,0,5,7]},"overall_summary":{"summary":"전반적으로 서비스가 친절하고 음식 만족도가 높아요."}}
"""


def load_model_and_tokenizer(adapter_path: str, base_model: str) -> Tuple[Any, Any]:
    """모델·토크나이저 로드 (base + PEFT adapter). (model, tokenizer) 반환."""
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


def extract_json_for_rouge(raw: str) -> str:
    """출력에서 JSON 블록만 추출해 반환. 실패 시 원문."""
    if not raw or not raw.strip():
        return raw or ""
    try:
        from .json_parse_utils import extract_json_block, parse_json_relaxed
        from .schema_repair import repair_summary_schema
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


def get_category_lengths(instruction: str) -> Dict[str, int]:
    """instruction(JSON)에서 카테고리별 리뷰 개수 반환."""
    out: Dict[str, int] = {"service": 0, "price": 0, "food": 0}
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


def postprocess_prediction(pred_json_str: str, instruction: str) -> str:
    """
    추론 결과 후처리: evidence 인덱스 검증·보정, 빈 카테고리 폴백 치환.
    """
    if not pred_json_str or not pred_json_str.strip():
        return pred_json_str
    try:
        pred = json.loads(pred_json_str)
    except json.JSONDecodeError:
        return pred_json_str
    if not isinstance(pred, dict) or not any(k in pred for k in ("service", "price", "food")):
        return pred_json_str

    lengths = get_category_lengths(instruction)
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
        evidence: List[int] = []
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
        if len(evidence) > len(bullets):
            evidence = evidence[: len(bullets)]
        elif len(evidence) < len(bullets):
            bullets = bullets[: len(evidence)]

        if not summary.strip() and not bullets:
            summary = fallback_price if cat == "price" else fallback_other
            bullets = []
            evidence = []

        pred[cat] = {"summary": summary.strip(), "bullets": bullets, "evidence": evidence}

    ov = pred.get("overall_summary")
    if isinstance(ov, dict):
        ov = {"summary": _as_str(ov.get("summary", "")).strip()[:200]}
        pred["overall_summary"] = ov

    return json.dumps(pred, ensure_ascii=False)


def _drop_evidence_fields(pred_json_str: str) -> str:
    """카테고리별 evidence 키를 제거한 JSON 문자열 반환."""
    if not pred_json_str or not pred_json_str.strip():
        return pred_json_str
    try:
        pred = json.loads(pred_json_str)
    except json.JSONDecodeError:
        return pred_json_str
    if not isinstance(pred, dict):
        return pred_json_str

    for cat in ("service", "price", "food"):
        cell = pred.get(cat)
        if isinstance(cell, dict):
            cell.pop("evidence", None)
            if "summary" not in cell:
                cell["summary"] = ""
            if "bullets" not in cell or not isinstance(cell.get("bullets"), list):
                cell["bullets"] = []
        else:
            pred[cat] = {"summary": "", "bullets": []}
    return json.dumps(pred, ensure_ascii=False)


def generate_one(
    model: Any,
    tokenizer: Any,
    instruction: str,
    max_new_tokens: int = 1024,
    postprocess: bool = True,
    no_evidence_output: bool = False,
) -> str:
    """system + few-shot + instruction으로 추론 후 JSON 추출. postprocess=True면 후처리까지 적용."""
    messages = [
        {"role": "system", "content": SCHEMA_ENFORCEMENT_SYSTEM},
        {"role": "user", "content": TINY_FEWSHOT_USER},
        {"role": "assistant", "content": TINY_FEWSHOT_ASSISTANT},
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
    extracted = extract_json_for_rouge(raw)
    if postprocess:
        out = postprocess_prediction(extracted, instruction)
    else:
        out = extracted
    if no_evidence_output:
        return _drop_evidence_fields(out)
    return out


def generate_summary_from_payload(
    payload: Dict[str, List[str]],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 1024,
    no_evidence_output: bool = False,
) -> Dict[str, Any]:
    """
    payload(service/price/food 리스트)로 instruction 문자열을 만든 뒤 추론·후처리하여
    summary 구조(dict)로 반환. API summary 파이프라인에서 사용.
    """
    instruction = json.dumps(payload, ensure_ascii=False)
    pred_str = generate_one(
        model,
        tokenizer,
        instruction,
        max_new_tokens=max_new_tokens,
        no_evidence_output=no_evidence_output,
    )
    out = json.loads(pred_str)
    return out


# ----- 싱글톤 (API에서 USE_DISTILL_SUMMARY 시 lazy 로드) -----
_distill_model: Optional[Any] = None
_distill_tokenizer: Optional[Any] = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _adapter_files_present(adapter_dir: Path) -> bool:
    if not adapter_dir.is_dir():
        return False
    if (adapter_dir / "adapter_config.json").exists():
        return True
    return any(adapter_dir.glob("*.safetensors")) or any(adapter_dir.glob("*.bin"))


def ensure_distill_adapter_local(adapter_path_str: str) -> str:
    """
    로컬에 PEFT 어댑터가 없으면 wandb Python API로 artifact를 받아 저장한 뒤 절대 경로 반환.
    - DISTILL_ADAPTER_ARTIFACT가 있으면 그 qualified 이름 사용 (entity/project/name:alias).
    - 없으면 WANDB_ENTITY + DISTILL_ADAPTER_PATH의 .../artifacts/<run_id>/adapter 에서 run_id로
      qlora-adapter-<run_id>:latest 조합.
    """
    from .config import Config

    p = Path(adapter_path_str)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    if _adapter_files_present(p):
        return str(p)
    if not os.environ.get("WANDB_API_KEY"):
        raise FileNotFoundError(
            f"어댑터가 {p} 에 없습니다. 로컬에 배치하거나 WANDB_API_KEY 를 설정해 wandb에서 받으세요."
        )
    qualified = getattr(Config, "DISTILL_ADAPTER_ARTIFACT", None)
    if not qualified:
        entity = (os.environ.get("WANDB_ENTITY") or "").strip()
        project = (os.environ.get("WANDB_PROJECT") or "tasteam-distill").strip()
        run_folder = p.parent.name
        if not entity or not run_folder:
            raise ValueError(
                "로컬 어댑터가 없고 artifact 이름을 알 수 없습니다. "
                "DISTILL_ADAPTER_ARTIFACT=entity/project/qlora-adapter-<id>:latest 를 설정하거나, "
                "WANDB_ENTITY 와 DISTILL_ADAPTER_PATH=.../artifacts/<run_id>/adapter 형태를 맞추세요."
            )
        qualified = f"{entity}/{project}/qlora-adapter-{run_folder}:latest"
    import wandb

    api = wandb.Api()
    art = api.artifact(qualified)
    download_root = p.parent
    download_root.mkdir(parents=True, parents=True)
    art.download(root=str(download_root))
    if not _adapter_files_present(p):
        raise RuntimeError(
            f"artifact {qualified} 다운로드 후에도 예상 경로에 어댑터가 없습니다: {p}"
        )
    return str(p)


def get_distill_model():
    """Config.USE_DISTILL_SUMMARY일 때만 호출. 싱글톤 (model, tokenizer) 반환."""
    global _distill_model, _distill_tokenizer
    from .config import Config
    if not getattr(Config, "USE_DISTILL_SUMMARY", False):
        raise RuntimeError("USE_DISTILL_SUMMARY is not enabled")
    adapter_path = getattr(Config, "DISTILL_ADAPTER_PATH", None)
    base_model = getattr(Config, "DISTILL_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    if not adapter_path:
        raise ValueError("DISTILL_ADAPTER_PATH is required when USE_DISTILL_SUMMARY=true")
    if _distill_model is None:
        resolved = ensure_distill_adapter_local(adapter_path)
        _distill_model, _distill_tokenizer = load_model_and_tokenizer(resolved, base_model)
        logger.info("Distill summary model loaded: base=%s adapter=%s", base_model, resolved)
    return _distill_model, _distill_tokenizer


def generate_summary_sync(payload: Dict[str, List[str]]) -> Dict[str, Any]:
    """동기: payload → summary dict. USE_DISTILL_SUMMARY일 때만 사용."""
    from .config import Config
    model, tokenizer = get_distill_model()
    return generate_summary_from_payload(
        payload,
        model,
        tokenizer,
        no_evidence_output=getattr(Config, "DISTILL_NO_EVIDENCE_OUTPUT", False),
    )
