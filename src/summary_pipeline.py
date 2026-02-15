"""
새로운 Summary 파이프라인 모듈
하이브리드 검색 (Dense + Sparse) 및 Aspect 기반 카테고리별 요약
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any

from .config import Config
from .json_parse_utils import parse_json_relaxed

logger = logging.getLogger(__name__)

# Price 관련 힌트
PRICE_HINTS = [
    "가격", "가성비", "저렴", "비싸", "비쌈", "가격대", "합리", "구성", "구성비",
    "양", "푸짐", "리필", "무한", "만족", "혜자"
]

# 카테고리별 서치 결과 없음/키 누락 시 사용할 디폴트 (리뷰 없음·생성 실패 시 동일 문구)
CATEGORY_EMPTY_DEFAULT = {
    "service": {"summary": "아직 충분한 서비스 관련 리뷰 정보가 쌓이지 않았어요.", "bullets": [], "evidence": []},
    "price": {"summary": "아직 충분한 가격 관련 리뷰 정보가 쌓이지 않았어요.", "bullets": [], "evidence": []},
    "food": {"summary": "아직 충분한 음식 관련 리뷰 정보가 쌓이지 않았어요.", "bullets": [], "evidence": []},
}


def _clip(xs: List[str], n: int = 8) -> List[str]:
    """리스트를 최대 n개로 제한"""
    xs = [x.strip() for x in xs if x and str(x).strip()]
    return xs[:n]


def _has_price_signal(text: str) -> bool:
    """가격 관련 신호가 있는지 확인"""
    t = text.replace(" ", "")
    return any(k.replace(" ", "") in t for k in PRICE_HINTS)


def _estimate_tokens(text: str) -> int:
    """보수적 토큰 추정 (한글 많으면 1토큰 ≈ 2자, 4731 입력 초과 방지)."""
    return max(0, (len(text) + 1) // 2)


def _payload_within_context_limit(
    service_reviews: List[str],
    price_reviews: List[str],
    food_reviews: List[str],
    instructions: str,
    per_category_max: int,
    max_input_tokens: int,
) -> Dict[str, List[str]]:
    """
    컨텍스트 한도(4096 - max_new_tokens - margin) 이하가 되도록
    per_category_max를 줄여가며 payload를 만든다. evidence 인덱스 정렬 유지.
    """
    n = per_category_max
    while n >= 1:
        payload = {
            "service": _clip(service_reviews, n),
            "price": _clip(price_reviews, n),
            "food": _clip(food_reviews, n),
        }
        user_content = json.dumps(payload, ensure_ascii=False)
        if _estimate_tokens(instructions) + _estimate_tokens(user_content) <= max_input_tokens:
            return payload
        n -= 1
    # 최소 1개씩은 유지
    return {
        "service": _clip(service_reviews, 1),
        "price": _clip(price_reviews, 1),
        "food": _clip(food_reviews, 1),
    }


def summarize_aspects_new(
    service_reviews: List[str],
    price_reviews: List[str],
    food_reviews: List[str],
    service_evidence_data: List[Dict] = None,
    price_evidence_data: List[Dict] = None,
    food_evidence_data: List[Dict] = None,
    llm_utils: Optional[Any] = None,
    per_category_max: int = 8,
) -> Dict:
    """
    새로운 파이프라인: 카테고리별 요약 생성
    
    Args:
        service_reviews: 서비스 관련 리뷰 텍스트 리스트
        price_reviews: 가격 관련 리뷰 텍스트 리스트
        food_reviews: 음식 관련 리뷰 텍스트 리스트
        service_evidence_data: 서비스 evidence 데이터 (review_id, snippet, rank)
        price_evidence_data: 가격 evidence 데이터
        food_evidence_data: 음식 evidence 데이터
        llm_utils: LLMUtils 인스턴스
        per_category_max: 카테고리당 최대 리뷰 수
    
    Returns:
        카테고리별 요약 결과
    """
    instructions = """
너는 음식점 리뷰 분석가다.
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
- 각 카테고리 summary: 1문장, 과장 금지
- bullets: 3~5개, 중복 제거, 구체적으로
- evidence: 근거로 쓴 리뷰의 인덱스(각 카테고리 리스트에서 0-based)
- price는 '가격 숫자'가 없으면 '가성비/양/구성/만족감' 같은 우회표현을 근거로 요약하라.
- overall_summary는 2~3문장으로 종합 요약하라.
- 근거(입력 리뷰)에 없는 내용은 추측하지 말고 "언급이 적어요"처럼 해요체로 표현하라.
"""
    max_input_tokens = Config.LLM_MAX_CONTEXT_LENGTH - 1500 - 64
    payload = _payload_within_context_limit(
        service_reviews, price_reviews, food_reviews,
        instructions, per_category_max, max_input_tokens,
    )

    if not llm_utils:
        raise ValueError("llm_utils가 필요합니다.")

    # LLM 호출
    try:
        # 기존 LLMUtils 사용
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        text = llm_utils._generate_response(
            messages=messages,
            temperature=0.1,
            max_new_tokens=1500,
        )
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        return {
            **{k: dict(v) for k, v in CATEGORY_EMPTY_DEFAULT.items()},
            "overall_summary": {"summary": "아직 충분한 리뷰 정보가 쌓이지 않았어요."}
        }

    # JSON 파싱
    try:
        # JSON 부분만 추출
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        
        # JSON 매칭
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        out = parse_json_relaxed(text)
        if out is None or not isinstance(out, dict):
            raise json.JSONDecodeError("relaxed parse failed or not a dict", text, 0)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 재시도
        try:
            fix = llm_utils._generate_response(
                messages=[{"role": "user", "content": f"다음 텍스트를 유효한 JSON으로만 변환: {text}"}],
                temperature=0.1,
                max_new_tokens=500,
            )
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', fix, re.DOTALL)
            if json_match:
                fix = json_match.group(0)
            out = parse_json_relaxed(fix)
            if out is None or not isinstance(out, dict):
                raise ValueError("relaxed parse failed on fix")
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {
                **{k: dict(v) for k, v in CATEGORY_EMPTY_DEFAULT.items()},
                "overall_summary": {"summary": "아직 충분한 리뷰 정보가 쌓이지 않았어요."}
            }

    # Evidence 인덱스를 실제 evidence 객체로 변환 (payload와 동일 개수로 슬라이스해 정렬 유지)
    evidence_data_map = {
        "service": (service_evidence_data or [])[: len(payload["service"])],
        "price": (price_evidence_data or [])[: len(payload["price"])],
        "food": (food_evidence_data or [])[: len(payload["food"])],
    }
    for cat in ("service", "price", "food"):
        out.setdefault(cat, dict(CATEGORY_EMPTY_DEFAULT[cat]))
        n = len(payload[cat])
        ev_indices = out.get(cat, {}).get("evidence", [])
        if isinstance(ev_indices, list):
            valid_indices = [i for i in ev_indices if isinstance(i, int) and 0 <= i < n]
            evidence_data = evidence_data_map[cat]
            out[cat]["evidence"] = [
                {
                    "review_id": evidence_data[i]["review_id"],
                    "snippet": evidence_data[i]["snippet"],
                    "rank": evidence_data[i]["rank"]
                }
                for i in valid_indices
                if i < len(evidence_data)
            ]
        else:
            out[cat]["evidence"] = []

    # 서치 결과가 없었던 카테고리는 디폴트 문구로 통일
    for cat in ("service", "price", "food"):
        if len(payload[cat]) == 0:
            out[cat] = dict(CATEGORY_EMPTY_DEFAULT[cat])

    # Price 요약 게이트: evidence 리뷰들에 price 신호가 전혀 없으면 안전하게 다운그레이드
    price_ev = out.get("price", {}).get("evidence", [])
    # evidence는 이제 객체 리스트이므로 snippet을 직접 가져옴
    ev_texts = [ev.get("snippet", "") for ev in price_ev if isinstance(ev, dict)] if price_ev else []
    if ev_texts and not any(_has_price_signal(t) for t in ev_texts):
        out["price"]["summary"] = "가격 관련 언급이 많지 않아요. 전반적인 만족감이나 구성(양 등) 중심으로만 해석 가능해요."
        # bullets도 너무 단정적으로 쓰지 않게 최소화
        out["price"]["bullets"] = [
            "가격을 직접 언급한 리뷰가 많지 않아요.",
            "대신 만족/구성/양(푸짐함) 관련 표현이 간접적으로 나타나요."
        ]

    return out


async def summarize_aspects_new_async(
    service_reviews: List[str],
    price_reviews: List[str],
    food_reviews: List[str],
    service_evidence_data: List[Dict] = None,
    price_evidence_data: List[Dict] = None,
    food_evidence_data: List[Dict] = None,
    llm_utils: Optional[Any] = None,
    per_category_max: int = 8,
) -> Dict:
    """
    summarize_aspects_new의 비동기 버전. Config.SUMMARY_LLM_ASYNC=True(llm_async)일 때 배치에서 사용.
    httpx.AsyncClient / AsyncOpenAI로 LLM 호출.
    """
    instructions = """
너는 음식점 리뷰 분석가다.
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
- 각 카테고리 summary: 1문장, 과장 금지
- bullets: 3~5개, 중복 제거, 구체적으로
- evidence: 근거로 쓴 리뷰의 인덱스(각 카테고리 리스트에서 0-based)
- price는 '가격 숫자'가 없으면 '가성비/양/구성/만족감' 같은 우회표현을 근거로 요약하라.
- overall_summary는 2~3문장으로 종합 요약하라.
- 근거(입력 리뷰)에 없는 내용은 추측하지 말고 "언급이 적어요"처럼 해요체로 표현하라.
"""
    max_input_tokens = Config.LLM_MAX_CONTEXT_LENGTH - 1500 - 64
    payload = _payload_within_context_limit(
        service_reviews, price_reviews, food_reviews,
        instructions, per_category_max, max_input_tokens,
    )

    if not llm_utils:
        raise ValueError("llm_utils가 필요합니다.")

    try:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
        text = await llm_utils._generate_response_async(
            messages=messages,
            temperature=0.1,
            max_new_tokens=1500,
        )
    except Exception as e:
        logger.error(f"LLM 비동기 호출 실패: {e}")
        return {
            **{k: dict(v) for k, v in CATEGORY_EMPTY_DEFAULT.items()},
            "overall_summary": {"summary": "아직 충분한 리뷰 정보가 쌓이지 않았어요."}
        }

    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        out = parse_json_relaxed(text)
        if out is None or not isinstance(out, dict):
            raise json.JSONDecodeError("relaxed parse failed or not a dict", text, 0)
    except json.JSONDecodeError:
        try:
            fix = await llm_utils._generate_response_async(
                messages=[{"role": "user", "content": f"다음 텍스트를 유효한 JSON으로만 변환: {text}"}],
                temperature=0.1,
                max_new_tokens=500,
            )
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', fix, re.DOTALL)
            if json_match:
                fix = json_match.group(0)
            out = parse_json_relaxed(fix)
            if out is None or not isinstance(out, dict):
                raise ValueError("relaxed parse failed on fix")
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {
                **{k: dict(v) for k, v in CATEGORY_EMPTY_DEFAULT.items()},
                "overall_summary": {"summary": "아직 충분한 리뷰 정보가 쌓이지 않았어요."}
            }

    # payload와 동일 개수로 슬라이스해 evidence 인덱스 정렬 유지
    evidence_data_map = {
        "service": (service_evidence_data or [])[: len(payload["service"])],
        "price": (price_evidence_data or [])[: len(payload["price"])],
        "food": (food_evidence_data or [])[: len(payload["food"])],
    }
    for cat in ("service", "price", "food"):
        out.setdefault(cat, dict(CATEGORY_EMPTY_DEFAULT[cat]))
        n = len(payload[cat])
        ev_indices = out.get(cat, {}).get("evidence", [])
        if isinstance(ev_indices, list):
            valid_indices = [i for i in ev_indices if isinstance(i, int) and 0 <= i < n]
            evidence_data = evidence_data_map[cat]
            out[cat]["evidence"] = [
                {"review_id": evidence_data[i]["review_id"], "snippet": evidence_data[i]["snippet"], "rank": evidence_data[i]["rank"]}
                for i in valid_indices
                if i < len(evidence_data)
            ]
        else:
            out[cat]["evidence"] = []

    # 서치 결과가 없었던 카테고리는 디폴트 문구로 통일
    for cat in ("service", "price", "food"):
        if len(payload[cat]) == 0:
            out[cat] = dict(CATEGORY_EMPTY_DEFAULT[cat])

    price_ev = out.get("price", {}).get("evidence", [])
    ev_texts = [ev.get("snippet", "") for ev in price_ev if isinstance(ev, dict)] if price_ev else []
    if ev_texts and not any(_has_price_signal(t) for t in ev_texts):
        out["price"]["summary"] = "가격 관련 언급이 많지 않아요. 전반적인 만족감이나 구성(양 등) 중심으로만 해석 가능해요."
        out["price"]["bullets"] = [
            "가격을 직접 언급한 리뷰가 많지 않아요.",
            "대신 만족/구성/양(푸짐함) 관련 표현이 간접적으로 나타나요."
        ]

    return out
