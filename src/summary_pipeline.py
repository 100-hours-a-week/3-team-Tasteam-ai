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

logger = logging.getLogger(__name__)

# Price 관련 힌트
PRICE_HINTS = [
    "가격", "가성비", "저렴", "비싸", "비쌈", "가격대", "합리", "구성", "구성비",
    "양", "푸짐", "리필", "무한", "만족", "혜자"
]


def _clip(xs: List[str], n: int = 8) -> List[str]:
    """리스트를 최대 n개로 제한"""
    xs = [x.strip() for x in xs if x and str(x).strip()]
    return xs[:n]


def _has_price_signal(text: str) -> bool:
    """가격 관련 신호가 있는지 확인"""
    t = text.replace(" ", "")
    return any(k.replace(" ", "") in t for k in PRICE_HINTS)


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
    payload = {
        "service": _clip(service_reviews, per_category_max),
        "price": _clip(price_reviews, per_category_max),
        "food": _clip(food_reviews, per_category_max),
    }

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
- 각 카테고리 summary: 1문장, 과장 금지
- bullets: 3~5개, 중복 제거, 구체적으로
- evidence: 근거로 쓴 리뷰의 인덱스(각 카테고리 리스트에서 0-based)
- price는 '가격 숫자'가 없으면 '가성비/양/구성/만족감' 같은 우회표현을 근거로 요약하라.
- overall_summary는 2~3문장으로 종합 요약하라.
- 근거(입력 리뷰)에 없는 내용은 추측하지 말고 "언급이 적다"라고 표현하라.
"""

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
            max_new_tokens=1000,
        )
    except Exception as e:
        logger.error(f"LLM 호출 실패: {e}")
        # 폴백: 빈 결과 반환
        return {
            "service": {"summary": "", "bullets": [], "evidence": []},
            "price": {"summary": "", "bullets": [], "evidence": []},
            "food": {"summary": "", "bullets": [], "evidence": []},
            "overall_summary": {"summary": "요약 생성에 실패했습니다."}
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
        
        out = json.loads(text)
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
            out = json.loads(fix)
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {
                "service": {"summary": "", "bullets": [], "evidence": []},
                "price": {"summary": "", "bullets": [], "evidence": []},
                "food": {"summary": "", "bullets": [], "evidence": []},
                "overall_summary": {"summary": "요약 생성에 실패했습니다."}
            }

    # Evidence 인덱스를 실제 evidence 객체로 변환
    evidence_data_map = {
        "service": service_evidence_data or [],
        "price": price_evidence_data or [],
        "food": food_evidence_data or [],
    }
    
    for cat in ("service", "price", "food"):
        n = len(payload[cat])
        ev_indices = out.get(cat, {}).get("evidence", [])
        if isinstance(ev_indices, list):
            # 인덱스 검증 및 변환
            valid_indices = [i for i in ev_indices if isinstance(i, int) and 0 <= i < n]
            # 인덱스를 evidence 객체로 변환
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

    # Price 요약 게이트: evidence 리뷰들에 price 신호가 전혀 없으면 안전하게 다운그레이드
    price_ev = out.get("price", {}).get("evidence", [])
    # evidence는 이제 객체 리스트이므로 snippet을 직접 가져옴
    ev_texts = [ev.get("snippet", "") for ev in price_ev if isinstance(ev, dict)] if price_ev else []
    if ev_texts and not any(_has_price_signal(t) for t in ev_texts):
        out["price"]["summary"] = "가격 관련 언급이 많지 않아, 전반적인 만족감/구성(양 등) 중심으로만 해석 가능합니다."
        # bullets도 너무 단정적으로 쓰지 않게 최소화
        out["price"]["bullets"] = [
            "가격을 직접 언급한 리뷰가 많지 않습니다.",
            "대신 만족/구성/양(푸짐함) 관련 표현이 간접적으로 나타납니다."
        ]

    return out
