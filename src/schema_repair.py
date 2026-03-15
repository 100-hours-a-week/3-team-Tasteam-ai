"""
LLM 출력 스키마 보정(repair).

목표 스키마:
{
  "service": {"summary": str, "bullets": [str], "evidence": [int]},
  "price":   {"summary": str, "bullets": [str], "evidence": [int]},
  "food":    {"summary": str, "bullets": [str], "evidence": [int]},
  "overall_summary": {"summary": str}  # 선택
}
"""

from __future__ import annotations

from typing import Any

REQUIRED_TOP_KEYS = ("service", "price", "food")
OPTIONAL_TOP_KEYS = ("overall_summary",)


def repair_summary_schema(
    parsed: dict | None,
    *,
    bullet_max: int = 3,
    bullet_max_chars: int = 80,
    summary_max_chars: int = 200,
) -> dict | None:
    """
    파싱된 dict를 스키마에 맞게 보정한다.

    - 최상위 필수 키(service/price/food) 없으면 채움
    - 허용되지 않은 키 제거 (최상위는 REQUIRED + OPTIONAL만)
    - examples -> bullets 별칭 보정
    - summary/bullets/evidence 타입·기본값 정리
    - evidence는 int list로 정리하고 bullets 길이와 맞춤
    """
    if not isinstance(parsed, dict):
        return None

    def _as_str(x: Any) -> str:
        if isinstance(x, str):
            return x
        if x is None:
            return ""
        return str(x)

    def _aspect_cell(val: Any) -> dict:
        if not isinstance(val, dict):
            val = {}

        summary = _as_str(val.get("summary", "")).strip()[:summary_max_chars]

        bullets_raw = val.get("bullets")
        if bullets_raw is None:
            bullets_raw = val.get("examples")
        bullets: list[str] = []
        if isinstance(bullets_raw, list):
            for b in bullets_raw:
                s = _as_str(b).strip()
                if s:
                    bullets.append(s[:bullet_max_chars])
                if len(bullets) >= bullet_max:
                    break

        ev_raw = val.get("evidence", [])
        evidence: list[int] = []
        if isinstance(ev_raw, list):
            for x in ev_raw:
                try:
                    evidence.append(int(x))
                except (TypeError, ValueError):
                    continue
        # bullets 길이에 맞춤 (짧으면 자르고, 길면 0으로 패딩)
        if len(evidence) > len(bullets):
            evidence = evidence[: len(bullets)]
        elif len(evidence) < len(bullets):
            evidence = evidence + [0] * (len(bullets) - len(evidence))

        return {"summary": summary, "bullets": bullets, "evidence": evidence}

    out: dict[str, Any] = {k: _aspect_cell(parsed.get(k)) for k in REQUIRED_TOP_KEYS}

    if "overall_summary" in parsed:
        ov = parsed.get("overall_summary")
        if not isinstance(ov, dict):
            ov = {}
        out["overall_summary"] = {"summary": _as_str(ov.get("summary", "")).strip()[:summary_max_chars]}

    return out

