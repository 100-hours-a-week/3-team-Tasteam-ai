"""
LLM 출력용 완화된 JSON 파싱.
RunPod vLLM 등이 반환하는 비표준 JSON(작은따옴표, 미따옴표 키, trailing comma)을 처리.
"""

import json
import re
from typing import Any, Optional, Union


def extract_json_block(text: str, want_object: bool = True) -> Optional[str]:
    """
    텍스트에서 첫 번째 완전한 JSON 객체 { } 또는 배열 [ ] 를 괄호 매칭으로 추출.
    want_object=True면 { } 우선, False면 [ ] 우선. 실패 시 None.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # 마크다운 코드블록 제거
    for start in ("```json", "```"):
        if s.startswith(start):
            s = s[len(start):].lstrip()
            break
    if s.endswith("```"):
        s = s[:-3].rstrip()
    s = s.strip()
    if not s:
        return None

    def find_matching(s: str, open_ch: str, close_ch: str, start: int) -> Optional[int]:
        depth = 0
        i = start
        while i < len(s):
            if s[i] == open_ch:
                depth += 1
            elif s[i] == close_ch:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    # 먼저 나오는 [ 또는 { 기준으로 해당 블록만 추출 (중첩 괄호 정확히 매칭)
    idx_b = s.find("{")
    idx_a = s.find("[")
    candidates = []
    if idx_b >= 0:
        j = find_matching(s, "{", "}", idx_b)
        if j is not None:
            candidates.append((idx_b, j + 1, "object"))
    if idx_a >= 0:
        j = find_matching(s, "[", "]", idx_a)
        if j is not None:
            candidates.append((idx_a, j + 1, "array"))
    if not candidates:
        return None
    # want_object면 객체 우선, 아니면 배열 우선
    candidates.sort(key=lambda x: (0 if (x[2] == "object") == want_object else 1, x[0]))
    start_i, end_i, _ = candidates[0]
    return s[start_i:end_i]


def _fix_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ]."""
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _fix_unquoted_keys(s: str) -> str:
    """Fix unquoted object keys: { id: 1 } -> { "id": 1 }."""
    # After { or ,, optional whitespace, then identifier, then :
    return re.sub(r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', s)


def _fix_single_quoted_keys(s: str) -> str:
    """Fix single-quoted keys: 'key': -> "key":."""
    return re.sub(r"'([^']*)'\s*:", r'"\1":', s)


def _fix_single_quoted_string_values(s: str) -> str:
    """Fix single-quoted string values after colon: : 'value' -> : "value" (escape " inside)."""
    def repl(m: re.Match) -> str:
        val = m.group(1).replace("\\", "\\\\").replace('"', '\\"')
        return f': "{val}"'
    return re.sub(r":\s*'([^']*)'", repl, s)


def parse_json_relaxed(text: str) -> Optional[Union[dict, list]]:
    """
    Parse JSON with common LLM output relaxations.
    Tries json.loads first, then extracts JSON block by brace matching if needed,
    then applies fixes for unquoted keys, single-quoted keys, and trailing commas.
    Returns parsed dict/list or None on failure.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    # 0) 괄호 매칭으로 블록 추출 (앞뒤 설명/마크다운 제거)
    want_obj = not s.lstrip().startswith("[")
    extracted = extract_json_block(s, want_object=want_obj)
    if extracted is not None and len(extracted.strip()) > 0:
        s = extracted
    if not s.strip():
        return None

    # 1) Strict parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Trailing commas
    try:
        s2 = _fix_trailing_commas(s)
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 3) Unquoted keys (e.g. [{ id: 1, sentiment: "positive" }])
    try:
        s2 = _fix_unquoted_keys(s)
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 4) Single-quoted keys (e.g. {'interpretation': '...'})
    try:
        s2 = _fix_single_quoted_keys(s)
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 5) Single-quoted string values
    try:
        s2 = _fix_single_quoted_string_values(s)
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 6) Combine all fixes
    try:
        s2 = s
        for fix_fn in (_fix_single_quoted_keys, _fix_single_quoted_string_values, _fix_unquoted_keys, _fix_trailing_commas):
            s2 = fix_fn(s2)
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    return None
