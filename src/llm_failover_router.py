"""
벤더 API 페일오버 라우터 (failover_router_strategy.md).

429/5xx/타임아웃 시 다른 벤더로 자동 전환.
기존 self-hosted(RunPod Serverless) fallback 제거 → 벤더 API(OpenAI, Gemini) failover로 변경.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_SEC = 60
CIRCUIT_BREAKER_THRESHOLD = 3


def _is_retryable(e: Exception) -> bool:
    """429/5xx/타임아웃 등 재시도·failover 대상 여부."""
    if e is None:
        return False
    try:
        from openai import RateLimitError, APIStatusError
        if isinstance(e, RateLimitError):
            return True
        if isinstance(e, APIStatusError) and e.status_code and e.status_code >= 500:
            return True
    except ImportError:
        pass
    code = getattr(e, "status_code", None) or (
        getattr(getattr(e, "response", None), "status_code", None)
    )
    if code in (429, 500, 502, 503, 504):
        return True
    s = str(e).lower()
    return "rate_limit" in s or "429" in s or "timeout" in s or "503" in s or "502" in s


def _get_retry_after_sec(e: Exception) -> float:
    """Retry-After 헤더 값 추출 (초)."""
    try:
        resp = getattr(e, "response", None)
        if resp and hasattr(resp, "headers"):
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra), 60.0)
                except ValueError:
                    return 10.0
    except Exception:
        pass
    return 10.0


@dataclass
class Provider:
    name: str
    call_sync: Callable[..., str]
    call_async: Callable[..., Any]  # async, returns awaitable
    tier: int = 0
    cooldown_until: float = 0
    fail_count: int = 0


class LLMFailoverRouter:
    """
    벤더 API 페일오버 라우터.
    Tier 순으로 시도, 429 시 Retry-After 반영 1회 재시도 후 failover.
    Circuit breaker: 연속 실패 시 일정 시간 제외.
    """

    def __init__(
        self,
        providers: List[Provider],
        circuit_breaker_sec: float = CIRCUIT_BREAKER_SEC,
        circuit_breaker_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
    ):
        self.providers = sorted(providers, key=lambda p: p.tier)
        self.circuit_breaker_sec = circuit_breaker_sec
        self.circuit_breaker_threshold = circuit_breaker_threshold

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        top_p: Optional[float] = None,
    ) -> str:
        last_err: Optional[Exception] = None
        now = time.time()

        for p in self.providers:
            if now < p.cooldown_until:
                continue
            try:
                kwargs: Dict[str, Any] = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if top_p is not None:
                    kwargs["top_p"] = top_p
                return p.call_sync(**kwargs)
            except Exception as e:
                last_err = e
                p.fail_count += 1

                if _is_retryable(e) and p.fail_count == 1:
                    wait = _get_retry_after_sec(e)
                    logger.warning("Provider %s retryable error, retrying after %.0fs: %s", p.name, wait, e)
                    time.sleep(wait)
                    try:
                        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
                        if response_format:
                            kwargs["response_format"] = response_format
                        if top_p is not None:
                            kwargs["top_p"] = top_p
                        return p.call_sync(**kwargs)
                    except Exception as retry_e:
                        last_err = retry_e
                        p.fail_count += 1

                if p.fail_count >= self.circuit_breaker_threshold:
                    p.cooldown_until = time.time() + self.circuit_breaker_sec
                    p.fail_count = 0
                    logger.warning("Provider %s circuit breaker: excluded for %.0fs", p.name, self.circuit_breaker_sec)
                continue

        raise RuntimeError(f"All providers failed. last_err={last_err}")

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, Any]] = None,
        top_p: Optional[float] = None,
    ) -> str:
        import asyncio
        last_err: Optional[Exception] = None
        now = time.time()

        for p in self.providers:
            if now < p.cooldown_until:
                continue
            try:
                kwargs = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format
                if top_p is not None:
                    kwargs["top_p"] = top_p
                return await p.call_async(**kwargs)
            except Exception as e:
                last_err = e
                p.fail_count += 1

                if _is_retryable(e) and p.fail_count == 1:
                    wait = _get_retry_after_sec(e)
                    logger.warning("Provider %s retryable error, retrying after %.0fs: %s", p.name, wait, e)
                    await asyncio.sleep(wait)
                    try:
                        kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
                        if response_format:
                            kwargs["response_format"] = response_format
                        if top_p is not None:
                            kwargs["top_p"] = top_p
                        return await p.call_async(**kwargs)
                    except Exception as retry_e:
                        last_err = retry_e
                        p.fail_count += 1

                if p.fail_count >= self.circuit_breaker_threshold:
                    p.cooldown_until = time.time() + self.circuit_breaker_sec
                    p.fail_count = 0
                    logger.warning("Provider %s circuit breaker: excluded for %.0fs", p.name, self.circuit_breaker_sec)
                continue

        raise RuntimeError(f"All providers failed. last_err={last_err}")


def build_openai_provider(api_key: str, model: str) -> Provider:
    def _sync(messages, temperature=0.3, max_tokens=1024, response_format=None, top_p=None):
        from openai import OpenAI
        c = OpenAI(api_key=api_key)
        kw: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            kw["response_format"] = response_format
        if top_p is not None:
            kw["top_p"] = top_p
        r = c.chat.completions.create(**kw)
        return r.choices[0].message.content.strip()

    async def _async(messages, temperature=0.3, max_tokens=1024, response_format=None, top_p=None):
        from openai import AsyncOpenAI
        c = AsyncOpenAI(api_key=api_key)
        kw = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            kw["response_format"] = response_format
        if top_p is not None:
            kw["top_p"] = top_p
        r = await c.chat.completions.create(**kw)
        return r.choices[0].message.content.strip()

    return Provider(name="openai", call_sync=_sync, call_async=_async, tier=0)


def build_gemini_provider(api_key: str, model: str = "gemini-1.5-flash") -> Provider:
    base = "https://generativelanguage.googleapis.com/v1beta/openai"

    def _sync(messages, temperature=0.3, max_tokens=1024, response_format=None, top_p=None):
        from openai import OpenAI
        c = OpenAI(api_key=api_key, base_url=base)
        kw: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            kw["response_format"] = response_format
        if top_p is not None:
            kw["top_p"] = top_p
        r = c.chat.completions.create(**kw)
        return r.choices[0].message.content.strip()

    async def _async(messages, temperature=0.3, max_tokens=1024, response_format=None, top_p=None):
        from openai import AsyncOpenAI
        c = AsyncOpenAI(api_key=api_key, base_url=base)
        kw = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:
            kw["response_format"] = response_format
        if top_p is not None:
            kw["top_p"] = top_p
        r = await c.chat.completions.create(**kw)
        return r.choices[0].message.content.strip()

    return Provider(name="gemini", call_sync=_sync, call_async=_async, tier=1)
