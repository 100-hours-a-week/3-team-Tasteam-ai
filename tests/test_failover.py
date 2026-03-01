# tests/test_failover.py (예시)
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError

def test_failover_on_429():
    from src.llm_failover_router import LLMFailoverRouter, build_openai_provider, build_gemini_provider
    from src.config import Config

    # OpenAI provider가 항상 429를 던지게 mock
    def failing_openai_sync(*, messages, **kwargs):
        raise RateLimitError("rate_limit", response=MagicMock(status_code=429))

    async def failing_openai_async(*, messages, **kwargs):
        raise RateLimitError("rate_limit", response=MagicMock(status_code=429))

    from src.llm_failover_router import Provider
    openai_provider = Provider("openai", failing_openai_sync, failing_openai_async, tier=0)
    gemini_provider = build_gemini_provider(Config.GEMINI_API_KEY, Config.GEMINI_MODEL)

    router = LLMFailoverRouter(providers=[openai_provider, gemini_provider])
    result = router.chat(messages=[{"role": "user", "content": "hi"}])
    assert result  # Gemini 응답이 와야 성공
    
if __name__ == "__main__":
    test_failover_on_429()