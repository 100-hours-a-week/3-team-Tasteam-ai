"""
LLM 유틸리티 모듈 (Qwen2.5-7B-Instruct 사용)
RunPod 서버리스 엔드포인트를 통한 vLLM 서빙 지원
"""

import json
import logging
import os
import re
import time
import asyncio
import requests
import torch
import httpx
from typing import Dict, List, Optional, Any, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .json_parse_utils import parse_json_relaxed
from .llm_failover_router import (
    LLMFailoverRouter,
    build_gemini_provider,
    build_openai_provider,
)

logger = logging.getLogger(__name__)

# RunPod/vLLM: input + max_tokens <= model context. 보수적 추정(한글 많을수록 토큰 많음)
VLLM_MAX_TOKENS_ABSOLUTE_CAP = 1024  # RunPod/vLLM 호출 시 max_tokens 절대 상한
VLLM_CONTEXT_MARGIN = 256  # 출력 구간 여유


def _estimate_input_tokens(messages: List[Dict[str, str]]) -> int:
    """보수적 추정: 1토큰 ≈ 1.33자 (0647 948/540 재발 방지, 한글 많을수록 실제보다 많게)."""
    total = sum(len(m.get("content", "") or "") for m in messages)
    return max(0, (total * 3) // 4)


def _cap_max_tokens_for_context(max_new_tokens: int, messages: List[Dict[str, str]]) -> int:
    """RunPod/vLLM 컨텍스트 한도 내로 max_tokens 제한 (400 오류 방지). 보수적 추정·여유·절대상한 적용."""
    estimated = _estimate_input_tokens(messages)
    cap = Config.LLM_MAX_CONTEXT_LENGTH - estimated - VLLM_CONTEXT_MARGIN
    if cap < 1:
        # 입력이 이미 한도 근접 시 512 이하로 요청해 서버 400 방지
        return max(1, min(max_new_tokens, 512))
    capped = min(max_new_tokens, max(1, cap))
    return min(capped, VLLM_MAX_TOKENS_ABSOLUTE_CAP)


class LLMUtils:
    """LLM 관련 유틸리티 클래스 (Qwen 모델 사용, RunPod 서버리스 엔드포인트, OpenAI API, 또는 로컬 vLLM 지원)"""
    
    def __init__(
        self,
        model_name: str = Config.LLM_MODEL,
        device: Optional[str] = None,
        use_runpod: Optional[bool] = None,
        use_pod_vllm: Optional[bool] = None,
        llm_provider: Optional[str] = None,
    ):
        """
        Args:
            model_name: 사용할 모델명 (기본값: Qwen/Qwen2.5-7B-Instruct)
            device: 사용할 디바이스 (vLLM 사용 시 무시)
            use_runpod: RunPod 서버리스 엔드포인트 사용 여부 (None이면 Config에서 가져옴)
            use_pod_vllm: RunPod Serverless vLLM 엔드포인트 사용 여부 (None이면 Config에서 가져옴)
            llm_provider: LLM 제공자 ('openai', 'runpod', 'local') (None이면 Config에서 가져옴)
        """
        self.model_name = model_name
        self.llm_provider = llm_provider if llm_provider is not None else Config.LLM_PROVIDER
        
        # 토큰 사용량 추적 (OpenAI API 사용 시)
        self.last_tokens_used = None
        
        # 하위 호환성을 위한 설정
        self.use_runpod = use_runpod if use_runpod is not None else Config.USE_RUNPOD
        self.use_pod_vllm = use_pod_vllm if use_pod_vllm is not None else Config.USE_POD_VLLM
        
        # Router 패턴: 로컬 큐 기본, 오버플로우 시 OpenAI API 폴백
        # OpenAI 클라이언트는 폴백용으로 항상 초기화 시도 (선택적)
        self.openai_client = None
        self.openai_model = None
        self.use_openai_as_fallback = False
        # 벤더 API 페일오버: OpenAI 429/5xx 시 Gemini로 전환 (self-hosted RunPod Serverless 폴백은 제거됨)
        self._failover_router = None

        # LLM_PROVIDER 우선순위 적용
        if self.llm_provider == "openai":
            self.use_openai = True
            self.use_runpod = False
            self.use_pod_vllm = False
        elif self.llm_provider == "runpod":
            self.use_openai = False
            # use_runpod는 기존 로직 유지
        elif self.llm_provider == "local":
            self.use_openai = False
            self.use_runpod = False
            self.use_pod_vllm = False
        else:
            # 기본값: 기존 로직 유지
            self.use_openai = False
        
        # OpenAI API 폴백용 초기화 (로컬 큐 사용 시 오버플로우 대비, 선택적)
        if Config.ENABLE_OPENAI_FALLBACK and Config.OPENAI_API_KEY and not self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.openai_model = Config.OPENAI_MODEL
                self.use_openai_as_fallback = True
                logger.info(f"OpenAI API 폴백 활성화: {self.openai_model} (오버플로우 시 사용)")
            except ImportError:
                logger.warning("openai 패키지가 설치되지 않아 폴백 기능을 사용할 수 없습니다.")
            except Exception as e:
                logger.warning(f"OpenAI API 폴백 초기화 실패: {e}")
        elif Config.ENABLE_OPENAI_FALLBACK and not Config.OPENAI_API_KEY:
            logger.warning("ENABLE_OPENAI_FALLBACK이 활성화되었지만 OPENAI_API_KEY가 설정되지 않았습니다.")
        
        if self.use_openai:
            # OpenAI API 사용 (페일오버: Config.LLM_FAILOVER_ENABLED + GEMINI_API_KEY 시 LLMFailoverRouter 사용)
            if not Config.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                    "export OPENAI_API_KEY='your_api_key'를 설정하세요."
                )
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.openai_model = Config.OPENAI_MODEL
                logger.info(f"OpenAI API 사용: {self.openai_model} (빠른 검증용)")
            except ImportError:
                raise ImportError(
                    "openai 패키지가 설치되지 않았습니다. "
                    "pip install openai>=1.0.0를 실행하세요."
                )
            self.model = None
            self.tokenizer = None
            self.device = None
            self.batch_size = Config.LLM_BATCH_SIZE  # 기본 배치 크기 설정
        elif self.use_pod_vllm:
            # vLLM: RunPod Pod 직접 URL 우선, 없으면 Serverless 엔드포인트
            pod_base = getattr(Config, "VLLM_POD_BASE_URL", None) and (Config.VLLM_POD_BASE_URL or "").strip()
            if pod_base:
                self.use_runpod = True
                self.api_key = "dummy"  # Pod 직접 연결은 인증 없음
                self.endpoint_id = ""
                self._runpod_openai_base_url = pod_base.rstrip("/")
                if not self._runpod_openai_base_url.endswith("/v1"):
                    self._runpod_openai_base_url = f"{self._runpod_openai_base_url.rstrip('/')}/v1"
                self._pod_direct_url = True
                self.poll_interval = Config.RUNPOD_POLL_INTERVAL
                self.max_wait_time = Config.RUNPOD_MAX_WAIT_TIME
                logger.info("vLLM RunPod Pod 직접 연결: %s", self._runpod_openai_base_url)
                self.model = None
                self.tokenizer = None
                self.device = None
                self.batch_size = 1
            else:
                if not Config.RUNPOD_API_KEY:
                    raise ValueError(
                        "USE_POD_VLLM 사용 시 RUNPOD_API_KEY와 RunPod 엔드포인트가 필요합니다. "
                        "RUNPOD_API_KEY, RUNPOD_VLLM_ENDPOINT_ID(또는 RUNPOD_ENDPOINT_ID)를 설정하세요."
                    )
                self.use_runpod = True
                self.api_key = Config.RUNPOD_API_KEY
                raw_eid = getattr(Config, "RUNPOD_VLLM_ENDPOINT_ID", None) or Config.RUNPOD_ENDPOINT_ID
                self.endpoint_id = (raw_eid or "").strip()
                self._runpod_openai_base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/openai/v1"
                self._pod_direct_url = False
                self.poll_interval = Config.RUNPOD_POLL_INTERVAL
                self.max_wait_time = Config.RUNPOD_MAX_WAIT_TIME
                logger.info(
                    "vLLM RunPod Serverless: 엔드포인트 %s (OpenAI 호환 API, 요청 시 GPU 기동, 유휴 시 스케일다운)",
                    self.endpoint_id,
                )
                self.model = None
                self.tokenizer = None
                self.device = None
                self.batch_size = 1
        elif self.use_runpod:
            # RunPod 서버리스 엔드포인트 사용
            if not Config.RUNPOD_API_KEY:
                raise ValueError(
                    "RUNPOD_API_KEY 환경변수가 설정되지 않았습니다. "
                    "export RUNPOD_API_KEY='your_api_key' 또는 USE_RUNPOD=false로 설정하세요."
                )
            self.api_key = Config.RUNPOD_API_KEY
            raw_eid = Config.RUNPOD_ENDPOINT_ID or ""
            self.endpoint_id = raw_eid.strip()
            self._runpod_openai_base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/openai/v1"
            self.poll_interval = Config.RUNPOD_POLL_INTERVAL
            self.max_wait_time = Config.RUNPOD_MAX_WAIT_TIME
            logger.info(f"RunPod 서버리스 엔드포인트 사용: {self.endpoint_id} (OpenAI 호환 API)")
            self.model = None
            self.tokenizer = None
            self.device = None
            self.batch_size = 1  # RunPod는 배치 처리를 엔드포인트에서 처리
        else:
            # 로컬 모델 사용
            # 디바이스 자동 선택
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = device
            
            logger.info(f"로컬 Qwen 모델 로딩 중: {model_name} (device: {device})")
            
            # 모델과 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Flash Attention-2 적용 시도 (설치되어 있으면 사용)
            attn_implementation = None
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                logger.info("Flash Attention-2 사용 가능, 적용합니다.")
            except ImportError:
                logger.info("Flash Attention-2가 설치되지 않았습니다. 기본 attention을 사용합니다.")
            
            model_kwargs = {
                "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
                "device_map": "auto" if device != "cpu" else None,
                "trust_remote_code": True,
            }
            
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # 배치 크기 동적 조정
            self.batch_size = Config.get_optimal_batch_size("llm")
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            logger.info(f"✅ 로컬 Qwen 모델 로딩 완료: {model_name}")
    
    def _get_failover_router(self) -> Optional[LLMFailoverRouter]:
        """페일오버 라우터 지연 초기화. GEMINI_API_KEY 있으면 OpenAI+Gemini 구성."""
        if self._failover_router is not None:
            return self._failover_router
        if not Config.LLM_FAILOVER_ENABLED or not Config.GEMINI_API_KEY or not Config.OPENAI_API_KEY:
            return None
        providers = [
            build_openai_provider(Config.OPENAI_API_KEY, self.openai_model),
            build_gemini_provider(Config.GEMINI_API_KEY, Config.GEMINI_MODEL),
        ]
        self._failover_router = LLMFailoverRouter(providers=providers)
        logger.info("벤더 API 페일오버 활성화: OpenAI → Gemini")
        return self._failover_router

    def _is_openai_rate_limit(self, e: Exception) -> bool:
        """OpenAI 429 Rate limit 예외 여부."""
        if e is None:
            return False
        try:
            from openai import RateLimitError
            if isinstance(e, RateLimitError):
                return True
        except ImportError:
            pass
        code = getattr(e, "status_code", None) or (
            getattr(getattr(e, "response", None), "status_code", None)
        )
        if code == 429:
            return True
        return "rate_limit" in str(e).lower() or "429" in str(e)

    @staticmethod
    def _extract_runpod_output(output: Any) -> str:
        """
        RunPod vLLM status 응답의 output에서 생성 텍스트만 추출.
        output이 리스트면 [{"choices": [{"text": "..."}]}] 형식에서 텍스트 추출.
        """
        if output is None:
            raise Exception("출력이 없습니다")
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            return str(output.get("text", output)).strip()
        if isinstance(output, list) and len(output) > 0:
            first = output[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                if first.get("output") is not None:
                    return str(first["output"]).strip()
                choices = first.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    c0 = choices[0]
                    if c0.get("text") is not None:
                        return str(c0["text"]).strip()
                    msg = c0.get("message") or {}
                    if isinstance(msg, dict) and msg.get("content") is not None:
                        return str(msg["content"]).strip()
                    tokens = c0.get("tokens")
                    if isinstance(tokens, list):
                        return " ".join(str(t) for t in tokens).strip()
            return str(first).strip()
        return str(output).strip()

    def _messages_to_prompt_for_vllm(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI 형식 messages를 vLLM/RunPod용 단일 프롬프트 문자열로 변환."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

    def _runpod_openai_base_url_for(self, endpoint_id: Optional[str]) -> str:
        """endpoint_id에 대해 RunPod OpenAI 호환 base URL 반환 (폴백 등). Pod 직접 URL 사용 시 인스턴스 URL 반환."""
        if getattr(self, "_pod_direct_url", False):
            return self._runpod_openai_base_url
        eid = (endpoint_id or getattr(self, "endpoint_id", None) or "").strip()
        if not eid:
            raise ValueError("RunPod endpoint_id가 필요합니다.")
        return f"https://api.runpod.ai/v2/{eid}/openai/v1"

    def _call_runpod_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = None,
    ) -> str:
        """
        RunPod OpenAI 호환 API(/openai/v1)로 chat completions 호출 (동기).
        채팅 템플릿이 적용되어 JSON 등 구조화 출력에 유리.
        """
        from openai import OpenAI
        base_url = self._runpod_openai_base_url_for(endpoint_id)
        key = api_key if api_key is not None else getattr(self, "api_key", None)
        if not key:
            raise ValueError("RunPod api_key가 필요합니다.")
        retries = max_retries if max_retries is not None else Config.MAX_RETRIES
        capped_max = _cap_max_tokens_for_context(max_new_tokens, messages)
        client = OpenAI(api_key=key, base_url=base_url)
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=capped_max,
                )
                if getattr(response, "usage", None):
                    self.last_tokens_used = response.usage.total_tokens
                else:
                    self.last_tokens_used = None
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(
                    "RunPod OpenAI API 실패 (시도 %s/%s): %s",
                    attempt + 1, retries, e,
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise Exception("RunPod OpenAI API 재시도 초과")

    async def _call_runpod_openai_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = None,
    ) -> str:
        """
        RunPod OpenAI 호환 API(/openai/v1)로 chat completions 호출 (비동기).
        """
        from openai import AsyncOpenAI
        base_url = self._runpod_openai_base_url_for(endpoint_id)
        key = api_key if api_key is not None else getattr(self, "api_key", None)
        if not key:
            raise ValueError("RunPod api_key가 필요합니다.")
        retries = max_retries if max_retries is not None else Config.MAX_RETRIES
        capped_max = _cap_max_tokens_for_context(max_new_tokens, messages)
        client = AsyncOpenAI(api_key=key, base_url=base_url)
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=capped_max,
                )
                if getattr(response, "usage", None):
                    self.last_tokens_used = response.usage.total_tokens
                else:
                    self.last_tokens_used = None
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(
                    "RunPod OpenAI API 실패 (시도 %s/%s): %s",
                    attempt + 1, retries, e,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise Exception("RunPod OpenAI API 재시도 초과")

    def _call_runpod(self, prompt: str, max_retries: int = Config.MAX_RETRIES) -> str:
        """
        RunPod 서버리스 엔드포인트를 호출하여 텍스트를 생성합니다.
        
        Args:
            prompt: 생성할 프롬프트
            max_retries: 최대 재시도 횟수
            
        Returns:
            생성된 텍스트
        """
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            "input": {
                "prompt": prompt
            }
        }
        
        for attempt in range(max_retries):
            try:
                # 작업 시작
                response = requests.post(url, json=data, headers=headers, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(
                        f"RunPod 요청 실패 (시도 {attempt + 1}/{max_retries}): "
                        f"상태 코드 {response.status_code}, 응답: {response.text[:200]}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # 지수 백오프
                        continue
                    else:
                        raise Exception(f"RunPod 요청 실패: {response.text}")
                
                response_json = response.json()
                job_id = response_json.get("id")
                
                if not job_id:
                    raise Exception(f"작업 ID를 받지 못했습니다: {response_json}")
                
                logger.debug(f"RunPod 작업 시작: {job_id}")
                
                # 작업 완료 대기
                start_time = time.time()
                while True:
                    status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
                    status_response = requests.get(status_url, headers=headers, timeout=10)
                    
                    if status_response.status_code != 200:
                        logger.warning(f"상태 확인 실패: {status_response.status_code}")
                        time.sleep(self.poll_interval)
                        continue
                    
                    status_info = status_response.json()
                    status = status_info.get("status")
                    
                    if status == "COMPLETED":
                        output = status_info.get("output")
                        return self._extract_runpod_output(output)
                    elif status == "FAILED":
                        error = status_info.get("error", "알 수 없는 오류")
                        raise Exception(f"RunPod 작업 실패: {error}")
                    elif time.time() - start_time > self.max_wait_time:
                        raise Exception(f"시간 초과: {self.max_wait_time}초 동안 완료되지 않았습니다")
                    
                    time.sleep(self.poll_interval)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"RunPod 요청 중 오류 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"RunPod 호출 실패: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
        
        raise Exception("모든 재시도 실패")

    async def _call_runpod_async(
        self,
        prompt: str,
        max_retries: int = Config.MAX_RETRIES,
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """
        RunPod 서버리스 엔드포인트를 httpx.AsyncClient로 비동기 호출.
        endpoint_id/api_key를 넘기면 429 폴백 등 오버라이드 호출에 사용(요청 시 GPU 기동, 유휴 시 스케일다운).
        """
        eid = endpoint_id if endpoint_id is not None else getattr(self, "endpoint_id", None)
        key = api_key if api_key is not None else getattr(self, "api_key", None)
        if not eid or not key:
            raise ValueError("RunPod endpoint_id와 api_key가 필요합니다.")
        poll = (
            Config.RUNPOD_POLL_INTERVAL
            if endpoint_id is not None
            else getattr(self, "poll_interval", Config.RUNPOD_POLL_INTERVAL)
        )
        max_wait = (
            Config.RUNPOD_MAX_WAIT_TIME
            if endpoint_id is not None
            else getattr(self, "max_wait_time", Config.RUNPOD_MAX_WAIT_TIME)
        )
        url = f"https://api.runpod.ai/v2/{eid}/run"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        data = {"input": {"prompt": prompt}}

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, json=data, headers=headers)
                    if response.status_code != 200:
                        logger.warning(
                            f"RunPod 요청 실패 (시도 {attempt + 1}/{max_retries}): "
                            f"상태 코드 {response.status_code}, 응답: {response.text[:200]}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise Exception(f"RunPod 요청 실패: {response.text}")

                    response_json = response.json()
                    job_id = response_json.get("id")
                    if not job_id:
                        raise Exception(f"작업 ID를 받지 못했습니다: {response_json}")

                    logger.debug(f"RunPod 작업 시작: {job_id}")
                    start_time = time.time()
                    while True:
                        status_url = f"https://api.runpod.ai/v2/{eid}/status/{job_id}"
                        status_response = await client.get(status_url, headers=headers)
                        if status_response.status_code != 200:
                            logger.warning(f"상태 확인 실패: {status_response.status_code}")
                            await asyncio.sleep(poll)
                            continue

                        status_info = status_response.json()
                        status = status_info.get("status")

                        if status == "COMPLETED":
                            output = status_info.get("output")
                            return self._extract_runpod_output(output)
                        if status == "FAILED":
                            error = status_info.get("error", "알 수 없는 오류")
                            raise Exception(f"RunPod 작업 실패: {error}")
                        if time.time() - start_time > max_wait:
                            raise Exception(f"시간 초과: {max_wait}초 동안 완료되지 않았습니다")

                        await asyncio.sleep(poll)
            except httpx.HTTPError as e:
                logger.warning(f"RunPod 요청 중 오류 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                logger.error(f"RunPod 비동기 호출 실패: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise Exception("모든 재시도 실패")

    def _generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Router 패턴: 로컬 큐 기본 사용, 오버플로우 시 OpenAI API 폴백
        
        Args:
            messages: 대화 메시지 리스트 (OpenAI 형식)
            temperature: 생성 온도
            max_new_tokens: 최대 생성 토큰 수 (기본값: 50, 요약/비교 시 더 큰 값 필요)
            
        Returns:
            생성된 응답 텍스트
        """
        # Router: OpenAI 전용 모드
        if self.use_openai:
            router = self._get_failover_router()
            if router:
                try:
                    out = router.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        response_format={"type": "json_object"},
                    )
                    self.last_tokens_used = None
                    return out
                except Exception as e:
                    self.last_tokens_used = None
                    raise
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    response_format={"type": "json_object"},
                )
                if hasattr(response, 'usage') and response.usage:
                    self.last_tokens_used = response.usage.total_tokens
                else:
                    self.last_tokens_used = None
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.last_tokens_used = None
                raise
        else:
            # Router: 로컬 큐 기본 사용 (오버플로우 시 OpenAI API 폴백)
            try:
                # 1단계: 로컬 큐 시도 (vLLM, RunPod, 또는 로컬 모델)
                return self._generate_with_local_queue(messages, temperature, max_new_tokens)
            except Exception as e:
                # 2단계: 오버플로우/에러 발생 시 OpenAI API 폴백
                if self.use_openai_as_fallback and self.openai_client:
                    logger.warning(f"로컬 큐 처리 실패, OpenAI API로 폴백: {str(e)}")
                    try:
                        response = self.openai_client.chat.completions.create(
                            model=self.openai_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_new_tokens,
                            response_format={"type": "json_object"},
                        )
                        if hasattr(response, 'usage') and response.usage:
                            self.last_tokens_used = response.usage.total_tokens
                        else:
                            self.last_tokens_used = None
                        return response.choices[0].message.content.strip()
                    except Exception as fallback_error:
                        logger.error(f"OpenAI API 폴백도 실패: {str(fallback_error)}")
                        raise e  # 원래 에러를 다시 발생
                else:
                    # 폴백 불가능한 경우 원래 에러 발생
                    raise

    async def _generate_with_local_queue_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 50,
    ) -> str:
        """로컬 큐 비동기: RunPod Serverless OpenAI 호환 API 사용."""
        if self.use_runpod or self.use_pod_vllm:
            return await self._call_runpod_openai_async(
                messages, temperature, max_new_tokens
            )
        else:
            raise NotImplementedError("llm_async 모드에서는 RunPod 또는 OpenAI만 지원합니다.")

    async def _generate_response_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 50,
        top_p: Optional[float] = None,
    ) -> str:
        """
        비동기 LLM 응답 (Config.SUMMARY_LLM_ASYNC=True 시 배치에서 사용).
        OpenAI: AsyncOpenAI 사용. RunPod: httpx.AsyncClient 사용.
        """
        if self.use_openai:
            router = self._get_failover_router()
            if router:
                try:
                    out = await router.chat_async(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_new_tokens,
                        response_format={"type": "json_object"},
                        top_p=top_p,
                    )
                    self.last_tokens_used = None
                    return out
                except Exception as e:
                    self.last_tokens_used = None
                    raise
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                create_kwargs: Dict[str, Any] = {
                    "model": self.openai_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_new_tokens,
                    "response_format": {"type": "json_object"},
                }
                if top_p is not None:
                    create_kwargs["top_p"] = top_p
                response = await client.chat.completions.create(**create_kwargs)
                if hasattr(response, "usage") and response.usage:
                    self.last_tokens_used = response.usage.total_tokens
                else:
                    self.last_tokens_used = None
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.last_tokens_used = None
                raise
        else:
            try:
                return await self._generate_with_local_queue_async(
                    messages, temperature, max_new_tokens
                )
            except Exception as e:
                if self.use_openai_as_fallback and Config.OPENAI_API_KEY:
                    logger.warning(f"로컬 큐 비동기 실패, OpenAI API로 폴백: {str(e)}")
                    try:
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
                        fallback_kwargs: Dict[str, Any] = {
                            "model": Config.OPENAI_MODEL,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_new_tokens,
                            "response_format": {"type": "json_object"},
                        }
                        if top_p is not None:
                            fallback_kwargs["top_p"] = top_p
                        response = await client.chat.completions.create(**fallback_kwargs)
                        if hasattr(response, "usage") and response.usage:
                            self.last_tokens_used = response.usage.total_tokens
                        else:
                            self.last_tokens_used = None
                        return response.choices[0].message.content.strip()
                    except Exception as fallback_error:
                        logger.error(f"OpenAI API 폴백도 실패: {str(fallback_error)}")
                        raise e
                raise
    
    def _generate_with_local_queue(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_new_tokens: int = 50,
    ) -> str:
        """
        로컬 큐를 사용하여 응답 생성 (vLLM, RunPod, 또는 로컬 모델)
        
        Args:
            messages: 대화 메시지 리스트 (OpenAI 형식)
            temperature: 생성 온도
            max_new_tokens: 최대 생성 토큰 수
            
        Returns:
            생성된 응답 텍스트
        """
        if self.use_runpod or self.use_pod_vllm:
            # RunPod OpenAI 호환 API 사용 (채팅 템플릿 적용, JSON 출력 안정)
            return self._call_runpod_openai(
                messages, temperature=temperature, max_new_tokens=max_new_tokens
            )
        else:
            # 로컬 모델 사용
            # Gemma 모델은 system role을 지원하지 않으므로 변환 필요
            is_gemma_model = "gemma" in self.model_name.lower()
            
            if is_gemma_model:
                # Gemma 모델: system role을 user role로 변환
                processed_messages = []
                system_content = None
                
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        # system content를 누적
                        if system_content:
                            system_content += "\n" + content
                        else:
                            system_content = content
                    else:
                        processed_messages.append(msg)
                
                # system content가 있으면 첫 번째 user 메시지에 통합
                if system_content and processed_messages:
                    first_user_msg = processed_messages[0]
                    if first_user_msg.get("role") == "user":
                        first_user_msg["content"] = system_content + "\n\n" + first_user_msg["content"]
                    else:
                        processed_messages.insert(0, {"role": "user", "content": system_content})
                elif system_content:
                    processed_messages = [{"role": "user", "content": system_content}]
                
                messages_to_use = processed_messages
            else:
                # Qwen, Llama 등: system role 그대로 사용
                messages_to_use = messages
            
            # Qwen chat template 형식으로 변환
            try:
                text = self.tokenizer.apply_chat_template(
                    messages_to_use,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # chat template 실패 시 (예: 예상치 못한 모델)
                logger.warning(f"Chat template 적용 실패: {e}, system role 제거 후 재시도")
                # system role 제거 후 재시도
                fallback_messages = []
                system_content = None
                for msg in messages_to_use:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        if system_content:
                            system_content += "\n" + content
                        else:
                            system_content = content
                    else:
                        fallback_messages.append(msg)
                
                if system_content and fallback_messages:
                    first_user_msg = fallback_messages[0]
                    if first_user_msg.get("role") == "user":
                        first_user_msg["content"] = system_content + "\n\n" + first_user_msg["content"]
                    else:
                        fallback_messages.insert(0, {"role": "user", "content": system_content})
                elif system_content:
                    fallback_messages = [{"role": "user", "content": system_content}]
                
                text = self.tokenizer.apply_chat_template(
                    fallback_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # 토크나이징
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 생성 (최적화 옵션 적용)
            with torch.no_grad():
                generate_kwargs = {
                    **model_inputs,
                    "max_new_tokens": max_new_tokens,
                    "num_beams": 1,  # beam search 비활성화 (빠름)
                    "use_cache": True,  # KV 캐시 사용 (빠름)
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                # do_sample=True일 때만 샘플링 파라미터 추가 (경고 메시지 방지)
                if temperature > 0.1:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["do_sample"] = True
                else:
                    generate_kwargs["do_sample"] = False
                
                generated_ids = self.model.generate(**generate_kwargs)
            
            # 디코딩
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.strip()
            
            return response


    async def generate_comparison_interpretation_async(
        self,
        category: str,
        lift_pct: float,
        tone: str,
        n_reviews: int,
    ) -> Optional[str]:
        """
        비교 해석 전체 문장 생성.
        - 숫자(%) 정확히 1회 포함
        - 하지만/다만/그래도로 차이 크기만 해석
        - 좋은 편/긍정적인 편 같은 재진술 금지
        """
        raw: Optional[str] = None

        FORBIDDEN_PHRASES = (
            "리뷰 수", "표본 수", "신뢰", "믿을 수",
            "상당히", "매우", "굉장히", "확실히", "단연",
            "최고", "압도적", "완벽",
            "평균보다 높은 비율",  # 보고서형 표현 축소
        )
        QUALITATIVE_RESTATEMENT = ("좋은 편", "긍정적인 편")
        PCT_RE = r"\d+(?:\.\d+)?\s*(?:%|퍼센트)"
        INTERP_MARKERS = ("차이", "크지", "어느 정도", "크게", "나타", "비슷")

        def _extract_json_obj(text: str) -> Optional[str]:
            t = text.strip()
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
            i, j = t.find("{"), t.rfind("}")
            if i == -1 or j == -1 or j <= i:
                return None
            return t[i:j+1]

        def _enforce_one_sentence(s: str) -> str:
            """마침표 2개 나오면 첫 문장 or 숫자 포함 문장만 남김."""
            t = " ".join(s.strip().split())
            parts = re.split(r"(?<=[.!?])\s+|\s*\n+\s*", t)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) <= 1:
                return t
            for p in parts:
                if re.search(PCT_RE, p):
                    return p if re.search(r"[.!?]$", p) else (p + ".")
            p0 = parts[0]
            return p0 if re.search(r"[.!?]$", p0) else (p0 + ".")

        def is_valid_interp(s: str, abs_lift_val: float) -> bool:
            s = " ".join(s.strip().split())
            if not s:
                return False
            if any(p in s for p in FORBIDDEN_PHRASES):
                return False
            if any(q in s for q in QUALITATIVE_RESTATEMENT):
                return False
            if abs_lift_val < 30 and "강점" in s:
                return False
            pct_matches = re.findall(PCT_RE, s)
            if len(pct_matches) != 1:
                return False
            if "평균" not in s:
                return False
            if not any(m in s for m in INTERP_MARKERS):
                return False
            return True

        abs_lift = abs(lift_pct)
        if abs_lift < 10:
            diff_hint = "문장은 '... 높지만, 차이는 크지 않은 편입니다.'처럼 차이가 크지 않음을 해석하세요."
        elif abs_lift < 30:
            diff_hint = "문장은 '... 높지만, 차이가 어느 정도 있습니다.'처럼 완만한 차이를 해석하세요."
        else:
            diff_hint = "문장은 '... 높아, 강점이 보입니다.' 또는 '... 높아, 차이가 크게 나타납니다.'처럼 의미를 한 단계 더 명확히 하세요. '차이가 큰 편'만 쓰지 말고 강점/차이 나타남을 드러내세요."

        system_content = (
            "당신은 음식점 비교 해석 문장을 만드는 도우미입니다. "
            "반드시 다음을 지킵니다: "
            "(1) 문장에 숫자(%)를 정확히 1회 포함하세요. "
            "(2) 좋은 편/긍정적인 편 같은 재진술 금지. "
            "(3) 접속어(예: '높지만', '높아', '다만')를 사용해 차이 크기만 해석하세요. "
            "(4) 과장/강조 표현 금지. 예: '최고', '압도적', '완벽', '상당히', '매우', '굉장히'. "
            "(5) 리뷰수/표본수/신뢰도 언급 금지. "
            "(6) '평균보다 높은 비율은', '~비율은' 같은 보고서형 표현은 쓰지 마세요. 짧고 명확하게. "
            "(7) lift는 만족도가 평균 대비 얼마나 높은지. 가격 lift = 가격/가성비 만족. "
            "(8) 한 문장만 출력(줄바꿈 금지). "
            "반드시 JSON 형식으로만 답하세요: {\"interpretation\": \"한 문장\"}."
        )

        user_content = (
            f"카테고리: {category}. "
            f"lift 퍼센트: {int(round(lift_pct))}%. "
            f"표본 톤: {tone}. "
            f"{diff_hint} "
            "숫자(%)는 문장에 정확히 1회만 포함하세요."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self._generate_response_async(
                messages,
                temperature=0.0,
                max_new_tokens=80,
                top_p=0.2,
            )
            if not raw or not raw.strip():
                return None

            json_text = _extract_json_obj(raw) or raw
            data = parse_json_relaxed(json_text)
            if not isinstance(data, dict):
                data = {}
            interp = data.get("interpretation")
            if isinstance(interp, str):
                interp = _enforce_one_sentence(interp)
                interp = interp.replace("퍼센트", "%")
                if is_valid_interp(interp, abs_lift):
                    return interp.strip()
            return None

        except Exception as e:
            if raw and isinstance(raw, str):
                candidate = _enforce_one_sentence(raw)
                candidate = candidate.replace("퍼센트", "%")
                if is_valid_interp(candidate, abs_lift):
                    return candidate.strip()
            logger.warning("비교 해석 LLM 실패: %s", e)
            return None
