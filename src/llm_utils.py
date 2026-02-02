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
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config

logger = logging.getLogger(__name__)


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
            use_pod_vllm: RunPod Pod에서 vLLM 직접 사용 여부 (None이면 Config에서 가져옴)
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
            # OpenAI API 사용
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
            # RunPod Pod에서 vLLM 직접 사용
            logger.info(f"vLLM 직접 사용 모드: {model_name}")
            self._init_vllm()
            self.executor = ThreadPoolExecutor(max_workers=4)  # 비동기 실행용
            self.use_runpod = False  # vLLM 사용 시 RunPod Serverless 비활성화
        elif self.use_runpod:
            # RunPod 서버리스 엔드포인트 사용
            if not Config.RUNPOD_API_KEY:
                raise ValueError(
                    "RUNPOD_API_KEY 환경변수가 설정되지 않았습니다. "
                    "export RUNPOD_API_KEY='your_api_key' 또는 USE_RUNPOD=false로 설정하세요."
                )
            self.api_key = Config.RUNPOD_API_KEY
            self.endpoint_id = Config.RUNPOD_ENDPOINT_ID
            self.poll_interval = Config.RUNPOD_POLL_INTERVAL
            self.max_wait_time = Config.RUNPOD_MAX_WAIT_TIME
            logger.info(f"RunPod 서버리스 엔드포인트 사용: {self.endpoint_id}")
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
    
    def _init_vllm(self):
        """vLLM 초기화"""
        try:
            from vllm import LLM, SamplingParams
            
            # vLLM 초기화 옵션
            vllm_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "tensor_parallel_size": Config.VLLM_TENSOR_PARALLEL_SIZE,
            }
            
            if Config.VLLM_MAX_MODEL_LEN:
                vllm_kwargs["max_model_len"] = Config.VLLM_MAX_MODEL_LEN
            
            logger.info(f"vLLM 모델 로딩 중: {self.model_name}")
            self.llm = LLM(**vllm_kwargs)
            
            # 기본 SamplingParams
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=100,
            )
            
            # 토크나이저는 vLLM이 내부적으로 사용하므로 별도 로드 불필요
            # 하지만 프롬프트 생성용으로는 필요할 수 있음
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"토크나이저 로드 실패 (프롬프트 생성에 영향): {e}")
                self.tokenizer = None
            
            logger.info("✅ vLLM 모델 로딩 완료")
            
            # vLLM 모드에서도 batch_size 설정
            self.batch_size = Config.LLM_BATCH_SIZE
            
        except ImportError:
            raise ImportError(
                "vLLM이 설치되지 않았습니다. "
                "pip install vllm>=0.3.3 또는 USE_POD_VLLM=false로 설정하세요."
            )
    
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
                        if output:
                            # RunPod 엔드포인트에서 반환하는 형식에 따라 조정 필요
                            if isinstance(output, str):
                                return output
                            elif isinstance(output, dict):
                                # vLLM 엔드포인트의 경우 output.text 또는 output 형식일 수 있음
                                return output.get("text", str(output))
                            else:
                                return str(output)
                        else:
                            raise Exception("출력이 없습니다")
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

    async def _call_runpod_async(self, prompt: str, max_retries: int = Config.MAX_RETRIES) -> str:
        """
        RunPod 서버리스 엔드포인트를 httpx.AsyncClient로 비동기 호출.
        배치 경로에서 Config.SUMMARY_LLM_ASYNC=True(llm_async)일 때 사용.
        """
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
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
                        status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
                        status_response = await client.get(status_url, headers=headers)
                        if status_response.status_code != 200:
                            logger.warning(f"상태 확인 실패: {status_response.status_code}")
                            await asyncio.sleep(self.poll_interval)
                            continue

                        status_info = status_response.json()
                        status = status_info.get("status")

                        if status == "COMPLETED":
                            output = status_info.get("output")
                            if output:
                                if isinstance(output, str):
                                    return output.strip()
                                if isinstance(output, dict):
                                    return str(output.get("text", output)).strip()
                                return str(output).strip()
                            raise Exception("출력이 없습니다")
                        if status == "FAILED":
                            error = status_info.get("error", "알 수 없는 오류")
                            raise Exception(f"RunPod 작업 실패: {error}")
                        if time.time() - start_time > self.max_wait_time:
                            raise Exception(f"시간 초과: {self.max_wait_time}초 동안 완료되지 않았습니다")

                        await asyncio.sleep(self.poll_interval)
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
            # OpenAI API 사용 (빠른 검증용)
            try:
                # JSON 스키마 강제 (OpenAI API 지원)
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    response_format={"type": "json_object"},  # JSON 출력 강제
                )
                # 토큰 사용량 추출 및 저장
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
        """로컬 큐 비동기: RunPod만 httpx로 지원. vLLM/로컬은 NotImplementedError."""
        if self.use_runpod:
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
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            response = await self._call_runpod_async(prompt)
            return response.strip()
        elif self.use_pod_vllm:
            raise NotImplementedError("llm_async 모드에서 vLLM 직접 사용은 미구현입니다.")
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
        if self.use_runpod:
            # RunPod 서버리스 엔드포인트 사용
            # Qwen chat template 형식으로 변환 (로컬 토크나이저 없이 직접 구성)
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
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            # RunPod 호출
            response = self._call_runpod(prompt)
            return response.strip()
        elif self.use_pod_vllm:
            # vLLM 직접 사용 (비동기 처리 필요)
            # 동기 메서드에서는 vLLM을 직접 사용할 수 없으므로 에러 발생
            raise NotImplementedError("vLLM은 비동기 메서드를 사용해야 합니다.")
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
        )
        QUALITATIVE_RESTATEMENT = ("좋은 편", "긍정적인 편")
        PCT_RE = r"\d+(?:\.\d+)?\s*%"

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

        def is_valid_interp(s: str) -> bool:
            s = " ".join(s.strip().split())
            if not s:
                return False
            if any(p in s for p in FORBIDDEN_PHRASES):
                return False
            if any(q in s for q in QUALITATIVE_RESTATEMENT):
                return False
            pct_matches = re.findall(PCT_RE, s)
            if len(pct_matches) != 1:
                return False
            if "평균" not in s:
                return False
            return True

        abs_lift = abs(lift_pct)
        if abs_lift < 10:
            diff_hint = "문장은 '... 높지만, 차이는 크지 않은 편입니다.'처럼 차이가 크지 않음을 해석하세요."
        elif abs_lift < 30:
            diff_hint = "문장은 '... 높지만, 차이는 중간 정도입니다.'처럼 완만한 차이를 해석하세요."
        else:
            diff_hint = "문장은 '... 높아, 차이가 큰 편입니다.'처럼 차이가 큼을 해석하세요."

        system_content = (
            "당신은 음식점 비교 해석 문장을 만드는 도우미입니다. "
            "반드시 다음을 지킵니다: "
            "(1) 문장에 숫자(%)를 정확히 1회 포함하세요. "
            "(2) 좋은 편/긍정적인 편 같은 재진술 금지. "
            "(3) 하지만/다만/그래도로 차이 크기만 해석하세요. "
            "(4) 과장/강조 표현 금지. 예: '최고', '압도적', '완벽', '상당히', '매우', '굉장히'. "
            "(5) 리뷰수/표본수/신뢰도 언급 금지. "
            "(6) lift는 '만족도가 평균보다 높은 비율'. 가격 lift = 가격/가성비 만족. "
            "(7) 한 문장만 출력(줄바꿈 금지). "
            "반드시 JSON 형식으로만 답하세요: {\"interpretation\": \"한 문장\"}."
        )

        user_content = (
            f"카테고리: {category}. "
            f"lift 퍼센트: {round(lift_pct)}%. "
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
            data = json.loads(json_text)
            interp = data.get("interpretation")
            if isinstance(interp, str):
                interp = _enforce_one_sentence(interp)
                if is_valid_interp(interp):
                    return interp.strip()
            return None

        except Exception as e:
            if raw and isinstance(raw, str):
                candidate = _enforce_one_sentence(raw)
                if is_valid_interp(candidate):
                    return candidate.strip()
            logger.warning("비교 해석 LLM 실패: %s", e)
            return None
