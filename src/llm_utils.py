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
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .review_utils import estimate_reviews_tokens, estimate_tokens

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
    
    def _fix_truncated_json(self, text: str) -> str:
        """
        잘린 JSON 문자열을 복구합니다.
        
        Args:
            text: 잘린 JSON 문자열
            
        Returns:
            복구된 JSON 문자열
        """
        # 마지막 불완전한 문자열 필드를 닫기
        text = text.strip()
        
        # 마지막 따옴표가 닫히지 않은 경우
        if text.count('"') % 2 != 0:
            # 마지막 따옴표 뒤에 닫는 따옴표 추가
            last_quote_idx = text.rfind('"')
            if last_quote_idx != -1:
                # 마지막 따옴표 뒤에 닫는 따옴표와 중괄호 추가
                text = text[:last_quote_idx + 1] + '"'
        
        # 중괄호가 닫히지 않은 경우
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            # 닫는 중괄호 추가
            text += '}' * (open_braces - close_braces)
        
        return text
    
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
            max_new_tokens: 최대 생성 토큰 수 (기본값: 50, 요약/강점 추출 시 더 큰 값 필요)
            
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
    
    def classify_reviews(
        self,
        texts: List[str],
        max_retries: int = Config.MAX_RETRIES,
        batch_size: Optional[int] = None,
    ) -> List[Dict]:
        """
        LLM을 사용하여 텍스트들을 분류합니다. (배치 처리)
        
        Args:
            texts: 분류할 텍스트 리스트
            max_retries: 최대 재시도 횟수
            batch_size: 배치 크기 (None이면 자동으로 최적 크기 사용)
            
        Returns:
            분류 결과 리스트
        """
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = self.batch_size
        
        all_results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"총 {len(texts)}개 리뷰를 {total_batches}개 배치로 분류합니다 (배치 크기: {batch_size})")
        
        # 배치별로 처리
        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx:batch_idx + batch_size]
            current_batch_num = batch_idx // batch_size + 1
            
            logger.debug(f"배치 {current_batch_num}/{total_batches} 처리 중 ({len(batch)}개 리뷰)")
            
            # 각 배치에 대해 재시도 로직 적용
            batch_results = None
            for attempt in range(max_retries):
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "너는 긍정/부정 분류를 잘하는 AI 어시스턴트다.\n"
                                "다음 문장들에 대해 positive/negative를 판단하여라.\n"
                                "하나의 문장에 긍/부정이 섞였다면, 전체적인 톤에 따라 하나의 라벨을 선택하라.\n"
                                "반드시 **JSON 표준**(큰따옴표)으로 배열 형태 출력\n"
                                "예시: [{\"label\":\"positive\",\"text\":\"...\"}, {\"label\":\"negative\",\"text\":\"...\"}]\n"
                                "작은따옴표(') 사용 금지"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps({"reviews": batch}, ensure_ascii=False),
                        },
                    ]
                    
                    response_text = self._generate_response(messages, temperature=0.1)
                    logger.debug(f"배치 {current_batch_num} LLM 응답: {response_text[:200]}...")
                    
                    # JSON 파싱 시도
                    try:
                        llm_results = json.loads(response_text)
                        if isinstance(llm_results, list):
                            # 결과 개수 확인
                            if len(llm_results) == len(batch):
                                batch_results = llm_results
                                break  # 성공 시 재시도 루프 탈출
                            else:
                                logger.warning(
                                    f"배치 {current_batch_num}: 결과 개수 불일치 "
                                    f"(예상: {len(batch)}, 실제: {len(llm_results)}). 재시도합니다."
                                )
                        else:
                            logger.warning(f"배치 {current_batch_num}: LLM 응답이 리스트가 아닙니다. 재시도합니다.")
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"배치 {current_batch_num}: JSON 파싱 실패 "
                            f"(시도 {attempt + 1}/{max_retries}): {str(e)}"
                        )
                        logger.debug(f"원문: {response_text[:500]}")
                        
                        # 마지막 시도가 아니면 재시도
                        if attempt < max_retries - 1:
                            continue
                        else:
                            logger.error(f"배치 {current_batch_num}: 모든 재시도 실패. 빈 결과 반환.")
                            batch_results = []
                            break
                            
                except Exception as e:
                    logger.error(
                        f"배치 {current_batch_num}: LLM 호출 중 오류 발생 "
                        f"(시도 {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(f"배치 {current_batch_num}: 모든 재시도 실패.")
                        batch_results = []
                        break
            
            # 배치 결과 추가
            if batch_results is None:
                logger.warning(f"배치 {current_batch_num}: 결과를 얻지 못했습니다. 빈 결과 반환.")
                batch_results = []
            
            all_results.extend(batch_results)
            logger.debug(f"배치 {current_batch_num} 완료: {len(batch_results)}개 결과")
        
        logger.info(f"✅ 전체 분류 완료: {len(all_results)}개 결과 반환")
        return all_results
    
    async def _generate_with_vllm(
        self,
        prompts: List[str],
        temperature: float = 0.1,
        max_tokens: int = 100,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        vLLM을 사용하여 비동기로 응답 생성 + 메트릭 수집
        
        Args:
            prompts: 프롬프트 리스트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            
        Returns:
            (생성된 응답 리스트, 메트릭 딕셔너리)
        """
        if not self.use_pod_vllm:
            raise ValueError("vLLM이 초기화되지 않았습니다. USE_POD_VLLM=true로 설정하세요.")
        
        from vllm import SamplingParams
        
        # SamplingParams 설정
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 전체 시작 시간
        start_time = time.time()
        
        # 동기 메서드를 비동기로 실행
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            self.executor,
            self.llm.generate,
            prompts,
            sampling_params
        )
        
        total_time = time.time() - start_time
        
        # 메트릭 수집
        metrics = {
            "total_time_ms": total_time * 1000,
            "requests": []
        }
        
        responses = []
        total_prefill_time = 0
        total_decode_time = 0
        total_tokens = 0
        
        for output in outputs:
            text = output.outputs[0].text
            responses.append(text)
            
            # vLLM RequestOutput 메트릭 추출
            prefill_time_ms = 0
            decode_time_ms = 0
            n_tokens = 0
            
            # vLLM의 metrics 속성 확인 (버전에 따라 다를 수 있음)
            if hasattr(output, 'metrics') and output.metrics:
                first_token_time = getattr(output.metrics, 'first_token_time', None) or 0
                finished_time = getattr(output.metrics, 'finished_time', None) or 0
                
                if first_token_time > 0:
                    prefill_time_ms = first_token_time * 1000
                    if finished_time > first_token_time:
                        decode_time_ms = (finished_time - first_token_time) * 1000
                    else:
                        decode_time_ms = 0
                else:
                    # metrics가 없거나 first_token_time이 없는 경우 전체 시간을 decode로 간주
                    decode_time_ms = total_time * 1000 / len(outputs) if len(outputs) > 0 else 0
            
            # 토큰 수 계산
            if hasattr(output.outputs[0], 'token_ids') and output.outputs[0].token_ids:
                n_tokens = len(output.outputs[0].token_ids)
            else:
                # token_ids가 없는 경우 텍스트 길이로 추정 (대략적)
                n_tokens = len(text.split()) if text else 0
            
            total_prefill_time += prefill_time_ms
            total_decode_time += decode_time_ms
            total_tokens += n_tokens
            
            tpot_ms = decode_time_ms / n_tokens if n_tokens > 0 else 0
            
            metrics["requests"].append({
                "prefill_time_ms": prefill_time_ms,
                "decode_time_ms": decode_time_ms,
                "total_time_ms": prefill_time_ms + decode_time_ms,
                "n_tokens": n_tokens,
                "tpot_ms": tpot_ms,  # Time Per Output Token
            })
        
        # 평균 메트릭
        n_requests = len(outputs)
        if n_requests > 0:
            metrics["avg_prefill_time_ms"] = total_prefill_time / n_requests
            metrics["avg_decode_time_ms"] = total_decode_time / n_requests
            metrics["avg_tpot_ms"] = (total_decode_time / total_tokens) if total_tokens > 0 else 0
            metrics["total_tokens"] = total_tokens
            metrics["tps"] = total_tokens / total_time if total_time > 0 else 0  # Tokens Per Second
            metrics["ttft_ms"] = metrics["avg_prefill_time_ms"]  # Time To First Token
        
        return responses, metrics
    
    def _estimate_prefill_cost(self, prompt: str) -> int:
        """
        프롬프트의 Prefill 비용을 추정합니다.
        
        Prefill 비용은 입력 토큰 수에 비례하므로, 프롬프트의 토큰 수를 반환합니다.
        
        Args:
            prompt: 프롬프트 문자열
            
        Returns:
            추정된 Prefill 비용 (토큰 수)
        """
        return estimate_tokens(prompt)
    
    def _calculate_dynamic_batch_size(self, reviews: List[str], max_tokens_per_batch: Optional[int] = None) -> int:
        """
        리뷰 리스트를 기반으로 동적 배치 크기 계산
        
        Args:
            reviews: 리뷰 문자열 리스트
            max_tokens_per_batch: 배치당 최대 토큰 수 (None이면 Config 값 사용)
            
        Returns:
            계산된 배치 크기
        """
        return Config.calculate_dynamic_batch_size(reviews, max_tokens_per_batch)
    
    async def expand_query_for_dense_search(self, user_query: str) -> str:
        """
        LLM을 사용하여 쿼리를 Dense 검색에 적합한 키워드로 확장합니다.
        
        예시:
        - 입력: "데이트하기 좋은"
        - 출력: "분위기 좋다 로맨틱 조용한 데이트 분위기"
        
        Args:
            user_query: 사용자 검색 쿼리
            
        Returns:
            확장된 쿼리 문자열 (키워드 공백으로 구분)
        """
        prompt = f"""사용자 검색 의도: "{user_query}"

이 검색 의도를 리뷰 검색에 적합한 키워드로 확장하세요.
- 실제 리뷰에 나타날 수 있는 표현 사용
- 동의어, 유사어 포함
- 공백으로 구분된 키워드 문자열로 출력

예시:
- 입력: "데이트하기 좋은"
- 출력: "분위기 좋다 로맨틱 조용한 분위기 데이트"

- 입력: "가족 모임"
- 출력: "가족 단체 방문 넓은 자리 아이들 좋아함"

키워드만 출력하세요 (설명 없이):"""
        
        try:
            if self.use_pod_vllm:
                # vLLM 사용 (비동기)
                expanded_query, _ = await self._generate_with_vllm(
                    [prompt],
                    temperature=0.3,
                    max_tokens=50
                )
                return expanded_query[0].strip()
            else:
                # 동기 방식 (RunPod 또는 로컬 모델)
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                response = self._generate_response(
                    messages,
                    temperature=0.3,
                    max_new_tokens=50
                )
                return response.strip()
        except Exception as e:
            logger.warning(f"쿼리 확장 실패: {e}, 원본 쿼리 사용")
            return user_query
    
    async def expand_query_for_dense_search_with_metrics(
        self, 
        user_query: str
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLM을 사용하여 쿼리를 Dense 검색에 적합한 키워드로 확장 + 메트릭 반환
        
        Args:
            user_query: 사용자 검색 쿼리
            
        Returns:
            (확장된 쿼리 문자열, vLLM 메트릭 또는 None)
        """
        prompt = f"""사용자 검색 의도: "{user_query}"

이 검색 의도를 리뷰 검색에 적합한 키워드로 확장하세요.
- 실제 리뷰에 나타날 수 있는 표현 사용
- 동의어, 유사어 포함
- 공백으로 구분된 키워드 문자열로 출력

예시:
- 입력: "데이트하기 좋은"
- 출력: "분위기 좋다 로맨틱 조용한 분위기 데이트"

- 입력: "가족 모임"
- 출력: "가족 단체 방문 넓은 자리 아이들 좋아함"

키워드만 출력하세요 (설명 없이):"""
        
        try:
            if self.use_pod_vllm:
                # vLLM 사용 (비동기) - 메트릭 반환
                expanded_query, vllm_metrics = await self._generate_with_vllm(
                    [prompt],
                    temperature=0.3,
                    max_tokens=50
                )
                return expanded_query[0].strip(), vllm_metrics
            else:
                # 동기 방식 (RunPod 또는 로컬 모델) - 메트릭 없음
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                response = self._generate_response(
                    messages,
                    temperature=0.3,
                    max_new_tokens=50
                )
                return response.strip(), None
        except Exception as e:
            logger.warning(f"쿼리 확장 실패: {e}, 원본 쿼리 사용")
            return user_query, None
    
    def _extract_aspects_from_reviews(
        self,
        reviews: List[Dict[str, Any]],
        aspect_type: str = "positive",  # "positive" or "negative"
        max_tokens: int = 4000,
    ) -> List[Dict[str, Any]]:
        """
        리뷰에서 aspect 단위로 추출 (strength 추출의 Step B 로직 재사용)
        
        Args:
            reviews: 리뷰 딕셔너리 리스트
            aspect_type: "positive" (장점) 또는 "negative" (단점)
            max_tokens: 최대 토큰 수
            
        Returns:
            aspect 리스트 [{"aspect": "...", "claim": "...", "evidence_quotes": [...], "evidence_review_ids": [...]}]
        """
        if not reviews:
            return []
        
        # 1. 토큰 제한 고려해 샘플링
        sampled_reviews = []
        total_tokens = 0
        
        for review in reviews:
            text = review.get("content", "") or review.get("text", "") or review.get("review", "")
            if not text:
                continue
            
            tokens = estimate_tokens(text)
            if total_tokens + tokens > max_tokens:
                break
            
            sampled_reviews.append(review)
            total_tokens += tokens
        
        if not sampled_reviews:
            return []
        
        # 2. LLM 프롬프트 (strength 추출의 Step B 로직 재사용)
        reviews_text = "\n".join([
            f"[{r.get('review_id', r.get('id', ''))}] {r.get('content', r.get('text', r.get('review', '')))}"
            for r in sampled_reviews
        ])
        
        aspect_label = "장점" if aspect_type == "positive" else "단점"
        prompt = f"""다음 리뷰들을 읽고 이 레스토랑의 {aspect_label}을 aspect 단위로 추출하세요.

리뷰들:
{reviews_text}

각 {aspect_label}에 대해:
- aspect: {aspect_label}의 카테고리 (예: "불맛", "서비스", "가격")
- claim: 구체적 주장 (예: "숫불향과 화력이 좋아 불맛이 강함" 또는 "가격이 비싸다")
- evidence_quotes: 해당 {aspect_label}을 언급한 리뷰 인용문 (최대 3개)
- evidence_review_ids: 해당 리뷰 ID 리스트

JSON 형식:
{{
  "aspects": [
    {{
      "aspect": "불맛",
      "claim": "숫불향과 화력이 좋아 불맛이 강함",
      "evidence_quotes": ["숫불향이 진해서 맛있어요", "불맛이 강해서 고기 맛이 살아있어요"],
      "evidence_review_ids": ["rev_1", "rev_5"]
    }},
    ...
  ]
}}
"""
        
        try:
            response = self._generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_new_tokens=500,
            )
            
            # JSON 파싱
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:].strip()
            elif response.startswith("```"):
                response = response[3:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()
            
            # JSON 부분만 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            result = json.loads(response)
            aspects = result.get("aspects", [])
            
            logger.info(f"LLM으로 {len(aspects)}개 {aspect_label} aspect 추출")
            return aspects
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return []
        except Exception as e:
            logger.error(f"{aspect_label} aspect 추출 중 오류: {e}")
            return []
    
    def _create_final_summary_from_aspects(
        self,
        positive_aspects: List[Dict[str, Any]],
        negative_aspects: List[Dict[str, Any]],
    ) -> str:
        """
        positive_aspects와 negative_aspects를 기반으로 overall_summary 생성
        
        Args:
            positive_aspects: 긍정 aspect 리스트
            negative_aspects: 부정 aspect 리스트
            
        Returns:
            overall_summary 문자열
        """
        # aspect 정보를 텍스트로 변환
        positive_text = ""
        if positive_aspects:
            positive_items = []
            for aspect in positive_aspects:
                aspect_name = aspect.get("aspect", "")
                claim = aspect.get("claim", "")
                if aspect_name and claim:
                    positive_items.append(f"- {aspect_name}: {claim}")
            if positive_items:
                positive_text = "긍정 측면:\n" + "\n".join(positive_items)
        
        negative_text = ""
        if negative_aspects:
            negative_items = []
            for aspect in negative_aspects:
                aspect_name = aspect.get("aspect", "")
                claim = aspect.get("claim", "")
                if aspect_name and claim:
                    negative_items.append(f"- {aspect_name}: {claim}")
            if negative_items:
                negative_text = "부정 측면:\n" + "\n".join(negative_items)
        
        if not positive_text and not negative_text:
            return "요약할 내용이 없습니다."
        
        prompt = f"""다음은 음식점 리뷰에서 추출한 긍정/부정 측면입니다.

{positive_text}

{negative_text}

위 정보를 바탕으로 균형잡힌 전체 요약을 생성하세요. 긍정과 부정을 모두 고려하여 객관적이고 간결하게 요약하세요.

전체 요약:"""
        
        try:
            response = self._generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_new_tokens=200,
            )
            
            return response.strip()
        except Exception as e:
            logger.error(f"전체 요약 생성 중 오류: {e}")
            return "요약 생성 실패"
    
    def format_overall_summary_from_aspects(
        self,
        positive_aspects: List[Dict[str, Any]],
        negative_aspects: List[Dict[str, Any]],
        positive_count: int,
        negative_count: int,
    ) -> str:
        """
        검증된 aspect와 claim을 템플릿 기반으로 포맷팅하여 자연스러운 overall_summary 생성
        
        Args:
            positive_aspects: 검증된 긍정 aspect 리스트
            negative_aspects: 검증된 부정 aspect 리스트
            positive_count: 긍정 리뷰 개수
            negative_count: 부정 리뷰 개수
        
        Returns:
            포맷팅된 overall_summary
        """
        summary_parts = []
        
        # 1. 전체 평가 (긍정/부정 비율)
        total_count = positive_count + negative_count
        if total_count > 0:
            positive_ratio = int(round((positive_count / total_count) * 100))
            negative_ratio = int(round((negative_count / total_count) * 100))
            summary_parts.append(f"전체 리뷰 중 {positive_ratio}%가 긍정적, {negative_ratio}%가 부정적 평가를 했습니다.")
        
        # 2. 주요 장점
        if positive_aspects:
            # 상위 3개만 사용
            top_positive = positive_aspects[:3]
            positive_claims = []
            for a in top_positive:
                aspect = a.get('aspect', '')
                claim = a.get('claim', '')
                if aspect and claim:
                    # "맛에 대한 만족도가 높다는 언급이 많음" -> "맛에 대한 만족도가 높음"
                    # "언급이 많음" 제거하여 자연스럽게
                    claim_clean = claim.replace("는 언급이 많음", "").replace("은 언급이 많음", "").replace("라는 언급이 많음", "")
                    positive_claims.append(f"{aspect}에 대해 {claim_clean}")
            
            if positive_claims:
                if len(positive_claims) == 1:
                    summary_parts.append(f"주요 장점으로는 {positive_claims[0]}이 있습니다.")
                elif len(positive_claims) == 2:
                    summary_parts.append(f"주요 장점으로는 {positive_claims[0]}과 {positive_claims[1]}이 있습니다.")
                else:
                    summary_parts.append(f"주요 장점으로는 {', '.join(positive_claims[:-1])}, {positive_claims[-1]}이 있습니다.")
        
        # 3. 주요 단점
        if negative_aspects:
            # 상위 2개만 사용
            top_negative = negative_aspects[:2]
            negative_claims = []
            for a in top_negative:
                aspect = a.get('aspect', '')
                claim = a.get('claim', '')
                if aspect and claim:
                    # "언급이 많음" 제거하여 자연스럽게
                    claim_clean = claim.replace("는 언급이 많음", "").replace("은 언급이 많음", "").replace("라는 언급이 많음", "")
                    negative_claims.append(f"{aspect}에 대해 {claim_clean}")
            
            if negative_claims:
                if len(negative_claims) == 1:
                    summary_parts.append(f"주요 단점으로는 {negative_claims[0]}이 있습니다.")
                else:
                    summary_parts.append(f"주요 단점으로는 {', '.join(negative_claims[:-1])}, {negative_claims[-1]}이 있습니다.")
        
        # 4. 종합
        if not summary_parts:
            return "요약할 내용이 없습니다."
        
        return " ".join(summary_parts)
    
    def format_overall_summary_hybrid(
        self,
        positive_aspects: List[Dict[str, Any]],
        negative_aspects: List[Dict[str, Any]],
        positive_count: int,
        negative_count: int,
        use_llm_polish: bool = True,  # 기본값을 True로 변경
    ) -> str:
        """
        하이브리드 방식: 템플릿 기반 + 선택적 LLM 개선 (타임아웃 및 fallback 포함)
        
        Args:
            positive_aspects: 검증된 긍정 aspect 리스트
            negative_aspects: 검증된 부정 aspect 리스트
            positive_count: 긍정 리뷰 개수
            negative_count: 부정 리뷰 개수
            use_llm_polish: LLM으로 자연스럽게 개선할지 여부 (기본값: True)
        
        Returns:
            포맷팅된 overall_summary
        """
        # 1. 템플릿 기반 기본 생성 (항상 생성, fallback용)
        template_summary = self.format_overall_summary_from_aspects(
            positive_aspects, negative_aspects, positive_count, negative_count
        )
        
        if not use_llm_polish:
            return template_summary
        
        # 2. 선택적 LLM 개선 (타임아웃 및 fallback 포함)
        import time
        
        total_count = positive_count + negative_count
        positive_ratio = int(round((positive_count / total_count) * 100)) if total_count > 0 else 0
        negative_ratio = int(round((negative_count / total_count) * 100)) if total_count > 0 else 0
        
        # 검증된 aspect 정보만 전달
        positive_info = [
            f"- {a.get('aspect', '')}: {a.get('claim', '')}" 
            for a in positive_aspects[:3]
        ]
        negative_info = [
            f"- {a.get('aspect', '')}: {a.get('claim', '')}" 
            for a in negative_aspects[:2]
        ]
        
        prompt = f"""다음은 검증된 aspect 정보입니다. 이 정보만 사용하여 자연스러운 요약 문장을 작성하세요.

**긍정 aspect:**
{chr(10).join(positive_info) if positive_info else "- 없음"}

**부정 aspect:**
{chr(10).join(negative_info) if negative_info else "- 없음"}

**통계:**
- 긍정 리뷰: {positive_count}개 ({positive_ratio}%)
- 부정 리뷰: {negative_count}개 ({negative_ratio}%)

**작업 요청:**
위 정보만 사용하여 자연스러운 요약 문장을 작성하세요.
- 새로운 내용을 추가하지 마세요.
- 위에 나열된 aspect와 claim만 사용하세요.
- 문장을 자연스럽게 연결하세요.

요약 문장:"""
        
        # 타임아웃 및 레이턴시 측정
        start_time = time.time()
        
        try:
            polished = self._generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # 낮은 temperature로 일관성 유지
                max_new_tokens=150,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 타임아웃 체크: 실제로 fallback
            if elapsed_ms > Config.LLM_POLISH_TIMEOUT_SECONDS * 1000:
                logger.warning(
                    f"LLM 개선 타임아웃 ({elapsed_ms:.1f}ms > {Config.LLM_POLISH_TIMEOUT_SECONDS * 1000}ms), "
                    f"템플릿 결과 사용"
                )
                return template_summary
            
            # 레이턴시는 로깅만 (성능 모니터링용)
            if elapsed_ms > 500:  # 하드코딩된 경고 임계값
                logger.debug(f"LLM 개선 레이턴시: {elapsed_ms:.1f}ms (경고 임계값 초과)")
            
            polished_text = polished.strip() if polished else ""
            if not polished_text:
                logger.warning("LLM 개선 결과가 비어있음, 템플릿 결과 사용")
                return template_summary
            
            logger.debug(f"LLM 개선 완료 (레이턴시: {elapsed_ms:.1f}ms)")
            return polished_text
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"LLM 개선 실패 (레이턴시: {elapsed_ms:.1f}ms, 오류: {e}), 템플릿 결과 사용"
            )
            return template_summary
    
    def _calculate_dynamic_similarity_threshold(self, review_count: int) -> float:
        """
        리뷰 수에 따라 동적 유사도 임계값 계산
        
        Args:
            review_count: 전체 리뷰 수
            
        Returns:
            동적 유사도 임계값
        """
        if review_count < 5:
            return 0.4  # 리뷰가 적으면 임계값 낮춤
        elif review_count < 10:
            return 0.45  # 중간 수준
        else:
            return 0.5  # 리뷰가 많으면 기본값
    
    def validate_aspects_by_cosine_similarity(
        self,
        aspects: List[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        vector_search: Any,  # VectorSearch 타입 (순환 참조 방지)
        similarity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        대표 벡터 TOP-K 리뷰와 aspect claim 간 cosine 유사도로 필터링
        
        Args:
            aspects: LLM이 생성한 aspect 리스트
            reviews: 대표 벡터 TOP-K 리뷰 (payload 포함)
            vector_search: VectorSearch 인스턴스 (encoder 접근용)
            similarity_threshold: 최소 cosine 유사도 (0.0 ~ 1.0)
        
        Returns:
            검증된 aspect 리스트
        """
        if not aspects or not reviews:
            return []
        
        validated_aspects = []
        
        # 1. 리뷰 텍스트 추출 및 임베딩 (배치 처리)
        review_texts = []
        for review in reviews:
            text = review.get("content", "") or review.get("text", "") or review.get("review", "")
            if text:
                review_texts.append(text)
        
        if not review_texts:
            return []
        
        # 배치 임베딩 (GPU 최적화)
        try:
            # normalize 파라미터 지원 여부 확인 (일부 모델은 지원하지 않음)
            encode_kwargs = {
                "batch_size": vector_search.batch_size,
                "convert_to_numpy": True,
            }
            try:
                # get_model_kwargs()가 있는 경우 확인
                if hasattr(vector_search.encoder, "get_model_kwargs"):
                    model_kwargs = vector_search.encoder.get_model_kwargs()
                    if model_kwargs and "normalize" in model_kwargs:
                        encode_kwargs["normalize"] = True  # cosine 유사도 계산을 위해 정규화
            except Exception:
                # get_model_kwargs() 호출 실패 시 normalize 없이 시도
                pass
            
            review_embeddings = vector_search.encoder.encode(
                review_texts,
                **encode_kwargs
            )
        except TypeError as e:
            # normalize 파라미터로 인한 TypeError면 normalize 없이 재시도
            if "normalize" in str(e):
                try:
                    review_embeddings = vector_search.encoder.encode(
                        review_texts,
                        batch_size=vector_search.batch_size,
                        convert_to_numpy=True,
                    )
                except Exception as e2:
                    logger.error(f"리뷰 임베딩 생성 중 오류: {e2}")
                    return aspects  # 검증 실패 시 원본 반환
            else:
                logger.error(f"리뷰 임베딩 생성 중 오류: {e}")
                return aspects  # 검증 실패 시 원본 반환
        except Exception as e:
            logger.error(f"리뷰 임베딩 생성 중 오류: {e}")
            return aspects  # 검증 실패 시 원본 반환
        
        # 2. 각 aspect의 claim 임베딩 및 유사도 계산
        for aspect in aspects:
            claim = aspect.get("claim", "")
            if not claim:
                continue
            
            # claim 임베딩
            try:
                # normalize 파라미터 지원 여부 확인 (일부 모델은 지원하지 않음)
                encode_kwargs = {"convert_to_numpy": True}
                try:
                    # get_model_kwargs()가 있는 경우 확인
                    if hasattr(vector_search.encoder, "get_model_kwargs"):
                        model_kwargs = vector_search.encoder.get_model_kwargs()
                        if model_kwargs and "normalize" in model_kwargs:
                            encode_kwargs["normalize"] = True
                    else:
                        # get_model_kwargs()가 없으면 normalize 없이 시도
                        pass
                except Exception:
                    # get_model_kwargs() 호출 실패 시 normalize 없이 시도
                    pass
                
                claim_embedding = vector_search.encoder.encode(
                    [claim],
                    **encode_kwargs
                )[0]
            except TypeError as e:
                # normalize 파라미터로 인한 TypeError면 normalize 없이 재시도
                if "normalize" in str(e):
                    try:
                        claim_embedding = vector_search.encoder.encode(
                            [claim],
                            convert_to_numpy=True,
                        )[0]
                    except Exception as e2:
                        logger.warning(f"Claim 임베딩 생성 실패 (aspect: {aspect.get('aspect', '')}): {e2}")
                        continue
                else:
                    logger.warning(f"Claim 임베딩 생성 실패 (aspect: {aspect.get('aspect', '')}): {e}")
                    continue
            except Exception as e:
                logger.warning(f"Claim 임베딩 생성 실패 (aspect: {aspect.get('aspect', '')}): {e}")
                continue
            
            # 3. 각 리뷰와의 cosine 유사도 계산
            similarities = np.dot(review_embeddings, claim_embedding)
            
            # 4. 최고 유사도 확인 (threshold 이상이면 통과)
            max_similarity = float(np.max(similarities))
            matching_count = int(np.sum(similarities >= similarity_threshold))
            
            if max_similarity >= similarity_threshold:
                # 검증 통과: aspect에 매칭된 리뷰 정보 추가
                aspect["validation_score"] = max_similarity  # 최고 유사도
                aspect["matching_review_count"] = matching_count
                validated_aspects.append(aspect)
                logger.debug(
                    f"Aspect '{aspect.get('aspect', '')}' 검증 통과: "
                    f"최고 유사도 {max_similarity:.3f} (매칭 리뷰: {matching_count}개)"
                )
            else:
                logger.debug(
                    f"Aspect '{aspect.get('aspect', '')}' 검증 실패: "
                    f"최고 유사도 {max_similarity:.3f} (임계값: {similarity_threshold:.2f})"
                )
        
        return validated_aspects
    
    def summarize_reviews(
        self,
        positive_reviews: List[Dict[str, Any]],
        negative_reviews: List[Dict[str, Any]],
        vector_search: Optional[Any] = None,  # VectorSearch 타입 (순환 참조 방지)
        validate_aspects: bool = True,  # aspect 검증 여부
        min_positive_aspects: int = 2,  # 최소 긍정 aspect 개수
        min_negative_aspects: int = 1,  # 최소 부정 aspect 개수
        max_retries: int = 1,  # 최대 재시도 횟수 (기본값: 1, 총 2회 시도)
        use_llm_polish: bool = True,  # LLM으로 overall_summary 개선 여부 (기본값: True)
    ) -> Dict:
        """
        대표 벡터 TOP-K 리뷰를 직접 요약합니다 (C 방식 재시도 로직 포함).
        
        C 방식 재시도 로직:
        - 검증 통과한 aspect: 유지 (A)
        - 검증 실패한 aspect: claim 수정 후 재검증 (B)
        - 추가 생성 없음 (검증하지 않으면 할루시네이션 위험)
        
        프로세스:
        1. 모든 리뷰(positive + negative)를 합침
        2. 리뷰 텍스트 추출
        3. LLM에 직접 전달하여 요약 생성
        4. 검증 수행 (검증 통과/실패 분류)
        5. 최소 개수 미만이면 재시도 (검증 통과는 유지, 검증 실패는 수정)
        
        Args:
            positive_reviews: 긍정 리뷰 딕셔너리 리스트 (payload 포함)
            negative_reviews: 부정 리뷰 딕셔너리 리스트 (payload 포함)
            vector_search: VectorSearch 인스턴스 (aspect 검증용)
            validate_aspects: aspect 검증 여부
            min_positive_aspects: 최소 긍정 aspect 개수
            min_negative_aspects: 최소 부정 aspect 개수
            max_retries: 최대 재시도 횟수
            
        Returns:
            요약 결과 딕셔너리 (overall_summary, positive_aspects, negative_aspects 포함)
        """
        try:
            # 모든 리뷰 합치기
            all_reviews = positive_reviews + negative_reviews
            
            # 빈 리뷰 제거 및 텍스트 추출
            review_texts = []
            for r in all_reviews:
                text = r.get("content", "") or r.get("text", "") or r.get("review", "")
                if text:
                    review_texts.append(text)
            
            if not review_texts:
                logger.warning("요약할 리뷰가 없습니다.")
                return {
                    "overall_summary": "요약할 리뷰가 없습니다.",
                    "positive_aspects": [],
                    "negative_aspects": [],
                    "positive_reviews": positive_reviews,
                    "negative_reviews": negative_reviews,
                    "positive_count": len(positive_reviews),
                    "negative_count": len(negative_reviews),
                    "tokens_used": None,
                }
            
            # 동적 검증 기준 계산
            total_review_count = len(review_texts)
            dynamic_similarity_threshold = self._calculate_dynamic_similarity_threshold(total_review_count)
            
            # 부정 리뷰가 없으면 부정 aspect 최소 개수를 0으로 조정
            adjusted_min_negative_aspects = 0 if len(negative_reviews) == 0 else min_negative_aspects
            
            logger.info(
                f"동적 검증 기준 적용: 리뷰 수={total_review_count}, "
                f"유사도 임계값={dynamic_similarity_threshold:.2f}, "
                f"부정 aspect 최소 개수={adjusted_min_negative_aspects}"
            )
            
            # 토큰 사용량 초기화 (재시도 루프 전)
            total_tokens_used = 0
            self.last_tokens_used = None
            
            # 재시도 루프
            validated_positive_aspects = []
            validated_negative_aspects = []
            failed_positive_aspects = []
            failed_negative_aspects = []
            
            for attempt in range(max_retries + 1):
                # 1. LLM 호출 (aspect만 추출, overall_summary는 나중에 포맷팅)
                if attempt == 0:
                    # 첫 시도: 일반 프롬프트
                    prompt = f"""다음은 음식점에 대한 고객 리뷰들입니다. 이 리뷰들을 읽고 aspect 정보를 추출해주세요.

리뷰들:
{chr(10).join([f"- {text}" for text in review_texts])}

다음 JSON 형식으로 응답해주세요:
{{
  "positive_aspects": [
    {{
      "aspect": "맛",
      "claim": "구체적 주장 (예: 불맛이 강하고 고기 질이 좋다)",
      "evidence_quotes": ["인용문1", "인용문2"],
      "evidence_review_ids": []
    }}
  ],
  "negative_aspects": [
    {{
      "aspect": "가격",
      "claim": "구체적 주장 (예: 가격이 비싸다)",
      "evidence_quotes": ["인용문1"],
      "evidence_review_ids": []
    }}
  ]
}}

aspect 추출 시 다음을 고려하세요:
1. 주요 장점 (맛, 서비스, 분위기, 가격 등)
2. 주요 단점 (있는 경우)
3. 이 레스토랑의 특징적인 부분

간결하고 객관적으로 작성해주세요."""
                else:
                    # 재시도: C 방식 (검증 통과는 유지, 검증 실패는 수정)
                    prompt = f"""다음은 음식점에 대한 고객 리뷰들입니다.

리뷰들:
{chr(10).join([f"- {text}" for text in review_texts])}

**검증을 통과한 aspect들 (이것들은 반드시 그대로 유지하세요):**
긍정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')}" for a in validated_positive_aspects]) if validated_positive_aspects else "- 없음"}

부정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')}" for a in validated_negative_aspects]) if validated_negative_aspects else "- 없음"}

**검증에 실패한 aspect들 (이것들의 claim을 더 일반적으로 수정하세요):**
긍정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')} (실패: 리뷰에서 언급이 적음)" for a in failed_positive_aspects]) if failed_positive_aspects else "- 없음"}

부정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')} (실패: 리뷰에서 언급이 적음)" for a in failed_negative_aspects]) if failed_negative_aspects else "- 없음"}

**작업 요청:**
1. **검증을 통과한 aspect들은 그대로 유지**하세요.
2. **검증에 실패한 aspect들의 claim을 더 일반적으로 수정**하세요.
   - 예: "맛있다" → "맛에 대한 만족도가 높다는 언급이 많음"
   - 예: "서비스 좋다" → "서비스에 대한 만족도가 높다는 언급이 많음"
   - 예: "가격 비싸다" → "가격이 비싸다는 언급이 많음"
3. **새로운 aspect를 추가로 생성하지 마세요.** (검증을 통과한 aspect와 수정된 aspect만 포함하세요)

다음 JSON 형식으로 응답해주세요:
{{
  "positive_aspects": [
    {{
      "aspect": "맛",
      "claim": "수정된 claim (검증 실패한 경우) 또는 원래 claim (검증 통과한 경우)",
      "evidence_quotes": [],
      "evidence_review_ids": []
    }}
  ],
  "negative_aspects": [...]
}}
"""

                response = self._generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_new_tokens=1000,  # aspect 정보 포함으로 토큰 수 증가
                )
                
                # 토큰 사용량 누적
                if self.last_tokens_used is not None:
                    total_tokens_used += self.last_tokens_used
                
                # JSON 파싱
                response_text = response.strip()
                # 마크다운 코드 블록 제거
                if response_text.startswith("```json"):
                    response_text = response_text[7:].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:].strip()
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()
                
                # JSON 부분만 추출
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                try:
                    result = json.loads(response_text)
                    overall_summary = result.get("overall_summary", "")
                    positive_aspects = result.get("positive_aspects", [])
                    negative_aspects = result.get("negative_aspects", [])
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {e}, 원문: {response_text[:200]}")
                    # 파싱 실패 시 요약만 추출 시도
                    overall_summary = response_text
                    positive_aspects = []
                    negative_aspects = []
                
                # 2. 검증 수행
                if validate_aspects and vector_search:
                    # 검증 전 aspect 분류
                    positive_aspects_before = positive_aspects.copy()
                    negative_aspects_before = negative_aspects.copy()
                    
                    # 검증 수행 (동적 기준 사용)
                    validated_positive = self.validate_aspects_by_cosine_similarity(
                        aspects=positive_aspects,
                        reviews=all_reviews,  # 대표 벡터 TOP-K 리뷰
                        vector_search=vector_search,
                        similarity_threshold=dynamic_similarity_threshold,  # 동적 임계값
                    )
                    validated_negative = self.validate_aspects_by_cosine_similarity(
                        aspects=negative_aspects,
                        reviews=all_reviews,
                        vector_search=vector_search,
                        similarity_threshold=dynamic_similarity_threshold,  # 동적 임계값
                    )
                    
                    # 검증 통과/실패 분류
                    validated_positive_ids = {(a.get('aspect', ''), a.get('claim', '')) for a in validated_positive}
                    validated_negative_ids = {(a.get('aspect', ''), a.get('claim', '')) for a in validated_negative}
                    
                    failed_positive_aspects = [
                        a for a in positive_aspects_before 
                        if (a.get('aspect', ''), a.get('claim', '')) not in validated_positive_ids
                    ]
                    failed_negative_aspects = [
                        a for a in negative_aspects_before 
                        if (a.get('aspect', ''), a.get('claim', '')) not in validated_negative_ids
                    ]
                    
                    # 검증 통과한 aspect 누적 (중복 제거)
                    for a in validated_positive:
                        key = (a.get('aspect', ''), a.get('claim', ''))
                        if not any((va.get('aspect', ''), va.get('claim', '')) == key for va in validated_positive_aspects):
                            validated_positive_aspects.append(a)
                    
                    for a in validated_negative:
                        key = (a.get('aspect', ''), a.get('claim', ''))
                        if not any((va.get('aspect', ''), va.get('claim', '')) == key for va in validated_negative_aspects):
                            validated_negative_aspects.append(a)
                    
                    # 최소 개수 확인 (조정된 부정 aspect 최소 개수 사용)
                    if (len(validated_positive_aspects) >= min_positive_aspects and 
                        len(validated_negative_aspects) >= adjusted_min_negative_aspects):
                        # 충분한 aspect 확보 → 루프 탈출
                        logger.info(
                            f"Aspect 검증 완료: 긍정 {len(validated_positive_aspects)}개, 부정 {len(validated_negative_aspects)}개 통과"
                        )
                        break
                    elif attempt < max_retries:
                        # 부족하면 재시도 (C 방식: 검증 통과는 유지, 검증 실패는 수정)
                        logger.info(
                            f"Aspect 개수 부족 (긍정: {len(validated_positive_aspects)}/{min_positive_aspects}, "
                            f"부정: {len(validated_negative_aspects)}/{adjusted_min_negative_aspects}). "
                            f"재시도합니다 (검증 통과: 유지, 검증 실패: 수정)."
                        )
                        continue
                    else:
                        # 최대 재시도 횟수 도달
                        logger.warning(
                            f"최대 재시도 횟수 도달. 현재까지 확보한 aspect 사용 "
                            f"(긍정: {len(validated_positive_aspects)}, 부정: {len(validated_negative_aspects)})"
                        )
                        break
                else:
                    # 검증 비활성화 → 첫 시도 결과 사용
                    validated_positive_aspects = positive_aspects
                    validated_negative_aspects = negative_aspects
                    break
            
            # 검증 완료 후 하이브리드 방식으로 overall_summary 생성
            # format_overall_summary_hybrid에서도 토큰 사용량이 업데이트될 수 있으므로 저장
            polish_tokens = 0
            if use_llm_polish:
                polish_tokens_before = total_tokens_used
            
            overall_summary = self.format_overall_summary_hybrid(
                positive_aspects=validated_positive_aspects,
                negative_aspects=validated_negative_aspects,
                positive_count=len(positive_reviews),
                negative_count=len(negative_reviews),
                use_llm_polish=use_llm_polish,
            )
            
            # format_overall_summary_hybrid 호출 후 토큰 사용량 누적
            if use_llm_polish and self.last_tokens_used is not None:
                polish_tokens = self.last_tokens_used
                total_tokens_used += polish_tokens
            
            # 최종 토큰 사용량 (0이면 None으로 반환하여 메트릭에서 구분)
            final_tokens_used = total_tokens_used if total_tokens_used > 0 else None
            
            # 메타데이터 추가
            return {
                "overall_summary": overall_summary,
                "positive_aspects": validated_positive_aspects,
                "negative_aspects": validated_negative_aspects,
                "positive_reviews": positive_reviews,
                "negative_reviews": negative_reviews,
                "positive_count": len(positive_reviews),
                "negative_count": len(negative_reviews),
                "tokens_used": final_tokens_used,  # 토큰 사용량 추가
            }
        except Exception as e:
            logger.error(f"리뷰 요약 중 오류: {str(e)}")
            return {
                "overall_summary": "요약 실패",
                "positive_aspects": [],
                "negative_aspects": [],
                "positive_reviews": positive_reviews,
                "negative_reviews": negative_reviews,
                "positive_count": len(positive_reviews),
                "negative_count": len(negative_reviews),
                "tokens_used": None,
            }
    
    # ==================== Phase 2: 독립적 강점 추출 ====================
    
    def extract_absolute_strengths(
        self,
        reviews: List[Dict[str, Any]],
        restaurant_id: str = "unknown",
    ) -> List[str]:
        """
        레스토랑의 절대적 강점을 추출 (비교 없이)
        
        Args:
            reviews: 레스토랑의 긍정 리뷰 딕셔너리 리스트
            restaurant_id: 레스토랑 ID (로깅용)
            
        Returns:
            강점 리스트 ["맛이 좋다", "서비스가 친절하다", ...]
        """
        if not reviews:
            return []
        
        # 리뷰 텍스트 추출
        review_texts = [r.get("content", "") or r.get("review", "") 
                        if isinstance(r, dict) else r for r in reviews]
        review_texts = [t for t in review_texts if t]
        
        if not review_texts:
            return []
        
        messages = [{
            "role": "system",
            "content": (
                "음식점 리뷰 강점 추출 AI. **한국어로만 출력.**\n"
                "주어진 리뷰들을 읽고 이 레스토랑의 강점을 추출하세요.\n"
                "비교 없이 이 레스토랑 자체의 장점만 나열하세요.\n"
                "각 강점을 간결한 문장으로 나열하세요 (예: \"맛이 좋다\", \"서비스가 친절하다\").\n"
                "JSON: {\"strengths\": [\"강점1\", \"강점2\", ...]}"
            )
        }, {
            "role": "user",
            "content": json.dumps({
                "reviews": review_texts
            }, ensure_ascii=False)
        }]
        
        try:
            response_text = self._generate_response(
                messages, 
                temperature=0.1, 
                max_new_tokens=200
            )
            
            # JSON 파싱
            response_json = json.loads(response_text)
            strengths = response_json.get("strengths", [])
            
            # 리스트가 아니면 문자열을 파싱
            if isinstance(strengths, str):
                strengths = [s.strip() for s in strengths.split(",")]
            
            if isinstance(strengths, list) and strengths:
                return strengths
            
            # 폴백: 간단한 파싱
            return self._parse_strengths_fallback(response_text) if response_text else []
            
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 폴백
            logger.warning(f"레스토랑 {restaurant_id} 강점 추출 JSON 파싱 실패, 폴백 사용")
            return self._parse_strengths_fallback(response_text) if response_text else []
        except Exception as e:
            logger.error(f"레스토랑 {restaurant_id} 강점 추출 실패: {e}")
            return []
    
    def _parse_strengths_fallback(self, text: str) -> List[str]:
        """강점 텍스트를 리스트로 파싱 (폴백)"""
        import re
        # 문장 단위로 분리
        sentences = re.split(r'[.!?]\s+', text)
        strengths = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        return strengths[:10]  # 최대 10개
    
    def _create_summary_prompt(self, positive_texts: List[str], negative_texts: List[str]) -> str:
        """요약 프롬프트 생성 (전체 요약만)"""
        messages = [
                {
                    "role": "system",
                    "content": (
                    "너는 음식점 리뷰 요약 전문가다.\n"
                    "주어진 긍정 리뷰와 부정 리뷰를 모두 읽고, "
                    "전체 리뷰를 종합하여 균형잡힌 요약을 생성하라.\n"
                    "긍정과 부정을 모두 고려하여 객관적이고 간결하게 요약하라.\n"
                    "반드시 JSON 형식으로 출력하라: "
                    "{\"overall_summary\": \"...\"}\n"
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                        "positive_reviews": positive_texts,
                        "negative_reviews": negative_texts
                        },
                        ensure_ascii=False
                    )
            },
        ]
        return self._format_messages_for_vllm(messages)
    
    def _parse_summary_response(self, response_text: str) -> Dict[str, str]:
        """요약 응답 파싱 (전체 요약만)"""
        try:
            response_text = response_text.strip()
            
            # 마크다운 코드 블록 제거
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            response_text = response_text.strip()
            
            # JSON 부분만 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            summary = json.loads(response_text)
            
            # 필수 키 확인
            required_keys = ["overall_summary"]
            for key in required_keys:
                if key not in summary:
                    summary[key] = ""
            
            return summary
        except Exception as e:
            logger.warning(f"요약 응답 파싱 실패: {e}, 원문: {response_text[:200]}")
            return {
                "overall_summary": ""
            }
    
    async def summarize_multiple_restaurants_vllm(
        self,
        restaurants_data: List[Dict[str, Any]],  # [{"restaurant_id": 1, "positive_reviews": [...], "negative_reviews": [...]}, ...]
        max_tokens_per_batch: Optional[int] = None,
        max_retries: int = Config.MAX_RETRIES,
        vector_search: Optional[Any] = None,  # VectorSearch 타입 (순환 참조 방지)
        validate_aspects: bool = True,  # aspect 검증 여부
        use_llm_polish: bool = True,  # LLM으로 overall_summary 개선 여부 (기본값: True)
    ) -> List[Dict[str, Any]]:
        """
        여러 레스토랑의 리뷰를 "레스토랑 단위"로 비동기 처리하여 요약합니다.

        핵심:
        - 한 레스토랑 내부에서 긍정/부정 리뷰를 다시 배치로 쪼개지 않습니다.
        - 음식점 간에만 비동기 큐(세마포어)로 동시 처리 수를 제한합니다.
        - 대표 벡터 TOP-K 리뷰를 직접 LLM에 전달하여 요약 생성 (aspect 추출 없음)
        
        OOM 방지 전략:
        - 세마포어를 통한 동시 처리 수 제한 (VLLM_MAX_CONCURRENT_BATCHES)
        - 레스토랑 단위로만 동시성을 부여 (레스토랑 내부 배치 분할 X)
        
        Args:
            restaurants_data: 레스토랑 데이터 리스트
                - restaurant_id: 레스토랑 ID
                - positive_reviews: 긍정 리뷰 리스트 (payload 포함)
                - negative_reviews: 부정 리뷰 리스트 (payload 포함)
            max_tokens_per_batch: (호환성 유지용, 현재 사용 안함)
            max_retries: 최대 재시도 횟수
            
        Returns:
            각 레스토랑별 요약 결과 리스트
        """
        if not restaurants_data:
            return []

        logger.info(f"총 {len(restaurants_data)}개 레스토랑을 레스토랑 단위로 비동기 처리하여 요약합니다.")

        from asyncio import Semaphore

        max_concurrent = Config.VLLM_MAX_CONCURRENT_BATCHES
        semaphore = Semaphore(max_concurrent)

        async def process_single_restaurant(restaurant_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                restaurant_id = restaurant_data.get("restaurant_id")
                positive_reviews = restaurant_data.get("positive_reviews", []) or []
                negative_reviews = restaurant_data.get("negative_reviews", []) or []

                # 빈 리뷰 제거 및 텍스트 추출
                all_reviews = positive_reviews + negative_reviews
                review_texts = []
                for r in all_reviews:
                    if isinstance(r, dict):
                        text = r.get("content", "") or r.get("text", "") or r.get("review", "")
                        if text:
                            review_texts.append(text)

                if not review_texts:
                    return {
                        "restaurant_id": restaurant_id,
                        "overall_summary": "요약할 리뷰가 없습니다.",
                        "positive_aspects": [],
                        "negative_aspects": [],
                        "positive_reviews": positive_reviews,
                        "negative_reviews": negative_reviews,
                        "positive_count": len(positive_reviews),
                        "negative_count": len(negative_reviews),
                    }

                # LLM에 직접 aspect 추출 요청 (overall_summary는 나중에 포맷팅)
                prompt = f"""다음은 음식점에 대한 고객 리뷰들입니다. 이 리뷰들을 읽고 aspect 정보를 추출해주세요.

리뷰들:
{chr(10).join([f"- {text}" for text in review_texts])}

다음 JSON 형식으로 응답해주세요:
{{
  "positive_aspects": [
    {{
      "aspect": "맛",
      "claim": "구체적 주장 (예: 불맛이 강하고 고기 질이 좋다)",
      "evidence_quotes": ["인용문1", "인용문2"],
      "evidence_review_ids": []
    }}
  ],
  "negative_aspects": [
    {{
      "aspect": "가격",
      "claim": "구체적 주장 (예: 가격이 비싸다)",
      "evidence_quotes": ["인용문1"],
      "evidence_review_ids": []
    }}
  ]
}}

aspect 추출 시 다음을 고려하세요:
1. 주요 장점 (맛, 서비스, 분위기, 가격 등)
2. 주요 단점 (있는 경우)
3. 이 레스토랑의 특징적인 부분

간결하고 객관적으로 작성해주세요."""

                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    self.executor,
                    self._generate_response,
                    [{"role": "user", "content": prompt}],
                    0.3,  # temperature
                    1000,  # max_new_tokens (aspect 정보 포함)
                )
                
                # JSON 파싱
                response_text = response_text.strip()
                # 마크다운 코드 블록 제거
                if response_text.startswith("```json"):
                    response_text = response_text[7:].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:].strip()
                if response_text.endswith("```"):
                    response_text = response_text[:-3].strip()
                
                # JSON 부분만 추출
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                try:
                    result = json.loads(response_text)
                    overall_summary = result.get("overall_summary", "")
                    positive_aspects = result.get("positive_aspects", [])
                    negative_aspects = result.get("negative_aspects", [])
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {e}, 원문: {response_text[:200]}")
                    # 파싱 실패 시 요약만 추출 시도
                    overall_summary = response_text
                    positive_aspects = []
                    negative_aspects = []
                
                # Aspect 검증 및 재시도 로직 (C 방식)
                if validate_aspects and vector_search:
                    # 동적 검증 기준 계산
                    total_review_count = len(review_texts)
                    dynamic_similarity_threshold = self._calculate_dynamic_similarity_threshold(total_review_count)
                    
                    # 부정 리뷰가 없으면 부정 aspect 최소 개수를 0으로 조정
                    adjusted_min_negative_aspects = 0 if len(negative_reviews) == 0 else 1
                    
                    logger.debug(
                        f"레스토랑 {restaurant_id} 동적 검증 기준 적용: 리뷰 수={total_review_count}, "
                        f"유사도 임계값={dynamic_similarity_threshold:.2f}, "
                        f"부정 aspect 최소 개수={adjusted_min_negative_aspects}"
                    )
                    
                    # 재시도 루프
                    validated_positive_aspects = []
                    validated_negative_aspects = []
                    failed_positive_aspects = []
                    failed_negative_aspects = []
                    aspect_max_retries = 1  # aspect 재시도 횟수 (기본값: 1, 총 2회 시도)
                    min_positive_aspects = 2
                    min_negative_aspects = adjusted_min_negative_aspects
                    
                    for aspect_attempt in range(aspect_max_retries + 1):
                        if aspect_attempt == 0:
                            # 첫 시도: 이미 생성된 aspect 사용
                            current_positive = positive_aspects
                            current_negative = negative_aspects
                        else:
                            # 재시도: C 방식 프롬프트
                            retry_prompt = f"""다음은 음식점에 대한 고객 리뷰들입니다.

리뷰들:
{chr(10).join([f"- {text}" for text in review_texts])}

**검증을 통과한 aspect들 (이것들은 반드시 그대로 유지하세요):**
긍정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')}" for a in validated_positive_aspects]) if validated_positive_aspects else "- 없음"}

부정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')}" for a in validated_negative_aspects]) if validated_negative_aspects else "- 없음"}

**검증에 실패한 aspect들 (이것들의 claim을 더 일반적으로 수정하세요):**
긍정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')} (실패: 리뷰에서 언급이 적음)" for a in failed_positive_aspects]) if failed_positive_aspects else "- 없음"}

부정:
{chr(10).join([f"- {a.get('aspect', '')}: {a.get('claim', '')} (실패: 리뷰에서 언급이 적음)" for a in failed_negative_aspects]) if failed_negative_aspects else "- 없음"}

**작업 요청:**
1. **검증을 통과한 aspect들은 그대로 유지**하세요.
2. **검증에 실패한 aspect들의 claim을 더 일반적으로 수정**하세요.
   - 예: "맛있다" → "맛에 대한 만족도가 높다는 언급이 많음"
   - 예: "서비스 좋다" → "서비스에 대한 만족도가 높다는 언급이 많음"
   - 예: "가격 비싸다" → "가격이 비싸다는 언급이 많음"
3. **새로운 aspect를 추가로 생성하지 마세요.** (검증을 통과한 aspect와 수정된 aspect만 포함하세요)

다음 JSON 형식으로 응답해주세요:
{{
  "positive_aspects": [
    {{
      "aspect": "맛",
      "claim": "수정된 claim (검증 실패한 경우) 또는 원래 claim (검증 통과한 경우)",
      "evidence_quotes": [],
      "evidence_review_ids": []
    }}
  ],
  "negative_aspects": [...]
}}
"""
                            
                            loop = asyncio.get_event_loop()
                            retry_response = await loop.run_in_executor(
                                self.executor,
                                self._generate_response,
                                [{"role": "user", "content": retry_prompt}],
                                0.3,  # temperature
                                1000,  # max_new_tokens
                            )
                            
                            # JSON 파싱
                            retry_response_text = retry_response.strip()
                            if retry_response_text.startswith("```json"):
                                retry_response_text = retry_response_text[7:].strip()
                            elif retry_response_text.startswith("```"):
                                retry_response_text = retry_response_text[3:].strip()
                            if retry_response_text.endswith("```"):
                                retry_response_text = retry_response_text[:-3].strip()
                            
                            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', retry_response_text, re.DOTALL)
                            if json_match:
                                retry_response_text = json_match.group(0)
                            
                            try:
                                retry_result = json.loads(retry_response_text)
                                # overall_summary는 나중에 포맷팅으로 생성
                                current_positive = retry_result.get("positive_aspects", [])
                                current_negative = retry_result.get("negative_aspects", [])
                            except json.JSONDecodeError as e:
                                logger.warning(f"재시도 JSON 파싱 실패: {e}")
                                current_positive = []
                                current_negative = []
                        
                        # 검증 수행 (동적 기준 사용)
                        all_reviews = positive_reviews + negative_reviews
                        validated_positive = self.validate_aspects_by_cosine_similarity(
                            aspects=current_positive,
                            reviews=all_reviews,
                            vector_search=vector_search,
                            similarity_threshold=dynamic_similarity_threshold,  # 동적 임계값
                        )
                        validated_negative = self.validate_aspects_by_cosine_similarity(
                            aspects=current_negative,
                            reviews=all_reviews,
                            vector_search=vector_search,
                            similarity_threshold=dynamic_similarity_threshold,  # 동적 임계값
                        )
                        
                        # 검증 통과/실패 분류
                        validated_positive_ids = {(a.get('aspect', ''), a.get('claim', '')) for a in validated_positive}
                        validated_negative_ids = {(a.get('aspect', ''), a.get('claim', '')) for a in validated_negative}
                        
                        failed_positive_aspects = [
                            a for a in current_positive 
                            if (a.get('aspect', ''), a.get('claim', '')) not in validated_positive_ids
                        ]
                        failed_negative_aspects = [
                            a for a in current_negative 
                            if (a.get('aspect', ''), a.get('claim', '')) not in validated_negative_ids
                        ]
                        
                        # 검증 통과한 aspect 누적 (중복 제거)
                        for a in validated_positive:
                            key = (a.get('aspect', ''), a.get('claim', ''))
                            if not any((va.get('aspect', ''), va.get('claim', '')) == key for va in validated_positive_aspects):
                                validated_positive_aspects.append(a)
                        
                        for a in validated_negative:
                            key = (a.get('aspect', ''), a.get('claim', ''))
                            if not any((va.get('aspect', ''), va.get('claim', '')) == key for va in validated_negative_aspects):
                                validated_negative_aspects.append(a)
                        
                        # 최소 개수 확인 (조정된 부정 aspect 최소 개수 사용)
                        if (len(validated_positive_aspects) >= min_positive_aspects and 
                            len(validated_negative_aspects) >= min_negative_aspects):
                            # 충분한 aspect 확보 → 루프 탈출
                            logger.debug(
                                f"레스토랑 {restaurant_id} Aspect 검증 완료: "
                                f"긍정 {len(validated_positive_aspects)}개, 부정 {len(validated_negative_aspects)}개 통과"
                            )
                            break
                        elif aspect_attempt < aspect_max_retries:
                            # 부족하면 재시도
                            logger.debug(
                                f"레스토랑 {restaurant_id} Aspect 개수 부족 "
                                f"(긍정: {len(validated_positive_aspects)}/{min_positive_aspects}, "
                                f"부정: {len(validated_negative_aspects)}/{min_negative_aspects}). 재시도합니다."
                            )
                            continue
                        else:
                            # 최대 재시도 횟수 도달
                            logger.debug(
                                f"레스토랑 {restaurant_id} 최대 재시도 횟수 도달. "
                                f"현재까지 확보한 aspect 사용"
                            )
                            break
                    
                    positive_aspects = validated_positive_aspects
                    negative_aspects = validated_negative_aspects
                else:
                    # 검증 비활성화 → 첫 시도 결과 사용
                    pass

                # 검증 완료 후 하이브리드 방식으로 overall_summary 생성
                overall_summary = self.format_overall_summary_hybrid(
                    positive_aspects=positive_aspects,
                    negative_aspects=negative_aspects,
                    positive_count=len(positive_reviews),
                    negative_count=len(negative_reviews),
                    use_llm_polish=use_llm_polish,
                )

            return {
                    "restaurant_id": restaurant_id,
                    "overall_summary": overall_summary,
                    "positive_aspects": positive_aspects,
                    "negative_aspects": negative_aspects,
                    "positive_reviews": positive_reviews,
                    "negative_reviews": negative_reviews,
                    "positive_count": len(positive_reviews),
                    "negative_count": len(negative_reviews),
                }

        results = await asyncio.gather(*[process_single_restaurant(r) for r in restaurants_data])
        logger.info(f"✅ {len(results)}개 레스토랑 요약 완료 (레스토랑 간 비동기 처리)")
        return results


def summarize_reviews(
    llm_utils: LLMUtils,
    positive_reviews: List[Dict[str, Any]],
    negative_reviews: List[Dict[str, Any]],
) -> Dict:
    """
    긍정/부정 리뷰를 요약하는 편의 함수. (메타데이터 포함)
    
    Args:
        llm_utils: LLMUtils 인스턴스
        positive_reviews: 긍정 리뷰 딕셔너리 리스트 (payload 포함)
        negative_reviews: 부정 리뷰 딕셔너리 리스트 (payload 포함)
        
    Returns:
        요약 결과 딕셔너리 (메타데이터 포함)
    """
    return llm_utils.summarize_reviews(positive_reviews, negative_reviews)
