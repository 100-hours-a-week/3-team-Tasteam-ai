"""
환경변수 기반 모델 다운로드 스크립트 (메모리 효율적 버전)
"""
import os
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    """환경변수에 따라 모델 다운로드 (파일만 다운로드, 메모리 로드 없음)"""
    
    # HF_HOME 환경 변수 확인
    hf_home = os.getenv("HF_HOME", "/app/models")
    logger.info(f"HF_HOME: {hf_home}")
    
    # 다운로드 여부 제어
    download_llm = os.getenv("PRE_DOWNLOAD_LLM", "true").lower() == "true"
    download_sentiment = os.getenv("PRE_DOWNLOAD_SENTIMENT", "true").lower() == "true"
    download_embedding = os.getenv("PRE_DOWNLOAD_EMBEDDING", "true").lower() == "true"
    
    # LLM 모델 - Hugging Face 표준 구조로 다운로드
    if download_llm:
        llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        logger.info(f"LLM 모델 다운로드 시작: {llm_model}")
        try:
            # Hugging Face 표준 캐시 구조 사용
            # HF_HOME/hub/models--{org}--{model} 형식으로 자동 저장됨
            snapshot_download(
                repo_id=llm_model,
                cache_dir=hf_home,
                local_files_only=False,  # 네트워크에서 다운로드
                ignore_patterns=["*.md", "*.txt"]  # 불필요한 파일 제외
            )
            logger.info(f"✅ LLM 모델 다운로드 완료: {llm_model}")
        except Exception as e:
            logger.error(f"❌ LLM 모델 다운로드 실패: {e}")
    
    # 감성 분석 모델
    if download_sentiment:
        sentiment_model = os.getenv("SENTIMENT_MODEL", "Dilwolf/Kakao_app-kr_sentiment")
        logger.info(f"감성 분석 모델 다운로드 시작: {sentiment_model}")
        try:
            snapshot_download(
                repo_id=sentiment_model,
                cache_dir=hf_home,
                local_files_only=False,
                ignore_patterns=["*.md", "*.txt"]
            )
            logger.info(f"✅ 감성 분석 모델 다운로드 완료: {sentiment_model}")
        except Exception as e:
            logger.error(f"❌ 감성 분석 모델 다운로드 실패: {e}")
    
    # 임베딩 모델
    if download_embedding:
        embedding_model = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-multitask")
        logger.info(f"임베딩 모델 다운로드 시작: {embedding_model}")
        try:
            snapshot_download(
                repo_id=embedding_model,
                cache_dir=hf_home,
                local_files_only=False,
                ignore_patterns=["*.md", "*.txt"]
            )
            logger.info(f"✅ 임베딩 모델 다운로드 완료: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 다운로드 실패: {e}")

if __name__ == "__main__":
    download_models()