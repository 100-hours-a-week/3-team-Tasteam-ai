FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 시스템 의존성은 베이스 이미지에 이미 포함되어 있음
# 필요 시 런타임에 설치 가능

COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Flash Attention-2 설치 (Phase 1)
RUN (pip install flash-attn==2.5.6 --no-build-isolation || echo "Flash Attention-2 설치 실패")

# vLLM 설치 (RunPod Pod 환경에서 사용)
RUN (pip install vllm>=0.3.3 || echo "vLLM 설치 실패 (선택사항)")

# Hugging Face 설정
ENV HF_HOME=/workspace/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출
EXPOSE 8001

# RunPod Pod 환경에서 직접 실행
CMD ["python", "app.py"]