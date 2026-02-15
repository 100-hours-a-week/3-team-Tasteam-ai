# Tasteam FastAPI 앱 (CPU 전용, OpenAI / RunPod Serverless HTTP 사용)
# 인프로세스 vLLM 제거됨 — GPU 추론은 RunPod 엔드포인트로만 사용
#
# 빌드: docker build -t tasteam-app .
# 실행: docker run -p 8001:8001 --env-file .env tasteam-app
#
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
ENV "USE_GPU#"=false
WORKDIR /app

# 시스템 의존성: 빌드 도구 + OpenJDK (PySpark/비교 파이프라인용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# PyTorch CPU 버전 (requirements보다 먼저 설치)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8001

CMD ["python", "app.py"]
