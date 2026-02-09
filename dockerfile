# CUDA 포함 이미지 - GPU 환경에서 src 애플리케이션 실행
#
# 빌드: docker build -f dockerfile -t app-cuda .
# 실행: docker run --gpus all -p 8001:8001 app-cuda
#
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# config.py에서 GPU 사용 여부를 "USE_GPU#" 환경 변수로 읽음
ENV "USE_GPU#"=true
WORKDIR /app

# Python 3.11 + 시스템 의존성 (빌드 도구, OpenJDK for PySpark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    curl \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# pip (Python 3.11용)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch CUDA 12.1 버전 먼저 설치 (requirements보다 먼저 해야 충돌 방지)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출 (app.py 기본값 8001)
EXPOSE 8001

# GPU 환경에서 실행 (실행 시 --gpus all 필요)
CMD ["python", "app.py"]
