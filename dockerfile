# CPU 전용 이미지 - GPU 없는 환경에서 src 애플리케이션 실행
#
# 빌드: docker build -f Dockerfile.cpu -t app-cpu .
# 실행: docker run -p 8001:8001 app-cpu
# 포트 변경: docker run -p 8080:8080 -e PORT=8080 app-cpu
#
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
# config.py에서 GPU 사용 여부를 "USE_GPU#" 환경 변수로 읽음
ENV "USE_GPU#"=false
WORKDIR /app

# 시스템 의존성 (빌드 도구, 일부 패키지 컴파일용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU 버전 먼저 설치 (requirements보다 먼저 해야 충돌 방지)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 포트 노출 (app.py 기본값 8001)
EXPOSE 8001

# CPU 환경에서 실행 (USE_GPU는 기본 false)
CMD ["python", "app.py"]
