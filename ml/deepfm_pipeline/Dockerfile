# Admin DeepFM API + Pipeline (api_design.md)
# 이미지: 학습/스코어링 파이프라인 + Admin API 서버

FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (선택: 빌드 시 컴파일 필요 시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 (캐시 활용)
COPY ml-requirements.txt .
RUN pip install --no-cache-dir -r ml-requirements.txt

# 소스 (빌드 컨텍스트: Dockerfile 위치 기준)
COPY . .

# API 서버는 프로젝트 루트를 path로 사용
ENV PYTHONPATH=/app

# Admin API (학습 트리거, 스코어링, 모델 조회, 활성화)
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
