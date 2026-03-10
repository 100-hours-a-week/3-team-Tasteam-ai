"""
Admin DeepFM API 서버 (api_design.md).

실행: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import deepfm

app = FastAPI(
    title="DeepFM Pipeline Admin API",
    description="학습 트리거, 모델/버전 조회, 활성화 (api_design.md)",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(deepfm.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
