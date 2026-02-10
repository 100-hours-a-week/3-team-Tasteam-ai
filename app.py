"""
FastAPI 애플리케이션 실행 파일
"""

import os
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    # 환경 변수로 포트 설정 가능 (기본값: 8000)
    # RUNPOD base image의 /start.sh가 8000을 사용할 수 있음
    port = int(os.getenv("PORT", "8001").strip())
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )