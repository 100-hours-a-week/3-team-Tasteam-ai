#!/usr/bin/env python3
"""
Spark 마이크로서비스 진입점.
실제 앱은 servjces.spark.main에 정의. 프로젝트 루트를 path에 넣고 uvicorn 실행.

실행: python scripts/spark_service.py
포트: SPARK_SERVICE_PORT (기본 8002)
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from servjces.spark.main import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("SPARK_SERVICE_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
