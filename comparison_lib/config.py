"""
comparison_lib 전용 최소 설정 (Spark 서비스·메인 앱 공통).
src.config 의존 제거 — comparison_pipeline에서만 사용.
"""
import os
from typing import Optional

# Spark 마이크로서비스 URL. 설정 시 메인 앱은 해당 서비스로 HTTP 호출. Spark 서비스는 비워 둠.
SPARK_SERVICE_URL: Optional[str] = os.getenv("SPARK_SERVICE_URL", "http://localhost:8002").strip() or None
DISABLE_SPARK: bool = os.getenv("DISABLE_SPARK", "false").lower() == "true"
RECALL_SEEDS_SPARK_THRESHOLD: int = int(os.getenv("RECALL_SEEDS_SPARK_THRESHOLD", "2000"))
