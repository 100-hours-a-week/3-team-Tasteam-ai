#!/usr/bin/env python3
"""
Prometheus 메트릭 노출 서버.
포트에서 /metrics 엔드포인트를 제공하며 Prometheus가 스크래핑할 수 있습니다.
"""

import argparse
import logging
from prometheus_client import Gauge, Info, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 커스텀 메트릭 예시 (확장 가능)
metrics_info = Info("metrics_agent", "Metrics agent info")
metrics_info.info({"version": "1.0", "service": "jobmgr"})


def main():
    parser = argparse.ArgumentParser(description="Prometheus metrics HTTP server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=1041,
        help="HTTP 서버 포트 (기본: 1041)",
    )
    args = parser.parse_args()

    start_http_server(args.port)
    logger.info("Prometheus metrics server started on port %d (GET /metrics)", args.port)

    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()