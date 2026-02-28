#!/usr/bin/env python3
"""
Spark 마이크로서비스 (Comparison용 전체 평균·recall seeds 계산)

메인 앱/워커는 SPARK_SERVICE_URL로 이 서비스를 호출하고, 로컬에서는 Spark/JVM을 로드하지 않음.
이 프로세스만 pyspark/JVM을 사용.

실행:
  python scripts/spark_service.py
  uvicorn (또는 gunicorn)으로 실행 시: 모듈 경로로 로드 (프로젝트 루트가 path에 있어야 함)

엔드포인트:
  POST /all-average-from-file       body: {"path": "/abs/path", "project_root": null}
  POST /all-average-from-reviews    body: {"texts": ["리뷰1", "리뷰2", ...]}
  POST /recall-seeds-from-file      body: {"path": "/abs/path", "project_root": null}
  POST /recall-seeds-from-reviews   body: {"texts": ["리뷰1", "리뷰2", ...]}
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# comparison_pipeline은 이 프로세스에서만 Spark 로드 (메인 앱은 SPARK_SERVICE_URL로 여기 호출)
from src.comparison_pipeline import (
    calculate_all_average_ratios_from_file,
    calculate_all_average_ratios_from_texts,
    compute_recall_seeds_from_file,
    compute_recall_seeds_from_texts,
)
from src.config import Config

app = FastAPI(title="Comparison Spark Service", version="0.1.0")


class AllAverageRequest(BaseModel):
    path: str
    project_root: Optional[str] = None


class RecallSeedsRequest(BaseModel):
    path: str
    project_root: Optional[str] = None


class RecallSeedsFromReviewsRequest(BaseModel):
    texts: List[str]


class AllAverageFromReviewsRequest(BaseModel):
    texts: List[str]


@app.post("/all-average-from-file")
def all_average_from_file(req: AllAverageRequest) -> Dict[str, float]:
    """
    파일 경로로 전체 평균 비율(service, price) 계산. Spark 사용.
    """
    try:
        stopwords = None  # comparison_pipeline 내부에서 로드 가능
        result = calculate_all_average_ratios_from_file(
            req.path,
            stopwords=stopwords,
            project_root=req.project_root,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="파일 없음 또는 계산 실패")
        return {"service": result.get("service", 0.0), "price": result.get("price", 0.0)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/all-average-from-reviews")
def all_average_from_reviews(req: AllAverageFromReviewsRequest) -> Dict[str, float]:
    """
    리뷰 텍스트 리스트로 전체 평균 비율(service, price) 계산. Spark 사용.
    """
    try:
        result = calculate_all_average_ratios_from_texts(req.texts, stopwords=None)
        if result is None:
            raise HTTPException(status_code=500, detail="전체 평균 계산 실패(Spark 미사용/실패)")
        return {"service": result.get("service", 0.0), "price": result.get("price", 0.0)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recall-seeds-from-file")
def recall_seeds_from_file(req: RecallSeedsRequest) -> Dict[str, Any]:
    """
    파일 경로로 recall seeds(service, price, food, other) 계산. Spark 사용.
    """
    try:
        result = compute_recall_seeds_from_file(
            req.path,
            stopwords=None,
            project_root=req.project_root,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="파일 없음 또는 계산 실패")
        # Tuple[str, int] 리스트를 JSON 호환 [["phrase", count], ...] 로 반환
        return {
            k: [[p, c] for p, c in v]
            for k, v in result.items()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recall-seeds-from-reviews")
def recall_seeds_from_reviews(req: RecallSeedsFromReviewsRequest) -> Dict[str, Any]:
    """
    리뷰 텍스트 리스트로 recall seeds(service, price, food, other) 계산. Spark 사용.
    """
    try:
        result = compute_recall_seeds_from_texts(req.texts, stopwords=None)
        if result is None:
            raise HTTPException(status_code=500, detail="recall seeds 계산 실패(Spark 미사용/실패)")
        return {
            k: [[p, c] for p, c in v]
            for k, v in result.items()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    port = int(os.getenv("SPARK_SERVICE_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
