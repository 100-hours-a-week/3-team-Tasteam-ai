#!/usr/bin/env python3
"""
임베딩 모델 비교 평가 스크립트

여러 임베딩 모델에 대해 Precision@k 평가를 수행하고 결과를 비교합니다.
각 모델마다 EMBEDDING_MODEL 환경 변수를 변경하여 평가합니다.

사용 방법:
    # 서버가 실행 중이어야 함 (각 모델마다 서버 재시작 필요)
    python scripts/compare_embedding_models.py \
        --models "jhgan/ko-sbert-multitask" "dragonkue/BGE-m3-ko" "upskyy/bge-m3-korean" \
        --ground-truth scripts/Ground_truth_vector_search.json \
        --base-url http://localhost:8000 \
        --k-values 1 3 5 10 \
        --output embedding_comparison.json
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate_vector_search import PrecisionAtKEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingModelComparator:
    """임베딩 모델 비교 평가 클래스"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ground_truth_path: Optional[str] = None,
    ):
        """
        Args:
            base_url: API 서버 URL
            ground_truth_path: Ground Truth JSON 파일 경로
        """
        self.base_url = base_url
        self.ground_truth_path = ground_truth_path
    
    def check_server_health(self, timeout: int = 30) -> bool:
        """서버 상태 확인"""
        import requests
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"서버 상태 확인 실패: {e}")
            return False
    
    def wait_for_server(self, max_wait: int = 60, check_interval: int = 2) -> bool:
        """서버가 준비될 때까지 대기"""
        logger.info(f"서버 준비 대기 중... (최대 {max_wait}초)")
        
        for _ in range(max_wait // check_interval):
            if self.check_server_health():
                logger.info("서버가 준비되었습니다.")
                return True
            time.sleep(check_interval)
        
        logger.error(f"서버가 {max_wait}초 내에 준비되지 않았습니다.")
        return False
    
    def evaluate_model(
        self,
        model_name: str,
        k_values: List[int] = [1, 3, 5, 10],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        특정 임베딩 모델에 대한 평가 수행
        
        Args:
            model_name: 평가할 임베딩 모델명
            k_values: 평가할 k 값 리스트
            limit: 검색할 최대 개수
            min_score: 최소 유사도 점수
            
        Returns:
            평가 결과 딕셔너리
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"임베딩 모델 평가 시작: {model_name}")
        logger.info(f"{'='*60}")
        
        # 서버 상태 확인
        if not self.check_server_health():
            logger.warning(f"서버가 응답하지 않습니다. {model_name} 평가를 건너뜁니다.")
            logger.warning("서버를 재시작하고 EMBEDDING_MODEL 환경 변수를 설정했는지 확인하세요.")
            return {
                "model_name": model_name,
                "status": "skipped",
                "error": "서버가 응답하지 않음",
                "timestamp": datetime.now().isoformat(),
            }
        
        # 평가기 초기화
        evaluator = PrecisionAtKEvaluator(
            base_url=self.base_url,
            ground_truth_path=self.ground_truth_path,
        )
        
        # 평가 수행
        try:
            result = evaluator.evaluate(
                k_values=k_values,
                limit=limit,
                min_score=min_score,
            )
            
            # 모델명 추가
            result["model_name"] = model_name
            result["status"] = "completed"
            
            logger.info(f"\n{model_name} 평가 완료:")
            evaluator.print_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"{model_name} 평가 중 오류 발생: {e}", exc_info=True)
            return {
                "model_name": model_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
    
    def compare_models(
        self,
        model_names: List[str],
        k_values: List[int] = [1, 3, 5, 10],
        limit: int = 10,
        min_score: float = 0.0,
        wait_between_models: int = 5,
    ) -> Dict[str, Any]:
        """
        여러 임베딩 모델 비교 평가
        
        Args:
            model_names: 평가할 임베딩 모델명 리스트
            k_values: 평가할 k 값 리스트
            limit: 검색할 최대 개수
            min_score: 최소 유사도 점수
            wait_between_models: 모델 간 대기 시간 (초)
            
        Returns:
            비교 결과 딕셔너리
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"임베딩 모델 비교 평가 시작")
        logger.info(f"모델 수: {len(model_names)}")
        logger.info(f"모델 목록: {', '.join(model_names)}")
        logger.info(f"{'='*60}\n")
        
        all_results = []
        
        for idx, model_name in enumerate(model_names, 1):
            logger.info(f"\n[{idx}/{len(model_names)}] {model_name} 평가 중...")
            
            # 모델 평가
            result = self.evaluate_model(
                model_name=model_name,
                k_values=k_values,
                limit=limit,
                min_score=min_score,
            )
            
            all_results.append(result)
            
            # 마지막 모델이 아니면 대기
            if idx < len(model_names):
                logger.info(f"\n다음 모델 평가 전 {wait_between_models}초 대기...")
                logger.info("⚠️  다음 모델을 사용하도록 서버를 재시작하고 EMBEDDING_MODEL 환경 변수를 설정하세요.")
                time.sleep(wait_between_models)
        
        # 비교 결과 생성
        comparison_result = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(model_names),
            "k_values": k_values,
            "models": all_results,
            "comparison": self._create_comparison_table(all_results, k_values),
        }
        
        return comparison_result
    
    def _create_comparison_table(
        self,
        results: List[Dict[str, Any]],
        k_values: List[int],
    ) -> Dict[str, Any]:
        """모델 비교 테이블 생성"""
        comparison = {
            "average_precisions": {},
            "best_model_per_k": {},
        }
        
        # 각 k 값에 대한 평균 precision 비교
        for k in k_values:
            k_key = f"P@{k}"
            model_scores = {}
            
            for result in results:
                if result.get("status") == "completed":
                    model_name = result.get("model_name", "unknown")
                    avg_precisions = result.get("average_precisions", {})
                    precision = avg_precisions.get(k_key, 0.0)
                    model_scores[model_name] = precision
            
            comparison["average_precisions"][k_key] = model_scores
            
            # 각 k 값에서 최고 성능 모델
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])
                comparison["best_model_per_k"][k_key] = {
                    "model": best_model[0],
                    "precision": best_model[1],
                }
        
        return comparison
    
    def print_comparison(self, comparison_result: Dict[str, Any]):
        """비교 결과 출력"""
        print("\n" + "="*80)
        print("임베딩 모델 비교 결과")
        print("="*80)
        
        print(f"\n총 평가 모델 수: {comparison_result['total_models']}")
        print(f"평가 k 값: {comparison_result['k_values']}")
        
        # 각 모델별 결과 요약
        print("\n" + "-"*80)
        print("모델별 평균 Precision@k:")
        print("-"*80)
        
        for result in comparison_result["models"]:
            model_name = result.get("model_name", "unknown")
            status = result.get("status", "unknown")
            
            print(f"\n[{model_name}]")
            print(f"  상태: {status}")
            
            if status == "completed":
                avg_precisions = result.get("average_precisions", {})
                for k in comparison_result["k_values"]:
                    k_key = f"P@{k}"
                    precision = avg_precisions.get(k_key, 0.0)
                    print(f"  {k_key}: {precision:.4f} ({precision*100:.2f}%)")
            elif status == "error":
                print(f"  오류: {result.get('error', 'Unknown error')}")
        
        # 비교 테이블
        print("\n" + "-"*80)
        print("k 값별 최고 성능 모델:")
        print("-"*80)
        
        best_models = comparison_result.get("comparison", {}).get("best_model_per_k", {})
        for k in comparison_result["k_values"]:
            k_key = f"P@{k}"
            best = best_models.get(k_key, {})
            if best:
                print(f"  {k_key}: {best['model']} (Precision: {best['precision']:.4f})")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="임베딩 모델 비교 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
    # 1. 첫 번째 모델로 서버 시작
    export EMBEDDING_MODEL="jhgan/ko-sbert-multitask"
    python app.py
    
    # 2. 별도 터미널에서 평가 시작 (각 모델마다 서버 재시작 필요)
    python scripts/compare_embedding_models.py \\
        --models "jhgan/ko-sbert-multitask" "dragonkue/BGE-m3-ko" "upskyy/bge-m3-korean" \\
        --ground-truth scripts/Ground_truth_vector_search.json \\
        --base-url http://localhost:8000 \\
        --k-values 1 3 5 10 \\
        --output embedding_comparison.json

주의사항:
    - 각 모델 평가 전에 서버를 재시작하고 EMBEDDING_MODEL 환경 변수를 설정해야 합니다.
    - 스크립트는 각 모델 평가 전에 대기 시간을 제공합니다.
        """
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="비교할 임베딩 모델명 리스트 (예: 'jhgan/ko-sbert-multitask' 'dragonkue/BGE-m3-ko')"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Ground Truth JSON 파일 경로"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="API 서버 URL (기본값: http://localhost:8000)"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="평가할 k 값 리스트 (기본값: 1 3 5 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="검색할 최대 개수 (기본값: 10)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="최소 유사도 점수 (기본값: 0.0)"
    )
    parser.add_argument(
        "--wait-between-models",
        type=int,
        default=10,
        help="모델 간 대기 시간 (초, 기본값: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="결과 저장 파일 경로 (JSON 형식, 선택사항)"
    )
    
    args = parser.parse_args()
    
    # 비교기 초기화
    comparator = EmbeddingModelComparator(
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
    )
    
    # 비교 평가 수행
    try:
        comparison_result = comparator.compare_models(
            model_names=args.models,
            k_values=args.k_values,
            limit=args.limit,
            min_score=args.min_score,
            wait_between_models=args.wait_between_models,
        )
        
        # 결과 출력
        comparator.print_comparison(comparison_result)
        
        # 결과 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과가 저장되었습니다: {output_path}")
        else:
            # 기본 파일명으로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("embedding_comparison_results") / f"embedding_comparison_{timestamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과가 저장되었습니다: {output_path}")
        
    except Exception as e:
        logger.error(f"비교 평가 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
