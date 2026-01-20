#!/usr/bin/env python3
"""
Precision@k 평가 스크립트

벡터 검색 결과의 정확도를 Precision@k 지표로 평가합니다.
- Precision@k = (상위 k개 검색 결과 중 관련 있는 문서 수) / k
- 여러 k 값에 대한 평가 (k=1, 3, 5, 10 등)
- Ground Truth와 검색 결과 비교
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from datetime import datetime
import requests

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrecisionAtKEvaluator:
    """Precision@k 평가 클래스"""
    
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
        
        # Ground Truth 로드
        if ground_truth_path and Path(ground_truth_path).exists():
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
        else:
            self.ground_truth = None
            logger.warning(f"Ground Truth 파일을 찾을 수 없습니다: {ground_truth_path}")
    
    def calculate_precision_at_k(
        self,
        retrieved_ids: List[int],
        relevant_ids: Set[int],
        k: int
    ) -> float:
        """
        Precision@k 계산
        
        Args:
            retrieved_ids: 검색된 리뷰 ID 리스트 (상위 k개)
            relevant_ids: 관련 있는 리뷰 ID 집합
            k: 평가할 상위 k개 결과
            
        Returns:
            Precision@k 값 (0.0 ~ 1.0)
        """
        if k == 0:
            return 0.0
        
        # 상위 k개 결과만 사용
        top_k_ids = retrieved_ids[:k]
        
        # 관련 있는 문서 수 계산
        relevant_count = sum(1 for doc_id in top_k_ids if doc_id in relevant_ids)
        
        # Precision@k = 관련 있는 문서 수 / k
        precision = relevant_count / k
        
        return precision
    
    def search_reviews(
        self,
        query_text: str,
        restaurant_id: Optional[int] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[int]:
        """
        벡터 검색 API 호출하여 리뷰 ID 리스트 반환
        
        Args:
            query_text: 검색 쿼리 텍스트
            restaurant_id: 레스토랑 ID 필터 (선택사항)
            limit: 반환할 최대 개수
            min_score: 최소 유사도 점수
            
        Returns:
            검색된 리뷰 ID 리스트
        """
        url = f"{self.base_url}/api/v1/vector/search/similar"
        
        payload = {
            "query_text": query_text,
            "limit": limit,
            "min_score": min_score,
        }
        
        if restaurant_id is not None:
            payload["restaurant_id"] = restaurant_id
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # 리뷰 ID 추출
            # API 응답 구조: {"results": [{"review": {"id": ...}, "score": ...}, ...]}
            review_ids = []
            for result in results:
                # review 객체에서 id 추출
                review = result.get("review", {})
                review_id = review.get("id")
                
                if review_id is not None:
                    # id가 문자열일 수도 있으므로 int로 변환
                    try:
                        review_id = int(review_id)
                        review_ids.append(review_id)
                    except (ValueError, TypeError):
                        logger.warning(f"리뷰 ID를 int로 변환할 수 없습니다: {review_id}")
                        continue
            
            return review_ids
            
        except requests.exceptions.RequestException as e:
            logger.error(f"벡터 검색 API 호출 실패: {e}")
            return []
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {e}")
            return []
    
    def evaluate(
        self,
        k_values: List[int] = [1, 3, 5, 10],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Precision@k 평가 수행
        
        Args:
            k_values: 평가할 k 값 리스트
            limit: 검색할 최대 개수
            min_score: 최소 유사도 점수
            
        Returns:
            평가 결과 딕셔너리
        """
        if not self.ground_truth:
            raise ValueError("Ground Truth 데이터가 로드되지 않았습니다.")
        
        queries = self.ground_truth.get("queries", [])
        if not queries:
            raise ValueError("Ground Truth에 쿼리가 없습니다.")
        
        logger.info(f"총 {len(queries)}개 쿼리 평가 시작 (k 값: {k_values})")
        
        all_precisions = {k: [] for k in k_values}
        query_results = []
        
        for idx, query_data in enumerate(queries, 1):
            query_text = query_data.get("query")
            restaurant_id = query_data.get("restaurant_id")
            relevant_ids = set(query_data.get("relevant_review_ids", []))
            
            if not query_text:
                logger.warning(f"쿼리 {idx}: query_text가 없습니다. 건너뜁니다.")
                continue
            
            if not relevant_ids:
                logger.warning(f"쿼리 {idx}: relevant_review_ids가 없습니다. 건너뜁니다.")
                continue
            
            logger.info(f"쿼리 {idx}/{len(queries)}: '{query_text}' (관련 문서 수: {len(relevant_ids)})")
            
            # 벡터 검색 수행
            retrieved_ids = self.search_reviews(
                query_text=query_text,
                restaurant_id=restaurant_id,
                limit=limit,
                min_score=min_score,
            )
            
            if not retrieved_ids:
                logger.warning(f"쿼리 {idx}: 검색 결과가 없습니다.")
                query_result = {
                    "query": query_text,
                    "restaurant_id": restaurant_id,
                    "retrieved_count": 0,
                    "precisions": {f"P@{k}": 0.0 for k in k_values}
                }
                query_results.append(query_result)
                continue
            
            # 각 k 값에 대한 Precision@k 계산
            precisions = {}
            for k in k_values:
                precision = self.calculate_precision_at_k(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k=k
                )
                precisions[f"P@{k}"] = precision
                all_precisions[k].append(precision)
            
            query_result = {
                "query": query_text,
                "restaurant_id": restaurant_id,
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_ids),
                "retrieved_ids": retrieved_ids[:max(k_values)],  # 최대 k값까지만 저장
                "precisions": precisions
            }
            query_results.append(query_result)
            
            # 진행 상황 출력
            precisions_str = ", ".join([f"P@{k}={precisions[f'P@{k}']:.3f}" for k in k_values])
            logger.info(f"  검색된 문서 수: {len(retrieved_ids)}, {precisions_str}")
        
        # 전체 평균 계산
        avg_precisions = {}
        for k in k_values:
            if all_precisions[k]:
                avg_precisions[f"P@{k}"] = sum(all_precisions[k]) / len(all_precisions[k])
            else:
                avg_precisions[f"P@{k}"] = 0.0
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "evaluated_queries": len(query_results),
            "k_values": k_values,
            "average_precisions": avg_precisions,
            "query_results": query_results
        }
        
        return result
    
    def print_summary(self, result: Dict[str, Any]):
        """평가 결과 요약 출력"""
        print("\n" + "="*60)
        print("Precision@k 평가 결과 요약")
        print("="*60)
        
        print(f"\n총 쿼리 수: {result['total_queries']}")
        print(f"평가된 쿼리 수: {result['evaluated_queries']}")
        print(f"\n평균 Precision@k:")
        
        for k in result['k_values']:
            avg_precision = result['average_precisions'][f"P@{k}"]
            print(f"  P@{k}: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="벡터 검색 Precision@k 평가 스크립트"
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
        "--output",
        type=str,
        help="결과 저장 파일 경로 (JSON 형식, 선택사항)"
    )
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = PrecisionAtKEvaluator(
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
    )
    
    # 평가 수행
    try:
        result = evaluator.evaluate(
            k_values=args.k_values,
            limit=args.limit,
            min_score=args.min_score,
        )
        
        # 결과 출력
        evaluator.print_summary(result)
        
        # 결과 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과가 저장되었습니다: {output_path}")
        else:
            # 기본 파일명으로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("precision_at_k_results") / f"precision_at_k_{timestamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과가 저장되었습니다: {output_path}")
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
