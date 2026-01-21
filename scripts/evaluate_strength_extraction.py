#!/usr/bin/env python3
"""
강점 추출 평가 스크립트

LLM 기반 강점 추출 결과의 정확도를 평가합니다.
- Precision@K: 대표 강점의 정확도 (상위 K개 강점 중 실제 강점 비율)
- Coverage: Ground Truth 강점 중 추출된 비율
- False Positive Rate: 잘못 추출된 강점 비율
- Aspect-level Precision/Recall: Aspect 추출 정확도
- Distinct Score 정확도: 차별 강점 점수 정확도
- Evidence 정확도: 근거 리뷰 ID 일치도
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


class StrengthExtractionEvaluator:
    """강점 추출 평가 클래스"""
    
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
    
    def extract_strengths(
        self,
        restaurant_id: int,
        strength_type: str = "both",
        comparison_restaurant_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        강점 추출 API 호출
        
        Args:
            restaurant_id: 레스토랑 ID
            strength_type: "representative", "distinct", "both"
            comparison_restaurant_ids: 비교 레스토랑 ID 리스트
            
        Returns:
            강점 추출 결과 딕셔너리
        """
        url = f"{self.base_url}/api/v1/llm/extract/strengths"
        
        payload = {
            "restaurant_id": restaurant_id,
            "strength_type": strength_type,
        }
        
        if comparison_restaurant_ids:
            payload["comparison_restaurant_ids"] = comparison_restaurant_ids
        
        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"강점 추출 API 호출 실패: {e}")
            return {}
        except Exception as e:
            logger.error(f"강점 추출 중 오류 발생: {e}")
            return {}
    
    def normalize_aspect(self, aspect: str) -> str:
        """
        Aspect 텍스트 정규화 (비교용)
        
        Args:
            aspect: Aspect 텍스트
            
        Returns:
            정규화된 텍스트
        """
        if not aspect:
            return ""
        return aspect.strip().lower()
    
    def calculate_precision_at_k(
        self,
        predicted_strengths: List[Dict[str, Any]],
        ground_truth_strengths: List[Dict[str, Any]],
        k: int,
    ) -> float:
        """
        Precision@K 계산
        
        Args:
            predicted_strengths: 예측된 강점 리스트 (final_score 기준 정렬 가정)
            ground_truth_strengths: Ground Truth 강점 리스트
            k: 평가할 상위 k개 강점
            
        Returns:
            Precision@K 값 (0.0 ~ 1.0)
        """
        if k == 0 or not predicted_strengths:
            return 0.0
        
        # Ground Truth aspect 집합 생성
        gt_aspects = set()
        for strength in ground_truth_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect:
                gt_aspects.add(aspect)
        
        if not gt_aspects:
            return 0.0
        
        # 상위 k개 강점만 사용
        top_k_strengths = predicted_strengths[:k]
        
        # 관련 있는 강점 수 계산
        relevant_count = 0
        for strength in top_k_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect in gt_aspects:
                relevant_count += 1
        
        # Precision@K = 관련 있는 강점 수 / k
        precision = relevant_count / k
        
        return precision
    
    def calculate_recall_at_k(
        self,
        predicted_strengths: List[Dict[str, Any]],
        ground_truth_strengths: List[Dict[str, Any]],
        k: int,
    ) -> float:
        """
        Recall@K 계산
        
        정의:
        - GT(정답) aspect 집합을 만들고,
        - 예측 상위 k개에서 GT aspect를 몇 개 '맞췄는지'를 세어
        - Recall@K = (맞춘 GT aspect 수) / (전체 GT aspect 수)
        
        Args:
            predicted_strengths: 예측된 강점 리스트 (final_score 기준 정렬 가정)
            ground_truth_strengths: Ground Truth 강점 리스트
            k: 평가할 상위 k개 강점
            
        Returns:
            Recall@K 값 (0.0 ~ 1.0)
        """
        if k == 0 or not predicted_strengths:
            return 0.0
        
        # Ground Truth aspect 집합 생성
        gt_aspects = set()
        for strength in ground_truth_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect:
                gt_aspects.add(aspect)
        
        if not gt_aspects:
            return 0.0
        
        # 상위 k개 강점만 사용
        top_k_strengths = predicted_strengths[:k]
        
        # 관련 있는(GT에 포함되는) 강점 수 계산
        relevant_count = 0
        for strength in top_k_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect in gt_aspects:
                relevant_count += 1
        
        # Recall@K = 관련 있는 강점 수 / |GT|
        recall = relevant_count / len(gt_aspects)
        return recall
    
    def calculate_coverage(
        self,
        predicted_strengths: List[Dict[str, Any]],
        ground_truth_strengths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Coverage 계산
        
        Args:
            predicted_strengths: 예측된 강점 리스트
            ground_truth_strengths: Ground Truth 강점 리스트
            
        Returns:
            Coverage 메트릭 딕셔너리
        """
        if not ground_truth_strengths:
            return {
                "coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        
        # Ground Truth aspect 집합
        gt_aspects = set()
        for strength in ground_truth_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect:
                gt_aspects.add(aspect)
        
        # 예측된 aspect 집합
        pred_aspects = set()
        for strength in predicted_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect:
                pred_aspects.add(aspect)
        
        # Coverage = 추출된 Ground Truth aspect 수 / 전체 Ground Truth aspect 수
        covered_aspects = gt_aspects & pred_aspects
        coverage = len(covered_aspects) / len(gt_aspects) if gt_aspects else 0.0
        
        # Precision = 추출된 Ground Truth aspect 수 / 전체 예측된 aspect 수
        precision = len(covered_aspects) / len(pred_aspects) if pred_aspects else 0.0
        
        # Recall = Coverage와 동일
        recall = coverage
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "coverage": coverage,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "covered_aspects": len(covered_aspects),
            "total_ground_truth_aspects": len(gt_aspects),
            "total_predicted_aspects": len(pred_aspects),
        }
    
    def calculate_false_positive_rate(
        self,
        predicted_strengths: List[Dict[str, Any]],
        ground_truth_strengths: List[Dict[str, Any]],
    ) -> float:
        """
        False Positive Rate 계산
        
        Args:
            predicted_strengths: 예측된 강점 리스트
            ground_truth_strengths: Ground Truth 강점 리스트
            
        Returns:
            False Positive Rate (0.0 ~ 1.0)
        """
        if not predicted_strengths:
            return 0.0
        
        # Ground Truth aspect 집합
        gt_aspects = set()
        for strength in ground_truth_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect:
                gt_aspects.add(aspect)
        
        # 예측된 aspect 중 Ground Truth에 없는 것
        false_positives = 0
        for strength in predicted_strengths:
            aspect = self.normalize_aspect(strength.get("aspect", ""))
            if aspect and aspect not in gt_aspects:
                false_positives += 1
        
        # False Positive Rate = 잘못 추출된 강점 수 / 전체 예측된 강점 수
        fpr = false_positives / len(predicted_strengths) if predicted_strengths else 0.0
        
        return fpr
    
    def calculate_evidence_accuracy(
        self,
        predicted_strength: Dict[str, Any],
        ground_truth_strength: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Evidence 정확도 계산 (근거 리뷰 ID 일치도)
        
        Args:
            predicted_strength: 예측된 강점
            ground_truth_strength: Ground Truth 강점
            
        Returns:
            Evidence 정확도 메트릭 딕셔너리
        """
        predicted_evidence_ids = set(predicted_strength.get("evidence_review_ids", []))
        ground_truth_evidence_ids = set(ground_truth_strength.get("evidence_review_ids", []))
        
        if not ground_truth_evidence_ids:
            return {
                "evidence_precision": 0.0,
                "evidence_recall": 0.0,
                "evidence_f1": 0.0,
                "evidence_overlap": 0.0,
            }
        
        # 교집합
        overlap_ids = predicted_evidence_ids & ground_truth_evidence_ids
        
        # Precision = 교집합 / 예측된 근거 수
        evidence_precision = len(overlap_ids) / len(predicted_evidence_ids) if predicted_evidence_ids else 0.0
        
        # Recall = 교집합 / Ground Truth 근거 수
        evidence_recall = len(overlap_ids) / len(ground_truth_evidence_ids) if ground_truth_evidence_ids else 0.0
        
        # F1 Score
        evidence_f1 = 2 * (evidence_precision * evidence_recall) / (evidence_precision + evidence_recall) if (evidence_precision + evidence_recall) > 0 else 0.0
        
        # Overlap Ratio = 교집합 / 합집합
        union_ids = predicted_evidence_ids | ground_truth_evidence_ids
        evidence_overlap = len(overlap_ids) / len(union_ids) if union_ids else 0.0
        
        return {
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "evidence_f1": evidence_f1,
            "evidence_overlap": evidence_overlap,
        }
    
    def find_matching_strength(
        self,
        predicted_strength: Dict[str, Any],
        ground_truth_strengths: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        예측된 강점과 가장 유사한 Ground Truth 강점 찾기
        
        Args:
            predicted_strength: 예측된 강점
            ground_truth_strengths: Ground Truth 강점 리스트
            
        Returns:
            매칭된 Ground Truth 강점 또는 None
        """
        predicted_aspect = self.normalize_aspect(predicted_strength.get("aspect", ""))
        
        if not predicted_aspect:
            return None
        
        # Aspect 텍스트로 매칭
        for gt_strength in ground_truth_strengths:
            gt_aspect = self.normalize_aspect(gt_strength.get("aspect", ""))
            if predicted_aspect == gt_aspect:
                return gt_strength
        
        return None
    
    def evaluate(
        self,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        강점 추출 평가 수행
        
        Args:
            k_values: 평가할 k 값 리스트
            
        Returns:
            평가 결과 딕셔너리
        """
        if not self.ground_truth:
            raise ValueError("Ground Truth 데이터가 로드되지 않았습니다.")
        
        restaurants = self.ground_truth.get("restaurants", [])
        if not restaurants:
            raise ValueError("Ground Truth에 레스토랑 데이터가 없습니다.")
        
        logger.info(f"총 {len(restaurants)}개 레스토랑 평가 시작")
        
        all_results = []
        all_precisions = {k: [] for k in k_values}
        total_coverage_representative = 0.0
        total_coverage_distinct = 0.0
        total_fpr_representative = 0.0
        total_fpr_distinct = 0.0
        total_restaurants = 0
        
        for idx, restaurant_data in enumerate(restaurants, 1):
            restaurant_id = restaurant_data.get("restaurant_id")
            comparison_restaurant_ids = restaurant_data.get("comparison_restaurant_ids", [])
            ground_truth_strengths = restaurant_data.get("ground_truth_strengths", {})
            
            if not restaurant_id:
                logger.warning(f"레스토랑 {idx}: restaurant_id가 없습니다. 건너뜁니다.")
                continue
            
            logger.info(f"레스토랑 {idx}/{len(restaurants)}: ID={restaurant_id}")
            
            # Ground Truth 분리
            gt_representative = ground_truth_strengths.get("representative", [])
            gt_distinct = ground_truth_strengths.get("distinct", [])
            
            # API 호출 (both 모드)
            result = self.extract_strengths(
                restaurant_id=restaurant_id,
                strength_type="both",
                comparison_restaurant_ids=comparison_restaurant_ids if comparison_restaurant_ids else None,
            )
            
            if not result:
                logger.warning(f"레스토랑 {restaurant_id}: API 호출 실패. 건너뜁니다.")
                continue
            
            # 예측 결과 추출
            predicted_strengths = result.get("strengths", [])
            
            # Representative와 Distinct 분리
            predicted_representative = [
                s for s in predicted_strengths
                if s.get("strength_type") == "representative"
            ]
            predicted_distinct = [
                s for s in predicted_strengths
                if s.get("strength_type") == "distinct"
            ]
            
            # final_score 기준 정렬
            predicted_representative.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            predicted_distinct.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            
            # Precision@K 계산 (Representative)
            for k in k_values:
                precision = self.calculate_precision_at_k(
                    predicted_strengths=predicted_representative,
                    ground_truth_strengths=gt_representative,
                    k=k,
                )
                all_precisions[k].append(precision)
            
            # Coverage 계산
            representative_coverage = self.calculate_coverage(
                predicted_strengths=predicted_representative,
                ground_truth_strengths=gt_representative,
            )
            
            distinct_coverage = self.calculate_coverage(
                predicted_strengths=predicted_distinct,
                ground_truth_strengths=gt_distinct,
            )
            
            # False Positive Rate 계산
            representative_fpr = self.calculate_false_positive_rate(
                predicted_strengths=predicted_representative,
                ground_truth_strengths=gt_representative,
            )
            
            distinct_fpr = self.calculate_false_positive_rate(
                predicted_strengths=predicted_distinct,
                ground_truth_strengths=gt_distinct,
            )
            
            total_coverage_representative += representative_coverage["coverage"]
            total_coverage_distinct += distinct_coverage["coverage"]
            total_fpr_representative += representative_fpr
            total_fpr_distinct += distinct_fpr
            total_restaurants += 1
            
            # Evidence 정확도 계산 (매칭된 강점에 대해)
            evidence_metrics = []
            for pred_strength in predicted_representative[:5]:  # 상위 5개만
                matched_gt = self.find_matching_strength(pred_strength, gt_representative)
                if matched_gt:
                    evidence_metric = self.calculate_evidence_accuracy(pred_strength, matched_gt)
                    evidence_metrics.append(evidence_metric)
            
            restaurant_result = {
                "restaurant_id": restaurant_id,
                "precisions": {
                    k: all_precisions[k][-1] for k in k_values
                },
                "representative_coverage": representative_coverage,
                "distinct_coverage": distinct_coverage,
                "representative_fpr": representative_fpr,
                "distinct_fpr": distinct_fpr,
                "evidence_metrics": evidence_metrics,
                "predicted": {
                    "representative_count": len(predicted_representative),
                    "distinct_count": len(predicted_distinct),
                },
                "ground_truth": {
                    "representative_count": len(gt_representative),
                    "distinct_count": len(gt_distinct),
                },
            }
            
            all_results.append(restaurant_result)
        
        # 평균 계산
        avg_precisions = {
            k: sum(all_precisions[k]) / len(all_precisions[k]) if all_precisions[k] else 0.0
            for k in k_values
        }
        avg_coverage_representative = total_coverage_representative / total_restaurants if total_restaurants > 0 else 0.0
        avg_coverage_distinct = total_coverage_distinct / total_restaurants if total_restaurants > 0 else 0.0
        avg_fpr_representative = total_fpr_representative / total_restaurants if total_restaurants > 0 else 0.0
        avg_fpr_distinct = total_fpr_distinct / total_restaurants if total_restaurants > 0 else 0.0
        
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "total_restaurants": total_restaurants,
            "average_metrics": {
                "precisions": avg_precisions,
                "representative_coverage": avg_coverage_representative,
                "distinct_coverage": avg_coverage_distinct,
                "representative_fpr": avg_fpr_representative,
                "distinct_fpr": avg_fpr_distinct,
            },
            "restaurant_results": all_results,
        }
        
        return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="강점 추출 평가 스크립트")
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
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (선택사항, 기본값: strength_evaluation_YYYYMMDD_HHMMSS.json)"
    )
    
    args = parser.parse_args()
    
    evaluator = StrengthExtractionEvaluator(
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
    )
    
    try:
        result = evaluator.evaluate(k_values=args.k_values)
        
        # 결과 출력
        print("\n" + "="*80)
        print("강점 추출 평가 결과")
        print("="*80)
        print(f"총 레스토랑 수: {result['total_restaurants']}")
        print(f"\n평균 Precision@K:")
        for k, precision in result['average_metrics']['precisions'].items():
            print(f"  - Precision@{k}: {precision:.4f}")
        print(f"\n평균 Coverage:")
        print(f"  - Representative: {result['average_metrics']['representative_coverage']:.4f}")
        print(f"  - Distinct: {result['average_metrics']['distinct_coverage']:.4f}")
        print(f"\n평균 False Positive Rate:")
        print(f"  - Representative: {result['average_metrics']['representative_fpr']:.4f}")
        print(f"  - Distinct: {result['average_metrics']['distinct_fpr']:.4f}")
        
        # 결과 저장
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"strength_evaluation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
