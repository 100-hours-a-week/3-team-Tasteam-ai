#!/usr/bin/env python3
"""
리뷰 요약 평가 스크립트

LLM 기반 리뷰 요약 결과의 품질을 평가합니다.
- ROUGE Score: 요약 텍스트 유사도 (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU Score: 요약 텍스트 유사도
- Aspect Coverage: Ground Truth aspect가 얼마나 추출되었는지
- Aspect Precision/Recall: Aspect 추출 정확도
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
import re

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ROUGE 라이브러리 (선택적)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("rouge-score가 설치되지 않았습니다. ROUGE 점수 계산이 비활성화됩니다.")

# BLEU 라이브러리 (선택적)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logger.warning("nltk가 설치되지 않았습니다. BLEU 점수 계산이 비활성화됩니다.")


class SummaryEvaluator:
    """리뷰 요약 평가 클래스"""
    
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
        
        # ROUGE Scorer 초기화
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        else:
            self.rouge_scorer = None
    
    def summarize_reviews(
        self,
        restaurant_id: int,
    ) -> Dict[str, Any]:
        """
        리뷰 요약 API 호출
        
        Args:
            restaurant_id: 레스토랑 ID
            
        Returns:
            요약 결과 딕셔너리
        """
        url = f"{self.base_url}/api/v1/llm/summarize"
        
        payload = {
            "restaurant_id": restaurant_id,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"요약 API 호출 실패: {e}")
            return {}
        except Exception as e:
            logger.error(f"요약 중 오류 발생: {e}")
            return {}
    
    def calculate_rouge_scores(
        self,
        predicted_text: str,
        ground_truth_text: str,
    ) -> Dict[str, float]:
        """
        ROUGE 점수 계산
        
        Args:
            predicted_text: 예측된 텍스트
            ground_truth_text: Ground Truth 텍스트
            
        Returns:
            ROUGE 점수 딕셔너리
        """
        if not self.rouge_scorer:
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            }
        
        scores = self.rouge_scorer.score(ground_truth_text, predicted_text)
        
        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    
    def calculate_bleu_score(
        self,
        predicted_text: str,
        ground_truth_text: str,
    ) -> float:
        """
        BLEU 점수 계산
        
        Args:
            predicted_text: 예측된 텍스트
            ground_truth_text: Ground Truth 텍스트
            
        Returns:
            BLEU 점수
        """
        if not BLEU_AVAILABLE:
            return 0.0
        
        # 한국어 토큰화 (간단한 공백 기반)
        predicted_tokens = predicted_text.split()
        ground_truth_tokens = ground_truth_text.split()
        
        if not predicted_tokens or not ground_truth_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [ground_truth_tokens],
            predicted_tokens,
            smoothing_function=smoothing,
        )
        
        return score
    
    def calculate_aspect_coverage(
        self,
        predicted_aspects: List[Dict[str, Any]],
        ground_truth_aspects: List[Dict[str, Any]],
        aspect_type: str = "positive",
    ) -> Dict[str, float]:
        """
        Aspect Coverage 계산
        
        Args:
            predicted_aspects: 예측된 aspect 리스트
            ground_truth_aspects: Ground Truth aspect 리스트
            aspect_type: "positive" 또는 "negative"
            
        Returns:
            Coverage 메트릭 딕셔너리
        """
        if not ground_truth_aspects:
            return {
                "coverage": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        
        # Ground Truth aspect 집합 (aspect 텍스트 기준)
        gt_aspects = set()
        for aspect in ground_truth_aspects:
            aspect_text = aspect.get("aspect", "").strip().lower()
            if aspect_text:
                gt_aspects.add(aspect_text)
        
        # 예측된 aspect 집합
        pred_aspects = set()
        for aspect in predicted_aspects:
            aspect_text = aspect.get("aspect", "").strip().lower()
            if aspect_text:
                pred_aspects.add(aspect_text)
        
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
    
    def evaluate(
        self,
    ) -> Dict[str, Any]:
        """
        리뷰 요약 평가 수행
        
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
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        total_rougeL = 0.0
        total_bleu = 0.0
        total_positive_coverage = 0.0
        total_negative_coverage = 0.0
        total_positive_precision = 0.0
        total_negative_precision = 0.0
        total_positive_recall = 0.0
        total_negative_recall = 0.0
        total_restaurants = 0
        
        for idx, restaurant_data in enumerate(restaurants, 1):
            restaurant_id = restaurant_data.get("restaurant_id")
            ground_truth_summary = restaurant_data.get("ground_truth_summary", {})
            
            if not restaurant_id:
                logger.warning(f"레스토랑 {idx}: restaurant_id가 없습니다. 건너뜁니다.")
                continue
            
            logger.info(f"레스토랑 {idx}/{len(restaurants)}: ID={restaurant_id}")
            
            # API 호출
            result = self.summarize_reviews(restaurant_id=restaurant_id)
            
            if not result:
                logger.warning(f"레스토랑 {restaurant_id}: API 호출 실패. 건너뜁니다.")
                continue
            
            # 예측 결과 추출
            predicted_overall_summary = result.get("overall_summary", "")
            predicted_positive_aspects = result.get("positive_aspects", [])
            predicted_negative_aspects = result.get("negative_aspects", [])
            
            # Ground Truth 추출
            gt_overall_summary = ground_truth_summary.get("overall_summary", "")
            gt_positive_aspects = ground_truth_summary.get("positive_aspects", [])
            gt_negative_aspects = ground_truth_summary.get("negative_aspects", [])
            
            # ROUGE 점수 계산
            rouge_scores = self.calculate_rouge_scores(
                predicted_text=predicted_overall_summary,
                ground_truth_text=gt_overall_summary,
            )
            
            # BLEU 점수 계산
            bleu_score = self.calculate_bleu_score(
                predicted_text=predicted_overall_summary,
                ground_truth_text=gt_overall_summary,
            )
            
            # Aspect Coverage 계산
            positive_coverage = self.calculate_aspect_coverage(
                predicted_aspects=predicted_positive_aspects,
                ground_truth_aspects=gt_positive_aspects,
                aspect_type="positive",
            )
            
            negative_coverage = self.calculate_aspect_coverage(
                predicted_aspects=predicted_negative_aspects,
                ground_truth_aspects=gt_negative_aspects,
                aspect_type="negative",
            )
            
            total_rouge1 += rouge_scores["rouge1"]
            total_rouge2 += rouge_scores["rouge2"]
            total_rougeL += rouge_scores["rougeL"]
            total_bleu += bleu_score
            total_positive_coverage += positive_coverage["coverage"]
            total_negative_coverage += negative_coverage["coverage"]
            total_positive_precision += positive_coverage["precision"]
            total_negative_precision += negative_coverage["precision"]
            total_positive_recall += positive_coverage["recall"]
            total_negative_recall += negative_coverage["recall"]
            total_restaurants += 1
            
            restaurant_result = {
                "restaurant_id": restaurant_id,
                "rouge_scores": rouge_scores,
                "bleu_score": bleu_score,
                "positive_aspect_metrics": positive_coverage,
                "negative_aspect_metrics": negative_coverage,
                "predicted": {
                    "overall_summary": predicted_overall_summary[:200] + "..." if len(predicted_overall_summary) > 200 else predicted_overall_summary,
                    "positive_aspects_count": len(predicted_positive_aspects),
                    "negative_aspects_count": len(predicted_negative_aspects),
                },
                "ground_truth": {
                    "overall_summary": gt_overall_summary[:200] + "..." if len(gt_overall_summary) > 200 else gt_overall_summary,
                    "positive_aspects_count": len(gt_positive_aspects),
                    "negative_aspects_count": len(gt_negative_aspects),
                },
            }
            
            all_results.append(restaurant_result)
        
        # 평균 계산
        avg_rouge1 = total_rouge1 / total_restaurants if total_restaurants > 0 else 0.0
        avg_rouge2 = total_rouge2 / total_restaurants if total_restaurants > 0 else 0.0
        avg_rougeL = total_rougeL / total_restaurants if total_restaurants > 0 else 0.0
        avg_bleu = total_bleu / total_restaurants if total_restaurants > 0 else 0.0
        avg_positive_coverage = total_positive_coverage / total_restaurants if total_restaurants > 0 else 0.0
        avg_negative_coverage = total_negative_coverage / total_restaurants if total_restaurants > 0 else 0.0
        avg_positive_precision = total_positive_precision / total_restaurants if total_restaurants > 0 else 0.0
        avg_negative_precision = total_negative_precision / total_restaurants if total_restaurants > 0 else 0.0
        avg_positive_recall = total_positive_recall / total_restaurants if total_restaurants > 0 else 0.0
        avg_negative_recall = total_negative_recall / total_restaurants if total_restaurants > 0 else 0.0
        
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "total_restaurants": total_restaurants,
            "average_metrics": {
                "rouge1": avg_rouge1,
                "rouge2": avg_rouge2,
                "rougeL": avg_rougeL,
                "bleu": avg_bleu,
                "positive_aspect_coverage": avg_positive_coverage,
                "negative_aspect_coverage": avg_negative_coverage,
                "positive_aspect_precision": avg_positive_precision,
                "negative_aspect_precision": avg_negative_precision,
                "positive_aspect_recall": avg_positive_recall,
                "negative_aspect_recall": avg_negative_recall,
            },
            "restaurant_results": all_results,
        }
        
        return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="리뷰 요약 평가 스크립트")
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
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (선택사항, 기본값: summary_evaluation_YYYYMMDD_HHMMSS.json)"
    )
    
    args = parser.parse_args()
    
    evaluator = SummaryEvaluator(
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
    )
    
    try:
        result = evaluator.evaluate()
        
        # 결과 출력
        print("\n" + "="*80)
        print("리뷰 요약 평가 결과")
        print("="*80)
        print(f"총 레스토랑 수: {result['total_restaurants']}")
        print(f"\n평균 메트릭:")
        print(f"  - ROUGE-1: {result['average_metrics']['rouge1']:.4f}")
        print(f"  - ROUGE-2: {result['average_metrics']['rouge2']:.4f}")
        print(f"  - ROUGE-L: {result['average_metrics']['rougeL']:.4f}")
        print(f"  - BLEU: {result['average_metrics']['bleu']:.4f}")
        print(f"\n긍정 Aspect 메트릭:")
        print(f"  - Coverage: {result['average_metrics']['positive_aspect_coverage']:.4f}")
        print(f"  - Precision: {result['average_metrics']['positive_aspect_precision']:.4f}")
        print(f"  - Recall: {result['average_metrics']['positive_aspect_recall']:.4f}")
        print(f"\n부정 Aspect 메트릭:")
        print(f"  - Coverage: {result['average_metrics']['negative_aspect_coverage']:.4f}")
        print(f"  - Precision: {result['average_metrics']['negative_aspect_precision']:.4f}")
        print(f"  - Recall: {result['average_metrics']['negative_aspect_recall']:.4f}")
        
        # 결과 저장
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"summary_evaluation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
