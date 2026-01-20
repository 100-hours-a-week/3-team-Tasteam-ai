#!/usr/bin/env python3
"""
감성 분석 평가 스크립트

LLM 기반 감성 분석 결과의 정확도를 평가합니다.
- 리뷰 단위 정확도: 개별 리뷰의 긍정/부정 분류 정확도
- 비율 정확도: 전체 긍정/부정 비율의 차이
- 개수 정확도: 긍정/부정 개수의 차이
- Confusion Matrix: Positive/Negative 분류 결과
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Any
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


class SentimentAnalysisEvaluator:
    """감성 분석 평가 클래스"""
    
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
    
    def analyze_sentiment(
        self,
        restaurant_id: int,
        reviews: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        감성 분석 API 호출
        
        Args:
            restaurant_id: 레스토랑 ID
            reviews: 리뷰 리스트 (선택사항)
            
        Returns:
            감성 분석 결과 딕셔너리
        """
        url = f"{self.base_url}/api/v1/sentiment/analyze"
        
        payload = {
            "restaurant_id": restaurant_id,
        }
        
        if reviews:
            payload["reviews"] = reviews
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"감성 분석 API 호출 실패: {e}")
            return {}
        except Exception as e:
            logger.error(f"감성 분석 중 오류 발생: {e}")
            return {}
    
    def calculate_review_accuracy(
        self,
        predicted_sentiments: Dict[int, str],
        ground_truth_sentiments: Dict[int, str],
    ) -> Dict[str, float]:
        """
        리뷰 단위 정확도 계산
        
        Args:
            predicted_sentiments: 예측된 감성 {review_id: "positive"/"negative"}
            ground_truth_sentiments: Ground Truth 감성 {review_id: "positive"/"negative"}
            
        Returns:
            정확도 메트릭 딕셔너리
        """
        if not predicted_sentiments or not ground_truth_sentiments:
            return {
                "accuracy": 0.0,
                "precision_positive": 0.0,
                "recall_positive": 0.0,
                "precision_negative": 0.0,
                "recall_negative": 0.0,
                "f1_positive": 0.0,
                "f1_negative": 0.0,
            }
        
        # Confusion Matrix 계산
        tp = 0  # True Positive (긍정으로 맞게 예측)
        fp = 0  # False Positive (긍정으로 잘못 예측)
        tn = 0  # True Negative (부정으로 맞게 예측)
        fn = 0  # False Negative (부정으로 잘못 예측)
        
        # Ground Truth에 있는 리뷰만 평가
        common_review_ids = set(predicted_sentiments.keys()) & set(ground_truth_sentiments.keys())
        
        for review_id in common_review_ids:
            pred = predicted_sentiments[review_id].lower()
            truth = ground_truth_sentiments[review_id].lower()
            
            if truth == "positive":
                if pred == "positive":
                    tp += 1
                else:
                    fn += 1
            else:  # truth == "negative"
                if pred == "negative":
                    tn += 1
                else:
                    fp += 1
        
        total = tp + fp + tn + fn
        if total == 0:
            return {
                "accuracy": 0.0,
                "precision_positive": 0.0,
                "recall_positive": 0.0,
                "precision_negative": 0.0,
                "recall_negative": 0.0,
                "f1_positive": 0.0,
                "f1_negative": 0.0,
            }
        
        # 정확도
        accuracy = (tp + tn) / total
        
        # Precision, Recall, F1 (Positive)
        precision_positive = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_positive = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_positive = 2 * (precision_positive * recall_positive) / (precision_positive + recall_positive) if (precision_positive + recall_positive) > 0 else 0.0
        
        # Precision, Recall, F1 (Negative)
        precision_negative = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        recall_negative = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative) if (precision_negative + recall_negative) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision_positive": precision_positive,
            "recall_positive": recall_positive,
            "precision_negative": precision_negative,
            "recall_negative": recall_negative,
            "f1_positive": f1_positive,
            "f1_negative": f1_negative,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        }
    
    def calculate_ratio_accuracy(
        self,
        predicted_ratio: float,
        ground_truth_ratio: float,
    ) -> float:
        """
        비율 정확도 계산 (절대 오차)
        
        Args:
            predicted_ratio: 예측된 비율 (%)
            ground_truth_ratio: Ground Truth 비율 (%)
            
        Returns:
            절대 오차
        """
        return abs(predicted_ratio - ground_truth_ratio)
    
    def calculate_count_accuracy(
        self,
        predicted_count: int,
        ground_truth_count: int,
    ) -> int:
        """
        개수 정확도 계산 (절대 오차)
        
        Args:
            predicted_count: 예측된 개수
            ground_truth_count: Ground Truth 개수
            
        Returns:
            절대 오차
        """
        return abs(predicted_count - ground_truth_count)
    
    def evaluate(
        self,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        감성 분석 평가 수행
        
        Args:
            debug: 디버그 모드 (리뷰 단위 감성 정보 포함)
            
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
        total_review_accuracy = 0.0
        total_positive_ratio_error = 0.0
        total_negative_ratio_error = 0.0
        total_positive_count_error = 0
        total_negative_count_error = 0
        total_restaurants = 0
        
        for idx, restaurant_data in enumerate(restaurants, 1):
            restaurant_id = restaurant_data.get("restaurant_id")
            ground_truth_reviews = restaurant_data.get("reviews", [])
            ground_truth_positive_count = restaurant_data.get("ground_truth_positive_count", 0)
            ground_truth_negative_count = restaurant_data.get("ground_truth_negative_count", 0)
            ground_truth_positive_ratio = restaurant_data.get("ground_truth_positive_ratio", 0.0)
            ground_truth_negative_ratio = restaurant_data.get("ground_truth_negative_ratio", 0.0)
            
            if not restaurant_id:
                logger.warning(f"레스토랑 {idx}: restaurant_id가 없습니다. 건너뜁니다.")
                continue
            
            logger.info(f"레스토랑 {idx}/{len(restaurants)}: ID={restaurant_id}")
            
            # Ground Truth 감성 딕셔너리 생성
            ground_truth_sentiments = {}
            for review in ground_truth_reviews:
                review_id = review.get("review_id")
                sentiment = review.get("ground_truth_sentiment", "").lower()
                if review_id and sentiment:
                    ground_truth_sentiments[review_id] = sentiment
            
            # API 호출
            reviews_for_api = [
                {
                    "id": r.get("review_id"),
                    "content": r.get("content"),
                    "restaurant_id": restaurant_id,
                }
                for r in ground_truth_reviews
            ]
            
            result = self.analyze_sentiment(
                restaurant_id=restaurant_id,
                reviews=reviews_for_api if reviews_for_api else None,
            )
            
            if not result:
                logger.warning(f"레스토랑 {restaurant_id}: API 호출 실패. 건너뜁니다.")
                continue
            
            # 예측 결과 추출
            predicted_positive_count = result.get("positive_count", 0)
            predicted_negative_count = result.get("negative_count", 0)
            predicted_positive_ratio = result.get("positive_ratio", 0.0)
            predicted_negative_ratio = result.get("negative_ratio", 0.0)
            
            # 리뷰 단위 정확도 계산 (디버그 모드에서만 가능)
            review_accuracy_metrics = {}
            if debug and "reviews" in result:
                # API가 리뷰 단위 감성을 반환하는 경우
                predicted_sentiments = {}
                for review in result.get("reviews", []):
                    review_id = review.get("id")
                    sentiment = review.get("sentiment", "").lower()
                    if review_id and sentiment:
                        predicted_sentiments[review_id] = sentiment
                
                review_accuracy_metrics = self.calculate_review_accuracy(
                    predicted_sentiments=predicted_sentiments,
                    ground_truth_sentiments=ground_truth_sentiments,
                )
                total_review_accuracy += review_accuracy_metrics.get("accuracy", 0.0)
            
            # 비율 정확도 계산
            positive_ratio_error = self.calculate_ratio_accuracy(
                predicted_ratio=predicted_positive_ratio,
                ground_truth_ratio=ground_truth_positive_ratio,
            )
            negative_ratio_error = self.calculate_ratio_accuracy(
                predicted_ratio=predicted_negative_ratio,
                ground_truth_ratio=ground_truth_negative_ratio,
            )
            
            # 개수 정확도 계산
            positive_count_error = self.calculate_count_accuracy(
                predicted_count=predicted_positive_count,
                ground_truth_count=ground_truth_positive_count,
            )
            negative_count_error = self.calculate_count_accuracy(
                predicted_count=predicted_negative_count,
                ground_truth_count=ground_truth_negative_count,
            )
            
            total_positive_ratio_error += positive_ratio_error
            total_negative_ratio_error += negative_ratio_error
            total_positive_count_error += positive_count_error
            total_negative_count_error += negative_count_error
            total_restaurants += 1
            
            restaurant_result = {
                "restaurant_id": restaurant_id,
                "predicted": {
                    "positive_count": predicted_positive_count,
                    "negative_count": predicted_negative_count,
                    "positive_ratio": predicted_positive_ratio,
                    "negative_ratio": predicted_negative_ratio,
                },
                "ground_truth": {
                    "positive_count": ground_truth_positive_count,
                    "negative_count": ground_truth_negative_count,
                    "positive_ratio": ground_truth_positive_ratio,
                    "negative_ratio": ground_truth_negative_ratio,
                },
                "errors": {
                    "positive_ratio_error": positive_ratio_error,
                    "negative_ratio_error": negative_ratio_error,
                    "positive_count_error": positive_count_error,
                    "negative_count_error": negative_count_error,
                },
            }
            
            if review_accuracy_metrics:
                restaurant_result["review_accuracy"] = review_accuracy_metrics
            
            all_results.append(restaurant_result)
        
        # 평균 계산
        avg_positive_ratio_error = total_positive_ratio_error / total_restaurants if total_restaurants > 0 else 0.0
        avg_negative_ratio_error = total_negative_ratio_error / total_restaurants if total_restaurants > 0 else 0.0
        avg_positive_count_error = total_positive_count_error / total_restaurants if total_restaurants > 0 else 0.0
        avg_negative_count_error = total_negative_count_error / total_restaurants if total_restaurants > 0 else 0.0
        avg_review_accuracy = total_review_accuracy / total_restaurants if total_restaurants > 0 and debug else None
        
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "total_restaurants": total_restaurants,
            "average_metrics": {
                "positive_ratio_error": avg_positive_ratio_error,
                "negative_ratio_error": avg_negative_ratio_error,
                "positive_count_error": avg_positive_count_error,
                "negative_count_error": avg_negative_count_error,
            },
            "restaurant_results": all_results,
        }
        
        if avg_review_accuracy is not None:
            evaluation_result["average_metrics"]["review_accuracy"] = avg_review_accuracy
        
        return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="감성 분석 평가 스크립트")
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
        help="결과 저장 파일 경로 (선택사항, 기본값: sentiment_evaluation_YYYYMMDD_HHMMSS.json)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 (리뷰 단위 정확도 계산, API에 debug=true 전달)"
    )
    
    args = parser.parse_args()
    
    evaluator = SentimentAnalysisEvaluator(
        base_url=args.base_url,
        ground_truth_path=args.ground_truth,
    )
    
    try:
        result = evaluator.evaluate(debug=args.debug)
        
        # 결과 출력
        print("\n" + "="*80)
        print("감성 분석 평가 결과")
        print("="*80)
        print(f"총 레스토랑 수: {result['total_restaurants']}")
        print(f"\n평균 메트릭:")
        print(f"  - 긍정 비율 오차: {result['average_metrics']['positive_ratio_error']:.2f}%")
        print(f"  - 부정 비율 오차: {result['average_metrics']['negative_ratio_error']:.2f}%")
        print(f"  - 긍정 개수 오차: {result['average_metrics']['positive_count_error']:.2f}")
        print(f"  - 부정 개수 오차: {result['average_metrics']['negative_count_error']:.2f}")
        
        if "review_accuracy" in result['average_metrics']:
            print(f"  - 리뷰 단위 정확도: {result['average_metrics']['review_accuracy']:.4f}")
        
        # 결과 저장
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"sentiment_evaluation_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 저장되었습니다: {output_path}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
