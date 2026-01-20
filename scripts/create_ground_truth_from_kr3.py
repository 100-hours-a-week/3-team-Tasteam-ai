#!/usr/bin/env python3
"""
kr3.tsv 파일을 기반으로 Ground Truth 파일들을 생성하는 스크립트

사용 방법:
    python scripts/create_ground_truth_from_kr3.py --input kr3.tsv --restaurants 2 --reviews-per-restaurant 50
"""

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import random


def read_kr3_tsv(file_path: str, num_restaurants: int = 2, reviews_per_restaurant: int = 50, seed: int = 42) -> Dict[int, List[Dict]]:
    """kr3.tsv 파일을 읽어서 레스토랑별로 그룹화"""
    random.seed(seed)
    restaurants = defaultdict(list)
    all_reviews = []
    
    print(f"📖 kr3.tsv 파일 읽기 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if 'Rating' not in row or 'Review' not in row:
                continue
            
            rating = row['Rating'].strip()
            review = row['Review'].strip()
            
            if not review or rating not in ['0', '1', '2']:
                continue
            
            all_reviews.append({
                'rating': rating,
                'content': review
            })
    
    print(f"✅ 총 {len(all_reviews)}개 리뷰 읽기 완료")
    
    # 샘플링
    sample_size = num_restaurants * reviews_per_restaurant
    if sample_size < len(all_reviews):
        all_reviews = random.sample(all_reviews, sample_size)
        print(f"📊 {sample_size}개 리뷰로 샘플링 완료")
    
    # 레스토랑별로 분배
    review_id = 1
    for i, review_data in enumerate(all_reviews):
        restaurant_id = (i // reviews_per_restaurant) + 1
        if restaurant_id > num_restaurants:
            break
        
        restaurants[restaurant_id].append({
            'review_id': review_id,
            'rating': review_data['rating'],
            'content': review_data['content']
        })
        review_id += 1
    
    print(f"✅ {len(restaurants)}개 레스토랑 생성 완료")
    for rid, reviews in sorted(restaurants.items()):
        pos = sum(1 for r in reviews if r['rating'] in ['1', '2'])
        neg = sum(1 for r in reviews if r['rating'] == '0')
        neu = sum(1 for r in reviews if r['rating'] == '2')
        print(f"   - Restaurant {rid}: {len(reviews)}개 리뷰 (긍정:{pos}, 부정:{neg}, 중립:{neu})")
    
    return restaurants


def generate_sentiment_ground_truth(restaurants: Dict[int, List[Dict]]) -> Dict:
    """감성 분석 Ground Truth 생성"""
    result = {"restaurants": []}
    
    for restaurant_id, reviews in sorted(restaurants.items()):
        positive_reviews = [r for r in reviews if r['rating'] in ['1', '2']]
        negative_reviews = [r for r in reviews if r['rating'] == '0']
        
        positive_count = len(positive_reviews)
        negative_count = len(negative_reviews)
        total_count = len(reviews)
        
        positive_ratio = (positive_count / total_count * 100) if total_count > 0 else 0
        negative_ratio = (negative_count / total_count * 100) if total_count > 0 else 0
        
        restaurant_data = {
            "restaurant_id": restaurant_id,
            "reviews": [
                {
                    "review_id": r['review_id'],
                    "content": r['content'],
                    "ground_truth_sentiment": "positive" if r['rating'] in ['1', '2'] else "negative"
                }
                for r in reviews
            ],
            "ground_truth_positive_count": positive_count,
            "ground_truth_negative_count": negative_count,
            "ground_truth_positive_ratio": round(positive_ratio, 1),
            "ground_truth_negative_ratio": round(negative_ratio, 1)
        }
        
        result["restaurants"].append(restaurant_data)
    
    return result


def generate_summary_ground_truth(restaurants: Dict[int, List[Dict]]) -> Dict:
    """리뷰 요약 Ground Truth 생성"""
    result = {"restaurants": []}
    
    for restaurant_id, reviews in sorted(restaurants.items()):
        positive_reviews = [r for r in reviews if r['rating'] in ['1', '2']]
        negative_reviews = [r for r in reviews if r['rating'] == '0']
        
        positive_aspects = []
        negative_aspects = []
        
        # 긍정 aspect 추출
        taste_positive = [r for r in positive_reviews if any(kw in r['content'] for kw in ['맛', '맛있', '맛나', '맛집', '맛있다', '맛있어', '맛있었'])]
        if taste_positive:
            quotes = []
            for r in taste_positive[:3]:
                quote = r['content'][:50].strip()
                if len(r['content']) > 50:
                    quote += "..."
                quotes.append(quote)
            
            positive_aspects.append({
                "aspect": "맛",
                "claim": "맛에 대한 긍정 언급이 많음",
                "evidence_quotes": quotes,
                "evidence_review_ids": [r['review_id'] for r in taste_positive[:3]]
            })
        
        amount_positive = [r for r in positive_reviews if any(kw in r['content'] for kw in ['양', '푸짐', '많', '양이', '양도'])]
        if amount_positive:
            quotes = []
            for r in amount_positive[:3]:
                quote = r['content'][:50].strip()
                if len(r['content']) > 50:
                    quote += "..."
                quotes.append(quote)
            
            positive_aspects.append({
                "aspect": "양",
                "claim": "양에 대한 긍정 언급이 많음",
                "evidence_quotes": quotes,
                "evidence_review_ids": [r['review_id'] for r in amount_positive[:3]]
            })
        
        service_positive = [r for r in positive_reviews if any(kw in r['content'] for kw in ['친절', '서비스', '직원', '직원분'])]
        if service_positive:
            quotes = []
            for r in service_positive[:3]:
                quote = r['content'][:50].strip()
                if len(r['content']) > 50:
                    quote += "..."
                quotes.append(quote)
            
            positive_aspects.append({
                "aspect": "서비스",
                "claim": "서비스가 친절하다는 언급이 많음",
                "evidence_quotes": quotes,
                "evidence_review_ids": [r['review_id'] for r in service_positive[:3]]
            })
        
        # 부정 aspect 추출
        taste_negative = [r for r in negative_reviews if any(kw in r['content'] for kw in ['맛', '맛없', '별로', '맛이', '맛도'])]
        if taste_negative:
            quotes = []
            for r in taste_negative[:3]:
                quote = r['content'][:50].strip()
                if len(r['content']) > 50:
                    quote += "..."
                quotes.append(quote)
            
            negative_aspects.append({
                "aspect": "맛",
                "claim": "맛에 대한 부정 언급이 있음",
                "evidence_quotes": quotes,
                "evidence_review_ids": [r['review_id'] for r in taste_negative[:3]]
            })
        
        # overall_summary 생성
        summary_parts = []
        if positive_aspects:
            aspects = list(set([a['aspect'] for a in positive_aspects]))
            summary_parts.append(f"{', '.join(aspects)}에 대한 긍정 언급이 많고")
        if negative_aspects:
            aspects = list(set([a['aspect'] for a in negative_aspects]))
            summary_parts.append(f"{', '.join(aspects)} 관련 불만도 있습니다")
        
        overall_summary = ". ".join(summary_parts) + "." if summary_parts else "리뷰 요약이 필요합니다."
        
        restaurant_data = {
            "restaurant_id": restaurant_id,
            "ground_truth_summary": {
                "overall_summary": overall_summary,
                "positive_aspects": positive_aspects,
                "negative_aspects": negative_aspects
            }
        }
        
        result["restaurants"].append(restaurant_data)
    
    return result


def generate_strength_ground_truth(restaurants: Dict[int, List[Dict]]) -> Dict:
    """강점 추출 Ground Truth 생성"""
    result = {"restaurants": []}
    
    for restaurant_id, reviews in sorted(restaurants.items()):
        positive_reviews = [r for r in reviews if r['rating'] in ['1', '2']]
        
        representative_strengths = []
        distinct_strengths = []
        
        # 맛 관련 강점
        taste_reviews = [r for r in positive_reviews if any(kw in r['content'] for kw in ['맛', '맛있', '맛나', '맛집', '맛있다', '맛있어'])]
        if taste_reviews and len(taste_reviews) >= 3:
            strength = {
                "aspect": "맛",
                "claim": "맛에 대한 긍정 언급이 많음",
                "evidence_review_ids": [r['review_id'] for r in taste_reviews[:5]],
                "support_count": min(len(taste_reviews), 5),
                "type": "representative"
            }
            representative_strengths.append(strength)
            
            distinct_strength = strength.copy()
            distinct_strength["type"] = "distinct"
            distinct_strength["distinct_score"] = 0.85
            distinct_strengths.append(distinct_strength)
        
        # 양 관련 강점
        amount_reviews = [r for r in positive_reviews if any(kw in r['content'] for kw in ['양', '푸짐', '많', '양이', '양도'])]
        if amount_reviews and len(amount_reviews) >= 3:
            strength = {
                "aspect": "양",
                "claim": "양에 대한 긍정 언급이 많음",
                "evidence_review_ids": [r['review_id'] for r in amount_reviews[:5]],
                "support_count": min(len(amount_reviews), 5),
                "type": "representative"
            }
            representative_strengths.append(strength)
            
            distinct_strength = strength.copy()
            distinct_strength["type"] = "distinct"
            distinct_strength["distinct_score"] = 0.80
            distinct_strengths.append(distinct_strength)
        
        # 서비스 관련 강점
        service_reviews = [r for r in positive_reviews if any(kw in r['content'] for kw in ['친절', '서비스', '직원', '직원분'])]
        if service_reviews and len(service_reviews) >= 2:
            strength = {
                "aspect": "서비스",
                "claim": "서비스가 친절하다는 언급이 많음",
                "evidence_review_ids": [r['review_id'] for r in service_reviews[:3]],
                "support_count": min(len(service_reviews), 3),
                "type": "representative"
            }
            representative_strengths.append(strength)
            
            distinct_strength = strength.copy()
            distinct_strength["type"] = "distinct"
            distinct_strength["distinct_score"] = 0.75
            distinct_strengths.append(distinct_strength)
        
        restaurant_data = {
            "restaurant_id": restaurant_id,
            "comparison_restaurant_ids": [rid for rid in sorted(restaurants.keys()) if rid != restaurant_id][:3],
            "ground_truth_strengths": {
                "representative": representative_strengths,
                "distinct": distinct_strengths
            }
        }
        
        result["restaurants"].append(restaurant_data)
    
    return result


def generate_vector_search_ground_truth(restaurants: Dict[int, List[Dict]]) -> Dict:
    """벡터 검색 Ground Truth 생성"""
    result = {"queries": []}
    
    for restaurant_id, reviews in sorted(restaurants.items()):
        positive_reviews = [r for r in reviews if r['rating'] in ['1', '2']]
        negative_reviews = [r for r in reviews if r['rating'] == '0']
        
        if positive_reviews:
            result["queries"].append({
                "query": "맛있다 좋다 만족",
                "restaurant_id": restaurant_id,
                "relevant_review_ids": [r['review_id'] for r in positive_reviews[:5]]
            })
        
        if negative_reviews:
            result["queries"].append({
                "query": "맛없다 별로 불만",
                "restaurant_id": restaurant_id,
                "relevant_review_ids": [r['review_id'] for r in negative_reviews[:3]]
            })
        
        service_reviews = [r for r in reviews if '친절' in r['content'] or '서비스' in r['content']]
        if service_reviews:
            result["queries"].append({
                "query": "서비스 친절하다",
                "restaurant_id": restaurant_id,
                "relevant_review_ids": [r['review_id'] for r in service_reviews[:5]]
            })
        
        price_reviews = [r for r in reviews if any(kw in r['content'] for kw in ['가격', '비싸', '저렴'])]
        if price_reviews:
            result["queries"].append({
                "query": "가격 합리적",
                "restaurant_id": restaurant_id,
                "relevant_review_ids": [r['review_id'] for r in price_reviews[:5]]
            })
    
    return result


def main():
    parser = argparse.ArgumentParser(description="kr3.tsv 기반 Ground Truth 파일 생성")
    parser.add_argument("--input", type=str, default="kr3.tsv", help="입력 TSV 파일 경로")
    parser.add_argument("--restaurants", type=int, default=2, help="생성할 레스토랑 수")
    parser.add_argument("--reviews-per-restaurant", type=int, default=50, help="레스토랑당 리뷰 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output-dir", type=str, default="scripts", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("kr3.tsv 기반 Ground Truth 파일 생성")
    print("=" * 60)
    print()
    
    # 1. kr3.tsv 읽기
    restaurants = read_kr3_tsv(
        file_path=args.input,
        num_restaurants=args.restaurants,
        reviews_per_restaurant=args.reviews_per_restaurant,
        seed=args.seed
    )
    
    if not restaurants:
        print("❌ 오류: 레스토랑 데이터가 없습니다.")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 각 Ground Truth 파일 생성
    print("\n📝 Ground Truth 파일 생성 중...")
    
    # 2.1 Sentiment
    print("\n1. Ground_truth_sentiment.json 생성 중...")
    sentiment_gt = generate_sentiment_ground_truth(restaurants)
    with open(output_dir / "Ground_truth_sentiment.json", 'w', encoding='utf-8') as f:
        json.dump(sentiment_gt, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(sentiment_gt['restaurants'])}개 레스토랑 데이터 생성 완료")
    
    # 2.2 Summary
    print("\n2. Ground_truth_summary.json 생성 중...")
    summary_gt = generate_summary_ground_truth(restaurants)
    with open(output_dir / "Ground_truth_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary_gt, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(summary_gt['restaurants'])}개 레스토랑 데이터 생성 완료")
    print("   ⚠️  주의: overall_summary와 aspect claim은 수동 검수 및 수정이 필요합니다.")
    
    # 2.3 Strength
    print("\n3. Ground_truth_strength.json 생성 중...")
    strength_gt = generate_strength_ground_truth(restaurants)
    with open(output_dir / "Ground_truth_strength.json", 'w', encoding='utf-8') as f:
        json.dump(strength_gt, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(strength_gt['restaurants'])}개 레스토랑 데이터 생성 완료")
    print("   ⚠️  주의: claim과 distinct_score는 수동 검수 및 수정이 필요합니다.")
    
    # 2.4 Vector Search
    print("\n4. Ground_truth_vector_search.json 생성 중...")
    vector_gt = generate_vector_search_ground_truth(restaurants)
    with open(output_dir / "Ground_truth_vector_search.json", 'w', encoding='utf-8') as f:
        json.dump(vector_gt, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(vector_gt['queries'])}개 쿼리 데이터 생성 완료")
    print("   ⚠️  주의: relevant_review_ids는 수동 검수 및 관련도 순서 정렬이 필요합니다.")
    
    print("\n" + "=" * 60)
    print("✅ 모든 Ground Truth 파일 생성 완료!")
    print("=" * 60)
    print("\n⚠️  중요: 생성된 파일들은 자동 생성된 샘플 데이터입니다.")
    print("   실제 평가를 위해서는 수동 검수 및 수정이 필요합니다:")
    print("   - Summary: overall_summary, aspect claim 정확성 확인")
    print("   - Strength: claim 정확성, distinct_score 계산 확인")
    print("   - Vector Search: relevant_review_ids 관련도 순서 확인")
    
    return 0


if __name__ == "__main__":
    exit(main())
