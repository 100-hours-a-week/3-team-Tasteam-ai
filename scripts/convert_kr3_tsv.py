#!/usr/bin/env python3
"""
kr3.tsv 파일을 프로젝트 API 형식으로 변환하는 스크립트

테스트 목적:
    # 작은 샘플 (빠른 테스트)
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_small.json --sample 100 --restaurants 5

# 중간 샘플 (배치 처리 테스트)
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_medium.json --sample 1000 --restaurants 20

# 대규모 샘플 (성능 테스트)
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_large.json --sample 10000 --restaurants 50

프로덕션 테스트:
# 전체 데이터 변환 (시간이 오래 걸릴 수 있음)
python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_production.json

# 파워로우 분포: 상위 3개 5000/2000/1000, 나머지 min=10~max=5000
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output tasteam_app_all_review_data.json --power-law --restaurants 100

# Zipf(80-20) 요청 시나리오도 함께 출력 (부하테스트 시 인기 레스토랑이 더 자주 요청되도록)
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output data.json --power-law --output-scenario scenario.txt --scenario-requests 20000

실제 서비스와 유사한 불균등 분포 (파워로우 + 핫키 + Zipf 시나리오):
# 파워로우 + 핫키 3개(5000/2000/1000) + min=10, max=5000, 100개 레스토랑
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output tasteam_app_all_review_data.json \
  --power-law --restaurants 100

# 위와 동일 + Zipf 시나리오 2만 요청을 scenario.txt로 저장
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output tasteam_app_all_review_data.json \
  --power-law --restaurants 100 \
  --output-scenario scenario.txt --scenario-requests 20000

# 핫키만 2개, 3000/1500으로 지정
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output data.json \
  --power-law --hot-top-n 2 --hot-counts 3000 1500 --min-reviews 5 --max-reviews 3000
"""

import csv
import json
import argparse
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="kr3.tsv를 프로젝트 API 형식으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
파라미터 사용 예시:
  # 전체 변환 (64만 개 리뷰)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json
  
  # 샘플링 (1000개 리뷰만)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 1000
  
  # 특정 레스토랑 수와 리뷰 수 지정
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --restaurants 10 --reviews-per-restaurant 100
  
  # 단일 레스토랑으로 그룹화
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --single-restaurant
  
  # 랜덤 시드 지정 (재현 가능한 결과)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_data.json --sample 100 --seed 123
  
규모별 테스트:  
  # 작은 샘플 (빠른 테스트)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_small.json --sample 100 --restaurants 5

  # 중간 샘플 (배치 처리 테스트)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_medium.json --sample 1000 --restaurants 20

  # 대규모 샘플 (성능 테스트)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_large.json --sample 10000 --restaurants 50

  # 전체 데이터 변환 (전체 테스트) (시간이 오래 걸릴 수 있음) (프로덕션 테스트)
  python scripts/convert_kr3_tsv.py --input kr3.tsv --output test_production.json
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 TSV 파일 경로 (kr3.tsv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="랜덤 샘플링할 리뷰 수 (기본값: 전체 변환)"
    )
    
    parser.add_argument(
        "--restaurants",
        type=int,
        default=None,
        help="생성할 레스토랑 수 (기본값: 리뷰 수에 따라 자동 결정)"
    )
    
    parser.add_argument(
        "--reviews-per-restaurant",
        type=int,
        default=None,
        help="레스토랑당 리뷰 수 (기본값: 균등 분배)"
    )
    
    parser.add_argument(
        "--single-restaurant",
        action="store_true",
        help="모든 리뷰를 단일 레스토랑으로 그룹화"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )
    # 파워로우(파레토) 분포: 레스토랑별 리뷰 수를 실제 서비스처럼 불균등하게
    parser.add_argument(
        "--power-law",
        action="store_true",
        help="레스토랑별 리뷰 수를 파워로우(파레토) 분포로 생성 (대부분 소량, 소수 핫키 다량)"
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=10,
        help="레스토랑당 리뷰 수 하한 (--power-law 시, 기본값: 10)"
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=5000,
        help="레스토랑당 리뷰 수 상한 (--power-law 시, 기본값: 5000)"
    )
    parser.add_argument(
        "--hot-top-n",
        type=int,
        default=3,
        help="상위 N개 레스토랑을 핫키로 고정 (--power-law 시, 기본값: 3)"
    )
    parser.add_argument(
        "--hot-counts",
        type=int,
        nargs="+",
        default=[5000, 2000, 1000],
        help="핫키 레스토랑별 리뷰 수 (순서대로 상위 1~N개). 예: 5000 2000 1000 (기본값)"
    )
    parser.add_argument(
        "--power-law-alpha",
        type=float,
        default=2.0,
        help="파워로우 지수 (alpha>1, 클수록 소량 레스토랑 비중 증가, 기본값: 2.0)"
    )
    # Zipf/80-20 요청 시나리오 (부하테스트 시 인기 레스토랑이 더 자주 요청되도록)
    parser.add_argument(
        "--output-scenario",
        type=str,
        default=None,
        help="Zipf(80-20)로 샘플링한 요청 시나리오 파일 출력 (한 줄에 restaurant_id 하나)"
    )
    parser.add_argument(
        "--scenario-requests",
        type=int,
        default=10000,
        help="시나리오에 넣을 요청 수 (--output-scenario 시, 기본값: 10000)"
    )
    parser.add_argument(
        "--zipf-alpha",
        type=float,
        default=1.0,
        help="Zipf 분포 지수 (alpha=1이면 80-20에 가깝게, 기본값: 1.0)"
    )
    return parser.parse_args()


def read_tsv_file(file_path: str, sample_size: Optional[int] = None, seed: int = 42) -> List[Dict[str, str]]:
    """
    TSV 파일을 읽어서 리뷰 리스트로 변환
    
    Args:
        file_path: TSV 파일 경로
        sample_size: 샘플링할 리뷰 수 (None이면 전체)
        seed: 랜덤 시드
        
    Returns:
        리뷰 딕셔너리 리스트 [{"Rating": "1", "Review": "리뷰 내용"}, ...]
    """
    reviews = []
    
    print(f"📖 TSV 파일 읽기 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            if 'Rating' not in row or 'Review' not in row:
                continue
            
            rating = row['Rating'].strip()
            review = row['Review'].strip()
            
            # 빈 리뷰 스킵
            if not review:
                continue
            
            # Rating 값 검증 (0, 1, 2만 허용)
            if rating not in ['0', '1', '2']:
                continue
            
            reviews.append({
                "Rating": rating,
                "Review": review
            })
    
    print(f"✅ 총 {len(reviews)}개 리뷰 읽기 완료")
    
    # 샘플링
    if sample_size and sample_size < len(reviews):
        random.seed(seed)
        reviews = random.sample(reviews, sample_size)
        print(f"📊 {sample_size}개 리뷰로 샘플링 완료")
    
    return reviews


def convert_to_review_model(
    review_data: Dict[str, str],
    review_id: int,
    restaurant_id: int,
    member_id: Optional[int] = None,
    created_at: Optional[str] = None
) -> Dict:
    """
    TSV 리뷰 데이터를 ReviewModel 형식으로 변환
    
    Args:
        review_data: TSV 리뷰 데이터 {"Rating": "1", "Review": "리뷰 내용"}
        review_id: 리뷰 ID
        restaurant_id: 레스토랑 ID
        member_id: 회원 ID (None이면 랜덤 생성)
        created_at: 생성 시간 (None이면 랜덤 생성)
        
    Returns:
        ReviewModel 형식 딕셔너리
    """
    rating = int(review_data['Rating'])
    review_text = review_data['Review']
    
    # is_recommended 추정: Rating 1, 2는 추천, 0은 비추천
    is_recommended = rating in [1, 2]
    
    # member_id 생성 (없으면 랜덤)
    if member_id is None:
        member_id = (review_id % 10000) + 1  # 1~10000 범위
    
    # created_at 생성 (없으면 랜덤 날짜)
    if created_at is None:
        # 최근 1년 내 랜덤 날짜
        days_ago = random.randint(0, 365)
        created_time = datetime.now() - timedelta(days=days_ago)
        created_at = created_time.isoformat()
    
    return {
        "id": review_id,
        "restaurant_id": restaurant_id,
        "member_id": member_id,
        "group_id": None,
        "subgroup_id": None,
        "content": review_text,
        "is_recommended": is_recommended,
        "created_at": created_at,
        "updated_at": None,
        "images": []
    }


def compute_target_counts_powerlaw(
    num_restaurants: int,
    total_reviews: int,
    min_reviews: int,
    max_reviews: int,
    hot_top_n: int,
    hot_counts: List[int],
    power_law_alpha: float,
    seed: int,
) -> Dict[int, int]:
    """
    파워로우(파레토) + 상위 N개 핫키 고정으로 레스토랑별 리뷰 수 목표 생성.
    목표 합이 total_reviews가 되도록 정규화.
    """
    random.seed(seed)
    if hot_top_n <= 0 or not hot_counts:
        hot_top_n = 0
        hot_counts = []
    hot_counts = list(hot_counts)[:hot_top_n]
    while len(hot_counts) < hot_top_n:
        hot_counts.append(max_reviews)  # 부족하면 상한으로 채움
    hot_sum = sum(hot_counts)
    rest_n = num_restaurants - hot_top_n
    if rest_n <= 0:
        # 전부 핫키면 합이 total_reviews 되도록 잘라서 반환
        out = {i + 1: hot_counts[i] for i in range(min(num_restaurants, len(hot_counts)))}
        s = sum(out.values())
        if s > total_reviews:
            out[1] = max(0, out.get(1, 0) + total_reviews - s)
        return out
    # 나머지 레스토랑: 파워로우 샘플. P(x) ∝ 작은 값에 치우침 -> count = min + (max-min)*u^alpha (u~U(0,1), alpha>1)
    remaining_budget = total_reviews - hot_sum
    if remaining_budget < 0:
        # 핫키 합이 이미 초과면 핫키만 줄여서 합 맞춤
        scale = total_reviews / hot_sum
        target = {i + 1: max(min_reviews, int(hot_counts[i] * scale)) for i in range(hot_top_n)}
        adj = total_reviews - sum(target.values())
        target[1] = target.get(1, 0) + adj
        return target
    # 파워로우: 대부분 min 근처, 소수만 max 근처
    raw = []
    for _ in range(rest_n):
        u = random.random()
        # u^alpha: u가 작으면 더 작은 값 -> 리뷰 수는 min에 가깝게
        x = min_reviews + (max_reviews - min_reviews) * (u ** power_law_alpha)
        raw.append(int(round(min(max(x, min_reviews), max_reviews))))
    raw_sum = sum(raw)
    if raw_sum <= 0:
        raw = [min_reviews] * rest_n
        raw_sum = rest_n * min_reviews
    # 정규화: remaining_budget에 맞추기
    if raw_sum != remaining_budget:
        scale = remaining_budget / raw_sum
        raw = [max(min_reviews, min(max_reviews, int(round(r * scale)))) for r in raw]
        delta = remaining_budget - sum(raw)
        # 나머지는 첫 번째 비핫 레스토랑에
        if delta != 0 and raw:
            raw[0] = max(min_reviews, min(max_reviews, raw[0] + delta))
    target = {}
    for i in range(hot_top_n):
        target[i + 1] = hot_counts[i]
    for i in range(rest_n):
        target[hot_top_n + 1 + i] = raw[i]
    return target


def group_reviews_by_restaurant(
    reviews: List[Dict[str, str]],
    num_restaurants: Optional[int] = None,
    reviews_per_restaurant: Optional[int] = None,
    single_restaurant: bool = False,
    seed: int = 42,
    target_counts: Optional[Dict[int, int]] = None,
) -> Dict[int, List[Dict]]:
    """
    리뷰를 레스토랑별로 그룹화
    
    Args:
        reviews: 리뷰 데이터 리스트
        num_restaurants: 레스토랑 수 (None이면 자동 결정)
        reviews_per_restaurant: 레스토랑당 리뷰 수 (None이면 균등 분배)
        single_restaurant: 단일 레스토랑으로 그룹화 여부
        seed: 랜덤 시드
        target_counts: 레스토랑별 목표 리뷰 수 {rid: count} (지정 시 이 수만큼만 분배)
        
    Returns:
        {restaurant_id: [review1, review2, ...]} 형식 딕셔너리
    """
    random.seed(seed)
    
    # 파워로우 등으로 목표 개수 지정된 경우: 그 수만큼 순서대로 할당
    if target_counts is not None and target_counts:
        shuffled = list(reviews)
        random.shuffle(shuffled)
        restaurants: Dict[int, List[Dict]] = {}
        idx = 0
        for rid in sorted(target_counts.keys()):
            cnt = target_counts[rid]
            chunk = shuffled[idx : idx + cnt]
            idx += cnt
            if chunk:
                restaurants[rid] = chunk
        if idx < len(shuffled):
            restaurants[sorted(restaurants.keys())[0]].extend(shuffled[idx:])
        return restaurants
    
    # 단일 레스토랑으로 그룹화
    if single_restaurant:
        return {1: reviews}
    
    # 레스토랑 수 자동 결정
    if num_restaurants is None:
        if reviews_per_restaurant is None:
            # 기본값: 리뷰가 100개 이하면 1개, 1000개 이하면 10개, 그 이상이면 100개
            if len(reviews) <= 100:
                num_restaurants = 1
            elif len(reviews) <= 1000:
                num_restaurants = 10
            else:
                num_restaurants = min(100, len(reviews) // 10)  # 최대 100개 레스토랑
        else:
            num_restaurants = max(1, len(reviews) // reviews_per_restaurant)
    
    # 레스토랑당 리뷰 수 자동 결정
    if reviews_per_restaurant is None:
        reviews_per_restaurant = len(reviews) // num_restaurants
        if reviews_per_restaurant == 0:
            reviews_per_restaurant = 1
    
    # 리뷰를 레스토랑별로 분배
    restaurants = {i + 1: [] for i in range(num_restaurants)}
    
    # 리뷰를 순차적으로 또는 랜덤하게 분배
    for idx, review in enumerate(reviews):
        if num_restaurants == 1:
            restaurant_id = 1
        else:
            # 균등 분배 또는 랜덤 분배
            restaurant_id = (idx % num_restaurants) + 1
        
        # 레스토랑당 최대 리뷰 수 제한
        if len(restaurants[restaurant_id]) < reviews_per_restaurant:
            restaurants[restaurant_id].append(review)
        else:
            # 최대치에 도달한 레스토랑은 다음 레스토랑에 할당
            for rid in range(1, num_restaurants + 1):
                if len(restaurants[rid]) < reviews_per_restaurant:
                    restaurants[rid].append(review)
                    break
            else:
                # 모든 레스토랑이 최대치에 도달하면 첫 번째 레스토랑에 추가
                restaurants[1].append(review)
    
    # 빈 레스토랑 제거
    restaurants = {rid: reviews for rid, reviews in restaurants.items() if reviews}
    
    return restaurants


def convert_to_api_format(
    restaurants: Dict[int, List[Dict[str, str]]],
    seed: int = 42
) -> Dict:
    """
    레스토랑별 리뷰를 API 요청 형식으로 변환
    
    Args:
        restaurants: {restaurant_id: [review1, review2, ...]} 형식 딕셔너리
        seed: 랜덤 시드
        
    Returns:
        SentimentAnalysisBatchRequest 형식 딕셔너리
    """
    random.seed(seed)
    
    api_restaurants = []
    review_id_counter = 1
    
    for restaurant_id, review_list in sorted(restaurants.items()):
        # 리뷰를 ReviewModel 형식으로 변환
        converted_reviews = []
        
        for review_data in review_list:
            review_model = convert_to_review_model(
                review_data=review_data,
                review_id=review_id_counter,
                restaurant_id=restaurant_id,
                member_id=None,  # 자동 생성
                created_at=None  # 자동 생성
            )
            converted_reviews.append(review_model)
            review_id_counter += 1
        
        # 레스토랑 정보 추가
        api_restaurants.append({
            "restaurant_id": restaurant_id,
            "restaurant_name": f"Test Restaurant {restaurant_id}",
            "reviews": converted_reviews
        })
    
    # API 요청 형식
    api_format = {
        "restaurants": api_restaurants,
        "max_tokens_per_batch": 4000  # 기본값
    }
    
    return api_format


def generate_zipf_scenario(
    restaurant_ids: List[int],
    num_requests: int,
    zipf_alpha: float,
    seed: int,
) -> List[int]:
    """
    Zipf(80-20) 분포로 요청 시나리오 생성. rank 1(첫 번째 ID)이 가장 자주 나옴.
    """
    random.seed(seed)
    n = len(restaurant_ids)
    if n == 0:
        return []
    # weight[k] ∝ 1/(k+1)^alpha (rank 1 = index 0가 가장 큼)
    weights = [1.0 / ((i + 1) ** zipf_alpha) for i in range(n)]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return random.choices(restaurant_ids, weights=weights, k=num_requests)


def main():
    """메인 함수"""
    args = parse_args()
    
    # 입력 파일 존재 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 오류: 입력 파일이 존재하지 않습니다: {args.input}")
        return 1
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("kr3.tsv → 프로젝트 API 형식 변환 스크립트")
    print("=" * 60)
    print()
    
    # 1. TSV 파일 읽기
    reviews = read_tsv_file(
        file_path=args.input,
        sample_size=args.sample,
        seed=args.seed
    )
    
    if not reviews:
        print("❌ 오류: 읽은 리뷰가 없습니다.")
        return 1
    
    # 2. 레스토랑별로 그룹화
    print("\n🏪 레스토랑별로 그룹화 중...")
    target_counts = None
    if args.power_law:
        # 파워로우: 레스토랑 수 결정
        num_restaurants = args.restaurants
        if num_restaurants is None:
            if len(reviews) <= 100:
                num_restaurants = 1
            elif len(reviews) <= 1000:
                num_restaurants = 10
            else:
                num_restaurants = min(100, len(reviews) // 10)
        hot_top_n = min(args.hot_top_n, num_restaurants)
        hot_counts = (args.hot_counts or [5000, 2000, 1000])[:hot_top_n]
        while len(hot_counts) < hot_top_n:
            hot_counts.append(args.max_reviews)
        target_counts = compute_target_counts_powerlaw(
            num_restaurants=num_restaurants,
            total_reviews=len(reviews),
            min_reviews=args.min_reviews,
            max_reviews=args.max_reviews,
            hot_top_n=hot_top_n,
            hot_counts=hot_counts,
            power_law_alpha=args.power_law_alpha,
            seed=args.seed,
        )
        total_alloc = sum(target_counts.values())
        if total_alloc > len(reviews):
            # 정규화 후에도 초과할 수 있음: 잘라서 맞춤
            scale = len(reviews) / total_alloc
            new_counts = {}
            rem = len(reviews)
            for rid in sorted(target_counts.keys()):
                c = max(1, int(round(target_counts[rid] * scale)))
                c = min(c, rem)
                if c > 0:
                    new_counts[rid] = c
                    rem -= c
            target_counts = new_counts
        print(f"   파워로우 분포: 상위 {hot_top_n}개 핫키 {hot_counts}, 나머지 min={args.min_reviews}~max={args.max_reviews}")
    
    restaurants = group_reviews_by_restaurant(
        reviews=reviews,
        num_restaurants=args.restaurants,
        reviews_per_restaurant=args.reviews_per_restaurant,
        single_restaurant=args.single_restaurant,
        seed=args.seed,
        target_counts=target_counts,
    )
    
    print(f"✅ {len(restaurants)}개 레스토랑 생성 완료")
    for rid, review_list in sorted(restaurants.items()):
        print(f"   - Restaurant {rid}: {len(review_list)}개 리뷰")
    
    # 3. API 형식으로 변환
    print("\n🔄 API 형식으로 변환 중...")
    api_format = convert_to_api_format(
        restaurants=restaurants,
        seed=args.seed
    )
    
    # 4. JSON 파일로 저장
    print(f"\n💾 JSON 파일 저장 중: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(api_format, f, ensure_ascii=False, indent=2)
    
    # 5. Zipf 요청 시나리오 출력 (80-20 부하 시뮬레이션용)
    if args.output_scenario:
        rids = sorted(restaurants.keys())
        scenario = generate_zipf_scenario(
            restaurant_ids=rids,
            num_requests=args.scenario_requests,
            zipf_alpha=args.zipf_alpha,
            seed=args.seed,
        )
        scenario_path = Path(args.output_scenario)
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scenario_path, "w", encoding="utf-8") as f:
            for rid in scenario:
                f.write(f"{rid}\n")
        print(f"📋 Zipf 시나리오 저장: {args.output_scenario} ({len(scenario)} 요청, alpha={args.zipf_alpha})")
    
    # 6. 통계 정보 출력
    total_reviews = sum(len(r['reviews']) for r in api_format['restaurants'])
    
    print()
    print("=" * 60)
    print("✅ 변환 완료!")
    print("=" * 60)
    print(f"📊 통계:")
    print(f"   - 레스토랑 수: {len(api_format['restaurants'])}")
    print(f"   - 총 리뷰 수: {total_reviews}")
    print(f"   - 레스토랑당 평균 리뷰 수: {total_reviews / len(api_format['restaurants']):.1f}")
    print(f"   - 출력 파일: {args.output}")
    if args.output_scenario:
        print(f"   - 시나리오 파일: {args.output_scenario}")
    print()
    print("📝 사용 예시:")
    print(f'   curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f'        -d @{args.output}')
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())

