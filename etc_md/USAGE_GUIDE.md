# 새로운 파이프라인 사용 가이드

## 환경 변수 설정

### 1. Strength Extraction - 전체 평균 비율

```bash
# 전체 데이터셋 평균 긍정 비율 설정 (선택적)
export ALL_AVERAGE_SERVICE_RATIO=0.60  # 서비스 카테고리 평균 비율
export ALL_AVERAGE_PRICE_RATIO=0.55   # 가격 카테고리 평균 비율
```

**설명**:
- 단일 음식점의 비율과 비교하여 Lift를 계산
- 배치 작업 결과를 환경 변수로 설정하거나, 나중에 캐시/DB에서 자동 로드 가능
- 기본값: service=0.60, price=0.55

### 2. Aspect Seed 파일 경로

```bash
# Aspect seed JSON 파일 경로 지정 (선택적)
export ASPECT_SEEDS_FILE=/path/to/aspect_seeds.json
```

**Aspect Seed JSON 형식**:
```json
{
  "service": [
    "직원 친절",
    "사장 친절",
    "친절 기분",
    "서비스 친절",
    "사장 직원",
    "직원 서비스",
    "아주머니 친절"
  ],
  "price": [
    "가격 대비",
    "무한 리필",
    "가격 생각",
    "음식 가격",
    "합리 가격",
    "메뉴 가격",
    "가격 만족",
    "가격 퀄리티",
    "리필 가능",
    "커피 가격",
    "가격 사악",
    "런치 가격"
  ],
  "food": [
    "가락 국수",
    "평양 냉면",
    "수제 버거",
    "크림 치즈",
    "치즈 케이크",
    "크림 파스타",
    "당근 케이크",
    "오일 파스타",
    "카페 커피",
    "비빔 냉면",
    "커피 원두",
    "리코타 치즈",
    "비빔 막국수",
    "치즈 돈가스",
    "커피 산미",
    "치즈 파스타"
  ]
}
```

**Aspect Seed 생성 방법**:
1. `hybrid_search/final_pipeline/total_aspect.py` 실행
2. 결과에서 `all_seeds_for_summary` 추출
3. JSON 파일로 저장
4. `ASPECT_SEEDS_FILE` 환경 변수로 지정

## API 사용 예시

### 1. Sentiment Analysis

```python
POST /api/sentiment/analyze
{
  "restaurant_id": 123,
  "limit": 100
}

# 응답에 neutral_count, neutral_ratio 추가됨
{
  "restaurant_id": 123,
  "positive_count": 50,
  "negative_count": 30,
  "neutral_count": 20,  # 새로 추가
  "positive_ratio": 50,
  "negative_ratio": 30,
  "neutral_ratio": 20,  # 새로 추가
  ...
}
```

### 2. Summary

```python
POST /api/llm/summarize
{
  "restaurant_id": 123,
  "limit": 10
}

# 응답
{
  "restaurant_id": 123,
  "overall_summary": "...",
  "categories": {
    "service": {
      "summary": "...",
      "bullets": ["...", "..."],
      "evidence": [
        {
          "review_id": "rev_33",
          "snippet": "...",
          "rank": 0
        }
      ]
    },
    "price": {...},
    "food": {...}
  }
}
```

### 3. Strength Extraction

```python
POST /api/llm/strength/v2
{
  "restaurant_id": 123,
  "strength_type": "distinct"
}

# 응답에 lift_percentage, all_average_ratio, single_restaurant_ratio 추가됨
{
  "restaurant_id": 123,
  "strength_type": "distinct",
  "strengths": [
    {
      "aspect": "service",
      "claim": "이 음식점은 전체 평균 대비 서비스 긍정 평가 비율이 20% 높습니다",
      "strength_type": "distinct",
      "lift_percentage": 20.0,  # 새로 추가
      "all_average_ratio": 0.60,  # 새로 추가
      "single_restaurant_ratio": 0.72,  # 새로 추가
      "distinct_score": 0.20,
      "final_score": 0.20,
      ...
    }
  ]
}
```

## 하이브리드 검색

### 현재 상태

- **Dense 벡터**: 항상 사용 가능
- **Sparse 벡터**: Qdrant 컬렉션에 추가 필요
- **폴백**: Sparse 벡터가 없으면 자동으로 Dense 검색으로 전환

### Sparse 벡터 추가 방법

1. FastEmbed로 Sparse 벡터 생성
2. Qdrant 컬렉션에 Sparse 벡터 추가
3. 컬렉션 설정에서 `sparse` 벡터 필드 활성화

자세한 내용은 `qdrant_sparse.md` 참고

## 문제 해결

### 1. Kiwi 형태소 분석기 오류

```
kiwipiepy가 설치되지 않았습니다. 단일 음식점 비율 계산이 제한됩니다.
```

**해결**:
```bash
pip install kiwipiepy
```

### 2. 하이브리드 검색 실패

```
하이브리드 검색 실패 (Sparse 벡터 없음 또는 미지원): ...
```

**해결**:
- 자동으로 Dense 검색으로 폴백됨 (정상 동작)
- Sparse 벡터를 추가하려면 Qdrant 컬렉션 설정 필요

### 3. Aspect Seed 파일 로드 실패

```
Aspect seed 파일 로드 실패: ...
```

**해결**:
- 기본값 사용 (정상 동작)
- 파일 경로 확인: `ASPECT_SEEDS_FILE` 환경 변수
- JSON 형식 확인

## 성능 최적화

### 1. 단일 음식점 비율 계산

- Kiwi 형태소 분석기는 초기화 비용이 있음
- 여러 레스토랑 처리 시 재사용 고려

### 2. 하이브리드 검색

- Sparse 벡터가 있으면 검색 품질 향상
- Sparse 벡터 생성 비용 고려

### 3. Aspect Seed

- 파일에서 로드하면 배치 작업 결과 활용 가능
- 기본값 사용 시 하드코딩된 seed 사용
