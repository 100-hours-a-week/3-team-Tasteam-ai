# 기존 파이프라인 vs 새로운 파이프라인 비교 문서

## 목차
1. [개요](#개요)
2. [Sentiment Analysis (감성 분석)](#sentiment-analysis-감성-분석)
3. [Summary (요약)](#summary-요약)
4. [Strength Extraction (강점 추출)](#strength-extraction-강점-추출)
5. [종합 비교](#종합-비교)
6. [마이그레이션 전략](#마이그레이션-전략)

---

## 개요

본 문서는 기존 파이프라인과 새로운 파이프라인의 차이점을 상세히 비교합니다.

### 기존 파이프라인 위치
- `src/sentiment_analysis.py` - 감성 분석
- `src/api/routers/llm.py` - 요약
- `src/strength_extraction.py` - 강점 추출

### 새로운 파이프라인 위치
- `hybrid_search/final_pipeline/final_sentiment_pipeline.py` - 감성 분석
- `hybrid_search/final_pipeline/final_summary_pipeline.py` - 요약
- `hybrid_search/final_pipeline/total_aspect.py` + `aspect_positive_filtering.py` + `total_data_aspect_test.py` - 강점 추출

---

## Sentiment Analysis (감성 분석)

### 기존 파이프라인

**파일**: `src/sentiment_analysis.py`

**프로세스**:
1. HuggingFace sentiment 모델로 전체 리뷰 분류
2. 대표 벡터 기반 TOP-K 리뷰만 선택하여 분석
3. Positive/Negative 비율 계산

**특징**:
- 대표 벡터 기반 샘플링으로 효율성 향상
- Neutral 처리 없음 (Positive/Negative만)
- 모든 리뷰를 모델에 전달

**API 응답 형식**:
```json
{
  "restaurant_id": 1,
  "positive_count": 3,
  "negative_count": 2,
  "total_count": 5,
  "positive_ratio": 60,
  "negative_ratio": 40
}
```

**장점**:
- 단순하고 빠른 처리
- 대표 벡터 기반으로 효율적

**단점**:
- Neutral 리뷰 처리 불가
- 모든 리뷰를 모델에 전달 (비효율적)

---

### 새로운 파이프라인

**파일**: `hybrid_search/final_pipeline/final_sentiment_pipeline.py`

**프로세스**:
1. HuggingFace 모델로 1차 분류
   - `Dilwolf/Kakao_app-kr_sentiment` 모델 사용
   - Positive score > 0.8이면 positive, 아니면 negative로 1차 분류
2. Negative로 분류된 리뷰만 LLM으로 재판정
   - GPT-4o-mini 사용
   - Positive/Negative/Neutral 중 하나로 재분류
3. Positive/Negative/Neutral 비율 계산

**특징**:
- Negative만 LLM 재판정으로 비용 절감
- Neutral 처리 가능
- 2단계 하이브리드 접근

**API 응답 형식** (예상):
```json
{
  "restaurant_id": 4,
  "positive_count": 3,
  "negative_count": 1,
  "neutral_count": 1,
  "total_count": 5,
  "positive_rate": 0.75,
  "negative_rate": 0.25,
  "neutral_rate": 0.20
}
```

**장점**:
- Neutral 처리로 정확도 향상
- Negative만 LLM 재판정으로 비용 절감
- 더 정확한 감성 분류

**단점**:
- LLM 호출로 인한 지연 시간 증가
- API 응답 형식 변경 필요

---

### 비교 요약

| 항목 | 기존 | 새로운 |
|------|------|--------|
| **분류 방식** | HuggingFace 모델만 | HuggingFace + LLM 하이브리드 |
| **Neutral 처리** | ❌ 없음 | ✅ 있음 |
| **LLM 사용** | 없음 | Negative만 재판정 |
| **비용** | 낮음 | 중간 (Negative만 LLM) |
| **정확도** | 중간 | 높음 |
| **처리 속도** | 빠름 | 중간 |

---

## Summary (요약)

### 기존 파이프라인

**파일**: `src/api/routers/llm.py`

**프로세스**:
1. 대표 벡터 기반 TOP-K 리뷰 검색
2. Dense 벡터 검색만 사용
3. 단일 쿼리로 전체 리뷰 검색
4. LLM이 aspect 기반으로 긍정/부정 요약 생성

**특징**:
- 대표 벡터 기반으로 효율적
- Dense 벡터 검색만 사용
- Aspect 기반 구조화된 요약

**API 응답 형식**:
```json
{
  "restaurant_id": 1,
  "overall_summary": "전체 요약",
  "positive_aspects": [
    {
      "aspect": "맛",
      "claim": "맛이 좋다는 언급이 많음",
      "evidence_quotes": ["맛있어요", "맛이 좋아요"],
      "evidence_review_ids": ["rev_1", "rev_2"]
    }
  ],
  "negative_aspects": [...],
  "positive_count": 10,
  "negative_count": 5
}
```

**장점**:
- 단순하고 빠른 처리
- Aspect 기반 구조화

**단점**:
- Dense 검색만 사용 (정확도 제한)
- Aspect seed가 고정되어 있지 않음

---

### 새로운 파이프라인

**파일**: `hybrid_search/final_pipeline/final_summary_pipeline.py`

**프로세스**:
1. **하이브리드 검색** (Dense + Sparse)
   - Dense: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
   - Sparse: `Qdrant/bm25`
   - RRF (Reciprocal Rank Fusion) 방식으로 결합
2. **Aspect 기반 카테고리별 검색**
   - Service: "직원 친절", "사장 친절" 등 seed 사용
   - Price: "가격 대비", "무한 리필" 등 seed 사용
   - Food: "가락 국수", "평양 냉면" 등 seed 사용
3. **전체 데이터셋 기반 seed 생성**
   - `total_aspect.py`에서 전체 데이터셋 분석
   - 빈도 기반으로 대표 seed 추출
4. **카테고리별 요약 생성**
   - 각 카테고리별로 검색된 리뷰를 LLM에 전달
   - LLM이 카테고리별 summary, bullets, evidence 인덱스 생성
   - Evidence 인덱스를 실제 evidence 객체로 변환
     - `review_id`: 검색 결과에서 추출한 실제 리뷰 ID
     - `snippet`: 실제 리뷰 텍스트
     - `rank`: top_k 내 인덱스 (0부터 시작)

**특징**:
- 하이브리드 검색으로 정확도 향상
- 전체 데이터셋 기반 seed로 더 정확한 검색
- 카테고리별 구조화된 요약

**API 응답 형식**:
```json
{
  "restaurant_id": 1,
  "service": {
    "summary": "서비스 요약",
    "bullets": ["친절하다", "응대가 빠르다"],
    "evidence": [
      {
        "review_id": "rev_33",
        "snippet": "직원이 매우 친절하게 응대해주셨어요",
        "rank": 0
      },
      {
        "review_id": "rev_47",
        "snippet": "사장님이 직접 서빙해주시고 친절하세요",
        "rank": 1
      }
    ]
  },
  "price": {
    "summary": "가격 요약",
    "bullets": ["가성비 좋다", "리필 가능"],
    "evidence": [
      {
        "review_id": "rev_12",
        "snippet": "가격 대비 양이 많아서 만족스러워요",
        "rank": 0
      }
    ]
  },
  "food": {
    "summary": "음식 요약",
    "bullets": ["맛있다", "양이 많다"],
    "evidence": [
      {
        "review_id": "rev_25",
        "snippet": "라멘 국물이 진하고 맛있어요",
        "rank": 0
      }
    ]
  },
  "overall_summary": {
    "summary": "전체 요약"
  }
}
```

**Evidence 형식**:
- `review_id`: 실제 리뷰 ID (문자열)
- `snippet`: 실제 리뷰 텍스트
- `rank`: top_k 내 인덱스 (0부터 시작)

**장점**:
- 하이브리드 검색으로 정확도 향상
- 전체 데이터셋 기반 seed로 더 정확한 검색
- 카테고리별 구조화된 요약
- Evidence 형식 개선: review_id, snippet, rank를 포함한 구조화된 객체 배열

**단점**:
- 하이브리드 검색 인프라 구축 필요
- 전체 데이터셋 분석 필요 (배치 작업)
- API 응답 형식 변경 필요

---

### 비교 요약

| 항목 | 기존 | 새로운 |
|------|------|--------|
| **검색 방식** | Dense 벡터만 | Dense + Sparse 하이브리드 |
| **Seed 생성** | 없음 (고정 쿼리) | 전체 데이터셋 기반 동적 생성 |
| **카테고리** | 긍정/부정 | Service/Price/Food |
| **Evidence 형식** | 인덱스 배열 + 별도 필드 | 객체 배열 (review_id, snippet, rank) |
| **검색 정확도** | 중간 | 높음 |
| **인프라 요구사항** | 낮음 | 높음 (Sparse 벡터) |
| **배치 작업** | 불필요 | 필요 (전체 데이터셋 분석) |

---

## Strength Extraction (강점 추출)

### 기존 파이프라인

**파일**: `src/strength_extraction.py`

**프로세스 (Step A~H)**:

#### Step A: 타겟 긍정 근거 후보 수집
- 대표 벡터 기반 TOP-K 선택 (대표성)
- 최근 리뷰 추가 (최신성)
- 랜덤 샘플링 추가 (다양성)

#### Step B: LLM으로 강점 후보 생성 ⭐ 핵심
- **입력**: 리뷰 텍스트 리스트
- **출력**: 구조화된 강점 후보
  ```json
  {
    "aspect": "불맛",
    "claim": "불맛이 좋다",
    "type": "specific",
    "confidence": 0.9,
    "evidence_quotes": ["숫불향이 진해서 맛있어요"],
    "evidence_review_ids": ["rev_1", "rev_5"]
  }
  ```

#### Step C: Qdrant 벡터 검색으로 근거 확장/검증
- Aspect별로 벡터 검색
- Support_count 계산 (유효 근거 수)
- Consistency, recency 가중치 계산

#### Step D: 의미 중복 제거
- Connected Components (Union-Find)
- 유사한 강점 병합

#### Step D-1: Claim 후처리 재생성
- 템플릿 보정 + LLM

#### Step E~H: 벡터 유사도 기반 차별점 계산
- **비교군**: 유사 레스토랑 (최대 20개)
- **계산 방식**: 타겟 aspect 벡터 vs 비교군 aspect 벡터 유사도
- **차별점**: `distinct = 1 - max_sim`
- **최종 점수**: `rep × (1 + alpha × distinct)`

**특징**:
- LLM이 리뷰를 읽고 강점을 추출
- 구체적인 claim과 evidence 제공
- 벡터 유사도 기반 차별점 계산

**API 응답 형식**:
```json
{
  "restaurant_id": 1,
  "strength_type": "both",
  "strengths": [
    {
      "aspect": "불맛",
      "claim": "불맛이 좋다는 언급이 많음",
      "strength_type": "distinct",
      "support_count": 15,
      "support_ratio": 0.75,
      "distinct_score": 0.75,
      "closest_competitor_sim": 0.25,
      "closest_competitor_id": 5,
      "evidence": [
        {
          "review_id": "rev_1",
          "snippet": "숫불향이 진해서 맛있어요",
          "rating": 5.0,
          "created_at": "2024-01-01"
        }
      ],
      "final_score": 8.5
    }
  ],
  "total_candidates": 300,
  "validated_count": 10
}
```

**장점**:
- 구체적인 강점 표현 (claim, evidence)
- 의미적 차별점 계산
- 유연한 aspect 추출

**단점**:
- LLM 호출 비용/시간 증가
- LLM 환각 가능성
- 비교군 구성 복잡도

---

### 새로운 파이프라인

**파일**: 
- `hybrid_search/final_pipeline/total_aspect.py` - 전체 데이터셋 분석
- `hybrid_search/final_pipeline/aspect_positive_filtering.py` - 긍정 aspect 필터링
- `hybrid_search/final_pipeline/total_data_aspect_test.py` - 통합 실행

**프로세스**:

#### 1. 전체 데이터셋 분석 (배치 작업)
- `total_aspect.py`: 전체 데이터셋에서 aspect bigram 추출
  - PySpark + Kiwi 형태소 분석
  - 명사 bigram 생성 및 빈도 계산
  - 카테고리 분류 (service/price/food/other)
  - Quantile 분할 (head/mid/tail)
  - Seed 선택 (head 전체 + mid 가중 샘플링 + tail 랜덤)
- `aspect_positive_filtering.py`: 긍정 aspect만 필터링
  - Service: "친절" 토큰 포함만
  - Price: "가성비", "가격 합리", "가격 만족" 포함만
- 전체 평균 긍정 비율 계산
  - `all_serv_positive_ratio = 긍정_service_count / 전체_service_count`
  - `all_price_positive_ratio = 긍정_price_count / 전체_price_count`

#### 2. 단일 음식점 분석 (실시간)
- 동일한 방식으로 단일 음식점의 카테고리별 긍정 비율 계산
  - `store_service_ratio = 0.72`
  - `store_price_ratio = 0.65`

#### 3. Lift 계산 (통계적 차별점)
- `service_lift = (store_service_ratio - all_service_ratio) / all_service_ratio × 100`
- `price_lift = (store_price_ratio - all_price_ratio) / all_price_ratio × 100`
- 예시: `(0.72 - 0.60) / 0.60 × 100 = 20%`

#### 4. LLM 설명 생성 ⭐ 핵심
- **입력**: Lift 수치만
- **출력**: 자연어 설명
  ```
  "이 음식점은 전체 평균 대비 서비스 긍정 평가 비율이 20% 높습니다"
  ```

**특징**:
- 통계적 비율 기반 차별점 계산
- LLM은 설명 생성만 담당
- 전체 데이터셋 기반 비교

**API 응답 형식** (예상):
```json
{
  "restaurant_id": 1,
  "strength_type": "distinct",
  "strengths": [
    {
      "category": "service",
      "lift_percentage": 20,
      "all_average_ratio": 0.60,
      "single_restaurant_ratio": 0.72,
      "description": "이 음식점은 전체 평균 대비 서비스 긍정 평가 비율이 20% 높습니다"
    },
    {
      "category": "price",
      "lift_percentage": 18,
      "all_average_ratio": 0.55,
      "single_restaurant_ratio": 0.65,
      "description": "이 음식점은 전체 평균 대비 가격 긍정 평가 비율이 18% 높습니다"
    }
  ]
}
```

**장점**:
- 객관적이고 해석하기 쉬운 수치
- LLM 호출 최소화 (비용/시간 절감)
- 전체 데이터셋 기반 비교로 안정적
- 통계적으로 명확한 해석

**단점**:
- 구체적인 claim/evidence 없음
- 고정 카테고리만 지원 (service/price)
- 전체 데이터셋 분석 필요 (배치 작업)
- 의미적 차별점은 반영 안 됨

---

### 비교 요약

| 항목 | 기존 | 새로운 |
|------|------|--------|
| **강점 추출 방식** | LLM이 리뷰를 읽고 추출 | PySpark + Kiwi로 bigram 추출 |
| **Aspect 범위** | LLM이 자유롭게 결정 | 고정 카테고리 (service/price/food) |
| **Claim 생성** | LLM이 생성 (구체적 표현) | 없음 (lift 수치만) |
| **Evidence** | LLM이 인용문 추출 | 없음 |
| **차별점 계산** | 벡터 유사도 (의미적) | 통계적 비율 (빈도 기반) |
| **비교군** | 유사 레스토랑 (최대 20개) | 전체 데이터셋 (모든 음식점) |
| **LLM 역할** | 강점 추출 + Claim 생성 | 설명 생성만 |
| **LLM 호출 횟수** | 강점 개수에 비례 (여러 번) | 1회 (설명만) |
| **비용** | 높음 | 낮음 |
| **정확도** | 중간 (LLM 환각 가능) | 높음 (통계적) |
| **해석 용이성** | 중간 | 높음 |

---

## 종합 비교

### 기술적 차이점

| 항목 | 기존 파이프라인 | 새로운 파이프라인 |
|------|----------------|------------------|
| **검색 방식** | Dense 벡터만 | Dense + Sparse 하이브리드 |
| **Aspect 추출** | LLM 기반 | PySpark + Kiwi 기반 |
| **차별점 계산** | 벡터 유사도 | 통계적 비율 |
| **비교군** | 유사 레스토랑 | 전체 데이터셋 |
| **LLM 사용** | 강점 추출, Claim 생성 | 설명 생성만 |
| **배치 작업** | 불필요 | 필요 (전체 데이터셋 분석) |

### 비용 및 성능

| 항목 | 기존 파이프라인 | 새로운 파이프라인 |
|------|----------------|------------------|
| **LLM 호출 횟수** | 많음 (강점 추출 + Claim 생성) | 적음 (설명만) |
| **처리 시간** | 중간 | 빠름 (통계적 계산) |
| **비용** | 높음 | 낮음 |
| **인프라 요구사항** | 낮음 | 높음 (Sparse 벡터, PySpark) |

### 정확도 및 해석

| 항목 | 기존 파이프라인 | 새로운 파이프라인 |
|------|----------------|------------------|
| **정확도** | 중간 (LLM 환각 가능) | 높음 (통계적) |
| **해석 용이성** | 중간 | 높음 |
| **구체성** | 높음 (claim, evidence) | 낮음 (lift 수치만) |
| **객관성** | 중간 | 높음 |

---

## 마이그레이션 전략

### Phase 1: Sentiment Analysis (즉시 교체 가능)

**난이도**: ⭐⭐ (낮음)  
**작업량**: 중간  
**리스크**: 낮음

**작업 내용**:
1. `final_sentiment_pipeline.py` 로직을 `src/sentiment_analysis.py`에 통합
2. API 응답 모델에 `neutral_count`, `neutral_rate` 필드 추가
3. Negative만 LLM 재판정 로직 추가
4. 기존 코드 제거

**예상 소요 시간**: 1-2일

---

### Phase 2: Summary (인프라 준비 후 교체)

**난이도**: ⭐⭐⭐⭐ (높음)  
**작업량**: 많음  
**리스크**: 중간

**작업 내용**:
1. **하이브리드 검색 인프라 구축**
   - Qdrant에 Sparse Vector 지원 추가
   - FastEmbed SparseTextEmbedding 모델 통합
   - 기존 컬렉션 마이그레이션 또는 새 컬렉션 생성
2. **Aspect Seed 생성 (배치 작업)**
   - `total_aspect.py`를 주기적 배치 작업으로 실행 (예: 일 1회)
   - 결과를 캐시/DB에 저장
   - `total_data_aspect_test.py`로 긍정 비율 계산 및 저장
3. **Summary 파이프라인 통합**
   - `final_summary_pipeline.py` 로직을 `src/api/routers/llm.py`에 통합
   - 하이브리드 검색 사용
   - Aspect 기반 카테고리별 검색
   - 캐시된 seed 사용
4. **API 응답 형식 변경**
   - 버전 관리 (v1, v2)
   - 클라이언트 업데이트 필요

**예상 소요 시간**: 1-2주

---

### Phase 3: Strength Extraction (기능 보완 후 교체)

**난이도**: ⭐⭐⭐⭐⭐ (매우 높음)  
**작업량**: 매우 많음  
**리스크**: 높음

**작업 내용**:
1. **전체 데이터셋 분석 배치 작업 설정**
   - `total_aspect.py` + `aspect_positive_filtering.py` 배치 작업
   - 결과를 캐시/DB에 저장
2. **단일 음식점 분석 로직 추가**
   - 동일한 방식으로 단일 음식점의 카테고리별 긍정 비율 계산
3. **Lift 계산 로직 추가**
   - `total_data_aspect_test.py` 로직을 API에 통합
4. **LLM 설명 생성 로직 추가**
   - Lift 수치를 받아서 자연어 설명 생성
5. **API 응답 형식 변경**
   - 기존 `StrengthDetail` 형식 유지하되 내용 변경
   - 또는 새로운 형식으로 완전 교체

**예상 소요 시간**: 2-3주

---

### 권장 순서

1. **Phase 1 (Sentiment Analysis)** - 즉시 진행
   - 가장 간단하고 리스크가 낮음
   - 즉시 효과를 볼 수 있음

2. **Phase 2 (Summary)** - 인프라 준비 후 진행
   - 하이브리드 검색 인프라 구축 필요
   - 품질 향상 효과가 큼

3. **Phase 3 (Strength Extraction)** - 신중하게 진행
   - 기능 보완 필요
   - API 응답 형식 변경으로 인한 클라이언트 영향 큼

---

## 결론

### 완전 교체 가능한 기능
- ✅ **Sentiment Analysis**: 즉시 교체 가능
- ✅ **Summary**: 인프라 작업 후 교체 가능

### 부분 교체 가능한 기능
- ⚠️ **Strength Extraction**: 기능 보완 후 교체 가능

### 주요 개선점
1. **비용 절감**: LLM 호출 횟수 대폭 감소
2. **정확도 향상**: 통계적 방법으로 객관성 향상
3. **해석 용이성**: 명확한 수치 기반 해석
4. **검색 품질**: 하이브리드 검색으로 정확도 향상

### 주의사항
1. **인프라 요구사항**: Sparse 벡터, PySpark 등 추가 인프라 필요
2. **배치 작업**: 전체 데이터셋 분석을 위한 배치 작업 필요
3. **API 호환성**: 기존 클라이언트와의 호환성 고려 필요
4. **기능 제한**: 구체적인 claim/evidence 제공 불가

---

## 참고 자료

- 기존 파이프라인: `src/` 디렉토리
- 새로운 파이프라인: `hybrid_search/final_pipeline/` 디렉토리
- 아키텍처 문서: `LLM_SERVICE_STEP/FINAL_ARCHITECTURE.md`
- API 명세: `LLM_SERVICE_STEP/API_SPECIFICATION.md`
