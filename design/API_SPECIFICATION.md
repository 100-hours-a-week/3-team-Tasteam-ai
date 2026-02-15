# API 명세서

## 목차
1. [전체 엔드포인트 목록](#전체-엔드포인트-목록)
2. [단일/배치 엔드포인트 사용 가이드](#단일배치-엔드포인트-사용-가이드)
3. [입출력 스키마 명세](#입출력-스키마-명세)
4. [시스템 아키텍처](#시스템-아키텍처)
5. [API 호출 예시](#api-호출-예시)

---

## 전체 엔드포인트 목록

### 감성 분석 (Sentiment Analysis)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| POST | `/api/v1/sentiment/analyze` | 단일 레스토랑의 리뷰를 감성 분석하여 긍정/부정 비율 계산 |
| POST | `/api/v1/sentiment/analyze/batch` | 여러 레스토랑의 리뷰를 배치로 감성 분석 |

### 리뷰 요약 및 강점 추출 (LLM-based Analysis)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| POST | `/api/v1/llm/summarize` | 단일 레스토랑의 긍정/부정 리뷰를 벡터 검색으로 자동 검색하고 LLM으로 요약 |
| POST | `/api/v1/llm/summarize/batch` | 여러 레스토랑의 리뷰를 배치로 요약 |
| POST | `/api/v1/llm/extract/strengths` | 구조화된 강점 추출 파이프라인 (Step A~H: 근거 수집, LLM 추출, 검증, 클러스터링, 비교군 기반 차별 강점 계산) |

### 벡터 검색 (Vector Search)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| POST | `/api/v1/vector/search/similar` | 의미 기반 검색으로 유사한 리뷰 검색 (Query Expansion 지원) |
| POST | `/api/v1/vector/search/review-images` | 리뷰 이미지 검색 (Query Expansion 지원) |
| POST | `/api/v1/vector/upload` | 레스토랑 데이터를 벡터 데이터베이스에 업로드 |
| GET | `/api/v1/vector/restaurants/{restaurant_id}/reviews` | 레스토랑 ID로 해당 레스토랑의 모든 리뷰 조회 |

### 리뷰 관리 (Review Management)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| POST | `/api/v1/vector/reviews/upsert` | 리뷰를 upsert (있으면 업데이트, 없으면 삽입) |
| POST | `/api/v1/vector/reviews/upsert/batch` | 여러 리뷰를 배치로 upsert |
| DELETE | `/api/v1/vector/reviews/delete` | 리뷰를 삭제 |
| DELETE | `/api/v1/vector/reviews/delete/batch` | 여러 리뷰를 배치로 삭제 |

### 헬스 체크 (Health Check)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| GET | `/health` | 서버 상태 확인 |
| GET | `/` | API 기본 정보 반환 |

### 테스트 데이터 생성 (Test Data Generation)

| 메서드 | 엔드포인트 | 기능 설명 |
|--------|-----------|----------|
| POST | `/api/v1/test/generate` | kr3.tsv 파일에서 테스트 데이터 샘플링하여 생성 |

---

## 단일/배치 엔드포인트 사용 가이드

### 개요

본 API는 각 기능별로 **단일 처리 엔드포인트**와 **배치 처리 엔드포인트**를 제공합니다.

### 선택 가이드

**단일 처리 엔드포인트 (`/analyze`, `/summarize`, `/extract/strengths`)를 사용할 때:**
- ✅ 1-5개 레스토랑 처리
- ✅ 실시간 사용자 요청
- ✅ 이벤트 기반 처리

**배치 처리 엔드포인트 (`/analyze/batch`, `/summarize/batch`, `/extract/strengths/batch`)를 사용할 때:**
- ✅ 6개 이상 레스토랑 처리
- ✅ 일괄 데이터 처리
- ✅ 배치 작업/스케줄러
- ✅ 성능 최적화가 필요한 경우

### 참고 사항

1. **응답 형식 차이**: 
   - 단일 처리: `{"restaurant_id": 1, "positive_ratio": 60, ...}`
   - 배치 처리: `{"results": [{"restaurant_id": 1, ...}, {"restaurant_id": 2, ...}]}`

2. **타임아웃 설정**: 배치 처리 엔드포인트는 처리 시간이 길 수 있으므로 적절한 타임아웃을 설정하세요 (예: 600초).

3. **에러 처리**: 배치 처리에서 일부 레스토랑 처리 실패 시, 성공한 레스토랑 결과는 반환되므로 클라이언트에서 부분 실패를 처리할 수 있어야 합니다.

---

## 입출력 스키마 명세

### 1. 감성 분석

#### 요청: `POST /api/v1/sentiment/analyze`

```json
{
  "restaurant_id": 1,
  "reviews": [
    {
      "id": 1,
      "restaurant_id": 1,
      "member_id": 100,
      "group_id": 200,
      "subgroup_id": 300,
      "content": "음식이 맛있네요! 또 가고싶어요!",
      "is_recommended": true,
      "created_at": "2024-01-01T12:00:00",
      "updated_at": "2024-01-01T12:00:00",
      "deleted_at": null
    }
  ]
}
```

**필드 설명:**
- `restaurant_id` (required, int): 레스토랑 ID
- `reviews` (required, List[Dict]): 리뷰 리스트
  - `id` (optional, int): 리뷰 ID
  - `restaurant_id` (required, int): 레스토랑 ID
  - `member_id` (optional, int): 회원 ID
  - `group_id` (optional, int): 그룹 ID
  - `subgroup_id` (optional, int): 서브그룹 ID
  - `content` (required, str): 리뷰 내용
  - `is_recommended` (optional, bool): 추천 여부
  - `created_at` (optional, str): 생성 시간
  - `updated_at` (optional, str): 수정 시간
  - `deleted_at` (optional, str): 삭제 시간

**프로세스:**

1. **중복 실행 방지 체크 (세 레이어 전략)**
   - **레이어 3 (Redis 락)**: 동시 중복 실행 차단 (`lock:{restaurant_id}:sentiment`)
   - **레이어 2 (SKIP 로직)**: 최근 성공 실행이면 SKIP (`analysis_metrics`에서 `MAX(created_at)` 조회)
     - `error_count=0` 중 최신 `created_at` 확인
     - `SKIP_MIN_INTERVAL_SECONDS` (기본값: 3600초 = 1시간) 이내면 SKIP
     - SKIP 시: 메트릭 기록 후 빈 응답 반환 (LLM 실행 없음)
   - **레이어 1 (스케줄러)**: 외부 스케줄러가 tier별 호출 빈도 결정 (거시적 제어)

2. 리뷰 리스트에서 `content` 필드 추출
3. LLM을 통해 긍/부정 개수 분석 (LLM 출력: `positive_count`, `negative_count`만)
4. 코드에서 비율 계산:
   - LLM이 판단한 개수: `total_judged = positive_count + negative_count`
   - 실제 리뷰 수와 불일치 시 스케일링: `scale = len(review_list) / total_judged`
   - 조정된 개수: `positive_count = round(positive_count * scale)`, `negative_count = round(negative_count * scale)`
   - 최종 비율: `positive_ratio = (positive_count / total_count) * 100`, `negative_ratio = (negative_count / total_count) * 100`

**에러 응답:**
- **409 Conflict**: Redis 락 획득 실패 (동시 중복 실행 차단)
  ```json
  {
    "detail": "중복 실행 방지: 레스토랑 1의 sentiment 분석이 이미 진행 중입니다."
  }
  ```

#### 응답: `SentimentAnalysisResponse`

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

**필드 설명:**
- `restaurant_id` (int): 레스토랑 ID
- `positive_count` (int): 긍정 리뷰 개수 (LLM 반환 후 스케일링 조정)
- `negative_count` (int): 부정 리뷰 개수 (LLM 반환 후 스케일링 조정)
- `total_count` (int): 전체 리뷰 개수 (len(review_list))
- `positive_ratio` (int): 긍정 비율 (%) (코드 계산: (positive_count / total_count) * 100)
- `negative_ratio` (int): 부정 비율 (%) (코드 계산: (negative_count / total_count) * 100)

**참고:** LLM이 판단하지 못한 리뷰(중립 등)가 있을 경우, `total_judged < total_count`가 발생할 수 있습니다. 이 경우 스케일링 로직을 통해 실제 리뷰 수에 맞춰 개수를 조정합니다.

---

#### 요청: `POST /api/v1/sentiment/analyze/batch` (여러 레스토랑 배치 처리)

```json
{
  "restaurants": [
    {
      "restaurant_id": 1,
      "reviews": [
        {
          "id": 1,
          "restaurant_id": 1,
          "member_id": 100,
          "content": "음식이 맛있네요!",
          "is_recommended": true,
          "created_at": "2024-01-01T12:00:00"
        }
      ]
    },
    {
      "restaurant_id": 2,
      "reviews": [
        {
          "id": 2,
          "restaurant_id": 2,
          "member_id": 101,
          "content": "서비스가 좋아요!",
          "is_recommended": true,
          "created_at": "2024-01-01T12:00:00"
        }
      ]
    }
  ],
  "max_tokens_per_batch": 4000
}
```

**필드 설명:**
- `restaurants` (required, List[Dict]): 레스토랑 데이터 리스트
  - `restaurant_id` (required, int): 레스토랑 ID
  - `reviews` (required, List[Dict]): 리뷰 리스트
- `max_tokens_per_batch` (optional, int, default: 4000, range: 1000-8000): 배치당 최대 토큰 수

**프로세스 (대표 벡터 TOP-K 방식):**

1. **대표 벡터 기반 TOP-K 리뷰 선택**
   - 레스토랑의 대표 벡터를 계산 (모든 리뷰의 가중 평균)
   - 대표 벡터 주위에서 TOP-K 리뷰 검색 (`query_by_restaurant_vector`)
   - 기본값: `top_k=20` (가장 대표적인 리뷰만 선택)
   - `reviews` 필드가 제공되면 사용, 없으면 자동으로 대표 벡터 기반 검색

2. **LLM 추론**
   - 선택된 TOP-K 리뷰의 `content` 필드 추출
   - vLLM 모드: 배치 처리 및 병렬 처리 (동적 배치 크기 + 비동기 큐)
   - 기존 모드: 단일 LLM 호출

3. **결과 집계 및 비율 계산**
   - 레스토랑별로 결과 집계
   - LLM이 판단하지 못한 리뷰가 있을 경우 스케일링 적용:
     - `total_judged = positive_count + negative_count`
     - `scale = len(review_list) / total_judged` (total_judged > 0인 경우)
     - `positive_count = round(positive_count * scale)`, `negative_count = round(negative_count * scale)`
   - 최종 비율 계산: `positive_ratio = (positive_count / total_count) * 100`, `negative_ratio = (negative_count / total_count) * 100`

#### 응답: `SentimentAnalysisBatchResponse`

```json
{
  "results": [
    {
      "restaurant_id": 1,
      "positive_count": 3,
      "negative_count": 2,
      "total_count": 5,
      "positive_ratio": 60,
      "negative_ratio": 40
    },
    {
      "restaurant_id": 2,
      "positive_count": 4,
      "negative_count": 1,
      "total_count": 5,
      "positive_ratio": 80,
      "negative_ratio": 20
    }
  ]
}
```

**필드 설명:**
- `results` (List[SentimentAnalysisResponse]): 각 레스토랑별 감성 분석 결과 리스트
  - 각 항목은 `SentimentAnalysisResponse`와 동일한 구조

---

### 2. 리뷰 요약

#### 요청: `POST /api/v1/llm/summarize`

```json
{
  "restaurant_id": 1,
  "positive_query": "맛있다 좋다 만족",
  "negative_query": "맛없다 별로 불만",
  "limit": 10,
  "min_score": 0.0
}
```

**필드 설명:**
- `restaurant_id` (required, int): 레스토랑 ID
- `positive_query` (optional, str, deprecated): 이전 버전 호환을 위해 유지되나 사용되지 않음
- `negative_query` (optional, str, deprecated): 이전 버전 호환을 위해 유지되나 사용되지 않음
- `limit` (optional, int, default: 10, range: 1-100): 대표 벡터 기반으로 검색할 최대 리뷰 수
- `min_score` (optional, float, deprecated): 이전 버전 호환을 위해 유지되나 사용되지 않음

**프로세스 (대표 벡터 TOP-K + aspect 기반 요약):**

1. **중복 실행 방지 체크 (세 레이어 전략)**
   - **레이어 3 (Redis 락)**: 동시 중복 실행 차단 (`lock:{restaurant_id}:summary`)
   - **레이어 2 (SKIP 로직)**: 최근 성공 실행이면 SKIP (`analysis_metrics`에서 `MAX(created_at)` 조회)
     - `error_count=0` 중 최신 `created_at` 확인
     - `SKIP_MIN_INTERVAL_SECONDS` (기본값: 3600초 = 1시간) 이내면 SKIP
     - SKIP 시: 메트릭 기록 후 빈 응답 반환 (LLM 실행 없음)
   - **레이어 1 (스케줄러)**: 외부 스케줄러가 tier별 호출 빈도 결정 (거시적 제어)

2. **대표 벡터 기반 TOP-K 리뷰 선택**
   - 레스토랑의 대표 벡터를 계산 (모든 리뷰의 가중 평균)
   - 대표 벡터 주위에서 TOP-K 리뷰 검색 (`query_by_restaurant_vector`)
   - 기본값: `limit * 2` (긍정/부정 모두 포함할 수 있도록)
   - 대표 벡터 기반이므로 대부분 긍정적일 가능성 높음

3. **Aspect 기반 요약 (LLM 추론)**
   - LLM이 선택된 TOP-K 리뷰에서 aspect 단위 장점/단점 추출
   - `positive_aspects`: 긍정 aspect 리스트 (aspect, claim, evidence_quotes, evidence_review_ids)
   - `negative_aspects`: 부정 aspect 리스트 (동일 구조)
   - `overall_summary`: positive_aspects + negative_aspects 기반으로 전체 요약 생성

3. **결과 반환**
   - `overall_summary`: LLM이 생성한 전체 요약
   - `positive_aspects`: 구조화된 긍정 aspect 리스트
   - `negative_aspects`: 구조화된 부정 aspect 리스트
   - 메타데이터 및 카운트 포함

#### 응답: `SummarizeResponse`

```json
{
  "restaurant_id": "res_1234",
  "overall_summary": "가츠동과 빠른 회전이 장점인 반면, 음식이 다소 짜고 일부 메뉴는 만족스럽지 않다.",
  "positive_reviews": [
    {
      "id": 1,
      "restaurant_id": 1,
      "member_id": 100,
      "group_id": 200,
      "subgroup_id": 300,
      "content": "점심시간이라 사람이 많았지만 생각보다 빨리 나왔다.",
      "is_recommended": true,
      "created_at": "2026-01-03T12:10:00",
      "updated_at": "2026-01-03T12:10:00",
      "deleted_at": null
    }
  ],
  "negative_reviews": [
    {
      "id": 2,
      "restaurant_id": 1,
      "member_id": 101,
      "group_id": 201,
      "subgroup_id": 301,
      "content": "가츠동은 괜찮았는데 다른 메뉴는 좀 애매했다.",
      "is_recommended": false,
      "created_at": "2026-01-03T12:12:00",
      "updated_at": "2026-01-03T12:12:00",
      "deleted_at": null
    }
  ],
  "positive_count": 3,
  "negative_count": 2
}
```

**필드 설명:**
- `restaurant_id` (str): 레스토랑 ID
- `overall_summary` (str): 전체 요약 (LLM 반환, 긍정과 부정을 모두 고려한 통합 요약)
- `positive_reviews` (List[Dict]): 긍정 리뷰 메타데이터 리스트
- `negative_reviews` (List[Dict]): 부정 리뷰 메타데이터 리스트
- `positive_count` (int): 긍정 리뷰 개수
- `negative_count` (int): 부정 리뷰 개수

**참고:** LLM은 `overall_summary`만 반환합니다. `positive_summary`와 `negative_summary`는 더 이상 제공되지 않습니다.

---

#### 요청: `POST /api/v1/llm/summarize/batch` (여러 레스토랑 배치 요약)

```json
{
  "restaurants": [
    {
      "restaurant_id": 1,
      "positive_reviews": [
        {
          "id": 1,
          "restaurant_id": 1,
          "content": "음식이 맛있네요!",
          "is_recommended": true
        }
      ],
      "negative_reviews": [
        {
          "id": 2,
          "restaurant_id": 1,
          "content": "서비스가 아쉽네요",
          "is_recommended": false
        }
      ]
    },
    {
      "restaurant_id": 2,
      "positive_reviews": [
        {
          "id": 3,
          "restaurant_id": 2,
          "content": "가격이 합리적이에요!",
          "is_recommended": true
        }
      ],
      "negative_reviews": []
    }
  ],
  "max_tokens_per_batch": 4000
}
```

**필드 설명:**
- `restaurants` (required, List[Dict]): 레스토랑 데이터 리스트
  - `restaurant_id` (required, int): 레스토랑 ID
  - `positive_reviews` (required, List[Dict]): 긍정 리뷰 리스트
  - `negative_reviews` (required, List[Dict]): 부정 리뷰 리스트
- `max_tokens_per_batch` (optional, int, default: 4000, range: 1000-8000): 배치당 최대 토큰 수

**프로세스:**

1. 각 레스토랑의 긍정/부정 리뷰에서 `content` 필드 추출
2. 배치 처리 및 병렬 처리
3. 레스토랑별로 결과 집계 및 요약 합치기

#### 응답: `SummaryBatchResponse`

```json
{
  "results": [
    {
      "restaurant_id": 1,
      "overall_summary": "음식은 좋지만 서비스 개선이 필요합니다.",
      "positive_reviews": [
        {
          "id": 1,
          "restaurant_id": 1,
          "content": "음식이 맛있네요!",
          "is_recommended": true
        }
      ],
      "negative_reviews": [
        {
          "id": 2,
          "restaurant_id": 1,
          "content": "서비스가 아쉽네요",
          "is_recommended": false
        }
      ],
      "positive_count": 1,
      "negative_count": 1
    },
    {
      "restaurant_id": 2,
      "overall_summary": "가격이 합리적입니다.",
      "positive_reviews": [
        {
          "id": 3,
          "restaurant_id": 2,
          "content": "가격이 합리적이에요!",
          "is_recommended": true
        }
      ],
      "negative_reviews": [],
      "positive_count": 1,
      "negative_count": 0
    }
  ]
}
```

**필드 설명:**
- `results` (List[SummaryResponse]): 각 레스토랑별 요약 결과 리스트
  - 각 항목은 `SummaryResponse`와 동일한 구조

---

### 3. 강점 추출

#### 요청: `POST /api/v1/llm/extract/strengths`

```json
{
  "restaurant_id": 123,
  "strength_type": "both",
  "category_filter": 1,
  "region_filter": "서울",
  "price_band_filter": "중",
  "top_k": 10,
  "max_candidates": 300,
  "months_back": 6,
  "min_support": 5
}
```

**필드 설명:**
- `restaurant_id` (required, int): 타겟 레스토랑 ID
- `strength_type` (optional, str, default: "both"): 강점 타입
  - `"representative"`: 대표 강점만 (자주 언급되는 장점)
  - `"distinct"`: 차별 강점만 (비교군 대비 희소/유니크한 장점)
  - `"both"`: 대표 강점 + 차별 강점 모두
- `category_filter` (optional, int): 카테고리 필터
- `region_filter` (optional, str): 지역 필터
- `price_band_filter` (optional, str): 가격대 필터
- `top_k` (optional, int, default: 10, range: 1-50): 반환할 최대 강점 개수
- `max_candidates` (optional, int, default: 300, range: 50-1000): 근거 후보 최대 개수
- `months_back` (optional, int, default: 6, range: 1-24): 최근 N개월 리뷰만 사용
- `min_support` (optional, int, default: 5, range: 1-50): 최소 support_count (희소 환각 방지)

**프로세스 (Step A~H):**

1. **중복 실행 방지 체크 (세 레이어 전략)**
   - **레이어 3 (Redis 락)**: 동시 중복 실행 차단 (`lock:{restaurant_id}:strength`)
   - **레이어 2 (SKIP 로직)**: 최근 성공 실행이면 SKIP (`analysis_metrics`에서 `MAX(created_at)` 조회)
     - `error_count=0` 중 최신 `created_at` 확인
     - `SKIP_MIN_INTERVAL_SECONDS` (기본값: 3600초 = 1시간) 이내면 SKIP
     - SKIP 시: 메트릭 기록 후 빈 응답 반환 (LLM 실행 없음)
   - **레이어 1 (스케줄러)**: 외부 스케줄러가 tier별 호출 빈도 결정 (거시적 제어)

2. **Step A: 타겟 긍정 근거 후보 수집**
   - 대표 벡터 기반 TOP-K 선택 (대표성)
   - 최근 리뷰 추가 (최신성)
   - 랜덤 샘플링 추가 (다양성)
   - 중복 제거 (review_id 기준)

2. **Step B: 강점 후보 생성 (LLM 구조화 출력, Recall 단계)**
   - LLM으로 aspect 단위 강점 추출
   - 최소 5개 후보 보장 (부족하면 generic 후보 자동 생성)
   - 출력 형식: `[{aspect, claim, type, confidence, evidence_quotes[], evidence_review_ids[]}]`
   - Generic aspect도 허용 (Step C에서 필터링)

3. **Step C: 강점별 근거 확장/검증**
   - 각 aspect에 대해 Qdrant 벡터 검색으로 근거 확장
   - **유효 근거 수 계산**: score >= 0.3 필터링 + 긍정 리뷰만
   - **"강점은 긍정이어야 한다" 가드**: 감성 라벨이 있으면 positive만, 없으면 부정 키워드 제외
   - support_count_raw (전체 검색 결과 수), support_count_valid (score 기준 유효 수), support_count (긍정 필터링 후 최종 유효 수) 저장
   - support_ratio, consistency, recency 계산
   - `support_count < min_support` 또는 `consistency < 0.25`면 버림

4. **Step D: 의미 중복 제거 (Connected Components)**
   - 유사도 그래프 만들기 (모든 pair에 대해 cosine sim 계산)
   - Connected Components로 그룹 생성 (Union-Find 알고리즘)
   - 이중 임계값 가드레일 (T_high=0.88 즉시 merge, T_low=0.82~0.88 가드레일)
   - Evidence overlap 가드레일 (30% 이상 겹치면 merge)
   - Aspect type 체크 (다른 type은 merge 금지)
   - 대표 aspect명 선정 (support_count 가장 큰 member)
   - 대표 벡터 재계산 (evidence 리뷰 벡터의 centroid)

5. **Step D-1: Claim 후처리 재생성**
   - **템플릿 기반 보정** (LLM 없이, 우선 적용)
     - 15-28자 범위 (모바일 카드 1줄 기준)
     - 메타 표현 통일: "언급이 많음" 사용 ("리뷰가 자주 보임" 등 제거)
     - 예시: "맛있다" → "맛에 대한 만족도가 높다는 언급이 많음"
   - **LLM 기반 생성** (템플릿 실패 시)
     - 맛 관련 claim은 구체명사 1개 포함 필수 (국물/면/유자라멘/디저트/커피/육즙/불맛 등)
     - 예시: "유자라멘 국물이 진하다는 언급이 많음"

6. **Step E~H: 비교군 기반 차별 강점 계산** (distinct 또는 both일 때만)
   - 비교군 구성 (같은 카테고리/지역/가격대)
   - 타겟 aspect vs 비교군 aspect 유사도 계산
   - `distinct = 1 - max_sim`
   - 최종 점수: `rep × (1 + alpha × distinct)`

7. **Top-K 선택 (both 모드): 쿼터 적용**
   - 대표 최소 2개 확보 (대표 1개뿐이면 1개)
   - distinct 최대 3개
   - 같은 타입 중복 방지: 대표 섹션에 aspect가 있으면 차별 섹션의 같은 aspect는 1개만 허용, "맛"은 "시그니처 메뉴"로 변경

**Step A~H 파이프라인 실행 조건:**
- **대표 강점 (representative)**: Step A~D만 실행 (근거 수집 → LLM 추출 → 검증 → 중복 제거)
- **차별 강점 (distinct)**: Step A~H 모두 실행 (Step A~D + 비교군 기반 차별점 계산)
- **Both**: 대표 강점 + 차별 강점 모두 반환 (representative와 distinct를 합쳐서 반환)

#### 응답: `StrengthResponse`

```json
{
  "restaurant_id": 123,
  "strength_type": "both",
  "strengths": [
        {
      "aspect": "불맛",
      "claim": "유자라멘 국물이 진하다는 언급이 많음",
      "strength_type": "distinct",
      "support_count": 17,
      "support_ratio": 0.85,
      "distinct_score": 0.42,
      "closest_competitor_sim": 0.58,
      "closest_competitor_id": 456,
      "evidence": [
        {
          "review_id": "rev_1",
          "snippet": "숫불향이 진해서 맛있어요...",
          "rating": 5.0,
          "created_at": "2026-01-01T00:00:00"
        },
        {
          "review_id": "rev_5",
          "snippet": "불맛이 강해서 고기 맛이 살아있어요...",
          "rating": 4.5,
          "created_at": "2026-01-15T00:00:00"
        }
      ],
      "final_score": 15.2
    },
    {
      "aspect": "서비스",
      "claim": "직원이 친절하고 웨이팅이 빠름",
      "strength_type": "representative",
      "support_count": 25,
      "support_ratio": 0.92,
      "distinct_score": null,
      "closest_competitor_sim": null,
      "closest_competitor_id": null,
      "evidence": [
        {
          "review_id": "rev_10",
          "snippet": "직원이 친절해서 인상이 좋았어요...",
          "rating": 5.0,
          "created_at": "2026-01-20T00:00:00"
        }
      ],
      "final_score": 8.5
        }
      ],
  "total_candidates": 300,
  "validated_count": 5,
  "debug": {
    "request_id": "req_123",
    "processing_time_ms": 1250.5
    }
}
```

**필드 설명:**
- `restaurant_id` (int): 레스토랑 ID
- `strength_type` (str): 요청한 강점 타입
- `strengths` (List[StrengthDetail]): 강점 리스트
  - `aspect` (str): 강점 카테고리 (예: "불맛", "서비스", "시그니처 메뉴")
  - `claim` (str): 구체적 주장 (1문장, 15-28자, 모바일 카드 1줄 기준)
  - `strength_type` (str): "representative" 또는 "distinct"
  - `support_count` (int): 유효 근거 수 (긍정 필터링 후)
  - `support_count_raw` (int, optional): 전체 검색 결과 수 (디버깅용)
  - `support_count_valid` (int, optional): score 기준 유효 수 (디버깅용)
  - `support_ratio` (float): 지원 비율 (0~1)
  - `representative_evidence` (str, optional): 대표 근거 1줄 (요약+대표 장점 섹션용)
  - `distinct_score` (float or null): 차별성 점수 (distinct일 때만)
  - `closest_competitor_sim` (float or null): 가장 유사한 경쟁자 유사도 (distinct일 때만)
  - `closest_competitor_id` (int or null): 가장 유사한 경쟁자 ID (distinct일 때만)
  - `evidence` (List[EvidenceSnippet]): 근거 스니펫 리스트 (3~5개)
    - `review_id` (str): 리뷰 ID
    - `snippet` (str): 짧은 인용문
    - `rating` (float or null): 별점
    - `created_at` (str): 생성 시간
  - `final_score` (float): 최종 점수
- `total_candidates` (int): 근거 후보 총 개수
- `validated_count` (int): 검증 통과한 강점 개수
- `debug` (DebugInfo, optional): 디버그 정보


### 4. 리뷰 Upsert

#### 요청: `POST /api/v1/vector/reviews/upsert`

```json
{
  "restaurant": {
    "id": 1,
    "name": "비즐",
    "full_address": "서울시 강남구...",
    "location": null,
    "created_at": "2024-01-01T00:00:00",
    "deleted_at": null
  },
  "review": {
    "id": 1,
    "restaurant_id": 1,
    "member_id": 100,
    "group_id": 200,
    "subgroup_id": 300,
    "content": "맛있어요!",
    "is_recommended": true,
    "created_at": "2024-01-01T12:00:00",
    "updated_at": "2024-01-01T12:00:00",
    "deleted_at": null
  },
  "update_version": null
}
```

**필드 설명:**
- `restaurant` (required, Dict): 레스토랑 정보
  - `id` (optional, int): 레스토랑 ID
  - `name` (required, str): 레스토랑 이름
  - `full_address` (optional, str): 전체 주소
  - `location` (optional, Dict): 위치 정보
  - `created_at` (optional, str): 생성 시간
  - `deleted_at` (optional, str): 삭제 시간
- `review` (required, Dict): 리뷰 정보
  - `id` (optional, int): 리뷰 ID
  - `restaurant_id` (required, int): 레스토랑 ID
  - `member_id` (optional, int): 회원 ID
  - `group_id` (optional, int): 그룹 ID
  - `subgroup_id` (optional, int): 서브그룹 ID
  - `content` (required, str): 리뷰 내용
  - `is_recommended` (optional, bool): 추천 여부
  - `created_at` (optional, str): 생성 시간
  - `updated_at` (optional, str): 수정 시간
  - `deleted_at` (optional, str): 삭제 시간
- `update_version` (optional, int or null): 업데이트할 버전 (None이면 항상 업데이트/삽입, 지정하면 해당 버전일 때만 업데이트)

#### 응답: `UpsertReviewResponse`

```json
{
  "action": "inserted",
  "review_id": 1,
  "version": 2,
  "point_id": "abc123def456...",
  "reason": null,
  "requested_version": null,
  "current_version": null
}
```

**필드 설명:**
- `action` (str): 수행된 작업 ("inserted", "updated", "skipped")
- `review_id` (int): 리뷰 ID
- `version` (int): 새로운 버전 번호
- `point_id` (str): Point ID (MD5 해시)
- `reason` (str or null): skipped인 경우 이유 ("version_mismatch" 등)
- `requested_version` (int or null): 요청한 버전 (skipped인 경우)
- `current_version` (int or null): 현재 버전 (skipped인 경우)

---

### 5. 리뷰 배치 Upsert

#### 요청: `POST /api/v1/vector/reviews/upsert/batch`

```json
{
  "restaurant": {
    "id": 1,
    "name": "비즐",
    "full_address": "서울시 강남구...",
    "location": null,
    "created_at": "2024-01-01T00:00:00",
    "deleted_at": null
  },
  "reviews": [
    {
      "id": 1,
      "restaurant_id": 1,
      "member_id": 100,
      "group_id": 200,
      "subgroup_id": 300,
      "content": "맛있어요!",
      "is_recommended": true,
      "created_at": "2024-01-01T12:00:00",
      "updated_at": "2024-01-01T12:00:00",
      "deleted_at": null
    }
  ],
  "batch_size": 32
}
```

**필드 설명:**
- `restaurant` (required, Dict): 레스토랑 정보
- `reviews` (required, List[Dict]): 리뷰 딕셔너리 리스트
- `batch_size` (optional, int, default: 32, range: 1-100): 벡터 인코딩 배치 크기

#### 응답: `UpsertReviewsBatchResponse`

```json
{
  "results": [
    {
      "action": "inserted",
      "review_id": "rev_3001",
      "version": 2,
      "point_id": "abc123..."
    }
  ],
  "total": 1,
  "success_count": 1,
  "error_count": 0
}
```

---

### 6. 리뷰 삭제

#### 요청: `DELETE /api/v1/vector/reviews/delete`

```json
{
  "restaurant_id": 1,
  "review_id": 1
}
```

**필드 설명:**
- `restaurant_id` (required, int): 레스토랑 ID
- `review_id` (required, int): 리뷰 ID

#### 응답: `DeleteReviewResponse`

```json
{
  "action": "deleted",
  "review_id": 1,
  "point_id": "abc123def456..."
}
```

**필드 설명:**
- `action` (str): 수행된 작업 ("deleted", "not_found")
- `review_id` (int): 리뷰 ID
- `point_id` (str): Point ID

---

### 7. 의미 기반 리뷰 검색

#### 요청: `POST /api/v1/vector/search/similar`

```json
{
  "query_text": "맛있다",
  "restaurant_id": 1,
  "limit": 3,
  "min_score": 0.0,
  "expand_query": null
}
```

**필드 설명:**
- `query_text` (required, str): 검색 쿼리 텍스트
- `restaurant_id` (optional, int or null): 레스토랑 ID 필터 (None이면 전체 검색)
- `limit` (optional, int, default: 3, range: 1-100): 반환할 최대 개수
- `min_score` (optional, float, default: 0.0, range: 0.0-1.0): 최소 유사도 점수
- `expand_query` (optional, bool or null, default: null): 쿼리 확장 여부
  - `null`: 자동 판단 (복잡한 쿼리는 확장, 단순 쿼리는 확장 안함)
  - `true`: 강제 확장 (LLM을 사용하여 키워드 확장)
  - `false`: 확장 안함 (원본 쿼리 그대로 사용)

**Query Expansion (쿼리 확장):**
- **목적**: 사용자의 간단한 질의를 Dense 검색에 더 적합한 키워드로 확장하여 검색 품질 향상
- **예시**: "데이트하기 좋은" → "분위기 좋다 로맨틱 조용한 데이트 분위기"
- **자동 판단 기준**:
  - 확장 필요: 상황 표현("데이트", "가족", "친구"), 평가 표현("좋은", "나쁜", "추천")
  - 확장 불필요: 단순 키워드("분위기", "맛", "서비스", "가격")

#### 응답: `VectorSearchResponse`

```json
{
  "results": [
    {
      "id": 1,
      "restaurant_id": 1,
      "member_id": 100,
      "group_id": 200,
      "subgroup_id": 300,
      "content": "맛있어요!",
      "is_recommended": true,
      "created_at": "2026-01-03T12:10:00",
      "updated_at": "2026-01-03T12:10:00",
      "deleted_at": null
    }
  ],
  "total": 1
}
```

**필드 설명:**
- `results` (List[Dict]): 검색 결과 리스트
- `total` (int): 총 결과 개수

---

### 8. 리뷰 이미지 검색

#### 요청: `POST /api/v1/vector/search/review-images`

```json
{
  "query": "분위기 좋다",
  "restaurant_id": 1,
  "limit": 10,
  "min_score": 0.0,
  "expand_query": null
}
```

**필드 설명:**
- `query` (required, str): 검색 쿼리 (예: "분위기 좋다", "데이트하기 좋은")
- `restaurant_id` (optional, int): 레스토랑 ID 필터
- `limit` (optional, int, default: 10, range: 1-100): 반환할 최대 개수
- `min_score` (optional, float, default: 0.0, range: 0.0-1.0): 최소 유사도 점수
- `expand_query` (optional, bool or null, default: null): 쿼리 확장 여부
  - `null`: 자동 판단 (복잡한 쿼리는 확장, 단순 쿼리는 확장 안함)
  - `true`: 강제 확장 (LLM을 사용하여 키워드 확장)
  - `false`: 확장 안함 (원본 쿼리 그대로 사용)

**프로세스:**

1. 벡터 검색으로 리뷰 검색 (Query Expansion 적용 가능)
2. 리뷰와 이미지 정보 결합
3. restaurant_id, review_id, image_url, review 정보 반환

**Query Expansion (쿼리 확장):**
- **목적**: 사용자의 간단한 질의를 Dense 검색에 더 적합한 키워드로 확장하여 검색 품질 향상
- **예시**: "데이트하기 좋은" → "분위기 좋다 로맨틱 조용한 데이트 분위기"
- **자동 판단 기준**:
  - 확장 필요: 상황 표현("데이트", "가족", "친구"), 평가 표현("좋은", "나쁜", "추천")
  - 확장 불필요: 단순 키워드("분위기", "맛", "서비스", "가격")

#### 응답: `ReviewImageSearchResponse`

```json
{
  "results": [
    {
      "restaurant_id": 1,
      "review_id": 1,
      "image_url": "http://localhost:8000/image1.jpeg",
      "review": {
        "id": 1,
        "restaurant_id": 1,
        "member_id": 100,
        "group_id": 200,
        "subgroup_id": 300,
        "content": "분위기가 좋아요!",
        "is_recommended": true,
        "created_at": "2026-01-03T12:10:00",
        "updated_at": "2026-01-03T12:10:00",
        "deleted_at": null
      }
    }
  ],
  "total": 1
}
```

**필드 설명:**
- `results` (List[Dict]): 리뷰 이미지 검색 결과 리스트
  - `restaurant_id` (int): 레스토랑 ID
  - `review_id` (int): 리뷰 ID
  - `image_url` (str): 이미지 URL
  - `review` (Dict): 리뷰 정보
- `total` (int): 총 결과 개수

---

### 9. 테스트 데이터 생성

#### 요청: `POST /api/v1/test/generate`

**Query Parameters:**

| 파라미터 | 타입 | 필수 | 기본값 | 범위 | 설명 |
|---------|------|------|--------|------|------|
| `sample` | int | 선택 | None | 1-1000000 | 샘플링할 리뷰 수 (None이면 전체 변환, 권장: 100-10000) |
| `restaurants` | int | 선택 | None | 1-1000 | 생성할 레스토랑 수 (None이면 자동 결정) |
| `reviews_per_restaurant` | int | 선택 | None | 1-10000 | 레스토랑당 리뷰 수 (None이면 균등 분배) |
| `single_restaurant` | bool | 선택 | false | - | 모든 리뷰를 단일 레스토랑으로 그룹화 |
| `seed` | int | 선택 | 42 | 0+ | 랜덤 시드 (재현 가능한 결과) |
| `tsv_path` | str | 선택 | None | - | TSV 파일 경로 (기본값: 프로젝트 루트의 kr3.tsv) |

**프로세스:**

1. `kr3.tsv` 파일 읽기 (프로젝트 루트 또는 `tsv_path`에서)
2. 샘플링 옵션에 따라 리뷰 샘플링 및 레스토랑별 그룹화
3. API 형식으로 변환 (`SentimentAnalysisBatchRequest` 형식)

**데이터 매핑:**
- `Rating = 1 또는 2` → `is_recommended = true`
- `Rating = 0` → `is_recommended = false`
- `Review` → `content`
- 나머지 필드(restaurant_id, member_id, created_at, id)는 자동 생성

**참고:** `kr3.tsv` 파일의 `Rating` 값은 `0`, `1`, `2`만 허용됩니다. 다른 값은 자동으로 필터링됩니다.

#### 응답: `SentimentAnalysisBatchRequest`

```json
{
  "restaurants": [
    {
      "restaurant_id": 1,
      "restaurant_name": "Test Restaurant 1",
      "reviews": [
        {
          "id": 1,
          "restaurant_id": 1,
          "member_id": 10001,
          "group_id": null,
          "subgroup_id": null,
          "content": "음식이 맛있네요! 또 가고싶어요!",
          "is_recommended": true,
          "created_at": "2024-01-01T12:00:00",
          "updated_at": null,
          "deleted_at": null,
          "images": []
        }
      ]
    }
  ],
  "max_tokens_per_batch": 4000
}
```

**필드 설명:**

- `restaurants` (List[Dict]): 레스토랑별 리뷰 데이터 리스트
  - `restaurant_id` (int): 레스토랑 ID
  - `restaurant_name` (str): 레스토랑 이름 (`"Test Restaurant {restaurant_id}"` 형식)
  - `reviews` (List[Dict]): 리뷰 리스트
- `max_tokens_per_batch` (int): 배치당 최대 토큰 수 (기본값: 4000)

**사용 예시:**

```bash
# 작은 샘플 생성 (100개 리뷰, 5개 레스토랑)
curl -X POST "http://localhost:8000/api/v1/test/generate?sample=100&restaurants=5" \
  -H "Content-Type: application/json" \
  -o test_data.json

# 생성된 데이터로 배치 감성 분석 실행
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @test_data.json \
  --max-time 600
```

**참고:**
- **테스트 가이드**: 프로젝트 루트의 `README.md`에 상세한 테스트 가이드가 포함되어 있습니다. 주요 테스트 방법: 단일/배치 엔드포인트 테스트, Swagger UI를 통한 인터랙티브 테스트, Python 스크립트를 통한 자동화 테스트 등이 설명되어 있습니다.
- Python 스크립트로 테스트 데이터를 생성하려면 `scripts/convert_kr3_tsv.py`를 사용할 수 있습니다 (서버 실행 전에도 가능).

---

## 시스템 아키텍처

**시스템 아키텍처 요약:**
- **모듈화 아키텍처**: 단일 책임 원칙 기반 모듈 분리 (Sentiment, Vector, LLM 모듈)
- **RAG 패턴**: 벡터 검색 + LLM 추론으로 컨텍스트 최적화
- **멀티스텝 파이프라인**: 구조화된 Step A~H 파이프라인 (강점 추출)
- **배치 처리 최적화**: 동적 배치 크기 + 비동기 큐 + 세마포어
- **vLLM 통합**: Continuous Batching, 우선순위 큐, Prefill/Decode 분리 측정
- **메트릭 수집**: SQLite + 로그 파일 기반 성능 추적
- **세 레이어 중복 실행 방지**: 스케줄러(거시적) + SKIP 로직(미세) + Redis 락(동시성)

---

## 메트릭 수집 및 모니터링

### vLLM 메트릭 수집

API는 vLLM 성능 메트릭을 자동으로 수집하여 저장합니다:

**수집 메트릭:**
- **Prefill 시간**: 입력 처리 시간 (밀리초)
- **Decode 시간**: 토큰 생성 시간 (밀리초)
- **TTFT (Time To First Token)**: 첫 토큰 생성까지의 시간
- **TPS (Tokens Per Second)**: 초당 생성 토큰 수
- **TPOT (Time Per Output Token)**: 토큰당 생성 시간

**저장 위치:**
- **SQLite**: `metrics.db`의 `vllm_metrics` 테이블
- **로그 파일**: JSON 형식으로 상세 로그 저장

**Goodput 추적:**
- SLA (TTFT < 2초) 기반 실제 처리량 측정
- Throughput vs Goodput 비교로 품질 확인

**이미지 검색 쿼리 확장 메트릭:**
- 쿼리 확장 사용률 모니터링
- 확장 성능 및 효과 분석

**최적화 기법 요약:**
- **대표 벡터 TOP-K 방식**: RAG 패턴으로 컨텍스트 크기 최적화 (토큰 사용량 60-80% 감소, 처리 시간 50-70% 단축)
- **vLLM Continuous Batching**: 여러 요청 자동 배치 처리 (처리량 5-10배 향상)
- **우선순위 큐 (Prefill 비용 기반)**: 작은 요청 우선 처리로 SLA 보호 (TTFT 30-40% 개선)
- **동적 배치 크기**: 리뷰 길이에 따른 최적 배치 (GPU 활용률 2-3배 향상)
- **세마포어 제한**: OOM 방지
- **비동기 큐 방식**: 여러 레스토랑 병렬 처리 (10개 레스토랑 처리 시간 80-90% 단축)

---

## API 호출 예시

각 엔드포인트의 상세한 요청/응답 스키마와 예시는 Swagger UI (http://localhost:8000/docs) 또는 ReDoc (http://localhost:8000/redoc)에서 확인할 수 있습니다.

### 주요 엔드포인트 예시

#### 감성 분석 (단일)
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "reviews": [...]}'
```

#### 감성 분석 (배치)
```bash
curl -X POST "http://localhost:8000/api/v1/sentiment/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"restaurants": [...], "max_tokens_per_batch": 4000}' \
  --max-time 600
```

#### 리뷰 요약
```bash
curl -X POST "http://localhost:8000/api/v1/llm/summarize" \
  -H "Content-Type: application/json" \
  -d '{"restaurant_id": 1, "limit": 10}'
```

#### 강점 추출
```bash
curl -X POST "http://localhost:8000/api/v1/llm/extract/strengths" \
  -H "Content-Type: application/json" \
  -d '{"target_restaurant_id": 1, "limit": 1}'
```

---

## 참고 문서

- **프로젝트 개요**: [README.md](README.md)
- **서비스 아키텍처**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **배치 처리 개선 사항**: [BATCH_PROCESSING_IMPROVEMENT.md](BATCH_PROCESSING_IMPROVEMENT.md)
- **프로덕션 환경 문제점 및 개선방안**: [PRODUCTION_ISSUES_AND_IMPROVEMENTS.md](PRODUCTION_ISSUES_AND_IMPROVEMENTS.md)
- **GPU 서버 + vLLM 구현 가이드**: [RUNPOD_POD_VLLM_GUIDE.md](RUNPOD_POD_VLLM_GUIDE.md)

---

## API 문서 (Swagger/ReDoc)

실행 중인 서버에서 다음 URL로 상세한 API 문서를 확인할 수 있습니다:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

이 문서들은 Pydantic 모델을 기반으로 자동 생성되며, 모든 엔드포인트의 요청/응답 스키마를 확인할 수 있습니다.

---

## 관련 문서

- [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md): 통합 아키텍처 개요
- [ARCHITECTURE.md](ARCHITECTURE.md): 모듈화 아키텍처 상세
- [LLM_SERVICE_DESIGN.md](LLM_SERVICE_DESIGN.md): LLM 서비스 설계 상세
- [RAG_ARCHITECTURE.md](RAG_ARCHITECTURE.md): RAG 아키텍처 상세
- [PRODUCTION_INFRASTRUCTURE.md](PRODUCTION_INFRASTRUCTURE.md): 인프라 및 배포 계획
- [EXTERNAL_SYSTEM_INTEGRATION.md](EXTERNAL_SYSTEM_INTEGRATION.md): 외부 시스템 통합 설계
