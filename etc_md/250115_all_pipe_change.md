# 파이프라인 변경 사항 (2025-01-15)

## 개요

모든 작업(Sentiment, Summary, Strength)을 **대표 벡터 기반 TOP-K 방식**으로 통일하여 일관성과 효율성을 개선했습니다. 또한 `is_recommended`와 `rating` 필터를 제거하여 더 단순하고 유연한 구조로 변경했습니다.

## 주요 변경 사항

### 1. 대표 벡터 기반 TOP-K 검색 메서드 추가

**파일**: `src/vector_search.py`

**추가된 메서드**:
- `query_by_restaurant_vector(restaurant_id, top_k, months_back)`
  - 레스토랑 대표 벡터를 쿼리로 사용하여 TOP-K 리뷰 검색
  - 날짜 필터링 지원 (선택)
  - 대표 벡터와 가장 가까운 리뷰를 우선 선택

**구현 로직**:
1. 레스토랑 대표 벡터 계산 (`compute_restaurant_vector`)
2. 대표 벡터로 Qdrant 검색 수행
3. 클라이언트 측 날짜 필터링 (선택)
4. 날짜 기준 정렬 (최신순)
5. TOP-K 반환

### 2. Sentiment 분석 개선

**파일**: `src/sentiment_analysis.py`, `src/api/routers/sentiment.py`, `src/api/dependencies.py`

**변경 사항**:
- **이전**: 모든 리뷰를 LLM에 전달하여 긍/부정 비율 계산
- **개선**: 대표 벡터 주위 TOP-K 리뷰만 선택하여 분석

**장점**:
- LLM 입력 토큰 수 감소 → 비용 절감
- 처리 시간 단축
- 대표적인 리뷰만 사용하여 더 정확한 감성 분석

**파라미터**:
- `top_k`: 기본값 20 (대표 벡터 주위에서 선택할 리뷰 수)
- `months_back`: 선택적 날짜 필터

### 3. Summary 요약 개선

**파일**: `src/api/routers/llm.py`

**변경 사항**:
- **이전**: 
  - 긍정 쿼리로 벡터 검색 → 긍정 리뷰 N개
  - 부정 쿼리로 벡터 검색 → 부정 리뷰 N개
  - 각각에서 aspect 추출 (LLM 2번)
  - 전체 요약 생성 (LLM 1번)
  - **총 LLM 호출: 3번**
- **개선**: 
  - 대표 벡터 주위 TOP-K 리뷰 선택
  - TOP-K 리뷰만 LLM에 넣어 요약 생성
  - **총 LLM 호출: 1번**

**장점**:
- LLM 호출 횟수 66% 감소 (3번 → 1번)
- 로직 단순화
- 대표적인 리뷰만 사용하여 일관성 향상

### 4. Strength 추출 개선

**파일**: `src/strength_extraction.py`

**변경 사항**:
- **이전**: 
  - 필터링된 모든 리뷰 수집 (날짜, `is_recommended`, `rating` 필터)
  - Scroll API로 모든 리뷰 검색 후 클라이언트 측 필터링
- **개선**: 
  - 대표 벡터 기반 TOP-K 리뷰로 시작
  - `is_recommended` 필터 제거
  - `rating` 필터 제거
  - 대표 벡터와 가장 가까운 리뷰 우선 선택

**제거된 필터**:
- `is_recommended`: 제거 (대표 벡터가 이미 레스토랑의 특성을 반영)
- `rating`: 제거 (대표 벡터가 이미 평점 정보를 포함)
- `min_rating` 파라미터: 제거

**유지된 필터**:
- `months_back`: 선택적 날짜 필터 (유지)

**장점**:
- 필터링 로직 단순화
- 대표 벡터 기반으로 일관성 향상
- 모든 리뷰를 스캔하지 않고 TOP-K만 처리하여 효율성 개선

## 전체 파이프라인 비교

### 이전 방식
