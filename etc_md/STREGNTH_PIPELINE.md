# 강점 추출 파이프라인 문서

아래는 **"강점(Strength) 추출"을 프로덕션 관점으로 끝까지** 굴릴 수 있게 만든 전체 파이프라인입니다.
(네 스택: FastAPI + Qdrant + Qwen2.5 + vLLM/RunPod 기준으로 바로 붙일 수 있는 형태)

---

## 0) 목표 정의

* **대표 강점(Representative strengths)**: 이 음식점에서 *자주* 언급되는 장점
* **차별 강점(Distinct strengths)**: 같은 카테고리/조건의 비교군 대비 *희소/유니크*한 장점
* 모든 강점은 **근거 리뷰(evidence)** 를 반드시 붙임(신뢰성 핵심)

---

## 1) 오프라인 준비: 인덱싱 & 메타데이터

### 1-1. 리뷰 전처리

* 언어 정규화(이모지/중복문자/공백)
* 문장 분리(가능하면)
* 리뷰 메타데이터 정리

  * `restaurant_id, category, region, price_band, created_at, rating(optional)`

**구현 위치**: `src/review_utils.py`
- `preprocess_review_text()`: 언어 정규화
- `split_sentences()`: 문장 분리
- `preprocess_reviews()`: 전체 전처리

### 1-2. 임베딩 생성 & Qdrant 적재

* **리뷰 단위 벡터** (또는 문장 단위면 더 좋음)
* payload에 필수로 저장:

  * `restaurant_id, category, sentiment(optional), created_at, rating(optional), text`

> Qdrant는 "근거 검색"과 "비교군 구축"에 계속 쓰임

**구현 위치**: `src/vector_search.py`
- `prepare_points()`: 리뷰를 Qdrant 포인트로 변환
- `upload_points()`: Qdrant에 업로드

---

## 2) 온라인 요청: Strength Extraction API 파이프라인 (핵심)

**API 엔드포인트**: `POST /api/v1/llm/extract/strengths/v2`

입력: `restaurant_id`, (옵션: `category_filter, region_filter, price_band_filter`, `top_k`)

### Step A. 타겟 긍정 근거 후보 수집 (Vector Search / Filter)

**목표:** "장점이 담긴 리뷰 풀"을 모음 (Recall 단계)

* 가장 안정적인 방식:

  1. `restaurant_id`로 필터링
  2. 최신/별점/길이 기준으로 샘플링(예: 최신 6개월 + 상위 300개)
  3. (옵션) sentiment가 있으면 positive만

* sentiment가 없다면:

  * 간단한 LLM/규칙으로 "긍정 문장 포함 리뷰"만 선별하거나,
  * 그냥 전부 모은 뒤 Step B에서 LLM이 장점만 추출

**구현 위치**: `src/strength_extraction.py`
- `collect_positive_evidence_candidates()`: 근거 후보 수집

### Step B. 강점 후보 생성 (LLM 1회, 구조화 출력)

**목표:** "맛있다/좋다"가 아니라 **aspect(무엇이 좋은지)** 를 뽑음

* LLM 입력: 후보 리뷰 텍스트(토큰 제한 고려해 샘플링/요약)
* LLM 출력(JSON 고정):

  * `[{aspect, claim, evidence_quotes[], evidence_review_ids[]}]`
  * 예: aspect="불맛", claim="숫불향과 화력이 좋아 불맛이 강함"

> 여기서 "장점 후보 리스트"가 생김

**구현 위치**: `src/strength_extraction.py`
- `extract_strength_candidates()`: LLM으로 강점 후보 생성

### Step C. 강점별 근거 확장/검증 (Aspect → Qdrant 벡터 검색)

**목표:** LLM이 뽑은 강점이 **실제 리뷰에서 반복되는지** 검증 + 근거 더 모으기 (Precision 단계)

각 aspect에 대해:

1. 쿼리 문장 생성: `"불맛이 강함"`, `"숫불향 좋다"` 같은 짧은 문장
2. Qdrant 검색(restaurant_id 필터)으로 top-N 근거 리뷰/문장 확보
3. support 계산:

   * `support_count`, `support_ratio`, (옵션) 평균 rating, recency 가중치

**통과 조건 예시**

* support_count < 5 → 버림(희소 환각 가능성)
* 근거들이 너무 분산(서로 다른 얘기) → 버림 또는 claim 수정

**구현 위치**: `src/strength_extraction.py`
- `expand_and_validate_evidence()`: 근거 확장 및 검증
- `_calculate_consistency()`: 일관성 계산
- `_calculate_recency_weight()`: 최근 가중치 계산

### Step D. 의미 중복 제거 (Connected Components)

**목표:** "불맛/숫불향/화력"처럼 사실상 같은 강점을 합치기

**프로세스:**

1. **유사도 그래프 만들기**: 모든 pair (i, j)에 대해 cosine sim 계산
   - `sim(i,j) >= T_high (0.88)`: 즉시 merge
   - `T_low (0.82) <= sim(i,j) < T_high`: 가드레일 적용
2. **Connected Components로 그룹 생성**: Union-Find 알고리즘 사용
   - 체인 케이스(A-B 0.86, B-C 0.86, A-C 0.82)도 한 그룹으로 묶임
3. **클러스터별 병합**:
   - Evidence 합치기 (리뷰 ID dedup)
   - 대표 벡터 재계산 (evidence 리뷰 벡터의 centroid)
   - 대표 aspect명 선정 (support_count 가장 큰 member)
4. **가드레일 (과병합 방지)**:
   - **이중 임계값**: T_high=0.88 (즉시 union), T_low=0.82~0.88 (가드레일)
   - **Evidence overlap**: 두 aspect의 evidence 리뷰가 일정 비율(30%) 이상 겹치면 merge

**구현 위치**: `src/strength_extraction.py`
- `merge_similar_strengths()`: Connected Components (Union-Find) 방식으로 중복 제거
- `_union_find()`: Union-Find 알고리즘으로 Connected Components 찾기
- `_calculate_evidence_overlap()`: Evidence 겹침 비율 계산
- `_compute_evidence_centroid()`: Evidence 리뷰 벡터의 centroid 계산

---

## 3) 비교군 기반 "차별 강점" 계산 (Distinct Layer)

### Step E. 비교군 구성

* 기준 필터:

  * same `category`
  * (가능하면) `region`, `price_band`까지 맞추기
* 비교 음식점 M개 샘플(예: 20~100)

**구현 위치**: `src/strength_extraction.py`
- `_find_comparison_restaurants()`: 대표 벡터 기반 유사 레스토랑 검색

### Step F. 비교군 강점 인덱스 만들기 (캐시/사전계산 강추)

비교군까지 실시간으로 Step B~D를 매번 돌리면 비싸요. 추천은:

* **오프라인/배치로** 음식점별 strength profile을 만들어 저장:

  * `restaurant_id -> [{aspect, vector, support_stats}]`
* Qdrant에 "strength-aspect 컬렉션"을 따로 만들어도 좋음

**현재 구현**: 간단한 대표 벡터 기반 비교 (실시간 계산)
**향후 개선**: 오프라인/배치로 strength profile 캐싱

### Step G. 타겟 aspect vs 비교군 aspect 유사도

각 타겟 강점 aspect 벡터에 대해:

* 비교군 전체 aspect 벡터 풀에서 `max_sim` 구하기
* `distinct = 1 - max_sim`

> **max_sim**이 핵심임: "비교군 중 가장 비슷한 강점이 있나?"를 보는 게 가장 직관적

**구현 위치**: `src/strength_extraction.py`
- `calculate_distinct_strengths()`: 차별 강점 계산

### Step H. 최종 점수(대표성 + 차별성 + 일관성)

추천 점수:

* `rep = log(1 + support_count) * positivity(optional) * recency(optional)`
* `final = rep * (1 + alpha * distinct)`

  * alpha는 0.5~2.0 사이 튜닝

**구현 위치**: `src/strength_extraction.py`
- `calculate_distinct_strengths()`: 최종 점수 계산

---

## 4) 최종 응답 포맷 (설득력 있는 결과)

각 강점마다 꼭 포함:

* `aspect` (짧게)
* `claim` (1문장)
* `support_count`, `support_ratio`
* `distinct_score` (distinct일 때만)
* `closest_competitor_sim`(max_sim 값, distinct일 때만)
* `evidence` (리뷰 id + 짧은 스니펫 3~5개)

**구현 위치**: `src/models.py`
- `StrengthDetail`: 강점 상세 정보 모델
- `EvidenceSnippet`: 근거 스니펫 모델
- `StrengthResponseV2`: 최종 응답 모델

---

## 5) 캐싱/성능 운영 팁 (프로덕션 포인트)

* 타겟 음식점 결과 캐시(TTL 1~7일)
* 비교군 strength profile은 배치로 미리 생성
* "리뷰가 새로 들어왔을 때만" 증분 업데이트
* LLM 호출 최소화:

  * 온라인: Step B 1회 + (선택) merge 라벨링 1회
  * 나머지는 벡터 검색/계산으로 처리

**현재 구현**: 실시간 계산 (캐싱 미구현)
**향후 개선**: Redis 캐싱 또는 DB 캐싱 추가

---

## 6) 평가(precision@k로 붙이기 좋은 형태)

* GT: "이 음식점의 대표 강점 Top-K"를 사람이 라벨링(또는 소량)
* 측정:

  * 대표 강점: precision@k, coverage
  * 차별 강점: "비교군에도 흔한 강점인데 유니크로 뽑히는지" false-positive 비율

---

## 7) 구현 상태

### 완료된 기능

✅ Step A: 타겟 긍정 근거 후보 수집
✅ Step B: 강점 후보 생성 (LLM 구조화 출력)
✅ Step C: 강점별 근거 확장/검증
✅ Step D: 의미 중복 제거 (클러스터링)
✅ Step E~H: 비교군 기반 차별 강점 계산
✅ API 엔드포인트: `POST /api/v1/llm/extract/strengths/v2`
✅ 모델 정의: `StrengthRequestV2`, `StrengthResponseV2`, `StrengthDetail`, `EvidenceSnippet`

### 향후 개선 사항

- [ ] 비교군 strength profile 오프라인/배치 생성 및 캐싱
- [ ] 타겟 결과 캐싱 (Redis 또는 DB)
- [ ] 문장 단위 임베딩 지원
- [ ] 평가 메트릭 수집 및 모니터링
- [ ] 성능 최적화 (벡터 검색 최적화, 배치 처리)

---

## 8) 사용 예시

### API 요청

```bash
curl -X POST "http://localhost:8000/api/v1/llm/extract/strengths/v2" \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant_id": 123,
    "strength_type": "both",
    "category_filter": 1,
    "top_k": 10,
    "max_candidates": 300,
    "months_back": 6,
    "min_support": 5
  }'
```

### 응답 예시

```json
{
  "restaurant_id": 123,
  "strength_type": "both",
  "strengths": [
    {
      "aspect": "불맛",
      "claim": "숫불향과 화력이 좋아 불맛이 강함",
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
        }
      ],
      "final_score": 15.2
    }
  ],
  "total_candidates": 300,
  "validated_count": 5
}
```

---

## 9) 관련 파일

- `src/strength_extraction.py`: 강점 추출 파이프라인 구현
- `src/models.py`: API 모델 정의
- `src/api/routers/llm.py`: API 엔드포인트
- `src/review_utils.py`: 리뷰 전처리 함수
- `src/vector_search.py`: 벡터 검색 및 Qdrant 연동
