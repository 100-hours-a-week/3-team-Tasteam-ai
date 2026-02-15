# LLM 서비스 설계 문서

## 목차
1. [개요](#개요)
2. [멀티스텝 체인/파이프라인 다이어그램](#멀티스텝-체인파이프라인-다이어그램)
3. [사용한 모델·도구·프레임워크](#사용한-모델도구프레임워크)
4. [구현 코드](#구현-코드)
5. [멀티스텝 구조 도입 이유 및 기대 효과](#멀티스텝-구조-도입-이유-및-기대-효과)

---

## 개요

본 프로젝트는 **LangChain이나 유사 프레임워크를 사용하지 않고**, 직접 구현한 멀티스텝 LLM 파이프라인을 통해 레스토랑 리뷰 분석 서비스를 제공합니다.

### 주요 특징
- **RAG (Retrieval-Augmented Generation) 패턴**: 벡터 검색 + LLM 추론
- **구조화된 강점 추출**: Step A~H 파이프라인 (근거 수집, LLM 추출, 검증, 클러스터링, 비교군 기반 차별 강점 계산)
- **배치 처리 최적화**: 동적 배치 크기 + 비동기 큐 방식
- **우선순위 큐**: Prefill 비용 기반 태스크 스케줄링으로 SLA 보호
- **vLLM 메트릭 수집**: Prefill/Decode 분리 측정, TTFT, TPS, TPOT 계산
- **프레임워크 독립적**: LangChain 없이 직접 구현하여 유연성 확보

### 서비스 기능
1. **감성 분석**: 리뷰 → 긍/부정 비율 추출
2. **리뷰 요약**: 벡터 검색으로 긍/부정 리뷰 검색 → LLM 요약
3. **강점 추출**: 구조화된 파이프라인 (Step A~H)으로 타겟 레스토랑 강점 추출

---

## 멀티스텝 체인/파이프라인 다이어그램

### 1. 감성 분석 파이프라인 (단일 스텝)

```
┌─────────────────────────────────────────────────────────────┐
│                    감성 분석 파이프라인                      │
└─────────────────────────────────────────────────────────────┘

입력: restaurant_id, reviews (List[Dict])
  │
  ▼
┌─────────────────┐
│ 1. 데이터 준비   │
│ - content_list  │
│   추출          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 동적 배치     │
│    크기 계산     │
│ - 리뷰 길이 기반 │
│ - 토큰 수 제한   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 배치 분할     │
│ - 여러 배치로    │
│   나누기         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. 비동기 큐     │
│    처리          │
│ - 세마포어 제한  │
│ - vLLM          │
│   Continuous    │
│   Batching      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. LLM 추론     │
│ - 프롬프트 생성  │
│ - 긍/부정 개수   │
│   집계          │
│   (LLM 출력:    │
│   positive_     │
│   count,        │
│   negative_     │
│   count만)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. 결과 집계     │
│   및 비율 계산   │
│ - 스케일링 조정  │
│   (LLM이 판단    │
│   못한 리뷰      │
│   있을 경우):    │
│   total_judged = │
│   positive_count │
│   + negative_    │
│   count          │
│   scale = len(   │
│   review_list) / │
│   total_judged   │
│   (if total_     │
│   judged > 0)    │
│   positive_count │
│   = round(       │
│   positive_count │
│   * scale)       │
│   negative_count │
│   = round(       │
│   negative_count │
│   * scale)       │
│ - 최종 비율 계산:│
│   positive_ratio │
│   = (positive_   │
│   count / total_ │
│   count) * 100   │
│   negative_ratio │
│   = (negative_   │
│   count / total_ │
│   count) * 100   │
└────────┬────────┘
         │
         ▼
출력: {
  restaurant_id,
  positive_count,    // LLM 반환 후 스케일링 조정
  negative_count,    // LLM 반환 후 스케일링 조정
  total_count,       // len(review_list)
  positive_ratio,    // 코드 계산
  negative_ratio     // 코드 계산
}
```
```

### 2. 리뷰 요약 파이프라인 (RAG 패턴)

```
┌─────────────────────────────────────────────────────────────┐
│                    리뷰 요약 파이프라인                      │
└─────────────────────────────────────────────────────────────┘

입력: restaurant_id, limit (positive_query, negative_query는 deprecated)
  │
  ▼
┌─────────────────────────────────────────┐
│ 1. 대표 벡터 기반 TOP-K 리뷰 선택       │
│    (Retrieval)                           │
│ - 레스토랑 대표 벡터 계산                │
│ - 대표 벡터 주위 TOP-K 리뷰 검색        │
│   (query_by_restaurant_vector)          │
│ - 기본값: limit * 2 (긍정/부정 포함)    │
│ - 대표 벡터 기반이므로 대부분 긍정적    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 2. Aspect 기반 요약 (LLM 추론)          │
│    (Generation)                           │
│ - LLM이 TOP-K 리뷰에서 aspect 추출      │
│   - positive_aspects: 긍정 aspect       │
│   - negative_aspects: 부정 aspect       │
│ - 각 aspect: aspect, claim, evidence    │
│ - overall_summary: aspect 기반 전체 요약│
└────────┬────────────────────────────────┘
         │
         ▼
출력: {
  restaurant_id,
  overall_summary,      // LLM 반환 (aspect 기반)
  positive_aspects,     // LLM 반환 (구조화된 aspect)
  negative_aspects,     // LLM 반환 (구조화된 aspect)
  positive_reviews,     // 메타데이터
  negative_reviews,     // 메타데이터
  positive_count,       // 코드 계산
  negative_count        // 코드 계산
}
```
```

### 3. 강점 추출 파이프라인 (구조화된 Step A~H 파이프라인)

```
┌─────────────────────────────────────────────────────────────┐
│          강점 추출 파이프라인 (Step A~H)                      │
└─────────────────────────────────────────────────────────────┘

입력: restaurant_id, strength_type, category_filter, ...
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step A: 타겟 긍정 근거 후보 수집                │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 1. Qdrant 필터  │
│    검색          │
│ - restaurant_id │
│   필터          │
│ - 최신 6개월     │
│ - 상위 300개    │
│ - (옵션) rating │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step B: 강점 후보 생성 (LLM 구조화 출력)        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 1. LLM 추론     │
│ - 프롬프트 생성  │
│ - 구조화 출력    │
│   (aspect,      │
│   claim,        │
│   evidence_     │
│   quotes,       │
│   evidence_     │
│   review_ids)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step C: 강점별 근거 확장/검증                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 1. 각 aspect별  │
│    Qdrant 검색  │
│ - 쿼리 생성      │
│   (aspect 기반) │
│ - restaurant_id │
│   필터          │
│ - top-N 근거    │
│   확보          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Support      │
│    계산         │
│ - support_count │
│ - support_ratio │
│ - avg rating    │
│ - recency       │
└────────┬────────┘
  │
  ▼
┌─────────────────┐
│ 3. 필터링       │
│ - min_support   │
│   통과 확인      │
│ - 일관성 검증    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step D: 의미 중복 제거 (Connected Components)  │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 1. 유사도 그래프 │
│    생성          │
│ - 모든 pair     │
│   cosine sim    │
│ - T_high=0.88   │
│   즉시 merge    │
│ - T_low=0.82    │
│   가드레일      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Connected    │
│    Components   │
│    (Union-Find) │
│ - 체인 케이스    │
│   처리          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. 클러스터별    │
│    병합          │
│ - evidence 합치기│
│ - 대표 벡터      │
│   재계산        │
│   (centroid)    │
│ - 대표 aspect   │
│   선정          │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│        Step D-1: Claim 후처리 재생성 (템플릿 보정 + LLM)     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ 1. 템플릿 기반   │
│    보정          │
│ - LLM 없이      │
│   가능          │
│ - 15-28자 범위  │
│ - 메타 표현      │
│   통일          │
│   ("언급이 많음")│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. LLM 기반      │
│    생성 (선택)   │
│ - 템플릿 실패 시 │
│ - 맛 claim은    │
│   구체명사 포함  │
│   (국물/면/     │
│   유자라멘 등)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│        Step E~H: 비교군 기반 차별 강점 계산 (distinct일 때만)│
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Step E: 비교군  │
│    구성          │
│ - category      │
│   필터          │
│ - 대표 벡터 기반 │
│   유사 레스토랑  │
│   검색          │
│ - Top-20 선정   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step F: 비교군  │
│    강점 인덱스   │
│    (실시간 계산) │
│ - 각 비교군      │
│   강점 추출      │
│   (간단 요약)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step G: 타겟 vs │
│    비교군 유사도 │
│ - 각 타겟 aspect│
│   벡터           │
│ - 비교군 aspect  │
│   풀에서 max_sim │
│ - distinct =    │
│   1 - max_sim   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step H: 최종    │
│    점수 계산     │
│ - rep = log(1+  │
│   support_count)│
│   * recency     │
│ - final = rep * │
│   (1 + alpha *  │
│   distinct)     │
└────────┬────────┘
         │
         ▼
출력: {
  restaurant_id,
  strength_type,
  strengths: [
    {
      aspect,
      claim,
      strength_type,
      support_count,
      support_ratio,
      distinct_score,      // distinct일 때만
      closest_competitor_  // distinct일 때만
      sim,
      closest_competitor_id,
      evidence: [          // 3~5개 스니펫
        {
          review_id,
          snippet,
          rating,
          created_at
        }
      ],
      final_score
    }
  ],
  total_candidates,
  validated_count
}
```

**Step A~H 파이프라인 특징:**
- **Step A**: 타겟 긍정 근거 후보 수집 (Qdrant 필터링으로 효율적 수집)
- **Step B**: LLM으로 구조화된 강점 후보 생성 (aspect, claim, evidence)
- **Step C**: Qdrant 벡터 검색으로 근거 확장 및 검증 (support_count, support_ratio)
- **Step D**: Connected Components (Union-Find)로 의미 중복 제거
  - 이중 임계값 (T_high=0.88, T_low=0.82)
  - Evidence overlap 가드레일 (30%)
- **Step E~H**: 비교군 기반 차별 강점 계산 (representative vs distinct)
  - Step E: 비교군 구성 (category 필터 + 대표 벡터 검색)
  - Step F: 비교군 강점 인덱스 (실시간 계산 또는 캐시)
  - Step G: 타겟 vs 비교군 유사도 (distinct = 1 - max_sim)
  - Step H: 최종 점수 계산 (rep * (1 + alpha * distinct))

### 4. 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (API Consumer)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  API Routers                                          │ │
│  │  - /api/v1/sentiment/analyze                          │ │
│  │  - /api/v1/llm/summarize                             │ │
│  │  - /api/v1/llm/extract/strengths                     │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Vector       │    │ LLM Utils    │    │ Sentiment   │
│ Search       │    │              │    │ Analyzer    │
│              │    │              │    │             │
│ - Qdrant     │    │ - vLLM       │    │ - LLM       │
│ - Sentence   │    │ - Qwen2.5-   │    │   추론      │
│   Transformer│    │   7B-        │    │             │
│              │    │   Instruct   │    │             │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 사용한 모델·도구·프레임워크

### 1. LLM 모델

#### Qwen2.5-7B-Instruct
- **선택 이유**:
  - 한국어 성능 우수
  - 긴 컨텍스트 길이 지원 (131,072 tokens)
  - 7B 파라미터로 GPU 메모리 효율적 (약 14GB)
  - Instruction tuning으로 구조화된 출력 가능
- **기대 효과**:
  - 한국어 리뷰 분석 정확도 향상
  - 긴 리뷰 리스트를 한 번에 처리 가능
  - JSON 형식 출력으로 파싱 안정성 확보
- **실제 성능 지표** (벤치마크 측정값):
  - **감성 분석**: 평균 0.843초, P95 0.874초, 처리량 1.19 req/s (목표 달성 ✅)
  - **배치 감성 분석**: 평균 9.15초 (목표: 5.0-10.0초, 달성 ✅)
  - **리뷰 요약**: 평균 0.629초, P95 0.639초, 처리량 1.59 req/s (목표 달성 ✅)
  - **배치 리뷰 요약**: 평균 83.05초 (목표: 5.0-10.0초, 최적화 필요 ⚠️)
  - **강점 추출**: 평균 0.614초, P95 0.653초, 처리량 1.63 req/s (목표 달성 ✅)
  - **리뷰 이미지 검색**: 평균 0.614초, P95 0.649초, 처리량 1.63 req/s (목표 달성 ✅)
  - **테스트 통과율**: 6/6 (100% 성공률) ✅

### 2. LLM 서빙 프레임워크

#### vLLM (v0.3.3+)
- **선택 이유**:
  - Continuous Batching 자동 지원
  - PagedAttention으로 메모리 효율성
  - 높은 처리량 (throughput)
  - GPU 서버 환경에서 로컬 실행 가능
- **기대 효과**:
  - GPU 활용률 70-90% 달성
  - 처리량 5배 향상 (2 req/s → 10 req/s)
  - 네트워크 오버헤드 제거 (로컬 실행)

### 3. 벡터 데이터베이스

#### Qdrant (on-disk 모드)
- **선택 이유**:
  - HNSW 구조로 빠른 유사도 검색
  - on-disk 모드로 메모리 효율성
  - MMAP, 페이지 캐시 활용 (OS 레벨 최적화)
  - 대규모 컬렉션에 실용적 성능
- **기대 효과**:
  - RAM 사용 최소화로 클라우드 비용 절감
  - 데이터 영속성 및 안정성 확보
  - 빠른 벡터 검색 (0.5-0.6초)

### 4. 임베딩 모델

#### SentenceTransformer (jhgan/ko-sbert-multitask)
- **선택 이유**:
  - 한국어 특화 모델
  - 멀티태스크 학습으로 다양한 도메인 적응
  - GPU + FP16 최적화 지원
- **기대 효과**:
  - 한국어 리뷰 의미 검색 정확도 향상
  - 메모리 50% 절감 (FP16)
  - 배치 처리로 처리 속도 향상

### 5. 웹 프레임워크

#### FastAPI
- **선택 이유**:
  - 비동기 지원 (async/await)
  - 자동 API 문서 생성 (Swagger/ReDoc)
  - Pydantic으로 타입 검증
  - 높은 성능
- **기대 효과**:
  - 비동기 처리로 동시 요청 처리 가능
  - 타입 안정성 확보
  - 개발 생산성 향상

### 6. 프레임워크 선택: LangChain 미사용

#### LangChain을 사용하지 않은 이유
- **유연성**: 직접 구현으로 세밀한 제어 가능
- **성능**: 불필요한 추상화 레이어 제거
- **의존성**: 추가 라이브러리 의존성 최소화
- **디버깅**: 코드 흐름 추적 용이
- **최적화**: 프로젝트 특화 최적화 가능

#### 직접 구현의 장점
- **커스터마이징**: 프로젝트 요구사항에 맞춘 최적화
- **성능 최적화**: 동적 배치 크기, 세마포어 등 세밀한 제어
- **메모리 관리**: OOM 방지 전략 직접 구현
- **비용 최적화**: GPU 서버 자동 종료 등 비용 관리

---

## 구현 코드

### 1. 멀티스텝 파이프라인: 강점 추출 (Step A~H 구조화된 파이프라인)

```python
async def extract_strengths(
    self,
    restaurant_id: int,
    strength_type: str = "both",
    category_filter: Optional[int] = None,
    region_filter: Optional[str] = None,
    price_band_filter: Optional[str] = None,
    top_k: int = 10,
    max_candidates: int = 300,
    months_back: int = 6,
    min_support: int = 5,
) -> Dict[str, Any]:
    """
    구조화된 강점 추출 파이프라인 (Step A~H)
    
    프로세스:
    1. Step A: 타겟 긍정 근거 후보 수집 (대표 벡터 TOP-K + 다양성 샘플링)
    2. Step B: 강점 후보 생성 (LLM 구조화 출력, 최소 5개 보장)
    3. Step C: 강점별 근거 확장/검증 (Qdrant 벡터 검색, 유효 근거 수 계산, 긍정 리뷰만)
    4. Step D: 의미 중복 제거 (Connected Components, aspect type 체크)
    5. Step D-1: Claim 후처리 재생성 (템플릿 보정 + LLM, 15-28자, 메타 표현 통일, 맛 claim은 구체명사 포함)
    6. Step E~H: 비교군 기반 차별 강점 계산 (distinct일 때만)
    7. Top-K 선택 (both 모드): 쿼터 적용, 같은 타입 중복 방지
    """
    start_time = time.time()
    
    # Step A: 타겟 긍정 근거 후보 수집
    evidence_candidates = self.collect_positive_evidence_candidates(
        restaurant_id=restaurant_id,
        max_candidates=max_candidates,
        months_back=months_back,
    )
    
    if not evidence_candidates:
        return {
            "restaurant_id": restaurant_id,
            "strength_type": strength_type,
            "strengths": [],
            "total_candidates": 0,
            "validated_count": 0,
        }
    
    # Step B: 강점 후보 생성 (LLM 구조화 출력)
    strength_candidates = self.extract_strength_candidates(
        evidence_candidates=evidence_candidates,
        max_tokens=4000,
    )
    
    if not strength_candidates:
        return {
            "restaurant_id": restaurant_id,
            "strength_type": strength_type,
            "strengths": [],
            "total_candidates": len(evidence_candidates),
            "validated_count": 0,
        }
    
    # Step C: 강점별 근거 확장/검증
    validated_strengths = self.expand_and_validate_evidence(
        strength_candidates=strength_candidates,
        restaurant_id=restaurant_id,
        min_support=min_support,
    )
    
    if not validated_strengths:
            return {
                "restaurant_id": restaurant_id,
            "strength_type": strength_type,
            "strengths": [],
            "total_candidates": len(evidence_candidates),
            "validated_count": 0,
            }
    
    # Step D: 의미 중복 제거 (Connected Components/Union-Find)
    merged_strengths = self.merge_similar_strengths(
        validated_strengths=validated_strengths,
        threshold_high=0.88,
        threshold_low=0.82,
        evidence_overlap_threshold=0.3,
    )
    
    # Step D-1: Claim 후처리 재생성 (템플릿 보정 + LLM)
    merged_strengths = self.regenerate_claims(merged_strengths)
    
    # Step E~H: 비교군 기반 차별 강점 계산 (distinct 또는 both일 때만)
    if strength_type in ["distinct", "both"]:
        distinct_strengths = self.calculate_distinct_strengths(
            target_strengths=merged_strengths,
            restaurant_id=restaurant_id,
            category_filter=category_filter,
            region_filter=region_filter,
            price_band_filter=price_band_filter,
        )
    
        # distinct 타입만 반환하거나, both일 때는 distinct_strengths 사용
        if strength_type == "distinct":
            merged_strengths = distinct_strengths
        else:  # both
            # representative와 distinct를 합침
            merged_strengths = merged_strengths + distinct_strengths
    
    # 최종 점수로 정렬 및 Top-K 선택
    merged_strengths.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    final_strengths = merged_strengths[:top_k]
    
    # 응답 형식으로 변환
    strengths = []
    for strength in final_strengths:
        strengths.append({
            "aspect": strength.get("aspect", ""),
            "claim": strength.get("claim", ""),
            "strength_type": strength.get("strength_type", "representative"),
            "support_count": strength.get("support_count", 0),
            "support_ratio": strength.get("support_ratio", 0.0),
            "distinct_score": strength.get("distinct_score"),
            "closest_competitor_sim": strength.get("closest_competitor_sim"),
            "closest_competitor_id": strength.get("closest_competitor_id"),
            "evidence": strength.get("evidence", []),
            "final_score": strength.get("final_score", 0.0),
        })
            
            return {
                "restaurant_id": restaurant_id,
        "strength_type": strength_type,
        "strengths": strengths,
        "total_candidates": len(evidence_candidates),
        "validated_count": len(merged_strengths),
        "processing_time_ms": (time.time() - start_time) * 1000,
    }
```

### 2. 동적 배치 크기 계산

```python
@classmethod
def calculate_dynamic_batch_size(
    cls,
    reviews: List[str],
    max_tokens_per_batch: Optional[int] = None
) -> int:
    """
    리뷰 길이 기반 동적 배치 크기 계산
    
    - 리뷰당 평균 토큰 수 추정 (한국어 기준 약 3.5 문자/토큰)
    - max_tokens_per_batch 제한 내에서 최적 배치 크기 계산
    - 최소/최대 배치 크기 제한 적용
    """
    if not reviews:
        return cls.VLLM_DEFAULT_BATCH_SIZE
    
    if max_tokens_per_batch is None:
        max_tokens_per_batch = cls.VLLM_MAX_TOKENS_PER_BATCH
    
    # 샘플링하여 평균 토큰 수 추정
    sample_size = min(50, len(reviews))
    sample_reviews = reviews[:sample_size]
    
    # 문자 수 기반 추정
    total_chars = sum(len(review) for review in sample_reviews)
    avg_chars_per_review = total_chars / sample_size if sample_size > 0 else 100
    
    # 평균 토큰 수 추정 (한국어 기준 약 3.5 문자/토큰)
    avg_tokens_per_review = max(1, int(avg_chars_per_review / 3.5))
    
    # 배치 크기 계산
    calculated_batch_size = max(1, int(max_tokens_per_batch / avg_tokens_per_review))
    
    # 최소/최대 제한 적용
    batch_size = max(
        cls.VLLM_MIN_BATCH_SIZE,
        min(calculated_batch_size, cls.VLLM_MAX_BATCH_SIZE)
    )
    
    return batch_size
```

### 3. 비동기 vLLM 추론 (메트릭 수집 포함)

```python
async def _generate_with_vllm(
    self,
    prompts: List[str],
    temperature: float = 0.1,
    max_tokens: int = 100,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    vLLM을 사용하여 비동기로 응답 생성 + 메트릭 수집
    
    ThreadPoolExecutor를 사용하여 vLLM의 동기 API를 비동기로 변환
    Prefill/Decode 시간 분리 측정 및 TTFT, TPS, TPOT 계산
    
    Returns:
        (생성된 응답 리스트, 메트릭 딕셔너리)
    """
    if not self.use_gpu_server_vllm:
        raise ValueError("vLLM이 초기화되지 않았습니다.")
    
    from vllm import SamplingParams
    
    # SamplingParams 설정
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # 동기 메서드를 비동기로 실행
    loop = asyncio.get_event_loop()
    outputs = await loop.run_in_executor(
        self.executor,
        self.llm.generate,
        prompts,
        sampling_params
    )
    
    # 결과 추출
    responses = [output.outputs[0].text for output in outputs]
    
    # 메트릭 수집 (Prefill/Decode 분리)
    metrics = {
        "avg_prefill_time_ms": 0.0,
        "avg_decode_time_ms": 0.0,
        "total_tokens": 0,
        "ttft_ms": 0.0,
        "tps": 0.0,
        "avg_tpot_ms": 0.0
    }
    
    total_prefill_time = 0
    total_decode_time = 0
    total_tokens = 0
    
    for output in outputs:
        # vLLM metrics에서 Prefill/Decode 시간 추출
        if hasattr(output, 'metrics') and output.metrics:
            first_token_time = getattr(output.metrics, 'first_token_time', 0)
            finished_time = getattr(output.metrics, 'finished_time', 0)
            
            prefill_time_ms = first_token_time * 1000 if first_token_time > 0 else 0
            decode_time_ms = (finished_time - first_token_time) * 1000 if finished_time > first_token_time else 0
            
            total_prefill_time += prefill_time_ms
            total_decode_time += decode_time_ms
            
            # 토큰 수 계산
            n_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
            total_tokens += n_tokens
    
    n_requests = len(outputs)
    if n_requests > 0:
        metrics["avg_prefill_time_ms"] = total_prefill_time / n_requests
        metrics["avg_decode_time_ms"] = total_decode_time / n_requests
        metrics["total_tokens"] = total_tokens
        metrics["ttft_ms"] = metrics["avg_prefill_time_ms"]  # TTFT = Prefill 시간
        metrics["tps"] = total_tokens / (total_decode_time / 1000) if total_decode_time > 0 else 0
        metrics["avg_tpot_ms"] = (total_decode_time / total_tokens) if total_tokens > 0 else 0
    
    return responses, metrics
```

### 3-1. Prefill 비용 추정

```python
def _estimate_prefill_cost(self, prompt: str) -> int:
    """
    프롬프트의 Prefill 비용을 추정합니다.
    
    Prefill 비용은 입력 토큰 수에 비례하므로, 프롬프트의 토큰 수를 반환합니다.
    
    Args:
        prompt: 프롬프트 문자열
        
    Returns:
        추정된 Prefill 비용 (토큰 수)
    """
    from src.review_utils import estimate_tokens
    return estimate_tokens(prompt)
```

### 4. 프롬프트 구성 예시

```python
def _create_sentiment_prompt(self, restaurant_id: Union[int, str], reviews: List[str]) -> str:
    """
    감성 분석 프롬프트 생성
    
    LLM은 개수만 반환하고, 비율은 코드에서 계산합니다.
    """
    reviews_text = "\n".join([f"{i+1}. {review}" for i, review in enumerate(reviews)])
    
    prompt = f"""레스토랑 ID: {restaurant_id}

다음 리뷰들을 분석하여 긍정과 부정 개수를 세어주세요.
각 리뷰는 하나의 긍정 또는 부정으로 분류됩니다.

리뷰들:
{reviews_text}

JSON 형식으로 응답하세요 (개수만 반환):
{{
  "positive_count": <긍정 개수>,
  "negative_count": <부정 개수>
}}

참고: 비율은 코드에서 계산합니다 (positive_ratio = positive_count / total_count * 100)."""
    
    return prompt

def _create_summary_prompt(self, positive_texts: List[str], negative_texts: List[str]) -> str:
    """
    리뷰 요약 프롬프트 생성
    
    LLM은 overall_summary만 반환합니다.
    """
    positive_text = "\n".join([f"- {text}" for text in positive_texts])
    negative_text = "\n".join([f"- {text}" for text in negative_texts])
    
    prompt = f"""다음 긍정/부정 리뷰를 모두 고려하여 전체 요약을 생성하세요.

긍정 리뷰:
{positive_text}

부정 리뷰:
{negative_text}

JSON 형식으로 응답하세요 (overall_summary만 반환):
{{
  "overall_summary": "<전체 요약 (긍정과 부정을 모두 고려한 통합 요약)>"
}}"""
    
    return prompt

def _create_strength_candidate_prompt(self, evidence_texts: List[str]) -> str:
    """
    Step B: 강점 후보 생성 프롬프트 (구조화된 JSON 출력)
    
    LLM은 aspect, claim, evidence_quotes, evidence_review_ids를 포함한 구조화된 JSON을 반환합니다.
    """
    evidence_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(evidence_texts)])
    
    prompt = f"""다음 리뷰들을 읽고 이 레스토랑의 강점을 추출하세요.

리뷰들:
{evidence_text}

각 강점을 다음과 같은 구조화된 형식으로 출력하세요:
- aspect: 강점의 카테고리 (예: "불맛", "서비스", "분위기")
- claim: 구체적 주장 (1문장, 15-28자, 모바일 카드 1줄 기준, 예: "유자라멘 국물이 진하다는 언급이 많음")
- evidence_quotes: 근거가 되는 리뷰 문장들 (2~3개)
- evidence_review_ids: 근거 리뷰 ID들

JSON 형식으로 응답하세요:
{{
  "strengths": [
    {{
      "aspect": "불맛",
      "claim": "유자라멘 국물이 진하다는 언급이 많음",
      "evidence_quotes": ["숫불향이 진해서 맛있어요", "화력이 좋아 고기가 맛있네요"],
      "evidence_review_ids": ["rev_1", "rev_2"]
    }}
  ]
}}"""
    
    return prompt
```

---

## Trade-off 분석

### Step A~H 구조화된 강점 추출 파이프라인

**Step A~H 파이프라인 특징:**

| Step | 목적 | 장점 | Trade-off |
|------|------|------|-----------|
| **Step A** | 근거 후보 수집 | Qdrant 필터링으로 효율적 | 메모리 사용량 증가 (후보 저장) |
| **Step B** | LLM 구조화 출력 | aspect, claim, evidence 명확화 | LLM 호출 1회 추가 |
| **Step C** | 근거 확장/검증 | support_count, support_ratio 객관화 | Qdrant 검색 N회 (aspect별) |
| **Step D** | 의미 중복 제거 | Connected Components로 정확한 클러스터링 | 유사도 계산 O(N²) |
| **Step E~H** | 차별 강점 계산 | distinct_score로 객관적 비교 | 비교군 강점 계산 오버헤드 |

**Step D (Connected Components) 선택 근거:**
- **Union-Find 알고리즘**: O(N log N) 시간 복잡도
- **이중 임계값 (T_high=0.88, T_low=0.82)**: 과병합 방지
- **Evidence overlap 가드레일 (30%)**: 정확한 병합 판단
- **DBSCAN 대비 장점**: 체인 케이스 처리 우수 (A-B, B-C 유사 → A-B-C 하나로)

### 우선순위 큐 vs FIFO 큐

| 기준 | FIFO 큐 | 우선순위 큐 (Prefill 비용) |
|------|---------|-------------------------|
| **공정성** | 높음 (순서 보장) | 낮음 (작은 요청 우선) |
| **SLA 준수율** | 낮음 (큰 요청 블로킹) | 높음 (작은 요청 보호) |
| **평균 응답 시간** | 높음 | 낮음 |
| **큰 요청 지연** | 낮음 | 높음 |
| **구현 복잡도** | 낮음 | 중간 |

**선택 근거**: SLA 보호가 공정성보다 중요 → 우선순위 큐 선택

---

## 멀티스텝 구조 도입 이유 및 기대 효과

### 1. 멀티스텝 구조 도입 이유

#### 1.1. 강점 추출의 복잡성
- **문제**: 타겟 레스토랑의 강점을 추출하려면 근거 수집, LLM 추출, 검증, 중복 제거, 비교군 기반 차별점 계산이 필요
- **해결**: Step A~H 구조화된 파이프라인으로 단계별 처리
  - Step A: 타겟 긍정 근거 후보 수집 (Qdrant 필터링)
  - Step B: LLM으로 구조화된 강점 후보 생성 (aspect, claim, evidence)
  - Step C: Qdrant 벡터 검색으로 근거 확장 및 검증 (support_count, support_ratio)
  - Step D: Connected Components로 의미 중복 제거
  - Step E~H: 비교군 기반 차별 강점 계산 (distinct_score)

#### 1.2. 근거 검증의 필요성
- **문제**: LLM이 생성한 강점이 실제 리뷰에서 반복되는지 검증 필요
- **해결**: Step C에서 Qdrant 벡터 검색으로 근거 확장 및 support_count 계산

#### 1.3. 의미 중복 제거의 정확성
- **문제**: "불맛", "숫불향", "화력"처럼 사실상 같은 강점을 하나로 합쳐야 함
- **해결**: Step D에서 Connected Components (Union-Find) 사용
  - 이중 임계값 (T_high=0.88, T_low=0.82)
  - Evidence overlap 가드레일 (30%)
  - 체인 케이스 처리 (A-B, B-C 유사 → A-B-C 하나로)

#### 1.4. 객관적 차별점 계산
- **문제**: LLM만으로 차별점 계산 시 변동성 높음
- **해결**: Step E~H에서 벡터 기반 유사도 계산 (distinct = 1 - max_sim)

#### 1.5. 구조화된 출력
- **문제**: 비구조화된 텍스트 출력은 파싱이 어렵고 일관성 없음
- **해결**: Step B에서 구조화된 JSON 출력 (aspect, claim, evidence)

### 2. 멀티스텝 구조의 기대 효과

#### 2.1. 정확도 향상
- **효과**: Step C에서 Qdrant 벡터 검색으로 근거 검증 및 확장하여 정확도 향상
- **측정 지표**: 추출된 강점의 관련성, support_count, support_ratio

#### 2.2. 일관성 향상
- **효과**: Step D에서 Connected Components로 의미 중복 제거하여 일관성 향상
- **측정 지표**: 중복 강점 제거율, 클러스터링 정확도

#### 2.3. 객관성 향상
- **효과**: Step E~H에서 벡터 기반 유사도 계산으로 객관적 차별점 도출
- **측정 지표**: distinct_score 일관성, LLM 변동성 감소

#### 2.4. 구조화된 출력
- **효과**: Step B에서 구조화된 JSON 출력으로 파싱 안정성 및 일관성 확보
- **측정 지표**: 파싱 성공률, 출력 형식 일관성

#### 2.5. 근거 검증
- **효과**: Step C에서 support_count, support_ratio 계산으로 신뢰성 있는 강점만 추출
- **측정 지표**: min_support 통과율, 근거 일관성

#### 2.6. 확장성
- **효과**: Step A~H 파이프라인으로 대량 레스토랑 처리 지원
- **측정 지표**: 처리량 (throughput), 응답 시간

### 3. 서비스 요구사항과의 관련성

#### 3.1. 비즈니스 요구사항
- **요구**: 레스토랑의 차별화된 강점 파악
- **해결**: 비교 대상 레스토랑과의 비교를 통한 강점 추출

#### 3.2. 성능 요구사항
- **요구**: 대량 레스토랑 처리 (100개 이상)
- **해결**: 배치 처리 + 비동기 큐 방식으로 확장성 확보

#### 3.3. 정확도 요구사항
- **요구**: 정확한 강점 추출 및 근거 검증
- **해결**: Step A~H 파이프라인으로 근거 수집, 검증, 중복 제거, 차별점 계산

#### 3.4. 비용 요구사항
- **요구**: GPU 사용 비용 최소화
- **해결**: 동적 배치 크기 + 세마포어로 OOM 방지, GPU 서버 자동 종료

#### 3.5. SLA 요구사항
- **요구**: 작은 요청의 응답 시간 보장
- **해결**: 우선순위 큐로 Prefill 비용이 작은 요청부터 처리하여 TTFT 개선

### 4. 멀티스텝이 불필요한 경우

#### 4.1. 감성 분석
- **이유**: 단순한 긍/부정 분류이므로 단일 스텝으로 충분
- **구조**: 단일 LLM 추론으로 처리

#### 4.2. 리뷰 요약
- **이유**: 벡터 검색으로 이미 관련 리뷰만 필터링되어 단일 스텝으로 충분
- **구조**: RAG 패턴 (벡터 검색 + LLM 추론)

### 5. 성능 개선 효과

| 항목 | 단일 스텝 | Step A~H 파이프라인 | 개선율 |
|------|----------|-------------------|--------|
| **정확도** | 중간 | 높음 (근거 검증) | - |
| **일관성** | 낮음 | 높음 (Connected Components) | - |
| **객관성** | 낮음 (LLM 의존) | 높음 (벡터 기반 유사도) | - |
| **구조화된 출력** | 없음 | 있음 (JSON 형식) | - |
| **근거 검증** | 없음 | 있음 (support_count) | - |
| **확장성** | 낮음 | 높음 | - |
| **메모리 효율성** | 중간 | 높음 (Qdrant 필터링) | - |

---

## 결론

본 프로젝트는 **LangChain 없이 직접 구현한 구조화된 멀티스텝 LLM 파이프라인 (Step A~H)**을 통해 다음과 같은 효과를 달성했습니다:

1. **유연성**: 프로젝트 요구사항에 맞춘 세밀한 제어
2. **정확도**: Step C에서 근거 검증 및 확장으로 정확도 향상
3. **일관성**: Step D에서 Connected Components로 의미 중복 제거하여 일관성 향상
4. **객관성**: Step E~H에서 벡터 기반 유사도 계산으로 객관적 차별점 도출
5. **구조화된 출력**: Step B에서 JSON 형식 출력으로 파싱 안정성 확보
6. **근거 검증**: Step C에서 support_count, support_ratio 계산으로 신뢰성 있는 강점만 추출
7. **확장성**: Step A~H 파이프라인으로 대량 레스토랑 처리 지원

구조화된 멀티스텝 파이프라인은 **복잡한 작업(강점 추출)에 필수적**이며, **단순한 작업(감성 분석, 요약)에는 불필요**합니다. 프로젝트의 요구사항에 맞춰 Step A~H로 단계별 처리하여 구현했습니다.
