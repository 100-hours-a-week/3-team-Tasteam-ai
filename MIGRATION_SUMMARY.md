# 파이프라인 교체 완료 요약

## 개요

기존 파이프라인을 새로운 파이프라인으로 완전 교체했습니다.

## 교체 완료된 기능

### 1. ✅ Sentiment Analysis (감성 분석)

**변경 파일**:
- `src/sentiment_analysis.py`
- `src/models.py`
- `src/api/routers/sentiment.py`

**주요 변경사항**:
- HuggingFace 모델: `Dilwolf/Kakao_app-kr_sentiment` 사용
- Negative만 LLM으로 재판정 (비용 절감)
- Neutral 처리 추가
- API 응답에 `neutral_count`, `neutral_ratio` 필드 추가

**새로운 프로세스**:
1. HuggingFace 모델로 1차 분류 (positive score > 0.8이면 positive, 아니면 negative)
2. Negative로 분류된 리뷰만 GPT-4o-mini로 재판정
3. Positive/Negative/Neutral 비율 계산

---

### 2. ✅ Summary (요약)

**변경 파일**:
- `src/vector_search.py` - 하이브리드 검색 메서드 추가
- `src/summary_pipeline.py` - 새로운 요약 파이프라인 모듈 생성
- `src/api/routers/llm.py` - Summary 라우터 업데이트
- `src/models.py` - CategorySummary 모델 추가

**주요 변경사항**:
- 하이브리드 검색 (Dense + Sparse) 지원
- Aspect 기반 카테고리별 검색 (service/price/food)
- 카테고리별 요약 생성
- Evidence 형식 개선 (review_id, snippet, rank 포함)

**새로운 프로세스**:
1. Aspect seed를 쿼리로 사용하여 하이브리드 검색
2. 카테고리별로 검색된 리뷰를 LLM에 전달
3. 카테고리별 summary, bullets, evidence 생성
4. Evidence 인덱스를 실제 객체로 변환

**주의사항**:
- Aspect Seed는 파일에서 로드 가능 (`ASPECT_SEEDS_FILE` 환경 변수)
- 기본값은 하드코딩된 seed 사용
- 하이브리드 검색은 Sparse 벡터가 컬렉션에 있는 경우에만 작동
- Sparse 벡터가 없으면 자동으로 Dense 검색으로 폴백

---

### 3. ✅ Strength Extraction (강점 추출)

**변경 파일**:
- `src/strength_extraction.py` - Step E~H 교체
- `src/strength_pipeline.py` - 새로운 강점 파이프라인 모듈 생성
- `src/models.py` - StrengthDetail 모델 업데이트

**주요 변경사항**:
- Step E~H: 벡터 유사도 기반 → 통계적 비율 기반
- Lift 계산: `(단일 - 전체) / 전체 × 100`
- LLM 설명 생성: Lift 수치를 기반으로 자연어 설명

**새로운 프로세스**:
1. 전체 데이터셋 평균 긍정 비율 (배치 작업 결과 사용)
2. 단일 음식점 긍정 비율 계산
3. Lift 계산
4. LLM으로 자연어 설명 생성

**주의사항**:
- 전체 평균 비율: 환경 변수 `ALL_AVERAGE_SERVICE_RATIO`, `ALL_AVERAGE_PRICE_RATIO`로 설정 가능 (기본값: 0.60, 0.55)
- 단일 음식점 비율: 자동 계산 (Kiwi 형태소 분석기 사용)
- 배치 작업 결과를 캐시/DB에 저장하면 환경 변수 대신 사용 가능

---

## 새로 생성된 파일

1. `src/summary_pipeline.py` - 새로운 Summary 파이프라인 모듈
2. `src/strength_pipeline.py` - 새로운 Strength Extraction 파이프라인 모듈
3. `src/aspect_seeds.py` - Aspect Seed 관리 모듈 (파일 로드/저장)

## 업데이트된 파일

1. `src/sentiment_analysis.py` - 새로운 감성 분석 로직
2. `src/vector_search.py` - 하이브리드 검색 메서드 추가
3. `src/api/routers/llm.py` - Summary 라우터 업데이트
4. `src/api/routers/sentiment.py` - Neutral 필드 추가
5. `src/strength_extraction.py` - Step E~H 교체
6. `src/models.py` - API 응답 모델 업데이트
7. `requirements.txt` - 새로운 의존성 추가

## 추가된 의존성

```txt
fastembed>=0.2.0  # Sparse 벡터 임베딩
kiwipiepy>=0.20.0  # 한국어 형태소 분석
pyspark>=3.5.0  # 대용량 데이터 처리 (Aspect 추출)
```

## TODO (추가 작업 필요)

### 1. 하이브리드 검색 인프라 구축
- Qdrant 컬렉션에 Sparse 벡터 추가
- 기존 컬렉션 마이그레이션 또는 새 컬렉션 생성
- FastEmbed SparseTextEmbedding 모델 통합

### 2. Aspect Seed 생성 배치 작업 ✅ (부분 완료)
- ✅ 파일에서 로드 가능 (`src/aspect_seeds.py` 모듈)
- ✅ 환경 변수 `ASPECT_SEEDS_FILE`로 파일 경로 지정 가능
- ⚠️ `total_aspect.py`를 주기적 배치 작업으로 실행 (수동)
- ⚠️ 결과를 JSON 파일로 저장 후 `ASPECT_SEEDS_FILE`로 지정

**사용 방법**:
```bash
# Aspect seed 파일 경로 지정
export ASPECT_SEEDS_FILE=/path/to/aspect_seeds.json

# JSON 형식:
{
  "service": ["직원 친절", "사장 친절", ...],
  "price": ["가격 대비", "무한 리필", ...],
  "food": ["가락 국수", "평양 냉면", ...]
}
```

### 3. Strength Extraction 완전 구현 ✅ (부분 완료)
- ✅ 단일 음식점 비율 계산 로직 추가
- ⚠️ 전체 데이터셋 분석 배치 작업 설정 (환경 변수로 임시 설정 가능)
- ⚠️ 배치 작업 결과를 캐시/DB에 저장 (환경 변수 사용 중)

**현재 상태**:
- 단일 음식점 비율 계산: `calculate_single_restaurant_ratios()` 함수로 구현 완료
- 전체 평균 비율: `Config.ALL_AVERAGE_SERVICE_RATIO`, `Config.ALL_AVERAGE_PRICE_RATIO` 환경 변수로 설정 가능
- 불용어 경로: 상대 경로로 수정 완료

### 4. 테스트 및 검증
- 각 파이프라인별 단위 테스트
- 통합 테스트
- 성능 테스트

## API 응답 형식 변경

### Sentiment Analysis
**추가된 필드**:
- `neutral_count`: 중립 리뷰 개수
- `neutral_ratio`: 중립 비율 (%)

### Summary
**추가된 필드**:
- `categories`: 카테고리별 요약 (service, price, food)
  - 각 카테고리는 `summary`, `bullets`, `evidence` 포함
  - `evidence`는 `[{"review_id": "...", "snippet": "...", "rank": 0}, ...]` 형식

### Strength Extraction
**추가된 필드** (새 파이프라인 사용 시):
- `lift_percentage`: Lift 퍼센트
- `all_average_ratio`: 전체 평균 비율
- `single_restaurant_ratio`: 단일 음식점 비율

## 마이그레이션 체크리스트

- [x] Sentiment Analysis 파이프라인 교체
- [x] Summary 파이프라인 교체
- [x] Strength Extraction 파이프라인 교체 (Step E~H)
- [x] API 응답 모델 업데이트
- [x] 의존성 추가
- [x] 단일 음식점 비율 계산 로직 구현
- [x] 전체 평균 비율 환경 변수 설정
- [x] 불용어 경로 상대 경로로 수정
- [x] Aspect Seed 파일 로드 기능 추가
- [x] 하이브리드 검색 에러 처리 강화
- [ ] 하이브리드 검색 인프라 구축 (Qdrant Sparse 벡터 추가)
- [ ] Aspect Seed 배치 작업 자동화
- [ ] 테스트 및 검증

## 참고

- 상세 비교 문서: `hybrid_search/final_pipeline/PIPELINE_COMPARISON.md`
- 새로운 파이프라인 위치: `hybrid_search/final_pipeline/`
