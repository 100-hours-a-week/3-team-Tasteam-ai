# API·Pydantic 모델 정리 기록

현재 API 입출력에 쓰이지 않는 Pydantic 모델·엔드포인트·라우터를 제거한 내역입니다.

**기준**: `test_all_task.py`가 호출하는 API만 “현재 API”로 간주  
- `/api/v1/sentiment/analyze`, `/api/v1/sentiment/analyze/batch`  
- `/api/v1/llm/summarize`, `/api/v1/llm/summarize/batch`  
- `/api/v1/llm/extract/strengths`  
- `/api/v1/vector/upload`, `/api/v1/vector/search/similar`  
- `/health`

---

## 1. 제거한 것

### 1.1 라우터·엔드포인트

| 위치 | 제거 내용 |
|------|-----------|
| **main.py** | `restaurant` 라우터 import 및 `include_router(restaurant.router, prefix="/api/v1/restaurants", ...)` 제거 |
| **vector.py** | `GET /restaurants/{restaurant_id}/reviews` (레스토랑 리뷰 목록) |
| **vector.py** | `POST /reviews/upsert` (리뷰 upsert) |
| **vector.py** | `DELETE /reviews/delete` (리뷰 1건 삭제) |
| **vector.py** | `DELETE /reviews/delete/batch` (리뷰 배치 삭제) |

### 1.2 Pydantic 모델 (src/models.py)

| 제거한 DTO | 용도 (제거 전) |
|------------|----------------|
| **SummaryResponse** | 요약 응답(디버그용, positive_reviews 등). 현재는 SummaryDisplayResponse만 사용 |
| **EvidenceSnippet** | 근거 스니펫. API 입출력에서 미사용 |
| **UpsertReviewsRequest** | 리뷰 upsert 요청 |
| **UpsertReviewsBatchResponse** | 리뷰 upsert 배치 응답 |
| **DeleteReviewRequest** | 리뷰 1건 삭제 요청 |
| **DeleteReviewResponse** | 리뷰 1건 삭제 응답 |
| **DeleteReviewsBatchRequest** | 리뷰 배치 삭제 요청 |
| **DeleteReviewsBatchResponse** | 리뷰 배치 삭제 응답 |
| **RestaurantReviewsResponse** | 레스토랑 리뷰 조회 응답 |
| **SummaryVectorUploadRequest** | 요약 벡터 업로드 요청. 어디서도 import/사용 안 함 |
| **SummaryVectorSearchRequest** | 요약 벡터 업로드/검색 요청. 어디서도 import/사용 안 함 |
| **ReviewImageModel** | 리뷰 이미지 모델. API 미사용 → 제거. ReviewModel/VectorUploadReviewInput의 images 필드도 제거 |
| **RestaurantModel** | 레스토랑 모델. API 미사용 |
| **FoodCategoryModel** | 음식 카테고리 모델. StrengthVectorUploadRequest에서만 참조 → 함께 제거 |
| **RestaurantFoodCategoryModel** | 레스토랑-음식 카테고리 관계. StrengthVectorUploadRequest에서만 참조 → 함께 제거 |
| **StrengthVectorUploadRequest** | 강점 벡터 업로드 요청. API 미사용 |
| **SummaryAspect** | 요약 aspect 모델. 어디서도 참조 안 함 |

**제거한 필드**: StrengthRequest의 `category_filter`, `region_filter`, `price_band_filter`, `max_candidates`, `months_back`. ReviewModel·VectorUploadReviewInput의 `images`. **메타 필드**: ReviewModel·VectorUploadReviewInput에서 `member_id`, `group_id`, `subgroup_id`, `is_recommended`, `created_at`, `updated_at` 제거 (선택 필드만 유지).

**제거한 코드**: `src/vector_search.py`의 `query_by_restaurant_vector` (미호출).

### 1.3 삭제한 파일

| 파일 | 비고 |
|------|------|
| **src/api/routers/restaurant.py** | RestaurantReviewsResponse만 참조하던 라우터. 라우터 제거에 따라 삭제 |

---

## 2. 남긴 것 (현재 API·DTO)

### 2.1 등록된 라우터 (main.py)

| prefix | 라우터 | 용도 |
|--------|--------|------|
| `/api/v1/sentiment` | sentiment | 감성 분석 (analyze, analyze/batch) |
| `/api/v1/vector` | vector | 벡터 검색·업로드 (search/similar, upload) |
| `/api/v1/llm` | llm | 요약·강점 (summarize, summarize/batch, extract/strengths) |
| `/api/v1/test` | test | 테스트 데이터 생성 (generate) |

### 2.2 Vector 엔드포인트 (현재)

| Method | path | 설명 |
|--------|------|------|
| POST | `/api/v1/vector/search/similar` | 유사 리뷰 검색 |
| POST | `/api/v1/vector/upload` | 리뷰·레스토랑 업로드 |

### 2.3 유지한 Pydantic 모델 (src/models.py)

- **공통**: ErrorResponse, DebugInfo, ReviewModel (id, restaurant_id, content만. 메타 필드 제거됨)  
- **Sentiment**: SentimentReviewInput, SentimentAnalysisRequest, SentimentAnalysisDisplayResponse, SentimentAnalysisResponse, SentimentRestaurantBatchInput, SentimentAnalysisBatchRequest, SentimentAnalysisBatchResponse  
- **Summary**: SummaryRequest, SummaryDisplayResponse, SummaryBatchRequest, SummaryBatchResponse, CategorySummary  
- **Strength**: StrengthRequest (restaurant_id, top_k만), StrengthDetail, StrengthResponse  
- **Vector**: VectorSearchRequest, VectorSearchResult, VectorSearchResponse, VectorUploadReviewInput (id, restaurant_id, content만. 메타 필드 제거됨), VectorUploadRestaurantInput, VectorUploadRequest, VectorUploadResponse  
- **Health**: HealthResponse  

---

## 3. 참고

- 상세 API 입출력·파이프라인: [PIPELINE_OPERATIONS.md](PIPELINE_OPERATIONS.md)  
- LLM 레거시 제거: [LLM_UTILS_LEGACY_REMOVAL.md](LLM_UTILS_LEGACY_REMOVAL.md)  
- 테스트: `test_all_task.py` (위 “현재 API”만 호출)

## 4. 추가 제거 (완료)

다음 항목은 코드 반영으로 제거 완료됨.  
- **모델**: ReviewImageModel, RestaurantModel, FoodCategoryModel, RestaurantFoodCategoryModel, StrengthVectorUploadRequest, SummaryAspect  
- **필드**: StrengthRequest의 category_filter, region_filter, price_band_filter, max_candidates, months_back. ReviewModel·VectorUploadReviewInput의 images  
- **코드**: `src/vector_search.py`의 `query_by_restaurant_vector`

이 문서는 “현재 API 입출력에 쓰이지 않는 Pydantic 모델·엔드포인트·필드·코드 제거” 작업 시점 기준으로 작성되었습니다.
