각 Phase의 구체적 구현 방법입니다.

## Phase 1: 비교군 선정 개선

### 현재 방식 (문제점)

```python
# src/api/routers/llm.py (현재)
# 모든 레스토랑에 대해 벡터 검색 수행
all_restaurant_ids = vector_search.get_all_restaurant_ids()  # 100개
comparison_restaurant_ids = [rid for rid in all_restaurant_ids if str(rid) != str(target_id)]

comparison_reviews = []
for comp_id in comparison_restaurant_ids:  # 100번 반복
    comp_results = vector_search.query_similar_reviews(
        query_text=request.query,
        restaurant_id=comp_id,
        limit=1,
    )
    if comp_results:
        comparison_reviews.append(comp_results[0]["payload"])
```

문제: 100개 레스토랑 → 100번 벡터 검색

### Phase 1 구현 방법

#### 1단계: 대표 벡터 생성 함수 추가

```python
# src/vector_search.py에 추가

def compute_restaurant_vector(
    self,
    restaurant_id: Union[int, str],
    weight_by_date: bool = True,
    weight_by_rating: bool = True,
) -> Optional[np.ndarray]:
    """
    레스토랑의 모든 리뷰 임베딩을 평균/가중 평균하여 대표 벡터 생성
    
    Args:
        restaurant_id: 레스토랑 ID
        weight_by_date: 최근 리뷰에 가중치 부여 (기본값: True)
        weight_by_rating: 높은 별점에 가중치 부여 (기본값: True)
        
    Returns:
        대표 벡터 (numpy array) 또는 None
    """
    # 1. 해당 레스토랑의 모든 리뷰 검색
    results = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="restaurant_id",
                    match=models.MatchValue(value=str(restaurant_id))
                )
            ]
        ),
        limit=10000,  # 충분히 큰 값
        with_payload=True,
        with_vectors=True,
    )
    
    if not results[0]:  # 리뷰가 없으면
        return None
    
    points = results[0]
    
    # 2. 각 리뷰의 벡터와 메타데이터 추출
    vectors = []
    weights = []
    
    for point in points:
        vector = point.vector
        payload = point.payload
        
        # 가중치 계산
        weight = 1.0
        
        if weight_by_date and "created_at" in payload:
            # 최근 리뷰일수록 높은 가중치 (예: 최근 1년 = 1.0, 2년 전 = 0.8)
            created_at = payload["created_at"]
            # 날짜 기반 가중치 계산 로직
            # weight *= date_weight
        
        if weight_by_rating and "is_recommended" in payload:
            # 추천 리뷰에 높은 가중치
            if payload["is_recommended"]:
                weight *= 1.5
        
        vectors.append(vector)
        weights.append(weight)
    
    # 3. 가중 평균 계산
    vectors = np.array(vectors)
    weights = np.array(weights)
    weights = weights / weights.sum()  # 정규화
    
    restaurant_vector = np.average(vectors, axis=0, weights=weights)
    
    return restaurant_vector
```

#### 2단계: Qdrant에 `restaurant_vectors` 컬렉션 생성 및 관리

```python
# src/vector_search.py에 추가

RESTAURANT_VECTORS_COLLECTION = "restaurant_vectors"

def upsert_restaurant_vector(
    self,
    restaurant_id: Union[int, str],
    restaurant_name: str,
    food_category_id: Optional[int] = None,
):
    """
    레스토랑 대표 벡터를 Qdrant에 저장/업데이트
    
    Args:
        restaurant_id: 레스토랑 ID
        restaurant_name: 레스토랑 이름
        food_category_id: 음식 카테고리 ID (선택)
    """
    # 1. 대표 벡터 계산
    restaurant_vector = self.compute_restaurant_vector(restaurant_id)
    
    if restaurant_vector is None:
        logger.warning(f"레스토랑 {restaurant_id}의 대표 벡터를 생성할 수 없습니다.")
        return
    
    # 2. restaurant_vectors 컬렉션이 없으면 생성
    try:
        self.client.get_collection(RESTAURANT_VECTORS_COLLECTION)
    except Exception:
        self.client.create_collection(
            collection_name=RESTAURANT_VECTORS_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE,
            ),
        )
    
    # 3. 포인트 생성 및 업로드
    point = PointStruct(
        id=str(restaurant_id),
        vector=restaurant_vector.tolist(),
        payload={
            "restaurant_id": str(restaurant_id),
            "restaurant_name": restaurant_name,
            "food_category_id": food_category_id,
        }
    )
    
    self.client.upsert(
        collection_name=RESTAURANT_VECTORS_COLLECTION,
        points=[point]
    )
    
    logger.info(f"레스토랑 {restaurant_id}의 대표 벡터를 업데이트했습니다.")
```

#### 3단계: 유사 레스토랑 검색 함수

```python
# src/vector_search.py에 추가

def find_similar_restaurants(
    self,
    target_restaurant_id: Union[int, str],
    top_n: int = 20,
    food_category_id: Optional[int] = None,
    exclude_self: bool = True,
) -> List[Dict]:
    """
    타겟 레스토랑과 유사한 레스토랑을 대표 벡터로 검색
    
    Args:
        target_restaurant_id: 타겟 레스토랑 ID
        top_n: 반환할 상위 N개 (기본값: 20)
        food_category_id: 음식 카테고리 필터 (선택)
        exclude_self: 타겟 레스토랑 제외 여부 (기본값: True)
        
    Returns:
        유사 레스토랑 리스트 [{"restaurant_id": ..., "score": ..., ...}, ...]
    """
    # 1. 타겟 레스토랑의 대표 벡터 가져오기
    target_vector = self.compute_restaurant_vector(target_restaurant_id)
    
    if target_vector is None:
        logger.warning(f"타겟 레스토랑 {target_restaurant_id}의 대표 벡터를 찾을 수 없습니다.")
        return []
    
    # 2. 필터 구성
    filter_conditions = []
    
    if exclude_self:
        filter_conditions.append(
            models.FieldCondition(
                key="restaurant_id",
                match=models.MatchValue(value=str(target_restaurant_id))
            )
        )
    
    if food_category_id:
        filter_conditions.append(
            models.FieldCondition(
                key="food_category_id",
                match=models.MatchValue(value=food_category_id)
            )
        )
    
    scroll_filter = None
    if filter_conditions:
        scroll_filter = models.Filter(
            must_not=filter_conditions if exclude_self else None,
            must=filter_conditions if food_category_id else None,
        )
    
    # 3. 유사도 검색 (1번만!)
    results = self.client.search(
        collection_name=RESTAURANT_VECTORS_COLLECTION,
        query_vector=target_vector.tolist(),
        limit=top_n + (1 if exclude_self else 0),  # 자기 자신 제외 고려
        query_filter=scroll_filter,
    )
    
    # 4. 결과 변환
    similar_restaurants = []
    for result in results:
        if exclude_self and str(result.payload["restaurant_id"]) == str(target_restaurant_id):
            continue  # 자기 자신 제외
        
        similar_restaurants.append({
            "restaurant_id": result.payload["restaurant_id"],
            "restaurant_name": result.payload.get("restaurant_name", ""),
            "score": result.score,
            "food_category_id": result.payload.get("food_category_id"),
        })
    
    return similar_restaurants[:top_n]
```

#### 4단계: API 라우터 수정

```python
# src/api/routers/llm.py 수정

# 기존 코드 (293-333줄) 대체
# 각 레스토랑에 대해 벡터 검색 (비효율)
# ↓
# Phase 1: 대표 벡터로 유사 레스토랑 검색 (효율적)

# 2. 비교 대상 레스토랑 선정 (Phase 1 적용)
similar_restaurants = vector_search.find_similar_restaurants(
    target_restaurant_id=request.target_restaurant_id,
    top_n=20,  # Top-20만 선택
    food_category_id=request.food_category_id,
    exclude_self=True,
)

if not similar_restaurants:
    raise HTTPException(
        status_code=404,
        detail="비교할 수 있는 유사 레스토랑을 찾을 수 없습니다."
    )

logger.info(
    f"타겟 레스토랑과 유사한 레스토랑 {len(similar_restaurants)}개 선택됨"
)

# 3. 선택된 레스토랑들의 긍정 리뷰 검색 (Query 기반)
comparison_reviews = []
for similar_rest in similar_restaurants:
    comp_id = similar_rest["restaurant_id"]
    comp_results = vector_search.query_similar_reviews(
        query_text=request.query,
        restaurant_id=comp_id,
        limit=request.limit,
        min_score=request.min_score,
        food_category_id=request.food_category_id,
    )
    if comp_results:
        comparison_reviews.append(comp_results[0]["payload"])
```

효과:
- 검색 횟수: 100회 → 1회 (대표 벡터 검색) + 20회 (리뷰 검색) = 21회
- 처리 시간: 약 5배 단축
- 관련성: 유사 레스토랑만 비교

---

## Phase 2: 차별점 계산 개선

### 현재 방식 (문제점)

```python
# 현재: LLM이 모든 리뷰를 보고 차별점 추출
# → LLM 능력에 의존, 비교 품질 불안정
target_restaurant_strength = llm_utils._generate_response(
    messages=[{
        "role": "user",
        "content": f"""
        타겟 레스토랑 리뷰들: {target_reviews}
        비교 대상 레스토랑 리뷰들: {comparison_reviews}
        차별점 추출하세요.
        """
    }]
)
```

### Phase 2 구현 방법

#### 1단계: 강점 임베딩 생성 함수

```python
# src/llm_utils.py에 추가

def extract_strengths_as_list(
    self,
    target_reviews: List[Dict[str, Any]],
    comparison_reviews: List[Dict[str, Any]],
) -> List[str]:
    """
    강점을 리스트 형태로 추출 (임베딩 비교용)
    
    Returns:
        강점 리스트 ["맛이 좋다", "서비스가 친절하다", ...]
    """
    # 기존 extract_strengths() 호출
    result = self.extract_strengths(
        target_reviews=target_reviews,
        comparison_reviews=comparison_reviews,
        target_restaurant_id="temp",
    )
    
    strength_text = result.get("strength_summary", "")
    
    # 강점을 리스트로 분리 (LLM 또는 간단한 파싱)
    # 예: "맛이 좋다. 서비스가 친절하다." → ["맛이 좋다", "서비스가 친절하다"]
    strengths = self._parse_strengths_to_list(strength_text)
    
    return strengths

def _parse_strengths_to_list(self, strength_text: str) -> List[str]:
    """강점 텍스트를 리스트로 파싱"""
    # 간단한 파싱: 문장 단위로 분리
    import re
    sentences = re.split(r'[.!?]\s+', strength_text)
    strengths = [s.strip() for s in sentences if s.strip()]
    return strengths
```

#### 2단계: 강점 임베딩 생성 및 Set 비교

```python
# src/vector_search.py에 추가

def compute_strength_embeddings(
    self,
    strengths: List[str],
) -> np.ndarray:
    """
    강점 리스트를 임베딩으로 변환
    
    Args:
        strengths: 강점 리스트 ["맛이 좋다", "서비스가 친절하다", ...]
        
    Returns:
        강점 임베딩 배열 (n_strengths, embedding_dim)
    """
    if not strengths:
        return np.array([])
    
    # 배치 인코딩
    embeddings = self.encoder.encode(
        strengths,
        batch_size=self.batch_size,
        convert_to_numpy=True,
    )
    
    return embeddings

def find_unique_strengths(
    self,
    target_strengths: List[str],
    comparison_strengths_list: List[List[str]],  # 각 비교군의 강점 리스트
    similarity_threshold: float = 0.7,
) -> List[str]:
    """
    타겟 레스토랑에만 있는 (또는 더 강한) 강점을 찾기
    
    Args:
        target_strengths: 타겟 레스토랑의 강점 리스트
        comparison_strengths_list: 비교군들의 강점 리스트 리스트
        similarity_threshold: 유사도 임계점 (이상이면 같은 강점으로 간주)
        
    Returns:
        차별화된 강점 리스트
    """
    if not target_strengths:
        return []
    
    # 1. 타겟 강점 임베딩
    target_embeddings = self.compute_strength_embeddings(target_strengths)
    
    # 2. 비교군 강점 임베딩 (모든 비교군 합치기)
    all_comparison_strengths = []
    for comp_strengths in comparison_strengths_list:
        all_comparison_strengths.extend(comp_strengths)
    
    if not all_comparison_strengths:
        # 비교군에 강점이 없으면 타겟의 모든 강점이 차별점
        return target_strengths
    
    comparison_embeddings = self.compute_strength_embeddings(all_comparison_strengths)
    
    # 3. 각 타겟 강점이 비교군 강점과 유사한지 확인
    unique_strengths = []
    
    for i, target_strength in enumerate(target_strengths):
        target_emb = target_embeddings[i]
        
        # 비교군 강점들과의 최대 유사도 계산
        similarities = np.dot(comparison_embeddings, target_emb) / (
            np.linalg.norm(comparison_embeddings, axis=1) * np.linalg.norm(target_emb)
        )
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0
        
        # 임계점 이하면 차별점으로 간주
        if max_similarity < similarity_threshold:
            unique_strengths.append(target_strength)
    
    return unique_strengths
```

#### 3단계: API 라우터 수정

```python
# src/api/routers/llm.py 수정

# Phase 2: 차별점 계산 개선

# 1. 타겟 레스토랑 강점 추출
target_strengths = llm_utils.extract_strengths_as_list(
    target_reviews=target_reviews,
    comparison_reviews=[],  # 일단 비교 없이 추출
)

# 2. 각 비교군의 강점 추출
comparison_strengths_list = []
for similar_rest in similar_restaurants:
    comp_id = similar_rest["restaurant_id"]
    comp_reviews = [r["payload"] for r in vector_search.query_similar_reviews(
        query_text=request.query,
        restaurant_id=comp_id,
        limit=request.limit,
    )]
    
    if comp_reviews:
        comp_strengths = llm_utils.extract_strengths_as_list(
            target_reviews=comp_reviews,
            comparison_reviews=[],
        )
        comparison_strengths_list.append(comp_strengths)

# 3. 차별화된 강점 계산 (임베딩 기반)
unique_strengths = vector_search.find_unique_strengths(
    target_strengths=target_strengths,
    comparison_strengths_list=comparison_strengths_list,
    similarity_threshold=0.7,
)

# 4. 최종 LLM 서술 (차별화된 강점만 사용)
if unique_strengths:
    final_prompt = f"""
    다음은 타겟 레스토랑이 경쟁군 대비 두드러지는 강점입니다:
    {chr(10).join([f"- {s}" for s in unique_strengths])}
    
    타겟 레스토랑 리뷰:
    {chr(10).join([f"- {r.get('content', '')}" for r in target_reviews[:5]])}
    
    위 강점들을 자연스러운 문장으로 서술하고, 근거가 되는 리뷰 예시를 포함하세요.
    """
    
    target_restaurant_strength = llm_utils._generate_response(
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3,
        max_new_tokens=300,
    ).strip()
else:
    target_restaurant_strength = "경쟁군 대비 두드러지는 차별점을 찾을 수 없습니다."
```

효과:
- 객관성: 임베딩 기반 비교
- 일관성: LLM 변동성 감소
- 명확성: 차별점이 명확히 구분됨

---

## Phase 3: 전체 통합

### Phase 3 구현 방법

Phase 1 + Phase 2 + Query 기반 검색 결합:

```python
# src/api/routers/llm.py (최종 통합)

# 1. Phase 1: 대표 벡터로 유사 레스토랑 선정
similar_restaurants = vector_search.find_similar_restaurants(
    target_restaurant_id=request.target_restaurant_id,
    top_n=20,
    food_category_id=request.food_category_id,
)

# 2. Query 기반 검색으로 각 레스토랑의 관련 리뷰 추출
target_reviews = vector_search.query_similar_reviews(
    query_text=request.query,  # 사용자 의도 반영
    restaurant_id=request.target_restaurant_id,
    limit=request.limit,
)

comparison_reviews_map = {}  # {restaurant_id: [reviews]}
for similar_rest in similar_restaurants:
    comp_id = similar_rest["restaurant_id"]
    comp_reviews = vector_search.query_similar_reviews(
        query_text=request.query,  # 사용자 의도 반영
        restaurant_id=comp_id,
        limit=request.limit,
    )
    comparison_reviews_map[comp_id] = [r["payload"] for r in comp_reviews]

# 3. Phase 2: 차별점 계산
target_strengths = llm_utils.extract_strengths_as_list(
    target_reviews=[r["payload"] for r in target_reviews],
    comparison_reviews=[],
)

comparison_strengths_list = []
for comp_id, comp_reviews in comparison_reviews_map.items():
    if comp_reviews:
        comp_strengths = llm_utils.extract_strengths_as_list(
            target_reviews=comp_reviews,
            comparison_reviews=[],
        )
        comparison_strengths_list.append(comp_strengths)

unique_strengths = vector_search.find_unique_strengths(
    target_strengths=target_strengths,
    comparison_strengths_list=comparison_strengths_list,
)

# 4. 최종 서술 (차별점 + 근거 리뷰)
final_strength = llm_utils.generate_final_strength_description(
    unique_strengths=unique_strengths,
    target_reviews=[r["payload"] for r in target_reviews],
)
```

효과:
- Phase 1: 효율적 비교군 선정
- Phase 2: 객관적 차별점 계산
- Query 기반: 사용자 의도 반영
- 통합: 최고의 품질과 효율성

---

## 구현 순서 권장

1. Phase 1 먼저 구현 (효과가 크고 구현이 상대적으로 단순)
2. Phase 2 추가 (차별점 계산 개선)
3. Phase 3 통합 (최종 최적화)

Phase 1부터 진행할까요?