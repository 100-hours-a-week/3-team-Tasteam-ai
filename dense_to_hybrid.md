[2 tools called]

현재 코드에서 “Dense 검색을 시도하는 전략”은 **(1) 기본은 하이브리드(RRF)** 이고, **(2) 하이브리드가 불가능/실패할 때만 Dense-only로 폴백**하는 형태입니다.

## 1) 현재 구현된 Dense-only 전략(폴백)

`VectorSearch.query_hybrid_search()` 안에서 아래 상황이면 **Dense-only** (`_query_dense_only`)로 떨어집니다.

- **컬렉션이 단일 벡터 형식**이라 하이브리드가 불가할 때  
- **Sparse 벡터 생성 실패**(sparse 모델 없음/에러)일 때  
- **Qdrant FusionQuery(RRF) 실패**(Sparse 미지원 등)일 때  
- 기타 예외

근거 코드:

```1095:1142:src/vector_search.py
def _query_dense_only(..., min_score: float = 0.2, limit: int = 20) -> List[Dict]:
    ...
    hits = self.client.query_points(..., limit=limit, using="dense").points
    for hit in hits:
        if hit.score and hit.score >= min_score:
            results.append(...)
```

```1167:1298:src/vector_search.py
def query_hybrid_search(..., fallback_min_score: float = 0.2, dense_prefetch_limit: int = 200, sparse_prefetch_limit: int = 300):
    if self._is_collection_single_vector():
        return self._query_dense_only(..., min_score=fallback_min_score)
    try:
        sparse_vector = ...  # 실패 시 Dense 폴백
    except:
        return self._query_dense_only(..., min_score=fallback_min_score)
    try:
        hits = self.client.query_points(... FusionQuery(RRF) ..., prefetch=[dense(limit=dense_prefetch_limit), sparse(limit=sparse_prefetch_limit)])
    except:
        return self._query_dense_only(..., min_score=fallback_min_score)
```

즉, 지금은 “**Dense를 별도로 먼저 시도한다**”가 아니라, “**하이브리드가 안 되면 Dense로만**”입니다.

## 2) Dense-only 결과 품질을 좌우하는 파라미터

Dense-only 폴백에서 중요한 건 두 가지입니다.

- **`limit`**: Dense-only에서 가져오는 최대 hit 수
- **`fallback_min_score`(= Dense-only min_score)**: 이 값보다 점수가 낮으면 버림

지금 구조에서는 하이브리드(RRF)에는 min_score를 적용하지 않고, **Dense-only 폴백에서만** `fallback_min_score`로 컷합니다.

## 3) “Dense를 적극적으로 쓰는” 일반적인 전략(제안)

현재 코드 외에, 보통은 이런 패턴들도 씁니다(아이디어/설계 옵션).

- **Cascading(2단계)**
  - 1차: Dense-only로 넓게(topK 크게, min_score 낮게) 가져와 **recall 확보**
  - 2차: 필요하면 sparse/RRF/재랭크로 **precision 개선**
- **0건 카테고리 구제**
  - 하이브리드가 “성공은 했는데 결과가 0건”인 카테고리에 한해
    - Dense-only를 **min_score=0.0 + limit 크게**로 한 번 더 시도해 “빈 입력”을 줄임
- **동적 튜닝**
  - 쿼리 길이/언어/카테고리별로 `dense_prefetch_limit`, `sparse_prefetch_limit`, `fallback_min_score`를 다르게 두기
  - 예: 짧은 쿼리면 sparse 비중↑ / 긴 문장이면 dense 비중↑

원하시는 “Dense 검색 전략”이 **지금처럼 ‘실패 시 폴백’**인지, 아니면 **‘0건이면 Dense-only 재시도’** 같은 적극적인 정책인지에 따라 접근이 달라집니다.  
원하는 동작을 한 문장으로만 정해주시면(예: “카테고리 0건이면 dense-only로 limit=50 재시도”), 그에 맞춰 적용 위치를 정확히 짚어드릴게요.