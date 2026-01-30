지금 함수가 깨지는 이유는 **2가지가 동시에** 있어요.

1. `dense_vector`, `sparse_vector`를 **포인트 1개짜리가 아니라 “전체 리스트”로 매번 만들어서** `PointStruct.vector`에 넣고 있음
2. `fastembed`의 `SparseTextEmbedding.embed()` 결과는 보통 `SparseEmbedding(indices, values)` 타입이라서, Qdrant가 기대하는 `qdrant_client.models.SparseVector(indices=[...], values=[...])`로 **변환**해줘야 함 ([qdrant.tech][1])

---

## ✅ 올바른 형태 (포인트 1개 = dense 1개 + sparse 1개)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector

def setup_qdrant_hybrid(passages, dense_embeddings, sparse_embeddings):
    client = QdrantClient(":memory:")
    collection_name = "hybrid_test"

    dense_dim = len(dense_embeddings[0])

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        },
    )

    points = []
    for i, (passage, dense_emb, sparse_emb) in enumerate(zip(passages, dense_embeddings, sparse_embeddings)):
        # ✅ dense는 "1개 벡터" (list[float])
        dense_vector = list(dense_emb)

        # ✅ sparse는 Qdrant SparseVector로 변환 (indices/values는 list여야 함)
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        point = PointStruct(
            id=i,
            vector={
                "dense": dense_vector,
                "sparse": sparse_vector,
            },
            payload={"text": passage},
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    return client
```

---

## 네 코드에서 “결정적으로” 잘못된 부분

```python
dense_vector = [dense_embeddings[i] for i in range(len(passages))]
sparse_vector = [sparse_embeddings[i] for i in range(len(passages))]
```

이건 **포인트 1개에 들어가야 할 벡터**가 아니라, **전체 데이터셋 벡터 리스트**를 매 포인트마다 만들어서 넣는 거라서 타입이 `list[list[float]]` / `list[SparseEmbedding]`가 되어버려요.

---

## 참고: Qdrant가 허용하는 sparse 업서트 형태

Qdrant는 named vector dict 안에 `SparseVector(indices, values)`로 넣는 형태를 문서에서도 그대로 보여줍니다. ([qdrant.tech][1])

---

원하면, 다음 단계로 **하이브리드 검색(RRF fusion)**까지 바로 이어서 예제도 붙여줄게요 (query_points + FusionQuery + prefetch).

[1]: https://qdrant.tech/articles/sparse-vectors/ "What is a Sparse Vector? How to Achieve Vector-based Hybrid Search - Qdrant"
