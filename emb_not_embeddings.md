에러 원인 딱 하나예요:

지금 루프 안에서 **`sparse_emb`(한 문서의 sparse 임베딩)**을 써야 하는데, 실수로 **`sparse_embeddings`(전체 리스트)**에 `.indices`를 붙였어요. 그래서 `list`에는 `indices`가 없어서 터진 겁니다.

그리고 같은 실수가 dense에도 있어요: `dense_vector = list(dense_embeddings)` ← 이것도 “전체 리스트”를 넣는 거라 잘못.

---

## ✅ 수정본 (그대로 붙여넣으면 됨)

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
        # ✅ 한 문서의 dense 벡터
        dense_vector = list(dense_emb)

        # ✅ 한 문서의 sparse 임베딩을 Qdrant SparseVector로 변환
        sparse_vector = SparseVector(
            indices=list(sparse_emb.indices),
            values=list(sparse_emb.values),
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

## 만약 `sparse_emb.indices`가 numpy/array가 아니라서 또 문제 나면

fastembed 버전/백엔드에 따라 `indices`, `values`가 이미 리스트거나, numpy일 수 있어요. 위처럼 `list()`로 감싸면 대부분 해결됩니다.

---

원하면, 바로 이어서 **hybrid 검색 쿼리 (dense + bm25 + RRF fusion)**까지 완성 예제도 붙여줄까요?
