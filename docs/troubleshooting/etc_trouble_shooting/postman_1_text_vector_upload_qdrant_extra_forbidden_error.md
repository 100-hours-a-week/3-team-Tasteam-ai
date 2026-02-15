로그에 Pydantic `extra_forbidden` 오류가 있고, 그 뒤에 포인트가 0개 생성됩니다. 원인 확인 중입니다.


[3 tools called]


원인: `PointStruct`(Qdrant) 검증 오류입니다. `vector`에 전달하는 형식이 Qdrant가 기대하는 형식과 맞지 않습니다. 포인트 생성 방식을 확인 중입니다.


[3 tools called]


Qdrant 컬렉션 스키마와 `PointStruct`의 `vector` 기대 형식 확인 중.


[7 tools called]


원인과 대응을 정리하면 아래와 같습니다.

---

## 원인

1. **에러 발생 위치**  
   `vector_search.prepare_points()` 안에서 `PointStruct(id=..., vector=vector_dict, payload=...)` 를 만들 때, **Qdrant 클라이언트의 `PointStruct` Pydantic 검증**에서 실패하고 있습니다.

2. **우리 코드가 넘기는 형식**  
   하이브리드(이름 있는 벡터)용으로 다음처럼 넘기고 있습니다.

   ```python
   vector_dict = {
       "dense": dense_vector.tolist(),   # list[float]
       "sparse": models.SparseVector(indices=..., values=...)
   }
   point = PointStruct(id=point_id, vector=vector_dict, payload=metadata)
   ```

3. **에러 메시지가 말하는 것**  
   - `vector` 필드에 들어갈 수 있는 타입이  
     `list[float]`, `list[list[float]]`,  
     또는 `Document` / `Image` / `InferenceObject` 같은 **추론용** dict 로 한정되어 있고,
   - 우리가 넘기는 `{"dense": list[float], "sparse": SparseVector}` 같은 **일반 named-vector dict**는 그 Union에 포함되지 않아서  
     `extra_forbidden` / `list_type` / `model_type` 등 여러 검증 오류가 납니다.

4. **포인트가 0개인 이유**  
   `prepare_points()` 내부의 `except`에서 포인트 생성 실패만 로그하고 `continue` 하기 때문에, 해당 리뷰는 포인트 리스트에 들어가지 않고, 그 결과 **생성된 포인트가 0개**가 됩니다.  
   업로드는 “포인트 0개”로 정상 완료되기 때문에 **HTTP 200**이 나옵니다.

---

## 결론

**설치된 `qdrant-client` 버전의 `PointStruct`가 named vector로 `dict[str, list[float] | SparseVector]` 를 허용하지 않는 것**이 직접 원인입니다.  
공식 문서 예시(`vector={"image": [...], "text": [...]}` 또는 `{"text": SparseVector(...)}`)와 같은 형태를 지원하던 이전 버전과, 지금 사용 중인 버전의 Pydantic 스키마가 맞지 않는 상황으로 보는 것이 맞습니다.

---

## 대응 방법

1. **사용 중인 qdrant-client 버전 확인**  
   - `pip show qdrant-client`  
   - 1.6.0 이상이면, 1.7.x / 1.11.x 등 세부 버전에 따라 `PointStruct.vector` 타입 정의가 다를 수 있습니다.

2. **PointStruct 검증 우회**  
   - `prepare_points()`에서는 **PointStruct를 쓰지 않고**  
     `(point_id, vector_dict, payload)` 형태로만 만들고,
   - `upload_collection()`에 넘길 때는  
     `ids`, `vectors`, `payload` 리스트만 만들어서 넘기기  
     (이미 `upload_collection()`이 `vectors=`, `payload=`, `ids=` 를 받도록 되어 있으므로, `PointStruct`를 한 번 거치지 않고 리스트만 채워서 넘기면 됨).  
   - 즉, **named vector용으로는 PointStruct 생성 단계를 건너뛰고**,  
     `vector_dict`를 그대로 `vectors` 리스트에 넣어서 `client.upload_collection(...)` 호출하는 방식으로 바꾸면, 현재 버전의 `PointStruct` 검증 문제를 피할 수 있습니다.

3. **버전 조정**  
   - 과거에 named vector dict를 지원하던 qdrant-client 버전을 알고 있다면, 그 버전으로 낮춰서 `PointStruct(vector={"dense": ..., "sparse": ...})` 가 통과하는지 시험해 볼 수 있습니다.  
   - 단, 다른 의존성과의 호환성을 반드시 확인하는 것이 좋습니다.

정리하면, **지금 로그의 `extra_forbidden`과 “0개 포인트 생성”은, named vector를 넣은 `PointStruct`가 현재 qdrant-client의 Pydantic 스키마와 맞지 않아서 발생한 것이고**,  
**PointStruct를 쓰지 않고 `vectors`/`ids`/`payload`만 만들어서 `upload_collection`에 넘기도록 코드를 바꾸는 것**이 가장 확실한 해결입니다.  
원하시면 Agent 모드에서 해당 경로(`prepare_points` → `upload_collection`) 수정용 패치 예시까지 작성해 드리겠습니다.

---

```bash
{
    "code": 500,
    "message": "데이터 업로드 중 오류 발생: 20 validation errors for PointStruct\nvector.list[float]\n  Input should be a valid list [type=list_type, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.list[list[float]]\n  Input should be a valid list [type=list_type, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.list[float]\n  Input should be a valid list [type=list_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.SparseVector\n  Input should be a valid dictionary or instance of SparseVector [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.list[list[float]]\n  Input should be a valid list [type=list_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.Document\n  Input should be a valid dictionary or instance of Document [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.Image\n  Input should be a valid dictionary or instance of Image [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.InferenceObject\n  Input should be a valid dictionary or instance of InferenceObject [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.Document.text\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Document.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Document.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Document.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Image.image\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Image.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Image.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Image.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.InferenceObject.object\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.InferenceObject.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.InferenceObject.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.InferenceObject.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden",
    "details": "데이터 업로드 중 오류 발생: 20 validation errors for PointStruct\nvector.list[float]\n  Input should be a valid list [type=list_type, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.list[list[float]]\n  Input should be a valid list [type=list_type, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.list[float]\n  Input should be a valid list [type=list_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.SparseVector\n  Input should be a valid dictionary or instance of SparseVector [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.list[list[float]]\n  Input should be a valid list [type=list_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/list_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.Document\n  Input should be a valid dictionary or instance of Document [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.Image\n  Input should be a valid dictionary or instance of Image [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.dict[str,union[list[float],SparseVector,list[list[float]],Document,Image,InferenceObject]].dense.InferenceObject\n  Input should be a valid dictionary or instance of InferenceObject [type=model_type, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/model_type\nvector.Document.text\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Document.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Document.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Document.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Image.image\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Image.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.Image.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.Image.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.InferenceObject.object\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.InferenceObject.model\n  Field required [type=missing, input_value={'dense': 0.0138650796725...s=[1.6877434821696136])}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.11/v/missing\nvector.InferenceObject.dense\n  Extra inputs are not permitted [type=extra_forbidden, input_value=0.01386507967254147, input_type=float]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden\nvector.InferenceObject.sparse\n  Extra inputs are not permitted [type=extra_forbidden, input_value=SparseVector(indices=[431...es=[1.6877434821696136]), input_type=SparseVector]\n    For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden",
    "request_id": "e4d00615-a8bb-43e3-b032-414e612ebf44"
}
```

참고: 에러 메시지에 vector.dense에 input_value=0.0138... (float)가 보이는 걸로 봐서, client.upload_collection(vectors=[{"dense": [...], "sparse": ...}, ...])를 호출할 때 qdrant-client가 내부에서 이걸로 PointStruct를 만들면서 검증하는 것으로 보입니다. 그래서 우리가 PointStruct를 안 만들고 dict만 넘겨도, 클라이언트 쪽에서 같은 검증이 나는 상황입니다.

---

테스트(142개 리뷰)는 성공하고, Postman으로 1개 리뷰만 보낸 요청에서만 `extra_forbidden`이 난다는 뜻이군요. 원인 확인 중입니다.


[2 tools called]


encoder가 단일 텍스트일 때 반환하는 shape 확인 중.


[1 tool called]


원인은 **리뷰가 1개일 때** encoder가 1차원 벡터를 반환하고, 그걸 그대로 쓰면서 `dense`에 **스칼라(float)**가 들어가는 것입니다.

---

## 원인

`_FastEmbedEncoderAdapter.encode()` (33~41행):

```bash 39:41:src/vector_search.py
        arrs = list(self._model.embed(sentences))
        out = np.array(arrs)
        if len(sentences) == 1:
            return out[0]
        return out
```

- **리뷰 1개** → `len(sentences) == 1` 이라 `out[0]` 반환 → shape `(768,)` **1차원 배열**
- **리뷰 2개 이상** → `out` 그대로 반환 → shape `(N, 768)` **2차원 배열**

그 다음 `prepare_points`에서는:

```bash 299:312:src/vector_search.py
                for text, dense_vector, sparse_emb, metadata in zip(batch_texts, batch_dense_vectors, batch_sparse_vectors, batch_metadata):
                    ...
                        if sparse_emb is not None:
                            vector_dict = {
                                "dense": dense_vector.tolist(),
```

- `batch_dense_vectors`가 1차원 `(768,)`이면  
  `zip(batch_texts, batch_dense_vectors, ...)`에서  
  `batch_dense_vectors`를 이터레이션할 때 **한 번에 하나의 float**만 나옵니다.
- 그래서 **첫 번째 반복**에서 `dense_vector` = `0.01386507967254147` (스칼라)가 되고,  
  `vector_dict["dense"]` = 그 float 하나만 들어갑니다.
- Qdrant는 `dense`에 **list[float]**를 기대하므로, 스칼라가 들어가면서 `list_type` / `extra_forbidden` 등 validation 에러가 난 것입니다.

테스트는 **리뷰 142개**라서 `batch_dense_vectors`가 항상 2차원 `(N, 768)`이고, `dense_vector`가 (768,) 벡터로만 나와서 정상 동작한 겁니다.

---

## 수정 제안

**리뷰 1개여도** `batch_dense_vectors`가 2차원이 되도록 하면 됩니다. 둘 중 하나만 적용하면 됩니다.

1. **encoder 쪽 수정 (권장)**  
   `len(sentences) == 1`일 때도 1차원으로 내리지 말고 2차원 유지:

   - `return out[0]` 제거하고, 항상 `return out`  
     → 리뷰 1개면 shape `(1, 768)` 반환.
   - 또는 `return np.atleast_2d(out[0])`  
     → 동일하게 `(1, 768)` 보장.

2. **prepare_points 쪽 수정**  
   `batch_dense_vectors`를 쓸 때만 2차원으로 강제:

   - `batch_dense_vectors = self.encoder.encode(batch_texts)` 직후  
     `batch_dense_vectors = np.atleast_2d(batch_dense_vectors)`  
   - 그러면 1개여도 shape `(1, 768)`이 되어, zip에서 `dense_vector`가 (768,) 벡터로만 나옵니다.

정리하면, **Postman으로 1개 리뷰만 보낼 때만** `extra_forbidden`이 나는 이유는, 그때만 encoder가 1차원을 반환하고, 그 결과 `vector_dict["dense"]`에 float 하나가 들어가기 때문입니다. 위 둘 중 한 곳만 고치면 해결됩니다.