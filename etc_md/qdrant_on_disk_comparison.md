# Qdrant 벡터 스토리지: on_disk=true vs on_disk=false 비교

컬렉션 생성 시 **벡터 저장 방식**을 선택하는 `on_disk` 옵션 비교입니다.  
(Qdrant **클라이언트 연결** 방식인 `:memory:` vs `./qdrant_data` 와는 별개입니다.)

---

## 1. 요약 비교

| 항목 | on_disk=false (In-Memory) | on_disk=true (Memmap) |
|------|---------------------------|------------------------|
| **벡터 저장 위치** | RAM 전부 로드 | 디스크 파일 + 메모리 매핑(MMAP) |
| **검색 속도** | 가장 빠름 (디스크 접근 최소) | RAM이 충분하면 in-memory에 근접 |
| **RAM 사용량** | 높음 (벡터 크기만큼) | 낮음 (페이지 캐시만 사용) |
| **디스크 I/O** | 영속성용만 | 검색 시 페이지 캐시 미스 시 디스크 읽기 |
| **권장 환경** | 소규모 컬렉션, 메모리 여유 있음 | 대규모 컬렉션, 메모리 제약/비용 절감 |
| **HNSW 인덱스** | 기본적으로 RAM | `hnsw_config.on_disk=true` 로 디스크 가능 |

---

## 2. 동작 방식

### on_disk=false (기본, In-Memory)

- 모든 벡터를 **RAM에 로드**.
- 디스크 접근은 **영속성(저장/복구)** 용으로만 사용.
- 지연 시간이 낮고 **검색이 가장 빠름**.

### on_disk=true (Memmap / On-Disk)

- 벡터 데이터는 **디스크 파일**에 두고, **메모리 매핑(MMAP)** 으로 접근.
- 파일 전체를 RAM에 올리지 않고, **OS 페이지 캐시**로 자주 쓰는 페이지만 메모리에 유지.
- RAM이 충분하면 자주 접근하는 구간은 캐시되어 **in-memory에 가까운 성능**.
- 대용량 컬렉션에서 **RAM 사용을 크게 줄일 수 있음**.

---

## 3. 설정 방법

### 환경 변수 (이 프로젝트)

```bash
# 벡터를 디스크 메모리 매핑으로 저장 (대용량/비용 절감 시 권장)
QDRANT_VECTORS_ON_DISK=true

# 벡터를 RAM에 저장 (기본, 최고 속도)
QDRANT_VECTORS_ON_DISK=false
```

### 코드에서 컬렉션 생성 (Python)

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

# on_disk=False (기본): RAM에 벡터 저장
client.create_collection(
    collection_name="reviews_fast",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=False,
    ),
)

# on_disk=True: MMAP으로 디스크에 벡터 저장
client.create_collection(
    collection_name="reviews_large",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=True,
    ),
)
```

### HNSW 인덱스까지 디스크에 두기 (대용량 시)

벡터만 on_disk로 두고, HNSW 인덱스는 RAM에 둘 수도 있고, 인덱스까지 디스크에 두면 RAM을 더 줄일 수 있습니다.

```python
client.create_collection(
    collection_name="reviews_on_disk_full",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=True,
    ),
    hnsw_config=models.HnswConfigDiff(on_disk=True),
)
```

---

## 4. 언제 어떤 모드를 쓸지

| 상황 | 권장 |
|------|------|
| 리뷰/포인트 수 적고 메모리 여유 있음 | `on_disk=false` (기본) |
| 리뷰 수만~수십만 이상, RAM 비용/제약 | `on_disk=true` |
| 빠른 디스크(SSD) + 대용량 컬렉션 | `on_disk=true` + 필요 시 `hnsw_config.on_disk=true` |
| 최소 지연 시간이 최우선, RAM 충분 | `on_disk=false` |

---

## 5. 참고

- **컬렉션 생성 시에만** 지정 가능. 이미 만든 컬렉션의 스토리지 타입은 변경할 수 없음.
- **Payload 스토리지**는 별도 옵션 `on_disk_payload` 로 설정 (벡터 `on_disk`와 독립).
- 공식 문서: [Storage - Qdrant](https://qdrant.tech/documentation/concepts/storage/)
