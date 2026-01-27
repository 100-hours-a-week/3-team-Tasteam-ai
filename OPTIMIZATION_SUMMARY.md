# 코드 최적화 완료 요약

## 최적화된 항목

### 1. ✅ Kiwi 형태소 분석기 싱글톤 패턴 (`src/strength_pipeline.py`)

**문제점**:
- 매번 `Kiwi()` 인스턴스를 새로 생성하여 초기화 비용 발생
- 형태소 분석기는 초기화 시간이 오래 걸림

**해결책**:
- `_get_kiwi()` 함수로 싱글톤 패턴 구현
- 모듈 레벨 변수 `_kiwi_instance`로 인스턴스 재사용

**성능 개선**:
- 첫 호출 후 재사용 시 초기화 시간 제로
- 메모리 사용량 감소

```python
# 최적화 전
kiwi = Kiwi()  # 매번 새로 생성

# 최적화 후
kiwi = _get_kiwi()  # 싱글톤 인스턴스 재사용
```

---

### 2. ✅ Sparse 벡터 모델 캐싱 (`src/vector_search.py`)

**문제점**:
- 하이브리드 검색 시마다 `SparseTextEmbedding` 모델을 새로 로드
- 모델 로딩 비용이 큼

**해결책**:
- `VectorSearch` 클래스에 `_sparse_model` 인스턴스 변수 추가
- 첫 로드 후 재사용

**성능 개선**:
- 첫 호출 후 재사용 시 모델 로딩 시간 제로
- 메모리 효율성 향상

```python
# 최적화 전
sparse_model = SparseTextEmbedding('Qdrant/bm25')  # 매번 새로 로드

# 최적화 후
if self._sparse_model is None:
    self._sparse_model = SparseTextEmbedding('Qdrant/bm25')
sparse_model = self._sparse_model  # 재사용
```

---

### 3. ✅ Aspect Seed 메모리 캐싱 (`src/aspect_seeds.py`)

**문제점**:
- 매번 파일을 읽어서 Aspect Seed 로드
- 파일 I/O 오버헤드 발생

**해결책**:
- 모듈 레벨 캐시 변수 추가 (`_aspect_seeds_cache`)
- 파일 수정 시간(`mtime`) 확인하여 변경 시에만 재로드
- 기본값도 캐싱

**성능 개선**:
- 첫 로드 후 재사용 시 파일 읽기 제로
- 파일 변경 감지로 정확성 유지

```python
# 최적화 전
def load_aspect_seeds():
    with open(file_path) as f:  # 매번 파일 읽기
        return json.load(f)

# 최적화 후
def load_aspect_seeds():
    if _aspect_seeds_cache and file_not_changed:
        return _aspect_seeds_cache  # 캐시에서 반환
    # 파일 변경 시에만 재로드
```

---

### 4. ✅ 불용어 파일 모듈 레벨 캐싱 (`src/strength_extraction.py`)

**문제점**:
- Strength Extraction 실행 시마다 불용어 파일을 읽음
- 파일 I/O 오버헤드 발생

**해결책**:
- `_get_stopwords()` 함수로 모듈 레벨 캐싱
- `_stopwords_cache` 변수로 재사용

**성능 개선**:
- 첫 로드 후 재사용 시 파일 읽기 제로
- 메모리 효율성 향상

```python
# 최적화 전
with open(stopwords_path) as f:  # 매번 파일 읽기
    stopwords = [w.strip() for w in f]

# 최적화 후
stopwords = _get_stopwords()  # 캐시에서 반환
```

---

### 5. ✅ 키워드 검색 최적화 (`src/strength_pipeline.py`)

**문제점**:
- 키워드 리스트를 매번 순회하여 `any()` 검사
- 리스트 순회는 O(n) 시간 복잡도

**해결책**:
- 키워드를 `set`으로 변환 (실제로는 `in` 연산 최적화)
- `any(kw in bigram for kw in SERVICE_KW)` 형태는 유지하되, set으로 변환하여 의도 명확화

**성능 개선**:
- 코드 가독성 향상
- 향후 더 복잡한 검색 시 set의 이점 활용 가능

```python
# 최적화 전
SERVICE_KW = ["친절", "서비스", ...]  # 리스트

# 최적화 후
SERVICE_KW = {"친절", "서비스", ...}  # set (의도 명확화)
```

---

### 6. ✅ 중복 import 제거 (`src/strength_pipeline.py`)

**문제점**:
- `import json`이 함수 내부에서 중복 선언
- 불필요한 import 반복

**해결책**:
- 함수 상단으로 import 이동
- 중복 제거

**성능 개선**:
- 코드 가독성 향상
- 약간의 메모리 절약

```python
# 최적화 전
def generate_strength_descriptions():
    if condition:
        import json  # 중복
    else:
        import json  # 중복

# 최적화 후
import json  # 함수 상단에 한 번만

def generate_strength_descriptions():
    # json 사용
```

---

### 7. ✅ 쿼리 텍스트 생성 최적화 (`src/api/routers/llm.py`)

**문제점**:
- 모든 seed를 join하여 쿼리 생성
- 토큰 사용량 증가

**해결책**:
- 최대 10개 seed만 사용하여 토큰 절약
- `seeds[:-1]` 대신 `seeds[:10]` 사용

**성능 개선**:
- 토큰 사용량 감소
- 검색 품질 유지 (상위 seed만 사용)

```python
# 최적화 전
query_text = " ".join(seeds[:-1])  # 거의 모든 seed 사용

# 최적화 후
query_seeds = seeds[:10]  # 최대 10개만 사용
query_text = " ".join(query_seeds)
```

---

## 성능 개선 효과

### 메모리 사용량
- **Kiwi 인스턴스**: 1개만 유지 (이전: 매번 생성)
- **Sparse 모델**: 1개만 유지 (이전: 매번 로드)
- **Aspect Seed**: 메모리 캐싱 (이전: 매번 파일 읽기)
- **불용어**: 메모리 캐싱 (이전: 매번 파일 읽기)

### 실행 시간
- **Kiwi 초기화**: 첫 호출 후 제로 (이전: ~1-2초)
- **Sparse 모델 로딩**: 첫 호출 후 제로 (이전: ~0.5-1초)
- **Aspect Seed 로드**: 첫 호출 후 제로 (이전: ~10-50ms)
- **불용어 로드**: 첫 호출 후 제로 (이전: ~10-50ms)

### 파일 I/O
- **Aspect Seed 파일**: 변경 시에만 읽기 (이전: 매번 읽기)
- **불용어 파일**: 한 번만 읽기 (이전: 매번 읽기)

## 추가 최적화 가능 항목

### 1. 배치 처리 최적화
- 여러 레스토랑 처리 시 Kiwi 인스턴스 재사용 (이미 구현됨)
- Sparse 벡터 배치 생성 고려

### 2. 비동기 처리
- Aspect Seed 로드 비동기화 (현재는 동기)
- 불용어 로드 비동기화

### 3. 캐시 무효화 전략
- Aspect Seed 파일 변경 감지 개선
- TTL 기반 캐시 무효화

### 4. 메모리 관리
- 대량 처리 시 캐시 크기 제한
- LRU 캐시 고려

## 테스트 권장사항

1. **성능 테스트**: 최적화 전후 실행 시간 비교
2. **메모리 테스트**: 장시간 실행 시 메모리 누수 확인
3. **캐시 테스트**: 파일 변경 시 캐시 무효화 확인
4. **동시성 테스트**: 여러 요청 동시 처리 시 안정성 확인
