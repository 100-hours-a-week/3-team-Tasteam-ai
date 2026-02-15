# 최근 변경사항 요약

이 문서는 벡터 API 복구, Spark 서비스 분리, Summary 음식점별 recall seed 도입 등 최근 적용된 변경을 정리합니다.

---

## 1. 벡터 API

### 복구된 API

| 항목 | 내용 |
|------|------|
| **엔드포인트** | `POST /api/v1/vector/upload` |
| **역할** | 리뷰·레스토랑 데이터를 벡터 DB(Qdrant)에 업로드. `VectorUploadRequest`(reviews, restaurants) → prepare_points → upload_collection. |
| **추가 동작** | 업로드 후 `restaurant_vectors` 컬렉션에 레스토랑 대표 벡터 자동 생성(비교군 검색용). |
| **코드** | `src/api/routers/vector.py` (업로드 전용), `src/api/main.py`에 라우터 등록. |

### 제거된 API (유지하지 않음)

- `POST /api/v1/vector/search/similar` — 외부 공개용 벡터 유사 검색 API. 제거됨.
- sentiment / summary / comparison은 **내부**에서만 `VectorSearch`(get_restaurant_reviews, query_hybrid_search 등)를 사용하며, 이 경로는 변경 없음.

---

## 2. Spark 마이크로서비스 (Comparison)

### 구성

- **Spark 서비스**: `scripts/spark_service.py` — FastAPI, pyspark 로드. 메인 앱/워커는 JVM 없이 HTTP로 호출.
- **메인 앱**: `SPARK_SERVICE_URL` 설정 시 로컬 Spark 대신 해당 URL 사용.

### 엔드포인트 (Spark 서비스)

| 메서드 | 경로 | 응답 |
|--------|------|------|
| POST | `/all-average-from-file` | `{"service": float, "price": float}` — 전체 평균 긍정 비율(service/price). |
| POST | `/recall-seeds-from-file` | `{"service": [[phrase, count], ...], "price": ..., "food": ..., "other": ...}` — 4-way recall 시드. |
| GET | `/health` | `{"status": "ok"}` |

### Comparison에서의 사용

- **개별 음식점 comparison**: 전체 평균을 구할 때만 Spark(또는 Spark 서비스) 사용. `all-average-from-file`만 호출하며, `recall-seeds-from-file`은 comparison 파이프라인에서 호출하지 않음.
- **전체 평균 경로**: `aspect_data_path`(또는 Config) 있으면 → `calculate_all_average_ratios_from_file` → `SPARK_SERVICE_URL` 있으면 HTTP, 없으면 로컬 Spark. 실패 시 Qdrant 리뷰로 계산 또는 Config fallback.
- **레거시**: 로컬 Spark, Python(Kiwi) 폴백 모두 유지.

상세: `docs/spark/SPARK_SERVICE.md`.

---

## 3. Summary — 음식점별 Recall Seed

### 변경 내용

- **기존**: 모든 음식점에 동일한 고정 시드(`DEFAULT_SERVICE_SEEDS`, `DEFAULT_PRICE_SEEDS`, `DEFAULT_FOOD_SEEDS`)로 하이브리드 검색.
- **변경**: **음식점마다** 해당 리뷰에서 recall seed를 생성해 카테고리별 검색 쿼리로 사용. 실패 또는 시드 부족 시 기존처럼 기본 시드로 폴백.

### Spark vs Python 기준

| 조건 | 사용 |
|------|------|
| 리뷰 수 **< RECALL_SEEDS_SPARK_THRESHOLD**(기본 2000) | Python(Kiwi) — `_python_recall_seeds` |
| 리뷰 수 **≥** threshold 이고 Spark 사용 가능 | Spark — `_spark_recall_seeds` |
| Spark 비활성/미설치 | Python(Kiwi) |

- **Config**: `RECALL_SEEDS_SPARK_THRESHOLD` (기본 2000). `Config.RECALL_SEEDS_SPARK_THRESHOLD`, 환경 변수 `RECALL_SEEDS_SPARK_THRESHOLD`.

### 추가된 코드

- `src/comparison_pipeline.py`: `_python_recall_seeds(texts, stopwords)` — Spark 없이 Kiwi만으로 4-way recall seed 계산.
- `compute_recall_seeds_from_reviews`: threshold·Spark 가용 여부에 따라 Spark 또는 Python 호출; Spark 예외 시 Python 폴백.
- `src/api/routers/llm.py`: `_get_seed_list_for_restaurant(vector_search, restaurant_id)` — 해당 음식점 리뷰로 recall seed 생성 후 `seed_list`/`name_list` 반환, 실패 시 기본 시드.
- 단일/배치 Summary 모두에서 위 시드 사용. 배치는 레스토랑마다 별도 recall seed 생성.

### 레거시 유지

- 기본 시드(`DEFAULT_*_SEEDS`)는 **폴백**으로 그대로 사용.
- 고정 시드만 쓰던 경로는 recall 실패/빈 결과일 때 자동으로 타게 됨.

상세: `docs/spark/SUMMARY_RECALL_SEEDS.md`.

---

## 4. Config 추가

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `RECALL_SEEDS_SPARK_THRESHOLD` | 2000 | 이 리뷰 수 미만이면 recall seed 계산에 Python(Kiwi), 이상이면 Spark 사용. `_SparkConfig`에 정의. |

---

## 5. 문서 변경

| 문서 | 변경 |
|------|------|
| **docs/spark/SUMMARY_RECALL_SEEDS.md** | 신규 — Summary용 음식점별 recall seed, Spark/Python 기준, 설정·코드 위치 정리. |
| **etc_md/SUMMARY_PIPELINE.md** | 시드 문구를 "음식점마다 recall seed 생성, 실패 시 기본 시드"로 수정, `docs/spark/SUMMARY_RECALL_SEEDS.md` 링크. |
| **docs/architecture/ARCHITECTURE_OVERVIEW.md** | §11 Summary: 시드 결정 단계·배치 시드 설명 수정, `RECALL_SEEDS_SPARK_THRESHOLD` Config 표 추가, `docs/spark/SUMMARY_RECALL_SEEDS.md` 참조. |
| **docs/spark/SPARK_SERVICE.md** | `Config.RECALL_SEEDS_SPARK_THRESHOLD` 설명 및 `docs/spark/SUMMARY_RECALL_SEEDS.md` 링크 추가. |

---

## 6. 요약 표

| 영역 | 변경 요약 | 레거시 |
|------|-----------|--------|
| 벡터 API | upload 복구, search/similar 제거 | 내부 VectorSearch 사용 동일 |
| Spark | 서비스 URL 설정 시 HTTP 호출 | 로컬 Spark·Python 폴백 유지 |
| Summary 시드 | 음식점별 recall seed 도입 | 기본 시드 폴백 유지 |
| Recall seed 계산 | threshold 기준 Spark/Python 분기, Python 구현 추가 | Spark 경로·기존 폴백 유지 |

---

## 7. RQ·LLM(Pod)·Spark 분리 문서 — 모놀리식(Monolithic) vs 마이크로서비스(MSA)

- **모놀리식 vs MSA**: **docs/architecture/ARCHITECTURE_OVERVIEW.md §4.1** 에 **모놀리식(Monolithic)** 과 **마이크로서비스(MSA)** 비교 표가 있음. (구조·리소스·장애·배포 관점에서 분리 이유·장점 정리.)
- **RQ**: `docs/batch/offline_batch_strategy.md`, `docs/batch/offline_batch_processing.md`, `docs/runpod/lambda_runpod_pod.md`, ARCHITECTURE §8.5에 배치 큐·enqueue·재시도·DLQ·오프라인 흐름 정리됨.
- **RunPod Pod에서 LLM 별도 프로세스(마이크로서비스) 분리**·**Spark 별도 마이크로서비스(MSA) 분리**의 **장점**은 **docs/architecture/ARCHITECTURE_OVERVIEW.md §4.1** 에 명시됨.  
  - LLM: 리소스 격리, 독립 스케일, 비용 제어, 안정성, Prometheus 스크래핑.  
  - Spark: JVM 미로드, Docker/경량 환경, 장애 격리, 리소스 분리.  
  - RQ: API 과부하 방지, 재시도·DLQ, 오프라인 배치.
