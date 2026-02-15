# Summary용 음식점별 Recall Seed

Summary 파이프라인에서 **카테고리별 하이브리드 검색**에 쓰는 쿼리 시드를 **음식점마다 해당 리뷰에서 도출한 recall seed**로 생성합니다.  
고정 시드만 쓰던 방식에서, **해당 가게 리뷰에서 자주 나오는 구문**으로 검색해 품질을 높이는 구조입니다.

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| **목적** | service / price / food 카테고리별 검색 쿼리를 **해당 음식점 리뷰 기반 recall seed**로 생성 |
| **이점** | 음식점·도메인별 실제 표현에 맞는 시드로 검색 → 검색 정확도·요약 품질 향상 |
| **폴백** | recall seed 생성 실패 또는 리뷰 부족 시 **기본 시드** (`DEFAULT_SERVICE_SEEDS` 등) 사용 |

---

## 2. 흐름

1. **단일 요약** (`POST /api/v1/llm/summarize`): `restaurant_id`로 해당 음식점 리뷰 조회 → recall seed 생성 → 시드 리스트로 카테고리별 하이브리드 검색 → LLM 요약.
2. **배치 요약** (`POST /api/v1/llm/summarize/batch`): **레스토랑마다** 위와 동일하게 해당 음식점 리뷰로 recall seed 생성 후 검색·요약.

```
[요청] → 해당 음식점 리뷰 조회 → recall seed 계산 (Spark 또는 Python) → seed_list/name_list
       → (실패/빈 시드 시 기본 시드) → 카테고리별 하이브리드 검색 → LLM 요약 → 응답
```

---

## 3. Recall Seed 계산 방식

- **입력**: 해당 음식점 리뷰 텍스트 리스트 (Qdrant `get_restaurant_reviews` 등으로 조회).
- **처리**: Kiwi 형태소 분석 → NNG/NNP bigram 추출 → 빈도 상위 2000개 후보 → **4-way 분류**(service, price, food, other) → quantile_split·pick_seeds_pairs·dedup → 카테고리별 구문 리스트.
- **출력**: `{ "service": [(phrase, count), ...], "price": [...], "food": [...], "other": [...] }` → Summary에서는 service/price/food만 사용해 `seed_list`, `name_list`로 변환.

분류에 쓰는 **키워드**(SERVICE_KW, PRICE_KW, FOOD_KW 등)는 `src/comparison_pipeline.py`에 하드코딩되어 있으며, Comparison 비율 계산과 동일한 정의를 사용합니다.

---

## 4. Spark vs Python (리뷰 수 기준)

Recall seed 계산은 **리뷰 수**에 따라 **Spark** 또는 **Python(Kiwi)** 중 하나를 사용합니다.

| 조건 | 사용 경로 |
|------|-----------|
| 리뷰 수 **< RECALL_SEEDS_SPARK_THRESHOLD** | **Python(Kiwi)** (`_python_recall_seeds`) |
| 리뷰 수 **≥ RECALL_SEEDS_SPARK_THRESHOLD** 이고 Spark 사용 가능 | **Spark** (`_spark_recall_seeds`) |
| Spark 비활성(`DISABLE_SPARK`) 또는 미설치 | **Python(Kiwi)** |

- **기본 threshold**: `RECALL_SEEDS_SPARK_THRESHOLD = 2000` (환경 변수로 변경 가능).
- **의도**: 소량(수백~2,000건 미만)은 단일 프로세스 Kiwi가 JVM/Spark 오버헤드 없이 유리하고, 그 이상은 Spark로 분산 처리.

Spark 사용 중 예외 발생 시(예: Py4J/JVM 오류)에는 **Python 폴백**으로 재시도합니다.

---

## 5. 설정 (Config / 환경 변수)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `RECALL_SEEDS_SPARK_THRESHOLD` | 2000 | 이 리뷰 수 미만이면 recall seed 계산에 **Python(Kiwi)** 사용, 이상이면 **Spark** 사용(가능 시). |
| `DISABLE_SPARK` | false | true 시 Spark 미사용 → recall seed는 항상 Python(Kiwi). |

- `Config.RECALL_SEEDS_SPARK_THRESHOLD`는 `src/config.py`의 `_SparkConfig`에 정의되어 있습니다.

---

## 6. 코드 위치

| 역할 | 파일 / 함수 |
|------|-------------|
| Recall seed 계산 (Python 경로) | `src/comparison_pipeline.py` — `_python_recall_seeds(texts, stopwords)` |
| Recall seed 계산 (Spark 경로) | `src/comparison_pipeline.py` — `_spark_recall_seeds(texts_rdd, stopwords)` |
| 리뷰 리스트 → recall seed (threshold·폴백 적용) | `src/comparison_pipeline.py` — `compute_recall_seeds_from_reviews(reviews, stopwords)` |
| Recall seed → Summary용 seed_list/name_list | `src/comparison_pipeline.py` — `recall_seeds_to_seed_lists(recall_seeds)` |
| 음식점별 시드 조회 (기본 시드 폴백) | `src/api/routers/llm.py` — `_get_seed_list_for_restaurant(vector_search, restaurant_id)` |
| 단일/배치 Summary에서 시드 사용 | `src/api/routers/llm.py` — `summarize_reviews`, `_process_one_restaurant_async`, 배치 루프 |

---

## 7. 관련 문서

- **Spark 마이크로서비스(MSA)**(Comparison 전체 평균·파일 기반 recall seeds): [SPARK_SERVICE.md](SPARK_SERVICE.md)
- **Summary 파이프라인 전체**: `etc_md/SUMMARY_PIPELINE.md`
- **아키텍처 요약**: [docs/architecture/ARCHITECTURE_OVERVIEW.md](../architecture/ARCHITECTURE_OVERVIEW.md) §11 Summary 파이프라인
