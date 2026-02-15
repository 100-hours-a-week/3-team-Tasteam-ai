strength_in_aspect.py

(env_ai) js@jinsoos-MacBook-Pro tasteam-aicode-gpu-all-python-process-runtime_for_github % /Users/js/miniconda
3/envs/env_ai/bin/python /Users/js/tasteam-aicode-gpu-a
ll-python-process-runtime_for_github/hybrid_search/fina
l_pipeline/strength_in_aspect.py
WARNING: Using incubator modules: jdk.incubator.vector
Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/01/27 18:23:24 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
26/01/27 18:23:25 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
root
 |-- content: string (nullable = true)
 |-- created_at: string (nullable = true)
 |-- deleted_at: string (nullable = true)
 |-- food_category_id: long (nullable = true)
 |-- food_category_name: string (nullable = true)
 |-- group_id: string (nullable = true)
 |-- id: long (nullable = true)
 |-- images: array (nullable = true)
 |    |-- element: string (containsNull = true)
 |-- is_recommended: boolean (nullable = true)
 |-- member_id: long (nullable = true)
 |-- restaurant_id: long (nullable = true)
 |-- restaurant_name: string (nullable = true)
 |-- subgroup_id: string (nullable = true)
 |-- updated_at: string (nullable = true)

[Stage 2:>                                             Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
Quantization is not supported for ArchType::neon. Fall back to non-quantized model.Quantization is not supported for ArchType::neon. Fall back to non-quantized model.

Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
[Stage 2:>                                             [Stage 2:>                                             [Stage 2:=========>                                    [Stage 2:===================>                          [Stage 2:=============================>                [Stage 2:=======================================>                                                             [Stage 3:>                                             [Stage 3:==============================================[Stage 4:>                                                                                                    Quantization is not supported for ArchType::neon. Fall back to non-quantized model.
[Stage 6:>                                                                                                    이 음식점의 서비스 만족도는 판교 평균의 0.97배 수준입니다.
이 음식점의 가격 만족도는 판교 평균의 4.83배 수준입니다.
/Users/js/tasteam-aicode-gpu-all-python-process-runtime_for_github/hybrid_search/final_pipeline/final_summary_pipeline.py:13: UserWarning: The model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 now uses mean pooling instead of CLS embedding. In order to preserve the previous behaviour, consider either pinning fastembed version to 0.5.1 or using `add_custom_model` functionality.
  dense_model = TextEmbedding('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

---

test_all_task.py result


✓ 강점 추출 성공 (소요 시간: 1.70초)
    레스토랑 1:
      * 추출된 강점 수: 0개
      * 카테고리별 lift: service: -100.0%, price: -100.0%
      * strength_display (판교 평균 N배):
        - 이 음식점의 서비스 만족도는 판교 평균의 0.00배 수준입니다.
        - 이 음식점의 가격 만족도는 판교 평균의 0.00배 수준입니다.
      * 강점(양수 lift): 없음 — 전체 평균 대비 양수인 카테고리 없음

---

## strength_in_aspect vs test_all_task(API) 결과가 다른 이유

| 구분 | strength_in_aspect.py | test_all_task / API (src) |
|------|------------------------|---------------------------|
| **타겟 레스토랑 ID** | `target_rid=4` (하드코딩) | `SAMPLE_RESTAURANT_ID=1` (test_data 첫 레스토랑) |
| **타겟 데이터 소스** | `test_data_sample.json` **파일**에서 `restaurant_id=4`만 필터 | **Qdrant** `get_restaurant_reviews(restaurant_id)` (기본 1) |
| **전체(판교) 평균** | `data/kr3.tsv` **전체** | ① `ALL_AVERAGE_ASPECT_DATA_PATH` ② Qdrant 전체 ③ `Config.ALL_AVERAGE_*` |

### 왜 test_all_task는 -100% / 0.00배인가?

1. **타겟 ID 불일치**  
   - strength_in_aspect: **restaurant_id=4** (test_data_sample에 4가 많음)  
   - test_all_task: **restaurant_id=1** (로드 시 `data['restaurants'][0]` = 첫 번째로 등장한 1번)

2. **타겟 리뷰 집합이 다름**  
   - 1번: 리뷰에 `친절`, `가성비`, `가격 합리`, `가격 만족`, `무한 리필`, `리필 가능` 등 **긍정 bigram**이 거의 없으면  
     → `service_ratio=0`, `price_ratio=0` →  
     `lift = (0 - 전체)/전체 * 100` = **-100%** → multiple=0 → **0.00배**  
   - 4번: 리뷰가 더 많고 위 키워드 bigram이 있으면 → 0.97배, 4.83배처럼 나옴.

3. **전체 평균 소스**  
   - `ALL_AVERAGE_ASPECT_DATA_PATH`가 비어 있으면 API는 Qdrant 전체 또는 Config 고정값 사용.  
   - strength_in_aspect는 항상 **kr3.tsv**로만 전체 평균을 계산.

### 결과를 맞추는 방법

- **타겟을 4번으로 통일**  
  - test_all_task: `SAMPLE_RESTAURANT_ID`를 4로 쓰거나, test_data 로드 시 `restaurant_id=4`인 레스토랑을 `data['restaurants'][0]`으로 두기.  
  - API 호출: `restaurant_id=4`로 요청.

- **전체 평균을 strength_in_aspect와 동일하게**  
  - `ALL_AVERAGE_ASPECT_DATA_PATH=data/kr3.tsv` (또는 strength_in_aspect와 같은 aspect_data 경로) 설정.

- **타겟 데이터를 같은 소스로**  
  - Qdrant에 test_data_sample을 업로드한 뒤, **restaurant_id=4**로 강점 추출 API를 호출.  
  - 또는 API/strength_extraction에서 테스트/로컬일 때만 `test_data_sample.json`의 `restaurant_id=4` 리뷰를 직접 쓰도록 분기 (구현 필요).