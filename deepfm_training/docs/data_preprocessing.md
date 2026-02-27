
수정 반영했습니다. 변경 내용만 요약합니다.

## 1. `deepfm_training/utils/dataPreprocess.py` (Tasteam 기준 재작성)

**기반 문서:** `tasteam_deepfm_data.md`의 user_feature, restaurant_feature, implicit_feedback, context 스냅샷.

- **연속형 8개**
  - `taste_preferences`: `spicy`, `sweet`, `savory`, `light` (0~1 클리핑)
  - `visit_time_distribution`: `breakfast`, `lunch`, `afternoon`, `dinner` (0~1 클리핑)

- **범주형 16개**
  - User: `user_identifier`(user_id 또는 anonymous_cohort_id), `avg_price_tier`
  - Item: `restaurant_id`, `primary_category`(categories 첫 항목), `price_tier`, `region_gu`, `region_dong`, `geohash`, `first_positive_segment`, `first_comparison_tag`
  - Context: `day_of_week`, `time_slot`, `admin_dong`, `distance_bucket`, `weather_bucket`, `dining_type`

- **입력:** `datadir` 아래 **train.csv**, **test.csv** (선택)
  - CSV 컬럼: `user_id`, `anonymous_cohort_id`, `preferred_categories`, `avg_price_tier`, `taste_preferences`, `visit_time_distribution`, `restaurant_id`, `categories`, `price_tier`, `region_gu`, `region_dong`, `geohash`, `positive_segments`, `comparison_tags`, `day_of_week`, `time_slot`, `admin_dong`, `distance_bucket`, `weather_bucket`, `dining_type`, `weight`(또는 `label`).
  - JSONB 컬럼(`taste_preferences`, `visit_time_distribution`, `categories`, `positive_segments`, `comparison_tags`)은 문자열이면 파싱해서 사용.

- **라벨:** `weight`가 있으면 0.5 기준 이진화, 없으면 `label` 또는 1.

- **출력:** 기존과 동일한 `train.txt`, `test.txt`, `feature_sizes.txt` (연속 8 + 범주 16 = 24필드).

`test.csv`가 없으면 `train`에서 앞쪽 일부를 잘라 `test.txt`로 씁니다.

## 2. `deepfm_training/data/dataset.py` (Tasteam/Criteo 공통)

- **연속형 개수:** `feature_sizes.txt`에서 **앞쪽 연속된 1의 개수**로 자동 계산 (Tasteam 8, Criteo 13 모두 가능).
- **테스트 데이터:** `self.test_data[idx, :]`로 접근하도록 수정 (numpy라 `.iloc` 제거).
- **Xv:** 연속값을 위해 `float32`로 통일.

이제 `data/raw`에 `train.csv`(및 선택적으로 `test.csv`)를 두고 `training_flow.py`의 `preprocess_task`를 실행하면 Tasteam 스키마에 맞춘 전처리와 학습이 동작합니다.  
DB에서 내보낼 때는 `user_feature`·`restaurant_feature`·`implicit_feedback`(및 필요 시 `context_snapshot`)를 조인한 뒤 위 CSV 컬럼명에 맞춰 export하면 됩니다.