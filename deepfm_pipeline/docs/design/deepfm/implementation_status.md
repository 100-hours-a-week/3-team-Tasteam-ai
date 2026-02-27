# deepfm_design.md 구현 현황

`docs/design/deepfm_design.md` 섹션(§0~§9) 기준으로 **구현 완료**, **부분 구현**, **미구현(이유·담당)**을 정리한 문서다.

---

## §0 목표 정의

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| 온라인 실시간 추론 제외, 배치 precompute 유지 | ✅ | 학습·스코어링 모두 배치. |
| DeepFM 역할: score·rank 계산 후 recommendation 저장 | ✅ | §6-3 `utils/score_batch.py`에서 score/rank 출력 → CSV. DB insert는 호출 측. |
| 성과지표가 추천 테크스펙 KR(CTR/Save 전환)과 정합 | ✅ | NDCG@K, Recall@K, AUC로 오프라인 평가. |

---

## §1 데이터 소스 & 스키마 매핑

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| 관측 단위: recommendation(노출) + implicit_feedback(라벨) | ✅ | 전처리 입력 CSV가 해당 join 결과라는 전제. |
| user_feature, restaurant_feature, context_snapshot 사용 | ✅ | `dataPreprocess`가 해당 피처 컬럼을 사용. |

---

## §2 DeepFM 입력 피처 설계

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| **§2-1** user_id / anonymous_cohort_id 임베딩 | ✅ | 범주형 `user_id`, `anon_cohort_id` + 연속형 `is_anonymous`. |
| preferred_categories weighted multi-hot | ⚠️ 부분 | 상위 K=3 슬롯(`pref_cat_1~3`, `pref_w_1~3`)으로 대체. |
| taste_preferences, visit_time_distribution numeric/top1 | ✅ | 연속형 4+4. |
| avg_price_tier categorical | ✅ | |
| **§2-2** restaurant_id, categories, price_tier, region_gu/dong, geohash | ✅ | `primary_category` 1개, region_gu/dong, geohash. |
| positive_segments, comparison_tags multi-hot | ⚠️ 부분 | `first_positive_segment`, `first_comparison_tag` 1개씩. |
| **§2-3** context_snapshot 필드 전부 categorical | ✅ | day_of_week, time_slot, admin_dong, geohash, distance_bucket, weather_bucket, dining_type. |

---

## §3 라벨링(학습 타깃)

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| 라벨 윈도우: generated_at ~ expires_at | ✅ | ETL에서 적용 후 CSV로 전달하는 전제. |
| Binary label + sample_weight (max(weight)) | ✅ | `use_sample_weight=True`, Dataset·DeepFM에서 BCE weight 적용. |

---

## §4 학습/검증 데이터 분할 전략

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| **§4-1** 시간 기준 split, row 랜덤 split 금지 | ✅ | `time_column` + train_end/valid_end/test_end, val.txt 또는 train 뒤쪽. |
| **§4-2** recommendation 단위 보존 | ✅ | `group_column`으로 그룹 단위 분할. |
| Feature cutoff (computed_at / tags_generated_at ≤ t) | ❌ 이 레포 미구현 | ETL/데이터 생성 단계에서 join 조건으로 처리. |

---

## §5 평가 지표 (오프라인)

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| **§5-1** Weighted NDCG@K (K=5/10), gain=max(weight), 모델 점수 재정렬 | ✅ | `utils/evaluate.run_evaluation()`. |
| **§5-2** Recall@K | ✅ | 동일. |
| **§5-3** AUC (모니터링) | ✅ | `run_evaluation()` 결과에 포함. |

---

## §6 배치 파이프라인

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| **§6-1** Feature pipeline (analytics → user/restaurant feature) | ❌ 이 레포 밖 | 기존 추천/분석 파이프라인. |
| **§6-2** Training pipeline | ✅ | |
| 1) 학습 데이터셋 생성 (recommendation + feedback + feature cutoff join) | ✅ 전제 | ETL이 CSV 생성, `dataPreprocess`가 train/val/test.txt 생성. |
| 2) DeepFM 학습 | ✅ | `training_flow.py` |
| 3) 모델 아티팩트 저장 + **pipeline_version 발급** | ✅ | `pipeline_version.txt`, `run_manifest.json` |
| 4) 오프라인 지표(NDCG@K/Recall@K/AUC) **기록** | ✅ | `run_metrics.json`, `run_manifest.json`에 metrics 저장. |
| **§6-3** Scoring / Recommendation generation pipeline | ✅ | |
| 사용자·코호트 목록 + 후보 음식점 → DeepFM score | ✅ | `utils/score_batch.py` |
| TopN → score, rank, context_snapshot, pipeline_version, generated_at, expires_at 출력 | ✅ | `--out` CSV. recommendation 테이블 insert는 호출 측(ETL/DB). |
| TTL 24h | ✅ | `--ttl-hours` (기본 24). |

---

## §7 서빙 API 설계

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| §7-1 컨텍스트 키 정규화 (cache key = context_snapshot bucket) | ❌ 이 레포 밖 | 추천 API/캐시 레이어. |
| §7-2 노출 로그(exposure / requestId) | ❌ 이 레포 밖 | BE/프론트 연동. |

---

## §8 운영/데이터 품질 가드레일

| 요구사항 | 상태 | 비고 |
|----------|------|------|
| §8-1 멱등/추천 피드백 dedup 규칙 문서화 | ❌ | 별도 운영 문서에서 정리 권장. |
| §8-2 데이터 보관/파티셔닝, 학습 윈도우(예: 90일) | ❌ | ETL/인프라 정책. |

---

## §9 최소 구현 로드맵 (3단계)

| 단계 | 요구사항 | 상태 | 비고 |
|------|----------|------|------|
| **1** | 오프라인 학습/평가 루프: dataset 생성 + 학습 + NDCG@K 평가 + pipeline_version 발급 | ✅ | 전처리 → training_flow → run_metrics.json·pipeline_version.txt |
| **2** | 배치 추천 생성 → recommendation 저장 (score/rank/context_snapshot/pipeline_version/TTL) | ✅ | `score_batch.py`가 CSV 출력. DB insert는 ETL/스크립트에서 수행. |
| **3** | 서빙 + 피드백 루프 연결, 가능하면 노출 로그 | ❌ 이 레포 밖 | 추천 API·Analytics. |

---

## 파일별 역할 (deepfm_design 기준)

| 파일 | 대응 설계 | 설명 |
|------|-----------|------|
| `utils/dataPreprocess.py` | §1, §2, §3, §4 | 시간/그룹 분할, 피처 추출, sample_weight, split_meta·test_meta. |
| `data/dataset.py` | §3 | train/val/test 로드, sample_weight 반환. |
| `model/DeepFM.py` | §2 | 피처 임베딩·FM·DNN, fit 시 sample_weight BCE. |
| `training_flow.py` | §6-2 | 전처리 → 학습 → pipeline_version 발급 → run_metrics·run_manifest 기록. |
| `utils/evaluate.py` | §5 | NDCG@K, Recall@K, AUC, warm/cold 구간. |
| `utils/score_batch.py` | §6-3 | 배치 스코어링 → recommendation 형식 CSV (score, rank, pipeline_version, TTL). |

---

## 이 레포에서 하지 않는 것 (어디서 할지)

| 항목 | 설계 | 담당 | 비고 |
|------|------|------|------|
| Feature cutoff | §4-2 | ETL/데이터 생성 | generated_at 시점 이전 user_feature·restaurant_feature join. |
| 노출→반응 join (recommendation + implicit_feedback 윈도우) | §3-1 | ETL/데이터 생성 | train.csv는 이미 해당 join 결과 전제. |
| recommendation 테이블 INSERT | §6-3 | ETL/DB 스크립트 | `score_batch.py` 출력 CSV를 DB에 적재. |
| 서빙·캐시·노출 로그 | §7 | BE/추천 API | 이 레포는 학습·스코어링만. |
| 코호트 생성 (anonymous_cohort_id) | §2-1 전제 | user_feature 배치 | 해시/룰/클러스터. |

---

## 사용 예시

### 학습 (§6-2)

```bash
python deepfm_training/training_flow.py
# 또는 Prefect flow 호출. 출력: output/<run_id>/model.pt, pipeline_version.txt, run_metrics.json, run_manifest.json
```

### 배치 스코어링 (§6-3)

```bash
cd deepfm_training
python -m utils.score_batch \
  --run-dir output/manual \
  --candidates data/candidates.txt \
  --meta data/candidates_meta.csv \
  --out output/recommendations.csv \
  --ttl-hours 24
```

후보 CSV는 전처리와 동일한 컬럼 순서(연속+범주 인덱스, 32열). meta CSV는 user_id, anonymous_id, restaurant_id, context_snapshot.
