# wandb_design.md 구현 현황

`docs/design/wandb/wandb_design.md` 기준으로 반영한 내용.

---

## 1) W&B Artifacts

| Artifact | 타입 | 로깅 시점 | 비고 |
|----------|------|-----------|------|
| **split_metadata** | split_metadata | preprocess_task 종료 | `split_meta.json` (시간 범위, group_column, use_sample_weight 등) |
| **feature_sizes** | feature_sizes | preprocess_task 종료 | `feature_sizes.txt` |
| **dataset_stats** | dataset_stats | preprocess_task 종료 | n_train/n_val/n_test, split_meta 요약, train 상위 100행 SHA256 |
| **model_checkpoint** | model | train_task 종료 | run 디렉터리 전체 (model.pt, feature_sizes.txt, pipeline_version.txt 등), metadata에 pipeline_version·학습 config |
| **evaluation_report** | evaluation | train_task 종료 (test 있을 때) | `run_metrics.json`, metadata에 pipeline_version |
| **scoring_output** | scoring | score_batch_task 종료 | recommendation CSV, metadata에 pipeline_version |

---

## 2) pipeline_version ↔ W&B 매핑

- `run_manifest.json`에 **wandb_run_id** 저장 (train_task에서 wandb run id 기록).
- W&B run summary에 **pipeline_version**, **wandb_run_id** 저장.
- DB/로드 시 `pipeline_version`으로 어떤 run·artifact인지 추적 가능.

---

## 3) Prefect 레이어링

| Task | 역할 |
|------|------|
| **deepfm-preprocess** (build_dataset) | 전처리 + split 메타·feature_sizes·dataset_stats artifact |
| **deepfm-train** (train_model) | 학습 + model checkpoint artifact + (evaluate 후) metrics 로깅 + evaluation_report artifact + pipeline_version 매핑 |
| **deepfm-score-batch** | 스코어링 실행 + scoring_output artifact |

`load_to_db`는 이 레포 밖(ETL/DB 스크립트). 해당 단계에서 `run_manifest.json`의 pipeline_version·wandb_run_id를 DB에 기록하면 됨.

---

## 4) Sweep

- **미구현.** 설계대로라면 고정 time-based val 구간에서만 sweep, 상위 2~3개 설정만 seed 반복/rolling 재검증 후 test 1회.
- 추후 필요 시 `wandb.sweep` + Prefect flow 파라미터화로 추가 가능.

---

## 5) 사용

- **wandb 미설치:** `use_wandb=False` 또는 `wandb` 패키지 제거 시 모든 로깅 no-op, 파이프라인 동작은 동일.
- **설치:** `pip install wandb` 후 `wandb login`. flow 기본값 `use_wandb=True`.
- **실행:** `python training_flow.py` (동일). flow 시작 시 `init_run(project="deepfm-pipeline", config=...)` 호출.
