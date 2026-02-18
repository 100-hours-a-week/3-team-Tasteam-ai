# Prefect 기반 DeepFM 파이프라인 설계

## 구현 현황

- **구현 파일**: `deepfm_training/training_flow.py`
- **Flow**: `deepfm_training_flow` (전처리 → 학습, 모델 저장 포함)
- **실행 방법·파라미터·배포**: [DeepFM Training Pipeline](deepfm_training_pipeline.md) 참고.

---

현재 `deepfm_training/` 구조와 Prefect 문서를 기준으로, **Prefect로 자동화 학습 파이프라인**을 어떻게 만들면 좋을지 정리했습니다.

---

## 현재 파이프라인 요약

| 단계 | 담당 | 입출력 |
|------|------|--------|
| 1. Raw 데이터 | 수동 | `data/raw/` (train.txt, test.txt) |
| 2. 전처리 | `utils/dataPreprocess.preprocess(datadir, outdir, ...)` | `data/` → train.txt, test.txt, feature_sizes.txt |
| 3. 학습 | `main.py` (스크립트) | CriteoDataset + DeepFM.fit (모델 저장 없음) |

---

## Prefect로 나눌 단위

문서의 “Flow + Task + Deployment + Schedule” 구조에 맞춰 다음처럼 나누는 것을 추천합니다.

### 1. Task 분리

- **`preprocess_task`**  
  - `dataPreprocess.preprocess(datadir, outdir, num_train_sample, num_test_sample)` 호출.  
  - **캐싱**: `datadir` + 샘플 수 등에 대한 cache key를 두면, 같은 raw 데이터/설정이면 전처리 한 번만 실행.
- **`train_task`**  
  - 전처리된 `data/` 경로를 받아서  
    - `CriteoDataset`, `feature_sizes` 로드  
    - DeepFM 생성 → DataLoader → `model.fit(...)`  
  - **출력**: 학습된 모델 저장 경로(및 선택적으로 validation 메트릭).  
  - 필요하면 **concurrency limit 1** (GPU 1대일 때 동시 학습 1개만).
- **(선택) `save_artifact_task`**  
  - 학습된 모델 경로를 Prefect Artifact로 등록하거나, S3/GCS 등으로 업로드.

### 2. Flow 하나로 묶기

```text
deepfm_training_flow( raw_data_dir, processed_data_dir, num_train, num_test, epochs, batch_size, lr, output_dir )
  → preprocess_task(raw_data_dir, processed_data_dir, num_train, num_test)
  → train_task(processed_data_dir, epochs, batch_size, lr, output_dir)
  → (선택) save_artifact_task(model_path, run_id)
```

- **파라미터**:  
  `raw_data_dir`, `processed_data_dir`, `num_train`, `num_test`, `epochs`, `batch_size`, `lr`, `output_dir` 등을 flow 인자로 두면, 나중에 Deployment에서 스케줄/수동 실행 시마다 바꿀 수 있습니다.

### 3. 디렉터리/파일 구성 제안

- **`deepfm_training/flows/`** (또는 `deepfm_training/pipeline/`)  
  - `flows.py` (또는 `training_flow.py`):  
    - `@flow` 한 개 (예: `deepfm_training_flow`)  
    - 그 안에서 `@task` 2~3개 호출 (preprocess, train, 필요 시 artifact).
- **기존 코드는 유지**  
  - `preprocess_task`는 `utils.dataPreprocess.preprocess`를 호출만 하도록 하고,  
  - `train_task`는 지금 `main.py`에 있는 로직(데이터 로드 → DeepFM 생성 → fit)을 함수로 빼서 그 함수를 task에서 호출하면 됩니다.  
  - 그러면 기존 `main.py`는 “Prefect 없이 직접 돌리는 진입점”으로 두고, Prefect 실행은 `flows.py`에서만 하면 됩니다.

---

## 구현 시 유의할 점

1. **경로**  
   - Task/Flow에서 `raw_data_dir`, `processed_data_dir`, `output_dir`를 모두 인자로 받고, `main.py`/`dataPreprocess`의 하드코딩된 `./data`, `../data/raw` 대신 이 인자를 쓰도록 바꾸는 것이 좋습니다.  
   - Prefect worker가 다른 CWD에서 돌 수 있으므로, “프로젝트 루트 기준”이 필요하면 `Path(__file__).resolve().parent` 등으로 기준 경로를 정한 뒤 그 루트와 조합해 사용하세요.

2. **모델 저장**  
   - 현재 `DeepFM.fit`은 저장을 하지 않습니다.  
   - `train_task` 마지막에 `torch.save(model.state_dict(), output_path)` (또는 전체 모델 + `feature_sizes`)를 추가하고, `output_path`를 flow 결과/다음 task로 넘기면 됩니다.

3. **캐싱**  
   - `@task(cache_key_includes=[...])` 또는 `cache_expiration`으로 전처리 task 캐싱 시,  
     “같은 raw 데이터 + 같은 num_train/num_test”일 때만 캐시 hit 되도록 key를 설계하면, 재학습만 자주 돌릴 때 이득이 큽니다.

4. **재시도**  
   - `@task(retries=2, retry_delay_seconds=60)` 등으로 전처리/학습 실패 시 자동 재시도 가능.

5. **GPU / Concurrency**  
   - GPU 1대만 쓰는 환경이면, 해당 flow를 실행하는 work pool에서 **concurrency limit = 1**로 두거나, Prefect의 tag + global concurrency limit으로 “train” 태그 동시 1개만 실행되게 하면 충돌을 막을 수 있습니다.

---

## 운영 레벨 (자동화)

- **Deployments**  
  - `prefect deployment build deepfm_training/flows/flows.py:deepfm_training_flow --name deepfm-weekly` 처럼 deployment를 만들고,
- **Schedule**  
  - cron으로 “매주 일요일 새벽” 같은 스케줄을 deployment에 붙이면, 새 데이터만 `raw_data_dir`에 넣어두고 자동 재전처리·재학습이 가능합니다.
- **Work pool**  
  - 로컬이면 `Process` work pool, GPU 서버/도커면 해당 환경용 work pool을 만들어서 worker를 띄우면 됩니다.

---

## 요약

- **Task**: 전처리 1개 + 학습 1개 (+ 선택적으로 아티팩트 저장 1개).  
- **Flow**: 위 세 단계를 순서대로 호출하는 단일 flow, 인자로 경로·샘플 수·epochs 등 전부 받기.  
- **기존 코드**: `dataPreprocess.preprocess`와 `main.py`의 학습 로직은 그대로 두고, “경로/하이퍼파라미터를 인자로 받는 래퍼 함수”를 만든 뒤 그걸 `@task`에서 호출.  
- **추가 작업**: 학습 끝난 뒤 `torch.save`로 모델 저장하고, 그 경로를 flow 출력/artifact로 노출.

이렇게 하면 `deepfm_training/`을 Prefect 기반 자동화 학습 파이프라인으로 정리할 수 있고, 스케줄·재시도·캐싱·모니터링까지 한 번에 다룰 수 있습니다.  
원하시면 `flows.py`에 넣을 구체적인 함수 시그니처와 `@flow`/`@task` 예시 코드도 단계별로 써 드리겠습니다.