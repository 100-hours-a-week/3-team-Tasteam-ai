# DeepFM Training Pipeline (Prefect)

Prefect로 자동화한 DeepFM 학습 파이프라인 사용 방법과 배포 가이드입니다.

## 개요

| 구분 | 내용 |
|------|------|
| **진입점** | `deepfm_training/training_flow.py` |
| **Flow** | `deepfm_training_flow` |
| **Task** | `preprocess_task`, `train_task` |
| **단계** | 1) 전처리 → 2) 학습 → 3) 모델 저장 |

## 사전 요구사항

- Python 3.8+
- PyTorch, numpy, pandas (기존 DeepFM 학습 환경)
- Prefect 2.x

```bash
pip install prefect
```

## 디렉터리 구조

```
deepfm_training/
├── training_flow.py   # Prefect flow + tasks
├── main.py            # 기존 수동 실행 스크립트 (그대로 사용 가능)
├── model/
├── data/
│   ├── raw/           # Criteo raw (train.txt, test.txt)
│   ├── train.txt      # 전처리 결과 (preprocess_task 생성)
│   ├── test.txt
│   └── feature_sizes.txt
├── output/            # 학습된 모델 저장 (flow run별 하위 디렉터리)
│   └── <run_id>/
│       ├── model.pt
│       └── feature_sizes.txt
└── utils/
```

## 실행 방법

### 1. 로컬에서 한 번 실행 (기본 인자)

프로젝트 루트에서:

```bash
python deepfm_training/training_flow.py
```

또는 `deepfm_training` 디렉터리에서:

```bash
cd deepfm_training && python training_flow.py
```

- **전제**: `data/raw/`에 Criteo 형식의 `train.txt`, `test.txt`가 있어야 합니다.
- **결과**: `data/`에 전처리 결과 생성 후 학습하고, `output/<run_id>/`에 `model.pt`, `feature_sizes.txt` 저장.

### 2. 파라미터 지정해서 실행

`training_flow.py` 하단의 `if __name__ == "__main__"` 블록을 수정하거나, 프로젝트 루트를 `PYTHONPATH`에 넣은 뒤 다른 스크립트에서 호출합니다.

```python
# 프로젝트 루트에서: PYTHONPATH=. python -c "..."
from deepfm_training.training_flow import deepfm_training_flow

result = deepfm_training_flow(
    raw_data_dir="/path/to/raw",
    processed_data_dir="/path/to/processed",
    num_train_sample=9000,
    num_test_sample=1000,
    num_val=1000,
    epochs=5,
    batch_size=100,
    lr=1e-4,
    output_dir="/path/to/output",
    use_cuda=False,
    skip_preprocess=False,
)
# result["model_path"], result["feature_sizes_path"] 등
```

### 3. 전처리 건너뛰고 재학습만

이미 전처리된 `data/`가 있을 때: `training_flow.py` 맨 아래를 다음처럼 바꿔 실행할 수 있습니다.

```python
if __name__ == "__main__":
    deepfm_training_flow(epochs=10, skip_preprocess=True)
```

또는 import로:

```python
deepfm_training_flow(
    processed_data_dir="/path/to/processed",
    num_train_sample=9000,
    num_val=1000,
    epochs=10,
    skip_preprocess=True,
)
```

## Flow 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `raw_data_dir` | `deepfm_training/data/raw` | Criteo raw 데이터 디렉터리 |
| `processed_data_dir` | `deepfm_training/data` | 전처리 결과 저장 경로 |
| `num_train_sample` | 9000 | 전처리 시 학습 샘플 수 |
| `num_test_sample` | 1000 | 전처리 시 테스트 샘플 수 |
| `num_val` | 1000 | 검증 구간 크기 (num_train ~ num_train+num_val) |
| `epochs` | 5 | 학습 에폭 |
| `batch_size` | 100 | 배치 크기 |
| `lr` | 1e-4 | 학습률 |
| `output_dir` | `deepfm_training/output` | 모델 저장 루트 |
| `use_cuda` | False | GPU 사용 여부 |
| `skip_preprocess` | False | True면 전처리 생략 |

## 출력

- **모델**: `output_dir/<flow_run_id>/model.pt` (state_dict)
- **메타**: `output_dir/<flow_run_id>/feature_sizes.txt` (추론 시 필요)
- Flow 반환값: `{"model_path": ..., "feature_sizes_path": ..., "processed_data_dir": ...}`

## Prefect Server / UI (선택)

실행 이력과 로그를 보려면 Prefect 서버를 띄운 뒤 실행합니다.

```bash
# 터미널 1: 서버
prefect server start

# 터미널 2: API 설정 후 flow 실행
export PREFECT_API_URL=http://127.0.0.1:4200/api
python deepfm_training/training_flow.py
```

브라우저에서 http://127.0.0.1:4200 으로 Flow Run 목록과 로그 확인 가능합니다.

## Deployment 및 스케줄 (자동화)

주기적 재학습을 위해 Deployment로 등록할 수 있습니다.

```bash
# 프로젝트 루트에서
prefect deployment build deepfm_training/training_flow.py:deepfm_training_flow \
  --name deepfm-weekly \
  --tag deepfm
```

스케줄 추가 예시 (매주 일요일 02:00):

```bash
prefect deployment schedule deepfm-weekly --cron "0 2 * * 0"
```

Worker 실행:

```bash
prefect worker start --pool default-pool
```

(실제로는 work pool 이름을 프로젝트에 맞게 생성한 뒤 사용합니다.)

## 재시도·캐싱 (확장)

- **재시도**: Task에 `@task(retries=2, retry_delay_seconds=60)` 추가 시 실패 시 자동 재시도.
- **캐싱**: Prefect 2의 task 결과 캐싱을 사용하면, 같은 `raw_data_dir`·샘플 수일 때 전처리 task를 생략할 수 있음.
- **GPU 1대**: 동시에 학습 1개만 돌리려면 work pool concurrency limit 1 또는 tag 기반 global concurrency를 설정.

## 참고

- 설계 배경·Task 분리·운영 포인트: [prefect_designe.md](prefect_designe.md)
- Prefect 개념: [what_is_prefect.md](what_is_prefect.md)
