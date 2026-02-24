# KD + QLoRA + Prefect + wandb 구현 전략

요약 KD 파이프라인에서 학습 로깅·추적을 위한 wandb 통합 전략.

---

## 1. SFTTrainer wandb 설정

별도 QLoRA 스크립트(`scripts/train_qlora.py`)에서 SFTTrainer 사용 시:

```python
training_args = SFTConfig(
    ...
    report_to="wandb",
    run_name=flow_run_id,
)
```

Prefect flow run id를 wandb run id로 연결:

```python
from prefect.context import get_run_context

flow_run_id = str(get_run_context().flow_run.id)
os.environ["WANDB_RUN_ID"] = flow_run_id
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_NAME"] = flow_run_id
```

`scripts/distill_flows.py`의 `train_student_task`에서 subprocess 호출 전에:

```python
ctx = get_run_context()
env["WANDB_RUN_ID"] = str(ctx.flow_run.id)
```

---

## 2. wandb.log()로 추가할 지표

Trainer 기본 로깅 외에, 아래 지표는 별도 `wandb.log()`로 기록 권장:

| 시점 | 지표 | 설명 |
|------|------|------|
| **데이터 확정 시** | `data/kept`, `data/filter_drop_rate_*`, `label_type_ratio` | 라벨링 결과·품질 필터 통계 |
| **학습 종료 직후** | teacher-student gap 요약 | 별도 계산 시 |
| **eval 직후** | `val/rouge1`, `val/rougeL`, `test/bertscore_f1` | ROUGE/BERTScore 요약 |
| **eval 직후** | GPT-judge 결과, 휴먼 스키마 점수 | 선택 구현 |
| **전체** | 샘플 단위 결과 테이블, artifact 업로드 | `wandb.Table`, `wandb.log_artifact` |

학습 완료 시 adapter를 artifact로 업로드: `scripts/train_qlora.py`에서 `wandb.log_artifact(artifact)` (같은 run에 `qlora-adapter-{run_id}` 타입 `model`).

---

## 3. 구현 위치

| 항목 | 위치 |
|------|------|
| SFTTrainer `report_to="wandb"` | `scripts/train_qlora.py` |
| 학습 완료 시 adapter artifact 업로드 | `scripts/train_qlora.py` (wandb.run 있을 때) |
| Sweep 설정·에이전트 | `scripts/wandb_sweep_qlora.yaml`, `scripts/run_qlora_sweep.py` |
| `WANDB_RUN_ID` 전달 | `scripts/distill_flows.py` → `train_student_task` |
| 데이터·필터 로깅 | `scripts/label_for_distill.py` (stdout → flow에서 wandb 연결 시) |
| eval 메트릭 로깅 | `scripts/eval_distill.py` (report.json → flow에서 wandb 연결 시) |

flow에서 wandb 로깅을 하려면 task 반환값(meta, report)을 받아 `wandb.log(meta)` 등을 호출할 수 있음.  
또는 각 스크립트 내부에서 `import wandb; wandb.log(...)` 호출 (같은 `WANDB_RUN_ID` 사용 시 같은 run에 기록).

---

## 4. 의존성

```bash
pip install wandb trl transformers peft bitsandbytes datasets
```

`.env` 또는 환경변수:

```
WANDB_API_KEY=your_key
WANDB_PROJECT=tasteam-distill
```

---

## 5. 실행 예

```bash
# 전체 파이프라인 (Prefect가 WANDB_RUN_ID 전달)
python scripts/distill_flows.py all --openai-cap 500

# QLoRA만 단독 실행 (wandb 사용 시)
WANDB_RUN_ID=manual-run-001 python scripts/train_qlora.py \
  --labeled-path labeled/xxx/train_labeled.json \
  --student-model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir distill_pipeline_output
```

---

## 6. wandb Sweep (하이퍼파라미터 탐색)

`train/loss`를 최소화하는 조합을 자동 탐색 (Bayesian 등). **사람이 직접 run을 비교할 필요 없음** — sweep이 메트릭 기준으로 다음 후보를 정함.

**설정**: `scripts/wandb_sweep_qlora.yaml`  
**에이전트**: `scripts/run_qlora_sweep.py` (wandb.config → `run_train` 호출)

```bash
# 1) sweep 등록 (프로젝트 루트에서)
wandb sweep scripts/wandb_sweep_qlora.yaml
# → sweep_id 출력 (예: entity/project/abc123)

# 2) 경로 환경변수 설정 후 에이전트 실행
export WANDB_SWEEP_LABELED_PATH=distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/train_labeled.json
export WANDB_SWEEP_OUTPUT_DIR=distill_pipeline_output

# 방법 A: sweep_id를 인자로 넘겨 스크립트 직접 실행
python scripts/run_qlora_sweep.py <sweep_id>

# 방법 B: wandb CLI 에이전트 (CLI가 program을 실행할 때 sweep_id 전달되는 경우)
wandb agent <sweep_id>
```

**Prefect에서 subprocess로 실행** (프로세스 격리, `docs/wandb/wandb_subprocess_oneprocess.md`):

```bash
# sweep 등록 후
python scripts/distill_flows.py run_sweep \
  --sweep-id <sweep_id> \
  --labeled-path distill_pipeline_output/labeled/YYYYMMDD_HHMMSS/train_labeled.json \
  --out-dir distill_pipeline_output
```

탐색 파라미터 예: `learning_rate`, `num_epochs`, `batch_size`, `r`, `alpha`, `max_seq_length`.  
고정값: `student_model`, `gold_oversample_ratio`, `target_modules`, `grad_accum`.  
`labeled_path`·`output_dir`는 환경변수 `WANDB_SWEEP_LABELED_PATH`, `WANDB_SWEEP_OUTPUT_DIR`로 전달.
