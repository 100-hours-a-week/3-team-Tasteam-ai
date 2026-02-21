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

---

## 3. 구현 위치

| 항목 | 위치 |
|------|------|
| SFTTrainer `report_to="wandb"` | `scripts/train_qlora.py` |
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
