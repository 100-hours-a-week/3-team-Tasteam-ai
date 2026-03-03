
현재 코드 기준으로 Teacher와 Student 설정 위치는 다음과 같습니다.

---

## 1. Teacher 모델

**사용처:** `labeling_flow` (self-hosted teacher로 라벨 생성)

| 설정 방법 | 위치 | 변수/설정 |
|-----------|------|-----------|
| **환경 변수** | `.env` | `LLM_MODEL` |
| **코드 기본값** | `src/config.py` | `DEFAULT_LLM_MODEL` = `/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct` |
| **OpenAI 사용 시** | `src/config.py` | `OPENAI_MODEL` = `gpt-4o-mini` |

RunPod vLLM에 Qwen 7B가 올라가 있으면, `LLM_MODEL`이 그 모델 경로/이름과 맞아야 합니다.  
`labeling_flow`는 아직 스텁이라, 구현 시 `Config.LLM_MODEL` 또는 `teacher_config` 같은 teacher 설정을 이쪽에서 가져와서 쓰게 됩니다.

---

## 2. Student 모델

**사용처:** `train_student_flow` (QLoRA SFT)

| 설정 방법 | 위치 | 변수/설정 |
|-----------|------|-----------|
| **Flow 인자** | `scripts/distill_flows.py` | `student_model` 기본값: `"Qwen/Qwen2.5-0.5B-Instruct"` |
| **CLI** | `scripts/distill_flows.py` | `--student-model` |

```161:176:scripts/distill_flows.py
@task(name="train-student-task", log_prints=True)
def train_student_task(
    labeled_path: str,
    student_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ...
)
```

`all` 명령 실행 시 예:

```bash
python scripts/distill_flows.py all --student-model Qwen/Qwen2.5-0.5B-Instruct
```

---

## 3. 요약

| 모델 | 설정 위치 | 설정 방식 |
|------|-----------|-----------|
| **Teacher** | `src/config.py`, `.env` | `LLM_MODEL` (RunPod vLLM), `OPENAI_MODEL` (OpenAI) |
| **Student** | `scripts/distill_flows.py` | flow 인자 `student_model` 또는 CLI `--student-model` |

현재는 **teacher**가 앱 LLM 설정(`Config.LLM_MODEL` 등)과 공유되고, **student**는 `distill_flows.py`에서만 지정됩니다.  
EasyDistill 같은 별도 도구를 쓰면 `models.teacher`, `models.student`처럼 별도 config 파일에서 설정할 수 있습니다.