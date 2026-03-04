# LLM-as-a-Judge 평가

`llm_as_a_judge_samples.json`의 `sample_ids`에 대해 student 모델 추론 후 OpenAI GPT-4o로 품질(1–5점 + 이유)을 평가한다.

## 흐름

1. `llm_as_a_judge_samples.json`에서 `sample_ids` 로드
2. `val_labeled.json`에서 해당 `sample_id`만 필터
3. Student 모델로 각 샘플의 `instruction`에 대해 추론 → `pred` 수집
4. `(instruction, reference, pred)`를 OpenAI GPT-4o에 전달하여 LLM-as-a-judge 평가
5. 결과를 `llm_as_a_judge_results.json`에 저장

## 사용법

```bash
python scripts/eval_llm_as_judge.py \
  --llm-judge-samples distill_pipeline_output/eval/llm_as_a_judge_samples.json \
  --val-labeled distill_pipeline_output/labeled/20260226_051037/val_labeled.json \
  --adapter-path distill_pipeline_output/runs/YYYYMMDD_HHMMSS/adapter \
  --output distill_pipeline_output/eval/llm_as_a_judge_results.json
```

**필수 인자**

| 인자 | 설명 |
|------|------|
| `--llm-judge-samples` | `llm_as_a_judge_samples.json` 경로 |
| `--val-labeled` | `val_labeled.json` 경로 |
| `--adapter-path` | Student adapter 경로 |
| `--output` | 결과 저장 경로 |

**선택 인자**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--base-model` | `Qwen/Qwen2.5-0.5B-Instruct` | 베이스 모델 |
| `--openai-model` | `gpt-4o` | Judge용 OpenAI 모델 |
| `--openai-api-key` | `OPENAI_API_KEY` | OpenAI API 키 |
| `--max-samples` | 0 (전부) | 평가할 최대 샘플 수 |

## 출력 형식

```json
{
  "meta": {
    "n_samples": 50,
    "avg_score": 3.8,
    "judge_model": "gpt-4o",
    "adapter_path": "..."
  },
  "results": [
    {
      "sample_id": 839,
      "instruction": "...",
      "ref": "...",
      "pred": "...",
      "score": 4,
      "reason": "참조와 비교해 핵심 내용을 잘 반영했고 표현이 자연스러움."
    }
  ]
}
```

## 요구사항

- `OPENAI_API_KEY` 환경변수 또는 `--openai-api-key`
- `openai>=1.0.0` (기존 `requirements.txt` 포함)
- Student 추론용 GPU (adapter + base model 로드)
