
`llm_as_a_judge_samples.json`은 `sample_ids`만 담고 있고, 레포 코드에서는 이 파일을 생성·참조하는 부분은 없습니다. 아래는 “그 샘플에 대해 student 추론 + GPT-4o judge”를 넣는 설계입니다.

---

## 1. 흐름

1. **`llm_as_a_judge_samples.json`** 에서 `sample_ids` 로드  
2. **val_labeled** 에서 해당 `sample_id`만 필터  
3. **Student 모델** 로 각 샘플의 `instruction`에 대해 추론 → `pred` 수집  
4. **(instruction, reference, pred)** 묶음을 **OpenAI GPT-4o**에 넘겨서 LLM-as-a-judge 점수/평가  
5. 결과를 파일로 저장 (예: `llm_as_a_judge_results.json`)

---

## 2. Judge 호출 방식

- **입력**: `instruction`(원문 리뷰/요약 요청), `reference`(정답 요약), `pred`(student 출력)  
- **역할**: 동일한 adapter/스크립트 체인으로 이미 쓰는 `eval_distill`의 `_load_model_and_tokenizer` + `_generate_one`로 student 추론만 추가하면 됨.  
- **Judge API**: OpenAI Chat Completions (GPT-4o) 한 번 호출 per sample (또는 배치로 여러 개 한 번에).  
- **출력 형태** 예:
  - **점수형**: 1–5 점수 + 짧은 이유 (예: relevance, faithfulness, completeness)
  - **선택형**: A/B 중 어떤 게 나은지 + 이유  
  - **통합**: “reference와 비교해 pred의 품질을 1–5로 매기고 한 줄 이유를 써라” 같은 단일 프롬프트

원하시는 게 “점수만”, “선택만”, “점수+이유” 중 어떤지 정해두면 프롬프트 문구까지 구체화할 수 있습니다.

---

## 3. 스크립트 위치/입출력 제안

- **파일**: `scripts/eval_llm_as_judge.py`  
- **입력**:  
  - `--llm-judge-samples` → `llm_as_a_judge_samples.json`  
  - `--val-labeled` → `val_labeled.json`  
  - `--adapter-path`, `--base-model` → student와 동일  
  - `--output` → 결과 저장 경로 (예: `distill_pipeline_output/eval/llm_as_a_judge_results.json`)  
  - `--openai-api-key` 또는 `OPENAI_API_KEY`  
- **출력**:  
  - 예: `[{ "sample_id", "pred", "ref", "score", "reason" }, ...]`  
  - 또는 judge가 반환하는 JSON 블록을 그대로 저장 (나중에 메트릭 집계용).

---

## 4. eval_distill과의 관계

- **Student 추론**: `eval_distill`의 `_load_model_and_tokenizer`, `_generate_one`를 그대로 재사용하거나, 같은 로직을 한 번 더 두어도 됨.  
- **ROUGE/BERTScore**: 기존 `eval_distill`은 그대로 두고,  
- **LLM-as-a-judge**: `llm_as_a_judge_samples`에 대해서만 위 스크립트로 “student 출력 + GPT-4o 평가”를 돌리는 식으로 분리하는 게 정리하기 좋습니다.  
- 필요하면 나중에 `run_eval_and_upload_artifact`나 flow에서 “eval 디렉터리 안에 `llm_as_a_judge_results.json`도 만들고 artifact에 포함”하도록 한 단계만 더 넣으면 됩니다.

---

## 5. 정리

- **llm_as_a_judge_samples**에 대해  
  - student 추론은 **지금 쓰는 adapter + `eval_distill` 스타일 로딩/생성**으로 하고,  
  - 평가는 **OpenAI GPT-4o**로 “instruction / reference / student output”을 주고 점수(또는 선택+이유)를 받으면 됩니다.  
- 위처럼 전용 스크립트 `eval_llm_as_judge.py` 하나 두고, 입력으로 `llm_as_a_judge_samples.json` + val_labeled + adapter, 출력으로 judge 결과 파일을 두면 “eval에서 student output을 못 보는” 부분은 이 경로로 충분히 볼 수 있습니다.

원하시면 Agent 모드에서 `eval_llm_as_judge.py` 골격(로드 → student 추론 → GPT-4o 호출 → 저장)까지 구현해 드리겠습니다. Judge 출력 형식(점수만 / 이유 포함 / A-B 선택)만 정해주시면 됩니다.