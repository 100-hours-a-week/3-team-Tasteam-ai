
**아니요. 지금은 report.json 만들 때 쓰던 프롬프트와 똑같이 따라하지 않습니다.**

---

## report.json 만들 때 (eval_distill.py)

```141:148:scripts/eval_distill.py
    messages = [
        {"role": "system", "content": _SCHEMA_ENFORCEMENT_SYSTEM},
        {"role": "user", "content": _TINY_FEWSHOT_USER},
        {"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT},
        {"role": "user", "content": _TINY_FEWSHOT_USER_2},
        {"role": "assistant", "content": _TINY_FEWSHOT_ASSISTANT_2},
        {"role": "user", "content": instruction},
    ]
```

- **system**: JSON 스키마 강제 프롬프트  
- **user/assistant**: few-shot 2쌍 (예시 입력→출력)  
- **마지막 user**: 실제 평가할 `instruction`  
→ 이 조합으로 추론한 결과가 report.json에 들어갑니다.

---

## LLM-as-a-Judge에서 재추론할 때 (eval_llm_as_judge.py)

```55:56:scripts/eval_llm_as_judge.py
    """이미 로드된 model/tokenizer로 instruction 한 건만 추론."""
    messages = [{"role": "user", "content": instruction}]
```

- **user 메시지 하나**: `val_labeled`에서 꺼낸 `instruction`만 넣음  
- system 없음, few-shot 없음  

그래서 **같은 instruction이라도, report 생성 시와 LLM-as-a-Judge 시의 추론 조건이 다릅니다.**

---

## 그래서 생기는 차이

- report.json은 **system + few-shot + instruction**으로 나온 예측이고  
- LLM-as-a-Judge의 `pred`는 **instruction만** 넣고 다시 뽑은 예측이라  

출력 형식(JSON 준수 여부)이나 품질이 달라질 수 있습니다. Judge가 “report에 들어간 그 예측”을 보는 게 아니라 “다시 뽑은 예측”을 보는 셈입니다.

**report.json 나올 때 쓰던 프롬프트 그대로 쓰려면**, `eval_llm_as_judge.py`의 `_generate_one`을 eval_distill과 동일하게 **system + 2개 few-shot + instruction** 메시지 구성으로 바꾸면 됩니다. (원하시면 Agent 모드에서 그 부분 수정해 드릴 수 있습니다.)