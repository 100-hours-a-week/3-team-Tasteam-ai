```bash
{
  "val": {
    "n_samples": 944,
    "rouge1": 0.0,
    "rouge2": 0.0,
    "rougeL": 0.0,
    "bertscore_p": 0.0,
    "bertscore_r": 0.0,
    "bertscore_f1": 0.0,
    "samples": [
      {
        "sample_id": 839,
        "pred_len": 1290,
        "ref_len": 661
      },
      {
        "sample_id": 3624,
        "pred_len": 1024,
        "ref_len": 525
      },
      {
        "sample_id": 853,
        "pred_len": 937,
        "ref_len": 549
      },
      {
        "sample_id": 2950,
        "pred_len": 802,
        "ref_len": 725
      },
      {
        "sample_id": 3562,
        "pred_len": 857,
        "ref_len": 470
      },
      {
        "sample_id": 3666,
        "pred_len": 999,
        "ref_len": 593
      },
      {
        "sample_id": 504,
        "pred_len": 1429,
        "ref_len": 549
      },
      {
        "sample_id": 2896,
        "pred_len": 868,
        "ref_len": 651
      },
      {
        "sample_id": 2961,
        "pred_len": 538,
        "ref_len": 554
      },
      {
        "sample_id": 936,
        "pred_len": 909,
        "ref_len": 420
      }
    ]
  },
  "test": {
    "n_samples": 1277,
    "rouge1": 0.0,
    "rouge2": 0.0,
    "rougeL": 0.0,
    "bertscore_p": 0.0,
    "bertscore_r": 0.0,
    "bertscore_f1": 0.0,
    "samples": [
      {
        "sample_id": 1930,
        "pred_len": 945,
        "ref_len": 523
      },
      {
        "sample_id": 1038,
        "pred_len": 1009,
        "ref_len": 456
      },
      {
        "sample_id": 2295,
        "pred_len": 821,
        "ref_len": 625
      },
      {
        "sample_id": 2097,
        "pred_len": 454,
        "ref_len": 654
      },
      {
        "sample_id": 1618,
        "pred_len": 576,
        "ref_len": 577
      },
      {
        "sample_id": 3230,
        "pred_len": 482,
        "ref_len": 719
      },
      {
        "sample_id": 1012,
        "pred_len": 990,
        "ref_len": 544
      },
      {
        "sample_id": 234,
        "pred_len": 1003,
        "ref_len": 563
      },
      {
        "sample_id": 2153,
        "pred_len": 711,
        "ref_len": 636
      },
      {
        "sample_id": 345,
        "pred_len": 1385,
        "ref_len": 699
      }
    ]
  },
  "meta": {
    "adapter_path": "/workspace/distill_pipeline_output/eval_input/20260303_121911/adapter",
    "base_model": "Qwen/Qwen2.5-0.5B-Instruct"
  }
}
```


코드와 라벨 스키마를 기준으로 네 가지 가능성을 정리했습니다.

---

## 1. prediction / reference 매핑 (가장 유력)

**코드 (148–155행):**
- `preds`와 `refs`는 같은 루프에서 한 샘플씩 쌓이므로 **길이는 항상 같음** (944, 1277).
- `ref = s.get("output", "")` 인데, **키는 "output"이지만 값 타입이 다를 수 있음.**

가능한 문제:

- **`"output"`이 리스트인 경우**  
  - 예: `"output": ["문장1", "문장2", ...]`  
  - `ref` = 리스트 → `len(ref)` = 원소 개수(661 등) → report의 `ref_len`과 일치.  
  - `_rouge(pred, ref)`에 리스트가 들어가면 `rouge_scorer`가 기대하는 문자열이 아니라서 **ROUGE가 0**이 되거나 예외가 날 수 있음.  
  - `_bertscore(preds, refs)`에 `refs`가 리스트의 리스트면 타입 오류나 NaN → 0으로 이어질 수 있음.

- **`"output"`이 없거나 `null`**  
  - `s.get("output", "")` → `""` 또는 `None`.  
  - `ref`가 `None`이면 `len(r)`에서 170행에서 **TypeError**가 나므로, 지금처럼 ref_len 661이 찍히는 걸로 보면 **최소한 앞 10개 샘플의 ref는 None이 아님**.  
  - 즉, “ref가 전부 None”이면 report까지 오지 못했을 가능성이 큼.

**정리:**  
**ref가 문자열이 아니라 리스트(또는 다른 타입)**인 경우가,  
`ref_len`은 크게 나오는데 ROUGE/BERTScore는 0이 되는 현상과 가장 잘 맞습니다.

---

## 2. tokenizer / 전처리

- `ref`에 대한 **strip이나 전처리는 없음** (153행 그대로 사용).
- `pred`만 `_generate_one` 안에서 `.strip()` 함 (88행).
- 따라서 “reference를 strip해서 비게 만든다”는 코드 경로는 없음.  
  ref가 비어 있다면 **데이터에서 이미 비어 있거나, 리스트인데 문자열로 안 합친 경우**에 해당.

---

## 3. BERTScore

- **104행:** `if not BERTSCORE_AVAILABLE or not preds or not refs` 일 때만 0 반환.  
  - `preds`/`refs` 길이가 944로 동일하므로 `not refs`는 False.
- **try/except 없음** → 예외 나면 스크립트가 죽어야 하고, report까지 온 걸로 보면 **BERTScore는 예외 없이 한 번은 실행된 상태**.
- 다만 `refs`가 **리스트의 리스트**면, 내부에서 NaN이 나와서 평균이 0으로 나오는 식의 동작은 가능.

---

## 4. ROUGE scorer

- **31–35행:** `rouge_score` import 실패 시 `ROUGE_AVAILABLE = False` → **93행**에서 전부 0 반환.
- **90–100행:** `_rouge` 안에는 try/except 없음.  
  - import는 됐는데, **인자 타입이 잘못되면**(예: ref가 리스트) 라이브러리 동작에 따라 0이 나오거나 예외가 날 수 있음.

---

## 결론 및 확인 방법

- **가장 유력한 원인:**  
  **labeled JSON의 `output`이 문자열이 아니라 리스트(또는 다른 타입)**이고,  
  그대로 `ref`로 써서 ROUGE/BERTScore에 넣고 있는 경우.  
  → `ref_len`은 `len(ref)`(리스트면 원소 수)로 커지고, 점수만 0이 됨.

**직접 확인 방법:**

1. **실제 labeled 파일 한두 개만 보기**
   - `distill_pipeline_output/labeled/20260226_051037/val_labeled.json`  
   - `samples[0]`의 `"output"` 타입 확인:  
     - 문자열이면 → 2번(전처리/토크나이저)이나 4번(ROUGE 초기화)을 더 봐야 하고,  
     - **리스트/딕셔너리면 → 1번(매핑/타입)이 원인**으로 보면 됨.

2. **코드 쪽 방어 로직 제안**  
   - `ref = s.get("output", "")` 다음에  
     - `None`이면 `""`로,  
     - 리스트/딕셔너리면 `" ".join(ref)` 또는 `json.dumps(ref, ensure_ascii=False)` 등으로 **항상 str로 통일**한 뒤  
     - `preds`/`refs`와 ROUGE/BERTScore에 넘기면,  
   “reference가 비어 있거나 타입이 어긋나서 0이 나오는” 경우를 줄일 수 있습니다.

실제로 `val_labeled.json`의 `samples[0]["output"]` 타입을 한 번만 확인해 보시면, 위 네 가지 중 어떤 케이스에 해당하는지 바로 좁힐 수 있습니다.

