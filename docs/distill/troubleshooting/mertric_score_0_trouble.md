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

---


`val_labeled.json` 7–8행을 보면 **`output`은 문자열**입니다. JSON을 문자열로 이어붙인 형태라서, `ref = s.get("output", "")` 도 **긴 문자열**이 됩니다.

---

## 정리

- **`output` 타입**: 문자열 (리스트 아님)  
  → `ref` 도 문자열, `ref_len` 661 등은 **문자 수**로 이해하면 됨.  
  → “reference가 리스트라서 ref_len만 크고 점수만 0” 가능성은 **이 파일 구조로는 해당 없음**.

- **reference가 비어 있다**는 가설도, 이 스키마에서는 맞지 않음.  
  (실제로 쓰인 ref는 600자대 문자열일 가능성이 큼.)

---

## 그런데도 ROUGE/BERTScore가 0일 수 있는 이유

1. **ROUGE**
   - reference는 **따옴표·중괄호가 많은 JSON 문자열**, prediction은 **다른 형식**(예: 자연문만, 또는 다른 JSON 포맷)일 수 있음.
   - `rouge_scorer`가 공백/구두점 기준으로 토큰을 나누면, **한국어 + JSON** 조합에서 overlap이 거의 없어 **0에 가깝게** 나올 수 있음.

2. **BERTScore**
   - 길거나 특수문자가 많은 입력에서 **NaN**이 나오고, 그걸 그대로 평균 내면 0으로 떨어지거나, 예외 처리에서 0으로 막혀 있을 수 있음.
   - Pod 환경에서 **BERTScore 모델/device** 이슈로 전부 0이 나올 수도 있음.

3. **Pod와 로컬 데이터 불일치**
   - 지금 보신 건 **로컬** `val_labeled.json`.
   - 실제 평가는 **Pod에 올라간 파일**로 돌았을 수 있음.  
     업로드/버전이 달라서 Pod에서는 `output`이 비어 있거나, 구조가 다를 수 있음.

---

## 다음에 할 확인

- **로컬에서 1개 샘플만** 같은 코드로 ROUGE/BERTScore 계산해 보기:  
  `ref = samples[0]["output"]`, `pred = "..."` (모델 출력 비슷한 문자열) 넣어서 0이 나오는지 확인.
- **Pod에서 쓴 파일**이 로컬과 동일한지:  
  eval 시 업로드한 `val_labeled.json` 내용을 Pod 로그/볼륨에서 한 번 확인.
- **ref/pred 정규화**:  
  JSON 문자열 그대로 말고, 요약만 뽑아서(예: `overall_summary.summary` 필드만) ROUGE/BERTScore에 넣으면 점수가 나오는지 비교.

지금 첨부해 주신 구조만 보면 **output은 문자열**이므로, “ref가 리스트/비어 있음”보다는 **형식/토크나이저/환경( Pod vs 로컬)** 쪽을 의심하는 게 맞습니다.

---

1) 모델/로딩/device/OOM 예외 → 0으로 fallback
코드:
_load_model_and_tokenizer, _generate_one, _rouge, _bertscore 호출부에 try/except 없음.
예외 나면 그대로 전파되고, report 쓰기 전에 스크립트가 죽음.
정리:
report까지 나왔다는 건 모델 로드 + 944/1277번 generate가 끝까지 돌았다는 뜻이라,
“로딩 실패/OOM으로 예외 났는데 0으로 fallback” 가능성은 낮음.
(그런 예외면 report.json이 안 쓰였을 가능성이 큼.)
2) NaN이 나와서 평균/후처리에서 0으로 덮음
코드:
_bertscore: float(P.mean()), float(R.mean()), float(F1.mean()) 그대로 반환 (108–110행).
NaN이 있으면 float(nan) → nan.
report[split] = { ..., **bs } 에 그대로 들어가고,
json.dump(report, ...) 시 표준 라이브러리 json은 NaN을 지원하지 않아
ValueError: Out of range float values are not JSON serializable 로 저장 단계에서 터짐.
정리:
“NaN이 생겼는데 평균/후처리에서 0으로 덮어써서 report에 0이 들어갔다”는 코드 경로는 없음.
NaN이 한 번이라도 들어가면 report 저장이 실패하는 구조라,
“길이/특수문자/이상 입력 → NaN → 0으로 덮음” 가능성은 낮음.
(다만 환경에 따라 NaN을 0으로 바꿔주는 커스텀 JSON 인코더가 있다면 그때는 가능.)
3) 실제로는 스코어가 있는데, 집계/저장 로직에서 0 초기화만 하고 업데이트가 안 됨
코드:
report[split] 은 한 번만 할당 (164–169행):
r1, r2, rl = 이미 계산된 평균, bs = _bertscore 반환값을 그대로 넣음.
“0으로 초기화해 두고, 나중에 업데이트”하는 패턴은 없음.
정리:
“집계/로그 저장 로직에서 0으로 초기화된 채 업데이트가 안 됨” 가능성은 낮음.
지금 구조는 “계산한 값으로 한 번에 dict 만든 뒤 report에 넣는” 방식이라,
스코어가 제대로 계산됐다면 그대로 report에 들어가야 함.
추가로 의심할 만한 것: import 실패로 인한 0
코드 (31–41행):
rouge_score import 실패 → ROUGE_AVAILABLE = False → 93행에서 항상 {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0} 반환.
bert_score import 실패 → BERTSCORE_AVAILABLE = False → 104행에서 not BERTSCORE_AVAILABLE 이 True가 되어 104행에서 {"bertscore_p": 0.0, ...} 반환.
정리:
Pod 이미지에 rouge_score / bert_score가 없거나 import가 실패하면,
예외 없이 전부 0으로 나오는 경로가 실제로 존재함.
이 경우 “모델/OOM/device 예외 → 0 fallback”이 아니라 의도된 fallback(미설치) 에 가깝고,
report는 정상적으로 쓰이므로 가능성 있음.
요약
가능성	결론
(1) 모델/로딩/device/OOM 예외 → 0 fallback	낮음 (예외면 report 자체가 안 나옴)
(2) NaN → 평균/후처리에서 0으로 덮음	낮음 (NaN이면 json.dump 단계에서 에러)
(3) 집계/저장 로직에서 0 초기화만 하고 미업데이트	낮음 (0 초기화 후 업데이트하는 코드 없음)
보너스 Pod에서 rouge_score/bert_score 미설치·import 실패	가능성 있음 (예외 없이 전부 0 반환하는 경로 있음)
추천 확인:
Pod(또는 eval 돌린 환경)에서
python -c "from rouge_score import rouge_scorer; from bert_score import score; print('ok')"
으로 import 가능 여부 확인해 보는 것이 좋습니다.

---


터미널에서 **로컬(env_ai)에 `bert_score`가 없어** import가 실패하는 걸 확인하셨네요.

이 상황이 **eval을 돌린 환경(Pod 포함)** 에도 그대로라면:

- `eval_distill.py` 39행 근처에서 `bert_score` import 실패 → `BERTSCORE_AVAILABLE = False`
- 104행 `if not BERTSCORE_AVAILABLE or ...` 에 걸려서  
  **bertscore_p/r/f1 이 항상 0.0으로 반환**하는 경로를 타게 됩니다.

report에서 **ROUGE까지 전부 0**이라면, Pod 이미지에 `rouge_score`도 없거나 import가 실패했을 가능성(즉 `ROUGE_AVAILABLE = False`)도 같이 의심하는 게 좋습니다.

**정리:**  
로컬에서 `bert_score`가 없다는 건, “미설치/import 실패 시 0으로 fallback”하는 코드 경로가 실제로 존재한다는 걸 보여줍니다. Pod에서 eval 돌릴 때 쓰는 이미지/환경에 `rouge_score`, `bert_score`가 둘 다 들어가 있는지, 그리고 그 환경에서 위 한 줄 import 테스트가 통과하는지 확인해 보시면 원인 좁히는 데 도움이 됩니다.

---

env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python -c "from rouge_score import rouge_scorer; from bert_score import score; print('ok')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'bert_score'

---

터미널에서 **로컬(env_ai)에 `bert_score`가 없어** import가 실패하는 걸 확인하셨네요.

이 상황이 **eval을 돌린 환경(Pod 포함)** 에도 그대로라면:

- `eval_distill.py` 39행 근처에서 `bert_score` import 실패 → `BERTSCORE_AVAILABLE = False`
- 104행 `if not BERTSCORE_AVAILABLE or ...` 에 걸려서  
  **bertscore_p/r/f1 이 항상 0.0으로 반환**하는 경로를 타게 됩니다.

report에서 **ROUGE까지 전부 0**이라면, Pod 이미지에 `rouge_score`도 없거나 import가 실패했을 가능성(즉 `ROUGE_AVAILABLE = False`)도 같이 의심하는 게 좋습니다.

**정리:**  
로컬에서 `bert_score`가 없다는 건, “미설치/import 실패 시 0으로 fallback”하는 코드 경로가 실제로 존재한다는 걸 보여줍니다. Pod에서 eval 돌릴 때 쓰는 이미지/환경에 `rouge_score`, `bert_score`가 둘 다 들어가 있는지, 그리고 그 환경에서 위 한 줄 import 테스트가 통과하는지 확인해 보시면 원인 좁히는 데 도움이 됩니다.