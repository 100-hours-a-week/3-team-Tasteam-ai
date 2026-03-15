
원인은 **다운로드 위치**와 **report.json 선택 방식**이 겹치면서 생긴 버전 혼동입니다.

---

## 무슨 일이 있었는지

1. **실행**: `download_eval_from_volume --eval-version 20260311_135015` (볼륨의 **20260311_135015** 버전 받기)
2. **결과**: Flow는 성공했지만, 반환된 경로와 아티팩트 버전은 **20260303_053420**입니다.
   - `report_path`: `.../eval/20260303_053420/report.json`
   - `artifact version`: `20260303_053420`

즉, **요청한 20260311_135015가 아니라, 예전에 있던 20260303_053420 쪽이 선택**된 상태입니다.

---

## 원인

1. **`output_dir` 기본값**  
   `main()`에서 `out_dir = args.out_dir or .../distill_pipeline_output` 이고, `download_eval_from_volume_flow`에는 `output_dir=out_dir`이 그대로 넘어갑니다.  
   그래서 다운로드 루트가 **`distill_pipeline_output`** 이 됩니다.

2. **다운로드 경로**  
   `download_directory_from_runpod(vol_id, "distill_pipeline_output/eval_output/20260311_135015/", out_dir)` 에서  
   prefix 아래 키들이 `report.json`, `llm_as_a_judge_results.json` 등 **이름만** 남도록 상대 경로가 잡혀서,  
   실제로 받는 파일 위치는  
   `distill_pipeline_output/report.json`,  
   `distill_pipeline_output/llm_as_a_judge_results.json`  
   처럼 **distill_pipeline_output 바로 아래**에 저장됩니다.

3. **report.json 선택**  
   그 다음에  
   `report_path = next(Path(out_dir).rglob("report.json"), None)`  
   로 **`out_dir`(= distill_pipeline_output) 아래의 “아무” report.json**을 하나 고릅니다.  
   이때 `distill_pipeline_output/eval/20260303_053420/report.json` 이 이미 있으면, rglob이 **그쪽을 먼저 반환**할 수 있어서,  
   `eval_dir = report_path.parent` → **eval/20260303_053420** 이 됩니다.

4. **이후 처리**  
   LLM-as-a-Judge, kd_sft_analysis, 아티팩트 업로드는 모두 이 **eval_dir(20260303_053420)** 기준으로 돌기 때문에,  
   방금 받은 20260311_135015가 아니라 **20260303_053420** 결과가 다시 업로드된 것입니다.

정리하면:

- **요청**: 볼륨에서 20260311_135015 받아서 judge → kd_sft → 아티팩트
- **실제**: 20260311_135015는 `distill_pipeline_output/` 직하에 받았는데,  
  `rglob("report.json")`이 기존 `eval/20260303_053420/report.json`을 골라서,  
  그 디렉터리 기준으로 플로우가 돌아가 버린 상황입니다.

---

## 어떻게 고치면 좋은지 (구조 개선 제안)

- **다운로드 디렉터리를 버전별로 분리**  
  예: `output_dir`을 `distill_pipeline_output/eval_from_pod` 로 두고,  
  그 안에 **버전 서브디렉터리**를 두어  
  `distill_pipeline_output/eval_from_pod/20260311_135015/report.json` 형태로만 받도록 하거나,
- **report.json을 “방금 받은 디렉터리”에서만 찾기**  
  예: 다운로드 시 버전 폴더를 하나 만들고, 그 폴더만 대상으로 `report.json`을 찾도록 하면,  
  기존 `eval/20260303_053420` 같은 디렉터리가 섞이지 않습니다.

지금 터미널에서 “무슨 일이 발생했지?”에 대한 답은:  
**요청한 eval 버전(20260311_135015)으로 받은 뒤 처리한 것이 아니라, 기존에 있던 다른 버전(20260303_053420)의 report.json이 선택되어, 그 버전 기준으로 judge·kd_sft·아티팩트가 실행·업로드된 것**입니다.