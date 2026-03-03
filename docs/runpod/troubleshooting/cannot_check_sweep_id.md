
원인: wandb API는 sweep을 **전체 경로**로만 조회합니다.

- `api.sweep("yqpn75wu")` → **Invalid path: 'yqpn75wu' (missing project)**
- `api.sweep("entity/project/sweep_id")` → 정상 (예: `jin-soo-none/tasteam-distill/yqpn75wu`)

`--sweep-id yqpn75wu`처럼 **sweep run id만** 넘기면 `check_sweep_complete_task`와 그 뒤의 `get_best_adapter_from_artifact_task`에서 같은 오류가 납니다.

## 해결 방법

### 1) 실행 시 전체 경로 넘기기 (당장 쓰기)

```bash
python scripts/distill_flows.py sweep_eval_merge \
  --labeled-path distill_pipeline_output/labeled/20260226_051037/train_labeled.json \
  --sweep-id "jin-soo-none/tasteam-distill/yqpn75wu" \
  --out-dir distill_pipeline_output --num-pods 2
```

wandb sweep URL이  
`https://wandb.ai/jin-soo-none/tasteam-distill/sweeps/yqpn75wu`  
이면 `--sweep-id`에는 **entity/project/sweep_id** 형식으로  
`jin-soo-none/tasteam-distill/yqpn75wu` 를 넣으면 됩니다.

### 2) 코드에서 short id 보정 (근본 처리)

`sweep_id`가 슬래시가 없으면(short id만 있으면) `WANDB_ENTITY`와 `WANDB_PROJECT`로 앞에 붙여서 전체 경로로 만들어 쓰면 됩니다.

- `check_sweep_complete_task` 진입 시, 그리고
- `get_best_adapter_from_artifact_task`에서 `api.sweep(...)` 호출 전에

같은 규칙으로 보정하면 됩니다.  
예: `sweep_id`에 `/`가 없으면 `entity = os.environ.get("WANDB_ENTITY","")`, `project = os.environ.get("WANDB_PROJECT", "tasteam-distill")` 로 `f"{entity}/{project}/{sweep_id}"` 를 만들어 사용. (이미 `run_sweep_on_pod_task`에서 entity/project 파싱하는 부분이 있으면, 그와 맞춰 한 곳에서 정규화하는 것도 좋습니다.)

지금 로그 기준으로는 **1)처럼 전체 경로로 `--sweep-id`를 넘기면** 바로 동작하고, **2)를 적용하면 short id만 넘겨도 동작**하게 할 수 있습니다.