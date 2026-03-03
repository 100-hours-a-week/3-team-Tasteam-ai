```bash
caffeinate -i python scripts/distill_flows.py sweep_eval_merge --labeled-path distill_pipeline_output/labeled/20260226_051037/train_labeled.json --sweep-id yqpn75wu --out-dir distill_pipeline_output --num-pods 2
```

---

코드상으로는 `num_pods=2`일 때 **run_sweep_on_pod_task가 2번만** submit 되고, 각 태스크가 Pod를 **1개씩** 만들므로 이번 실행만 보면 Pod는 2개가 나와야 합니다.

그런데 **4개**가 보인다면 아래 둘 중 하나 가능성이 큽니다.

---

## 1. Prefect 태스크 재시도 (가능성 높음)

`run_sweep_on_pod_task`가 **한 번이라도 실패**하면 (예: `wait_until_stopped` 중 ConnectionError, 타임아웃 등) Prefect가 그 태스크를 **다시 실행**할 수 있습니다.

- 재시도 시 태스크는 **처음부터** 다시 돌아가서 `create_pod`를 **한 번 더** 호출합니다.
- 실패한 Pod는 그대로 두고 새 Pod를 만드므로,  
  - 태스크 2개 × (1차 실행 + 1회 재시도) = **Pod 4개**  
  이렇게 보일 수 있습니다.

즉, **“2개 만들라고 했는데 4개 보인다”**는 건  
- 이번 run에서 **2개는 정상 생성**, **2개는 재시도로 인한 추가 생성**일 가능성이 큽니다.

---

## 2. 이전 실행에서 남은 Pod 2개

예전에 같은 flow를 돌렸다가  
- `wait_until_stopped`나 `delete_pod` 전에 크래시했거나  
- 네트워크 오류로 `finally`의 `delete_pod`가 실행되지 않았다면  

그때 만든 Pod 2개가 RunPod 콘솔에 그대로 남아 있을 수 있습니다.  
그 상태에서 이번에 다시 실행하면 **이번 run으로 2개 + 예전 run 남은 2개 = 4개**로 보일 수 있습니다.

RunPod 콘솔에서 Pod 생성 시각을 보면  
- 4개가 **이번 실행 시각에 몰려 있으면** → 1번(재시도) 가능성이 크고  
- 그중 2개는 **더 이전 시각**이면 → 2번(이전 run 잔여) 가능성이 큽니다.

---

## 정리 및 대응

- **원인**:  
  - **Prefect가 `run_sweep_on_pod_task` 실패 시 재시도**해서 같은 태스크가 두 번 돌고, 그때마다 `create_pod`가 호출되거나  
  - **이전 실행에서 삭제 안 된 Pod 2개**가 남아 있거나  
  둘 중 하나(또는 둘 다)일 가능성이 큽니다.

- **재시도로 4개가 나오는 걸 막으려면**  
  Pod를 만드는 태스크는 재시도하지 않도록 하는 게 안전합니다.  
  `run_sweep_on_pod_task`에 **`retries=0`** (Prefect 2 기준)을 주면, 실패 시 재시도 없이 한 번만 실행되어 이번 run에서는 최대 2개만 생성됩니다.

- **이미 남아 있는 Pod**는 RunPod 콘솔에서 수동으로 삭제하면 됩니다.

요약하면, **“2개만 만들라고 했는데 4개 보인다”**는 건 **Prefect 재시도** 또는 **예전 run 잔여 Pod** 때문일 가능성이 크고, 재시도로 인한 중복 생성을 막으려면 해당 태스크에 `retries=0`을 두는 게 좋습니다.