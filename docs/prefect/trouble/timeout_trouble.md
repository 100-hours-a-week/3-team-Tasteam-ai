
타임아웃이 걸린 이유와 현재 값은 아래와 같습니다.

---

## 왜 타임아웃이 있나

- **목적**: Pod가 **끝날 때까지** 기다리는 구간에서, Prefect task가 **무한 대기**하지 않도록 하는 **상한선**입니다.
- Pod가 죽지 않고 계속 RUNNING이거나, sweep agent가 멈춰 있어도 flow가 영원히 블로킹되지 않게 하려고 넣은 값입니다.

---

## 현재 설정

- **위치**: `run_sweep_on_pod_task` → `client.wait_until_stopped(..., timeout_sec=sweep_timeout_sec)`
- **기본값**: `sweep_timeout_sec=14400` → **4시간**

```775:776:scripts/distill_flows.py
    sweep_timeout_sec: int = 14400,
    sweep_poll_interval_sec: int = 60,
```

```833:836:scripts/distill_flows.py
        client.wait_until_stopped(
            pod_id,
            timeout_sec=sweep_timeout_sec,
            poll_interval_sec=sweep_poll_interval_sec,
```

---

## 4시간이 부족한 이유

- sweep에 **run_cap=50**이고 run당 20~40분이면, 이론상 **16~33시간** 정도 걸릴 수 있습니다.
- 그래서 4시간이 지나면 “학습이 아직 진행 중인데” 타임아웃으로 flow가 실패한 상황입니다.

---

## 어떻게 바꾸면 좋은지

- **타임아웃을 늘리기**: 예를 들어 `sweep_timeout_sec=86400`(24시간) 또는 43200(12시간)으로 올리기.
- **CLI에서 지정 가능하게**: `sweep_eval_merge` / `run_sweep` 호출 시 `--sweep-timeout-sec` 같은 인자를 받아서 `run_sweep_on_pod_task`에 넘기기.

정리하면, “학습을 짧게 자르려고” 걸어둔 게 아니라 **무한 대기 방지용**인데, 기본 4시간이 현재 sweep 규모에는 짧아서 타임아웃이 난 것이고, 위처럼 값을 늘리거나 인자로 조정하면 됩니다.