
두 방식의 차이를 정리하면 아래와 같습니다.

---

## 1. Subprocess로 실행

Prefect task가 **별도 프로세스**에서 `run_qlora_sweep.py` 또는 `wandb agent <sweep_id>` 를 실행하는 방식.

| 장점 | 단점 |
|------|------|
| **프로세스 격리**: 학습 OOM/크래시가 Prefect worker 프로세스를 죽이지 않음 | **한 task = sweep 전체**: N개 run이 끝날 때까지 task가 끝나지 않아, task 실행 시간이 매우 김 |
| **리소스 분리**: GPU/메모리를 subprocess가 쓰고, worker는 대기만 함 | **재시도 단위**: 실패 시 Prefect 재시도가 “sweep 전체” 단위라, 이미 끝난 run은 다시 돌지 않고 sweep 처음부터 다시 돌 수 있음 |
| **구현 단순**: `subprocess.run([sys.executable, "scripts/run_qlora_sweep.py", sweep_id], env=env)` 수준으로 붙이면 됨 | **로그**: 학습 로그가 subprocess stdout/stderr로만 오므로, Prefect 로그에 버퍼/타이밍 이슈가 생길 수 있음 |
| RunPod 등에서 “worker 컨테이너 1개 = sweep 1개 담당” 구조와 맞기 쉬움 | **진행 제어**: “sweep 중 3번째 run만 취소” 같은 세밀한 제어는 Prefect 단에서는 어렵고, wandb UI/agent 쪽에서만 가능 |

---

## 2. sweep_id를 받아서 같은 프로세스에서 에이전트 띄우기

Prefect task가 `sweep_id`를 인자로 받고, **같은 Python 프로세스**에서 `wandb.agent(sweep_id, function=train)` 를 호출하는 방식.

| 장점 | 단점 |
|------|------|
| **구현 단순**: task 내부에서 `wandb.agent(sweep_id, function=train)` 한 줄이면 됨 | **Worker 점유**: sweep이 끝날 때까지 Prefect worker가 그 프로세스에서 계속 돌아감 (run 1 → run 2 → … 순차 실행) |
| **config 주입 자연스러움**: wandb가 같은 프로세스에서 `train()`을 여러 번 호출하므로 `wandb.config` 등 사용이 직관적 | **격리 없음**: 학습 중 OOM/크래시가 **worker 프로세스까지** 죽일 수 있어, 같은 worker에서 도는 다른 flow/task도 영향 받음 |
| **로그**: `train()` 안의 print/logger가 그대로 worker stdout으로 가서 Prefect 로그에 붙기 쉬움 | **리소스**: worker가 GPU/메모리를 직접 쓰므로, “worker만 올려두고 학습은 다른 머신에서” 같은 분리가 어려움 |

---

## 요약 트레이드오프

| 기준 | Subprocess | 같은 프로세스 (sweep_id → agent) |
|------|------------|-----------------------------------|
| **Worker 안정성** | 학습 크래시가 worker와 분리됨 | 학습 크래시 시 worker까지 죽을 수 있음 |
| **Task 단위** | 한 task = sweep 전체 (길고 무거움) | 동일 (한 task = sweep 전체) |
| **재시도** | sweep 단위 재시도만 가능 | 동일 |
| **리소스** | 학습은 subprocess, worker는 대기만 | worker가 직접 학습까지 수행 |
| **로그** | subprocess 출력이라 버퍼/지연 가능 | 같은 프로세스라 로그 수집 단순 |
| **RunPod/원격** | “이 컨테이너에서 sweep 한 번 돌린다”고 두고 subprocess로 학습만 돌리기 좋음 | worker와 학습을 같은 머신에서 같이 돌릴 때만 적합 |

---

## 정리

- **안정성·격리·원격 실행**을 중시하면 → **subprocess로 `run_qlora_sweep.py`(또는 `wandb agent`) 실행**이 유리합니다. 대신 “한 번에 sweep 전체”가 하나의 길어진 task가 된다는 점은 감수하게 됩니다.
- **구현 단순함·로그 일원화**만 중요하고, worker가 죽어도 되는 환경이면 → **sweep_id를 받아서 같은 프로세스에서 `wandb.agent(sweep_id, function=train)`** 호출도 가능합니다.

실제로는 sweep이 여러 run을 순차 실행하는 구조라, **subprocess 방식**을 쓰는 편이 worker와 학습을 분리할 수 있어서 운영 측면에서 더 안전한 선택에 가깝습니다.

---

## Prefect 연동 (subprocess)

Prefect flow `run_sweep`로 sweep 에이전트를 subprocess 실행할 수 있음. `scripts/distill_flows.py`:

- **Task**: `run_sweep_agent_task(sweep_id, labeled_path, output_dir)` — `run_qlora_sweep.py`를 subprocess로 실행.
- **Flow**: `run_sweep_flow(sweep_id, labeled_path, output_dir)`.
- **CLI**: `python scripts/distill_flows.py run_sweep --sweep-id <id> --labeled-path <path> [--out-dir <dir>]`.

환경변수 `WANDB_SWEEP_LABELED_PATH`, `WANDB_SWEEP_OUTPUT_DIR`는 task에서 subprocess env로 전달됨. `WANDB_API_KEY`, `WANDB_PROJECT`는 호스트 환경이 그대로 상속됨.