
정리하면, **예전에는 Ctrl+C로 Prefect를 끄면 Pod까지 같이 종료됐는데, 지금은 Ctrl+C만으로는 Pod가 안 꺼진다**는 상황이죠.

## 왜 그렇게 바뀌었을 수 있는지

Prefect에서 **Ctrl+C(SIGINT)** 가 들어오면:

1. **예전 동작(Pod까지 종료됐을 때)**  
   - flow와 task가 **같은 프로세스·같은 스레드**에서 순서대로 실행되거나,  
   - Prefect가 종료할 때 **실행 중인 task에 취소/예외를 넣어 주고**, 그때 `try/finally`가 실행되면서 `delete_pod`가 호출됐을 가능성이 있습니다.

2. **지금 동작( Pod는 안 꺼짐)**  
   - **Prefect 3**나 최근 구조에서는 flow는 메인 스레드, task는 **다른 스레드/워커**에서 돌 수 있습니다.  
   - **멀티 Pod**에서 `run_sweep_on_pod_task.submit()`로 task를 던지면, 실제로는 그 task가 **워커 쪽**에서 `wait_until_stopped()` 안에서 대기 중일 수 있습니다.  
   - 이때 **Ctrl+C**는 보통 **메인 스레드**에만 전달되고, 메인에서 예외가 나면서 Prefect가 shutdown 하면:  
     - 워커에서 돌던 task는 **취소 통보를 받지 못하거나**,  
     - 받더라도 **프로세스가 먼저 종료**되어  
   - **그 task의 `finally`가 실행될 기회가 없을 수 있습니다.**  
   - 그래서 "Prefect만 꺼지고, Pod 삭제 코드는 실행 안 됨"이 됩니다.

즉, **Prefect 실행 모델이 바뀌었거나(스레드/워커 분리), 멀티 Pod로 task를 submit 하면서 "Ctrl+C가 걸린 쪽"과 "실제로 Pod 대기 중인 쪽"이 달라져서**, 예전처럼 같은 흐름 안에서 `finally`가 안 돌고 있는 상태로 보는 게 타당합니다.

## 대응 방향

- **원하는 동작**: "Prefect를 Ctrl+C로 종료할 때도, 그때까지 만든 Pod는 같이 삭제되게" 하는 것.
- **한계**:  
  - 지금처럼 **task 안의 `finally`에만** `delete_pod`를 두면, Ctrl+C 시 그 task가 정리될 기회를 Prefect/스레드 구조가 주지 않을 수 있음.

그래서 **Ctrl+C를 받는 쪽(프로세스/메인 스레드)** 에서 정리하도록 바꾸는 게 좋습니다.

- **프로세스 시그널 핸들러**  
  - `signal.signal(signal.SIGINT, ...)` / `SIGTERM` 를 등록해 두고,  
  - "이번 실행에서 생성한 `pod_id` 목록"을 flow나 전역 변수로 관리했다가,  
  - **SIGINT/SIGTERM 받을 때** 그 목록을 돌면서 `client.delete_pod(pod_id)` 호출.
- 이렇게 하면 **Ctrl+C로 Prefect를 종료하는 순간** 그 프로세스가 시그널을 받고, **같은 프로세스 안**에서 Pod 삭제가 실행되므로, 예전처럼 "Ctrl+C → Pod도 종료"에 가깝게 만들 수 있습니다.

정리하면, **Prefect를 Ctrl+C로 종료했을 때 Pod도 같이 종료됐다가 안 되는 이유**는, 지금 구조에서는 Ctrl+C가 task의 `finally`까지 도달하지 못하게 되어 있기 때문이고, **시그널 핸들러로 "생성한 Pod 목록 → Ctrl+C 시 일괄 delete_pod"** 를 넣으면 다시 예전처럼 동작하게 만들 수 있습니다.

---

## 적용 내용 (distill_flows.py)

- **SIGINT/SIGTERM 핸들러**: 프로세스에서 생성한 `pod_id`를 전역 리스트에 등록하고, 시그널 수신 시 해당 Pod들을 RunPod API로 일괄 삭제한 뒤 기존 핸들러(또는 `KeyboardInterrupt`) 호출.
- **추적**: Pod를 생성하는 5곳(labeling 2, train 1, sweep 1, merge 1)에서 생성 직후 `_track_pod_created(pod_id)`, `finally`에서 `delete_pod` 후 `_untrack_pod(pod_id)` 호출.
- Ctrl+C로 Prefect를 종료하면 위 핸들러가 실행되어 생성된 Pod가 정리됨.
