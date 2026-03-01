# sweep을 따로 안 만들고 run_sweep만 실행 (내부에서 sweep 등록 후 에이전트 실행)
python scripts/distill_flows.py run_sweep --labeled-path .../train_labeled.json

# 기존처럼 이미 만든 sweep id로 실행
python scripts/distill_flows.py run_sweep --sweep-id entity/project/xxxx --labeled-path .../train_labeled.json

# all_sweep도 sweep-id 없이 실행 가능 (build_dataset → labeling(Pod) → sweep 등록 → run_sweep → evaluate)
python scripts/distill_flows.py all_sweep

**1번·2번은 3번과 목적이 달라서 그대로 두는 게 좋습니다.**

- **3번 (all_sweep)**  
  - **데이터가 아직 없을 때** 쓰는 경로입니다.  
  - build_dataset → labeling(Pod) → sweep → evaluate 를 **처음부터 한 번에** 돌립니다.  
  - 이미 있는 `train_labeled.json`을 쓰는 게 아니라, flow가 만든 데이터를 씁니다.

- **1번 (run_sweep, sweep-id 없음)**  
  - **이미 `train_labeled.json`이 있을 때** “라벨링/빌드는 건너뛰고 sweep만 돌리고 싶을 때” 필요합니다.  
  - 예: 예전에 labeling 돌린 결과, 다른 스크립트로 만든 labeled 데이터를 그대로 쓰고 싶을 때.  
  - 3번을 쓰면 build_dataset + labeling을 또 돌리게 되므로, 1번이 있으면 불필요한 재실행을 피할 수 있습니다.

- **2번 (run_sweep, sweep-id 있음)**  
  - **이미 만든 sweep을 쓰고 싶을 때** 필요합니다.  
  - 예: 터미널에서 `wandb sweep ...` 으로 sweep을 만들어 둔 경우, 같은 sweep을 여러 에이전트에서 돌리는 경우, 또는 sweep 설정만 따로 수정해 둔 경우.  
  - 이런 경우엔 “sweep 등록” 단계를 건너뛰고, 기존 sweep id로 에이전트만 실행하는 2번이 맞습니다.

정리하면:

- **데이터 없음 → 처음부터 한 번에** → 3번 (all_sweep)  
- **labeled 데이터만 이미 있음 → sweep만 돌리기** → 1번  
- **sweep까지 이미 있음 → 에이전트만 돌리기** → 2번  

그래서 3번이 있어도 1번·2번은 각각 다른 사용 시나리오를 위해 유지하는 게 좋습니다.