로그 원인

---

로그를 보면 **`/app/scripts/run_qlora_sweep.py`는 이제 존재**하고(즉 “no such file” 문제는 해결됨), 현재 실패 원인은 **wandb 쪽 `project not found (404)`** 입니다.

### 원인
`run_qlora_sweep.py`가 아래처럼 wandb agent를 실행합니다(로그 L23–57에서 동일 경로/라인이 찍힘).

- `wandb.agent(..., project=os.environ.get("WANDB_PROJECT", "tasteam-distill"))`

즉 Pod 내부에서 `WANDB_PROJECT`가 없으면 **기본 프로젝트를 `tasteam-distill`로 가정**하는데,
- 그 프로젝트가 **해당 entity(계정/조직)에 존재하지 않거나**
- API 키가 그 프로젝트에 **권한이 없거나**
- sweep이 만들어진 프로젝트와 `WANDB_PROJECT`가 **불일치**하면  
wandb가 GraphQL에서 `project not found`를 내고 에이전트 등록이 실패합니다.

### 바로 할 수 있는 해결
- **W&B에 `tasteam-distill` 프로젝트를 실제로 생성**(해당 계정/조직 아래)하거나,
- **Pod에 `WANDB_PROJECT`(필요하면 `WANDB_ENTITY`도)를 정확히 주입**해서 sweep이 있는 프로젝트로 맞춰야 합니다.

### 체크 포인트 (가장 빠른 확인)
- **sweep id 형식 확인**: 보통 `entity/project/xxxx` 형태입니다. 여기서 `project`가 실제 존재하는지 확인.
- **Pod에 프로젝트가 안 들어가는 구조인지 확인**: 지금 코드의 Pod env에는 `WANDB_API_KEY`는 들어가지만 `WANDB_PROJECT`는 기본적으로 안 들어가서, 기본값(`tasteam-distill`)로 떨어질 가능성이 큽니다.

### 권장 수정(코드 레벨, 다음 액션)
Ask 모드라 제가 직접 수정은 못 하지만, 안정적으로 하려면 둘 중 하나로 바꾸는 걸 추천합니다.

- **옵션 A**: `run_sweep_on_pod_task`에서 payload env에 `WANDB_PROJECT`(및 필요 시 `WANDB_ENTITY`)를 명시적으로 넣기
- **옵션 B**: `run_qlora_sweep.py`에서 `wandb.agent(..., project=...)`를 강제하지 말고(또는 sweep_id에서 추론), sweep이 생성된 프로젝트를 그대로 쓰게 하기

원하시면 지금 사용 중인 **sweep id(문자열)** 와 W&B의 **entity / project 이름**을 알려주시면, 어떤 값으로 맞춰야 404가 사라지는지 딱 맞게 정리해드릴게요.

---

코드로 생성하는 법

---

가능합니다. W&B 문서에 따르면 **존재하지 않는 프로젝트 이름으로 `wandb.init(project="...")`를 호출하면 해당 프로젝트가 자동으로 생성**됩니다.

### 코드에서 할 수 있는 것

1. **프로젝트 확보 전용 task 추가**  
   sweep 등록 전에 한 번만 실행하면 됩니다.

   ```python
   @task(name="ensure-wandb-project-task", log_prints=True)
   def ensure_wandb_project_task(
       project: str = "tasteam-distill",
       entity: str | None = None,
   ) -> None:
       """프로젝트가 없으면 wandb.init으로 생성 (sweep/agent 404 방지)."""
       import wandb
       wandb.init(project=project, entity=entity)
       # run 종료해서 대시보드에 run 하나 안 남기려면:
       # wandb.finish()
   ```

2. **sweep 등록 전에 호출**  
   `register_sweep_task`를 부르는 모든 경로에서, 그 **앞**에 이 task를 넣습니다.

   - `run_sweep_flow`: `sweep_id`가 없을 때 `ensure_wandb_project_task` → `register_sweep_task`
   - `run_sweep_and_evaluate_flow`: 동일
   - `sweep_eval_merge_flow`: 동일
   - `distill_pipeline_all_sweep`: 동일

3. **sweep과 동일한 entity/project 사용**  
   `ensure_wandb_project_task`와 `wandb sweep`이 같은 프로젝트를 쓰려면, sweep 실행 시점의 환경과 맞춰야 합니다.

   - `wandb sweep`은 보통 `WANDB_PROJECT`(와 필요 시 `WANDB_ENTITY`)를 읽습니다.
   - `register_sweep_task`를 돌리는 subprocess에 `WANDB_PROJECT`(및 `WANDB_ENTITY`)를 설정해 두고,
   - `ensure_wandb_project_task`에도 같은 `project`(와 `entity`)를 넘기면, “코드에서 tasteam-distill 생성”과 “그 프로젝트에 sweep 등록”이 일치합니다.

4. **Pod 쪽과의 일치**  
   Pod에서 `wandb.agent(sweep_id, project=...)`를 부를 때 쓰는 프로젝트가, sweep이 등록된 그 프로젝트와 같아야 하므로,  
   Pod payload의 `env`에 `WANDB_PROJECT`(와 필요 시 `WANDB_ENTITY`)를 넣어 두는 것은 그대로 두는 것이 좋습니다.  
   (이미 sweep_id에 entity/project가 포함되어 있지만, agent 등록 시 project가 없으면 404가 나는 경우가 있어서, Pod에도 동일 프로젝트를 명시하는 편이 안전합니다.)

### 정리

- **tasteam-distill 생성** → 코드에서 `ensure_wandb_project_task(project="tasteam-distill")` 같은 task를 한 번 실행해 두고,
- 그 다음에만 `register_sweep_task`를 호출하도록 하면, “프로젝트가 없어서 404”는 코드만으로 해결할 수 있습니다.
- entity를 쓰는 경우에는 `ensure_wandb_project_task`와 sweep/agent 실행 시 모두 같은 `entity`를 쓰도록 맞추면 됩니다.

원하시면 Agent 모드로 전환해 주시면, `ensure_wandb_project_task` 추가와 위 flow들에 넣는 위치까지 구체적으로 수정해 드리겠습니다.