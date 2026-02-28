
이 로그는 **Pod 안에서** `run_qlora_sweep.py`가 `wandb.agent(...)`를 호출할 때, W&B API가 **project not found (404)** 를 반복해서 내는 상황을 보여줍니다.

### 로그에서 보이는 것
- 매번: `wandb.login()` 성공 → `wandb.agent(..., project=os.environ.get("WANDB_PROJECT", "tasteam-distill"))` → **register_agent** 단계에서 404 → `UsageError: project not found`
- 약 06:51 ~ 06:58 사이에 같은 에러가 반복 (재시도/재시작으로 보임).

### 가능한 원인
1. **이 로그가 수정 전 실행 결과인 경우**  
   `ensure_wandb_project_task`와 Pod env에 `WANDB_PROJECT` 넣는 변경이 반영되기 **전**에 돌린 run이면,  
   - 프로젝트가 한 번도 생성되지 않았거나  
   - Pod에 `WANDB_PROJECT`가 안 넘어가서 기본값/다른 프로젝트를 쓰다 404가 날 수 있습니다.

2. **sweep이 다른 entity/project에 있는 경우**  
   sweep id가 `entity_a/project_x/...` 인데,  
   - `ensure_wandb_project_task`는 `entity_b` 또는 다른 project로 프로젝트를 만들었거나  
   - `wandb sweep`이 `WANDB_PROJECT`를 안 읽고 다른 프로젝트에 sweep을 등록했을 수 있습니다.  
   그러면 Pod는 존재하지 않는 entity/project 조합으로 접속하게 되어 404가 납니다.

3. **Pod에 WANDB_PROJECT가 비어 있는 경우**  
   `run_sweep_on_pod_task`에서 `sweep_id`를 `entity/project/sweep_run_id`로 파싱해 `WANDB_PROJECT`를 넣는데,  
   - sweep_id 형식이 다르거나  
   - payload가 덮어쓰이거나 해서 Pod에 `WANDB_PROJECT`가 비어 있으면, `run_qlora_sweep.py`는 `"tasteam-distill"`을 쓰게 됩니다.  
   그때 그 프로젝트가 해당 entity에 없으면 404가 납니다.

### 다음에 확인할 것
- **최신 코드로 다시 실행**  
  `ensure_wandb_project_task` + Pod env에 `WANDB_PROJECT`(및 필요 시 `WANDB_ENTITY`)를 넣는 수정이 들어간 상태에서 sweep_eval_merge / all_sweep 를 다시 돌려보세요.
- **실제 sweep_id 확인**  
  Prefect 로그에서 `register_sweep_task` 이후 찍히는 `sweep_id`(예: `entity/project/sweep_run_id`)를 확인하고,  
  wandb UI에서 그 **entity** 아래에 **project**가 실제로 존재하는지 봅니다.
- **한 번만 프로젝트 생성**  
  로컬에서 `WANDB_API_KEY`를 넣고  
  `python -c "import wandb; wandb.init(project='tasteam-distill'); wandb.finish()"`  
  를 실행해, wandb 웹에서 해당 계정/팀에 `tasteam-distill` 프로젝트가 생겼는지 확인합니다.

정리하면, 이 로그는 “Pod에서 sweep에 붙을 때 쓰는 project가 W&B에 없다”는 뜻이므로, **최신 코드로 재실행**하고, **sweep_id에 들어 있는 entity/project와 실제 생성된 프로젝트가 일치하는지** 확인하면 원인을 좁힐 수 있습니다.

---


이 터미널 로그만 보면 **로컬에서 하는 부분은 정상 동작**한 상태입니다.

### 로그 요약
- **로그인**: `jin-soo (jin-soo-none)` / `https://api.wandb.ai`
- **ensure_wandb_project_task**: 완료 → 프로젝트 `tasteam-distill`이 **jin-soo-none** entity 아래에 생성/노출됨  
  - https://wandb.ai/jin-soo-none/tasteam-distill
- **register_sweep_task**: 완료 → sweep이 같은 entity/project에 등록됨

즉, sweep_id는 `jin-soo-none/tasteam-distill/<sweep_run_id>` 형태일 것이고, `run_sweep_on_pod_task`에서 이걸 파싱해 Pod env에 `WANDB_ENTITY=jin-soo-none`, `WANDB_PROJECT=tasteam-distill`을 넣어주면, Pod에서도 같은 프로젝트로 agent가 붙어야 합니다.

### 다음 단계
- 이어서 **run-sweep-on-pod-task**가 실행되면 Pod가 뜨고, 그 Pod에 동일한 `WANDB_API_KEY`와 위 env가 넘어가는지 확인하면 됩니다.
- 이전에 보이던 “project not found”는 **jin-soo-none/tasteam-distill**가 없을 때 발생한 것이므로, 지금처럼 ensure로 프로젝트가 생겼고 sweep도 같은 entity/project에 등록됐다면, Pod에서도 404가 나지 않을 가능성이 높습니다.

추가로, Pod 로그에서 다시 “project not found”가 나오면 그 시점의 sweep_id와 Pod env(`WANDB_ENTITY`, `WANDB_PROJECT`)를 같이 보면 원인 파악이 쉽습니다.

---


Pod env는 **두 군데**에서 정해집니다.

---

### 1. 기본 env (모든 Pod 공통)

**`scripts/runpod_cli/pod_create_delete_cli.py`** 174행:

```174:174:scripts/runpod_cli/pod_create_delete_cli.py
            "env": {"ENV_VAR": "value",**({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),},
```

- `RunPodClient.get_default_pod_payload(use="train"|"labeling"|"merge")` 할 때 들어가는 기본값.
- 항상: `ENV_VAR=value`
- 로컬에 `WANDB_API_KEY`가 있으면: `WANDB_API_KEY`도 포함.

---

### 2. Sweep Pod에서 덮어쓰는 env

**`scripts/distill_flows.py`** 657–674행, `run_sweep_on_pod_task` 안:

```657:674:scripts/distill_flows.py
    # sweep_id = entity/project/sweep_run_id → Pod에 동일 project/entity 전달해 404 방지
    parts = sweep_id.split("/")
    wandb_project = parts[1] if len(parts) >= 2 else DEFAULT_WANDB_PROJECT
    wandb_entity = parts[0] if len(parts) >= 2 else os.environ.get("WANDB_ENTITY", "")
    payload = RunPodClient.get_default_pod_payload(use="train", docker_start_cmd=[sweep_id])
    ...
    base_env = payload.get("env") or {}
    payload["env"] = {
        **base_env,
        "WANDB_SWEEP_LABELED_PATH": path_on_volume,
        "WANDB_SWEEP_OUTPUT_DIR": "/workspace/distill_pipeline_output",
        "WANDB_SWEEP_ID": sweep_id,
        "WANDB_PROJECT": wandb_project,
    }
    if wandb_entity:
        payload["env"]["WANDB_ENTITY"] = wandb_entity
```

- `base_env` = 위 1번 기본 env.
- 그 위에 sweep용으로 **추가/덮어쓰기**:  
  `WANDB_SWEEP_LABELED_PATH`, `WANDB_SWEEP_OUTPUT_DIR`, `WANDB_SWEEP_ID`, `WANDB_PROJECT`,  
  그리고 `wandb_entity`가 있으면 `WANDB_ENTITY`.

---

### 정리

| 설정 위치 | 파일 | 내용 |
|-----------|------|------|
| 기본 Pod env | `runpod_cli/pod_create_delete_cli.py` 174행 | `ENV_VAR`, (선택) `WANDB_API_KEY` |
| Sweep Pod env | `distill_flows.py` 665–674행 | `base_env` + `WANDB_SWEEP_*`, `WANDB_PROJECT`, (선택) `WANDB_ENTITY` |

그래서 **Pod env는**  
1) 공통은 `pod_create_delete_cli.py`의 `get_default_pod_payload`의 `"env"`,  
2) sweep 전용은 `distill_flows.py`의 `run_sweep_on_pod_task`에서 `payload["env"]`를 위처럼 갱신하는 곳에서 설정됩니다.

---

pod env

WANDB_SWEEP_LABELED_PATH=/workspace/labeled/20260226_051037/train_labeled.json
WANDB_SWEEP_OUTPUT_DIR=/workspace/distill_pipeline_output
WANDB_SWEEP_ID=Users/js/tasteam
WANDB_PROJECT=js
WANDB_ENTITY=Users
ENV_VAR=value
WANDB_API_KEY=""
PUBLIC_KEY=""

---


원인: **sweep_id가 wandb sweep id가 아니라 로컬 경로처럼 넘어가고 있습니다.**

지금 Pod에 들어간 값:
- `WANDB_SWEEP_ID=Users/js/tasteam` → **경로** (예: `/Users/js/tasteam...` 의 일부)
- `WANDB_PROJECT=js` → `sweep_id.split("/")[1]`
- `WANDB_ENTITY=Users` → `sweep_id.split("/")[0]`

즉 `sweep_id`가 `"Users/js/tasteam"` 같은 문자열로 전달되면서, `entity/project/sweep_run_id` 대신 **경로를 슬래시로 나눈 결과**가 entity/project로 쓰이고 있습니다.

---

### 원인 후보

1. **`register_sweep_task` 파싱**
   - `wandb sweep` 출력에서 sweep id를 정규식으로 뽑을 때,  
     `(\S+/\S+/\S+)` 가 **sweep id가 아닌 다른 슬래시 구분 문자열**(예: 경로, URL 일부)에 매칭된 경우.
2. **다른 환경에서의 wandb 출력**
   - Prefect worker 등 다른 머신에서 실행될 때, wandb가 sweep id 대신 경로나 다른 문구를 출력하고, 그게 그대로 파싱된 경우.

그래서 **sweep_id가 실제로는 “entity/project/sweep_run_id”가 아니라 “Users/js/tasteam”으로 들어가고 있고**, 그걸 그대로 `split("/")` 해서 Pod env가 잘못 설정되는 상황입니다.

---

### 수정 제안 (코드 레벨)

1. **sweep_id 검증**
   - `run_sweep_on_pod_task`에서 `sweep_id`를 받은 뒤,  
     경로처럼 보이면 파싱해서 쓰지 않고, **기본값만** 쓰기:
   - 예: `sweep_id`에 `"Users"`가 들어가거나 `/`로 나눈 첫 부분이 경로 스타일(예: `Users`, `home` 등)이면  
     `wandb_project = DEFAULT_WANDB_PROJECT`, `wandb_entity = os.environ.get("WANDB_ENTITY", "")` 로 고정.
2. **register_sweep_task 파싱 강화**
   - sweep id 패턴만 허용하도록 정규식을 좁히기:  
     예를 들어 entity/project가 **wandb 스타일**(영문, 숫자, 하이픈 등)일 때만 매칭하고,  
     `Users`, `home` 같은 경로 세그먼트는 제외.

이렇게 하면 “Users/js/tasteam”이 들어와도 Pod env가 `WANDB_PROJECT=tasteam-distill`, `WANDB_ENTITY=jin-soo-none`처럼 올바르게 설정되고, sweep_id가 진짜 wandb id일 때만 파싱해서 쓸 수 있습니다.

지금은 **Ask 모드**라 제가 직접 수정은 못 하니, Agent 모드로 바꿔 주시면 위 검증/파싱 로직을 코드에 반영해 드릴 수 있습니다.

---

WANDB_SWEEP_LABELED_PATH=/workspace/labeled/20260226_051037/train_labeled.json
WANDB_SWEEP_OUTPUT_DIR=/workspace/distill_pipeline_output
WANDB_SWEEP_ID=async/scripts/wandb
WANDB_PROJECT=scripts
WANDB_ENTITY=async

---


반영했습니다.

**원인**  
`sweep_id`가 `async/scripts/wandb`처럼 **경로 조각**(`tasteam-new-async/scripts/wandb`에서 잘린 형태)으로 파싱되어, `WANDB_ENTITY=async`, `WANDB_PROJECT=scripts`로 잘못 들어가고 있었습니다.

**수정**  
`entity/project/run_id` 세 부분 중 **project나 run_id가 흔한 폴더/도구 이름**이면 sweep id 후보에서 제외하도록 했습니다.

- **제외 세그먼트**: `scripts`, `wandb`, `src`, `lib`, `train`, `output`, `workspace`
- 따라서 `async/scripts/wandb`는 `scripts`, `wandb` 때문에 path-like로 판단되어 **사용하지 않고**, 그대로 두면 “Could not parse wandb sweep id ...” 예외가 나서 잘못된 값이 Pod로 넘어가지 않습니다.

실제 sweep id는 wandb가 출력하는 `entity/project/알파벳숫자_runid` 형태만 쓰이게 됩니다.


정리해서 말씀드리면 다음과 같습니다.

**1. `wandb/` 디렉터리에서 파싱하는지**

- 공식 문서에는 **“sweep id를 `wandb/` 아래 파일에 쓴다”**는 내용이 없습니다.
- `wandb sweep config.yaml`은 **sweep id를 터미널에 출력(print)**하는 방식만 설명되어 있고, 그 id를 로컬 파일로 저장한다는 말은 없습니다.
- `wandb/` 디렉터리는 보통 **run 로그·메타데이터**용이고, `.gitignore`에 `wandb/`가 있는 것도 그런 로그를 제외하기 위함입니다.
- 그래서 “sweep id를 `wandb/` 디렉터리에서 파싱해야 한다”는 동작은 **문서상/일반적인 사용법으로는 지원되지 않는 방식**에 가깝습니다.

**2. 더 안정적인 방법: wandb SDK에서 직접 받기**

sweep id를 **출력 문자열에서 파싱하지 않고** 쓰려면, **CLI subprocess 대신 Python SDK**를 쓰는 편이 좋습니다.

- 문서: [Initialize a sweep](https://docs.wandb.ai/guides/sweeps/initialize-sweeps/)
- 예시:
  ```python
  import wandb
  sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
  ```
- `wandb.sweep()`는 **sweep id를 반환값**으로 주기 때문에, `entity/project/scripts/wandb` 같은 경로 조각을 정규식으로 잘못 잡는 문제를 아예 피할 수 있습니다.

지금 코드는 **CLI**를 쓰고 있습니다:

```576:582:scripts/distill_flows.py
    """wandb sweep <yaml> 를 subprocess로 실행하고 stdout에서 sweep id를 파싱해 반환."""
    ...
    cmd = ["wandb", "sweep", str(path)]
```

**요약**

- **`wandb/`에서 파싱**: 공식 문서상 그런 방식은 없고, sweep id는 CLI가 **stdout에만** 뿌립니다.
- **권장**: stdout 파싱 대신 **`wandb.sweep(sweep=..., project=...)`를 Python에서 호출해 반환값을 `sweep_id`로 쓰는 방식**이 더 안정적입니다.  
  (원하시면 Agent 모드에서 `register_sweep_task`를 subprocess+파싱 대신 `wandb.sweep()` 호출로 바꾸는 예시까지 구체적으로 적어 드리겠습니다.)

  ---


반영했습니다.

**변경 요약**

1. **`register_sweep_task`**
   - **이전**: `wandb sweep <yaml>` subprocess 실행 후 stdout에서 sweep id 정규식 파싱.
   - **이후**: YAML을 `yaml.safe_load`로 읽어 `wandb.sweep(sweep=config, project=..., entity=...)`를 호출하고, **반환값**을 그대로 sweep id로 사용.

2. **제거**
   - subprocess 호출 및 stdout/stderr 처리.
   - 경로/폴더명 제외용 정규식·`_is_path_like` 등 파싱 로직.
   - 사용처가 없어진 `import re`.

3. **추가**
   - `import yaml` (YAML 설정 로드용).
   - task 내부에서 `import wandb` (SDK 호출용).

4. **project/entity**
   - `WANDB_PROJECT`가 있으면 사용, 없으면 `DEFAULT_WANDB_PROJECT` (`tasteam-distill`).
   - `WANDB_ENTITY`가 있으면 entity로 전달, 없으면 `None` (wandb 기본 계정).

이제 sweep id는 항상 wandb가 준 문자열만 쓰이므로, `async/scripts/wandb` 같은 경로 조각이 들어가는 일은 없습니다.