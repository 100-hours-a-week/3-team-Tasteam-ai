
세 가지 방식의 장단점을 정리하면 아래와 같습니다.

---

## 1. 환경변수 (예: `LLM_MODEL`, `LLM_ADAPTER_PATH` 등)

**장점**
- 이미 `LLM_MODEL` 등으로 쓰는 패턴과 맞음. 코드 변경 최소화.
- 배포/컨테이너에서 **실행 시점에만** 바꿔서 다른 모델·adapter로 전환 가능.
- 시크릿/경로를 코드에 안 넣어도 됨.
- CI/스테이징/프로덕션마다 다른 값을 쉽게 줄 수 있음.

**단점**
- 값이 흩어져 있어서 “지금 어떤 모델 쓰는지” 한눈에 보기 어렵다.
- 실수로 빈 값/잘못된 경로를 넣으면 런타임에 에러.
- adapter만 줄 경우, base 모델 경로도 따로 관리해야 할 수 있음 (예: `LLM_BASE_MODEL`, `LLM_ADAPTER_PATH`).

**적합한 경우**: 배포/실행 환경마다 모델을 바꾸고 싶을 때, DevOps/컨테이너 중심 운영.

---

## 2. 설정 파일 (예: `config.yaml`, `.env` 파일, 전용 `model_config.json`)

**장점**
- **한 파일**에 base 경로, adapter 경로, 옵션(quantization 등)을 모아서 관리 가능.
- 버전 관리(git)에 넣으면 “어떤 설정으로 서비스했는지” 추적 가능.
- 환경변수보다 구조화하기 좋고, 여러 adapter/모델 세트를 정의하기 편함.
- 문서화·온보딩 시 “이 파일만 보면 된다”로 안내하기 좋음.

**단점**
- 파일 경로/이름을 코드나 진입점에서 고정해야 함.
- 설정 파일을 배포물에 포함해야 하므로, 빌드/배포 단계가 한 번 더 생김.
- 실서비스에서 설정만 바꿔서 모델 전환할 때는, 배포 파이프라인 또는 설정 로딩 방식이 필요.

**적합한 경우**: 팀 단위로 설정을 공유하고, “이번 릴리스는 이 adapter”처럼 버전과 함께 관리하고 싶을 때.

---

## 3. Merge 후 경로 (adapter + base → 단일 디렉터리로 merge해서 그 경로 사용)

**장점**
- **서빙 측은 “풀 모델 하나”만 보면 됨.**  
  API/엔진이 adapter 지원이 없어도, merge된 디렉터리만 `LLM_MODEL` 등으로 지정하면 됨.
- vLLM 등 “풀 모델 경로만 받는” 스택과 호환 좋음.
- 추론 시 PEFT 로딩 분기 없이 단순해짐.
- 디스크만 충분하면, 같은 base에 여러 adapter를 각각 merge해 두고 경로만 바꿔 쓰기 가능.

**단점**
- Merge 단계(스크립트/flow)가 필요하고, **디스크 사용량**이 늘어남 (base + adapter 크기).
- Merge에 시간이 조금 걸림 (한 번만 하면 되지만, 자동화·배포 파이프라인에 단계 추가).
- “같은 base, adapter만 교체”가 필요할 때마다 merge를 다시 하거나, merge된 디렉터리를 여러 개 유지해야 함.

**적합한 경우**: vLLM 등 merge된 모델 경로만 받는 서빙을 쓰거나, “학습 → merge → 그 경로를 API가 사용”까지를 한 번에 자동화하고 싶을 때.

---

## 한 줄 비교

| 방식           | 유연성(런타임 전환) | 구조화/추적 | 서빙 스택 호환     | 구현/운영 복잡도 |
|----------------|---------------------|------------|---------------------|------------------|
| 환경변수       | 높음                | 낮음       | 현재 코드와 잘 맞음 | 낮음             |
| 설정 파일      | 중간                | 높음       | 코드에서 파일 참조  | 중간             |
| Merge 후 경로  | 중간(경로만 변경)   | 중간       | vLLM 등에 유리      | Merge 파이프라인 필요 |

**실무 조합 예시**
- **지금처럼 API가 base+adapter를 직접 로드**할 수 있다면:  
  **환경변수** 또는 **설정 파일**로 `base_model` + `adapter_path`만 넘겨 주는 방식이 구현·운영 모두 가볍습니다.
- **vLLM 등 merge된 모델만 쓸 예정**이면:  
  “학습 → (선택) best 선정 → merge → merge 경로를 환경변수/설정 파일에 넣기”처럼 **merge 후 경로**를 기준으로 두고, 그 경로를 환경변수나 설정 파일로 넘기는 식으로 쓰는 게 자연합니다.

---

## 구현: Merge 후 경로 (3번)

파이프라인에서 **학습된 adapter를 base와 merge해 서빙용 단일 디렉터리**로 저장하고, API는 해당 경로만 `LLM_MODEL`로 지정해 사용한다.

### 자동 실행

- **all**: `build_dataset` → `labeling_with_pod` → `train_student_with_pod` → `evaluate` → **merge_for_serving**
- **all_sweep**: `build_dataset` → `labeling_with_pod` → `run_sweep_and_evaluate` → best adapter 있으면 **merge_for_serving**

Merge 결과는 `out_dir/merged_for_serving/YYYYMMDD_HHMMSS/` 에 저장되고, `out_dir/merged_for_serving/latest_merged_path.json` 에 현재 서빙에 쓸 경로가 기록된다.

### 수동 실행 (adapter만 merge)

```bash
python scripts/distill_flows.py merge_for_serving --adapter-path .../runs/xxx/adapter [--out-dir distill_pipeline_output] [--student-model Qwen/Qwen2.5-0.5B-Instruct]
```

또는 스크립트만:

```bash
python scripts/merge_adapter_for_serving.py --adapter-path .../adapter --base-model Qwen/Qwen2.5-0.5B-Instruct --output-dir merged_models/20250101_120000
```

### API에서 사용

- **환경변수**: merge 후 출력된 경로를 그대로 사용  
  `export LLM_MODEL=/path/to/distill_pipeline_output/merged_for_serving/YYYYMMDD_HHMMSS`
- **포인터 파일**: 매번 최신 merge 경로를 쓰려면  
  `latest_merged_path.json` 의 `merged_model_path` 값을 읽어서 `LLM_MODEL`로 설정하거나, 스크립트/배포에서 해당 경로를 주입.

### 의존성

Merge 스크립트/flow는 `transformers`, `peft` 가 필요하다. 학습용 `requirements.train-llm.txt` 를 쓰는 환경에서 실행하면 된다.

---

## Merge on Pod (볼륨에 직접 저장)

API/추론을 **RunPod Pod**에서 돌릴 경우, merge도 **같은 네트워크 볼륨이 마운트된 Pod**에서 실행해 결과를 볼륨에 두면, 추론 Pod는 그 경로만 쓰면 된다.

### flow: merge_for_serving_with_pod

- **동작 (셋 중 하나)**: **--adapter-path** 로컬 adapter를 볼륨에 업로드 후 Pod에서 merge. **--run-id** 볼륨에 이미 있는 runs/RUN_ID/adapter 사용(업로드 없음) 후 Pod에서 merge. **인자 없음** 볼륨에서 adapter가 있는 run 중 최신 사용. 결과는 볼륨의 merged_for_serving/YYYYMMDD_HHMMSS/ 및 latest_merged_path.json.
- **실행**: `--adapter-path .../adapter` 또는 `--run-id RUN_ID` 또는 인자 없이 `merge_for_serving_with_pod`
- **필요 환경변수**: `RUNPOD_API_KEY`, `RUNPOD_S3_ACCESS_KEY`, `RUNPOD_S3_SECRET_ACCESS_KEY`, (선택) `RUNPOD_NETWORK_VOLUME_ID_TRAIN`.

### 추론 Pod에서 사용

- 같은 네트워크 볼륨을 마운트한 추론 Pod를 띄운다.
- `LLM_MODEL`을 볼륨 안 merge 경로로 설정. 최신 경로는 볼륨의 `distill_pipeline_output/merged_for_serving/latest_merged_path.json` 에서 `merged_model_path` 를 읽어 사용하면 된다.
