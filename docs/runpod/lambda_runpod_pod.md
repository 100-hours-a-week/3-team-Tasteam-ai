# Lambda + RunPod Pod + RQ 배치 운영 가이드

**“AWS 스케줄러 → Pod 기동(네트워크 볼륨 마운트) → RQ 워커로 작업 수행 → 완료 시 terminate”** 흐름을 위한 절차·체크포인트·실패 처리 정리.  
**RunPod Pod + Network Volume + Redis(RQ)** 조합 기준이며, 오케스트레이터는 **EventBridge Scheduler + Lambda** 를 전제로 함.

---

## 레포 내 구현 (이 코드베이스에 포함된 것)

| 항목 | 위치 | 비고 |
|------|------|------|
| **POST /api/v1/batch/enqueue** | `src/api/routers/batch.py` | `run_id` 선택 인자 지원. 전달 시 Redis `run:{run_id}:total/done/fail` 초기화 후 job meta에 저장. |
| **GET /api/v1/batch/run/{run_id}/status** | `src/api/routers/batch.py` | `total`, `done`, `fail`, `completed` 반환. Lambda 완료 판단용. |
| **run_id 집계 (성공 시 INCR done, 최종 실패 시 INCR fail)** | `src/queue_tasks.py` | `_run_id_incr_done`, `run_id_failure_callback`. job 성공 시 done, RQ 실패 콜백에서 fail. |
| **Worker ready 하트비트** | `scripts/rq_worker.py` | 환경변수 `RUNPOD_POD_ID` 가 있으면 기동 시 `worker:{pod_id}:ready` SET (TTL 120초). |
| **RQ job 정의 (sentiment/summary/comparison/all)** | `src/queue_tasks.py` | `run_id` 인자 추가. 전달 시 완료 시에만 집계 갱신. |
| **Redis 락 (API 진입)** | `src/cache.py` | `lock:{restaurant_id}:{analysis_type}`. 배치 job 내부 레스토랑별 락은 미구현(필요 시 추가). |
| **trigger_offline_batch.py** | `scripts/trigger_offline_batch.py` | JSON → enqueue 호출. `--run-id` 로 run_id 전달 가능. EventBridge/cron에서 호출 시 사용. |

**run_id 규칙 (레포 내)**  
- Redis 키: `run:{run_id}:total`, `run:{run_id}:done`, `run:{run_id}:fail` (TTL 2일).  
- 요청 1회 = job 1개이므로 현재는 **total=1** 로 설정. 여러 job을 같은 run_id로 쓰려면 오케스트레이터에서 total을 N으로 설정하는 등 확장 필요.

---

## 외부 구현 필요 (이 레포 밖에서 구현해야 하는 것)

| 단계 | 필요한 것 | 설명 |
|------|------------|------|
| **1) 스케줄 트리거** | **EventBridge Scheduler** | 정해진 시간에 Lambda 호출. (AWS 콘솔/CDK/Terraform 등으로 생성.) |
| **2) Pod 생성/terminate** | **Lambda → RunPod Control API** | Pod 생성(Network Volume attach, 환경변수, 커맨드 지정), Pod terminate 호출. 이 레포에는 RunPod **추론 URL** 호출만 있고, **Pod lifecycle API** 호출 코드는 없음. |
| **2) entrypoint** | **Pod 이미지 내 entrypoint.sh** | 볼륨 마운트 확인 → 모델/캐시 확인(또는 다운로드) → `rq worker` 실행. 레포에 예시 스크립트는 있으나, **실제 Pod에 넣고 커맨드로 지정하는 것은 배포/이미지 빌드 측에서 수행.** |
| **3) Worker ready 폴링** | **Lambda** | Redis `worker:{pod_id}:ready` 키 존재 여부를 주기적으로 확인. 레포는 워커에서 키만 설정함. |
| **4) enqueue 호출** | **Lambda** | FastAPI `POST /api/v1/batch/enqueue` 호출(payload에 `run_id`, `job_type`, `restaurants` 등). 레포는 API만 제공. |
| **6) 완료 판단** | **Lambda** | `GET /api/v1/batch/run/{run_id}/status` 로 `completed == true` 될 때까지 폴링 후 terminate 결정. |
| **8) terminate 실행** | **Lambda → RunPod API** | Pod terminate 호출. 레포에는 없음. |
| **9) Cleanup 스케줄러** | **Lambda + EventBridge 규칙** | 10~30분마다 `worker:*:ready` 만료·run 완료 후에도 살아 있는 Pod·최대 TTL 초과 Pod 찾아 terminate. 레포에는 없음. |
| **인프라** | **Redis(ElastiCache), FastAPI 호스트, 네트워크** | Pod(워커)와 Lambda가 같은 Redis·FastAPI에 접근 가능해야 함. VPC/보안 그룹 등은 AWS·RunPod 측 구성. |

---

## 0) 구성요소와 “역할 분리”

* **EventBridge Scheduler**: “언제 돌릴지”만 결정
* **Lambda(오케스트레이터)**: “Pod 생성/준비/종료 + enqueue 호출 + 상태 추적”
* **Redis(ElastiCache 권장)**: RQ queue + job registry + (락/하트비트)
* **RunPod Pod(워커)**: `rq worker` 실행해서 큐를 소비
* **Network Volume**: 모델/캐시/중간 산출물 등 **Pod 재기동에도 남아야 하는 것** 저장(모델 캐시가 핵심)

---

## 1) 스케줄 트리거 단계 (EventBridge → Lambda)

1. EventBridge Scheduler가 정해진 시간(예: 매일 01:00 KST)에 Lambda 호출
2. Lambda 입력 payload 예:

   * `job_type`: summary | sentiment | comparison
   * `scope`: all_restaurants | delta_since_last_run | restaurant_ids …
   * `run_id`: 추적용 UUID (권장)

> 여기서 `run_id`는 “이번 배치 1회 실행”을 대표하는 키라서, 로그/메트릭/재시도에서 엄청 편해져.

---

## 2) Pod 생성 단계 (Lambda → RunPod API)

### 2-1) 네트워크 볼륨/리전 결정

1. Lambda가 사용할 **Network Volume ID**를 알고 있어야 함
2. **Pod 생성 리전 = Network Volume 리전**(거의 항상 이렇게 맞춰야 실제로 attach 가능)

### 2-2) Pod 생성 요청(핵심: “볼륨 attach + 워커 커맨드”)

Lambda가 RunPod API로 Pod 생성할 때 포함해야 하는 것들:

* **Network Volume attach**

  * mount path 예: `/runpod-volume`
* **환경변수**

  * `REDIS_URL` (또는 host/port/db)
  * `RQ_QUEUE_NAME` (예: `batch`)
  * `RUN_ID` (= 이번 실행 식별자)
  * `MODEL_DIR=/runpod-volume/llm-models/...`
  * (옵션) `HF_HOME=/runpod-volume/hf-cache` 같은 캐시 경로
* **컨테이너 커맨드**

  * `./entrypoint.sh` 같은 스크립트로:

    1. 볼륨 마운트 확인
    2. 모델/캐시 존재 확인(없으면 다운로드)
    3. `rq worker` 실행

### 2-3) “볼륨 마운트 체크”가 최우선

워커가 뜨자마자 아래를 **반드시** 확인해야 해:

* `/runpod-volume` 존재 여부
* 쓰기 가능 여부(permissions)
* 기대 디렉토리(`/runpod-volume/llm-models`, `/runpod-volume/hf-cache` 등) 준비

실패하면: **즉시 종료(exit 1)** 해서 “불완전한 상태로 오래 살아있는 Pod”을 막는 게 중요.

---

## 3) 워커 Ready 확인 단계 (Lambda가 “기동 완료”를 판단)

Pod 생성 후 바로 enqueue 하면,

* 워커가 아직 Redis에 붙기 전이면 **작업은 큐에 쌓이지만** “Pod이 살아있나?”를 확인하기가 애매해져.

그래서 Lambda는 다음 중 하나로 “Ready”를 확인:

### 옵션 A) 워커가 Redis에 붙었는지 확인(권장)

* 워커 부팅 시 Redis에 `SET worker:<pod_id>:ready = 1 EX 60` 같은 하트비트를 넣게 함(주기 갱신)
* Lambda는 이 키가 생길 때까지 대기(짧게, 예: 1~3분)

### 옵션 B) Pod 로그 패턴 확인

* 로그에서 `Listening on ...` 같은 메시지 확인
  (다만 로그 API/권한/지연 이슈로 A보다 운영 난이도↑)

---

## 4) 큐잉 단계 (Lambda → FastAPI enqueue 또는 직접 Redis enqueue)

### 4-1) 추천: FastAPI 엔드포인트로 enqueue

Lambda가 **FastAPI**에 `POST /api/v1/batch/enqueue` 호출 → FastAPI가 RQ에 job push

* 장점:

  * 기존 파이프라인/검증/DTO 재사용
  * idempotency(중복 방지) 로직을 API에 넣기 쉬움

**중요**: 여기서 Lambda는 응답으로 받은 `job_id`들을 저장해야 해.

### 4-2) (대안) Lambda가 Redis에 직접 넣기

가능은 한데, job 함수 import 경로/직렬화/버전 관리가 복잡해지기 쉬워서 **권장하지 않음**.

---

## 5) 실행 단계 (RQ 워커가 job 처리)

워커 프로세스가 하는 일:

1. RQ에서 job pop → 실행 시작
2. (중복 방지) **락을 잡는다**

   * 예: `lock:restaurant:{id}:{job_type}` 를 Redis SET NX EX
   * 이미 있으면 SKIP 또는 재시도(정책 선택)
3. 필요한 데이터 로드(DB/VectorDB 등)
4. LLM/모델 수행

   * 모델 파일은 `/runpod-volume/...` 에 있으니 “매번 다운로드”가 아니라 “존재하면 바로 사용”
5. 결과 저장(DB/S3/VectorDB 등)
6. job 성공/실패 상태 업데이트 + 락 해제

---

## 6) 완료 판단 단계 (Lambda가 “언제 terminate할지” 결정)

여기가 start-terminate 전략의 핵심이야. “작업이 끝났는데 Pod이 떠있으면 돈이 샌다.”

### 가장 안정적인 방식: “러너(run_id) 단위 집계 키”

* enqueue 시점에 Lambda 또는 FastAPI가:

  * `run:{run_id}:total = N`
  * `run:{run_id}:done = 0`
  * `run:{run_id}:fail = 0`
* 워커가 job 끝날 때마다:

  * 성공: `INCR run:{run_id}:done`
  * 실패: `INCR run:{run_id}:fail`
* Lambda는 주기적으로:

  * `done + fail == total` 이 되면 “전체 종료 조건 달성”

이 방식은 **RQ job_id를 하나하나 폴링**하는 것보다 단순하고 견고해.

### 보조 안전장치: Idle timeout

* “최근 X분간 queue 비었고, 실행 중 job도 없다”면 terminate
* 단, **지연 enqueue/재시도**가 있는 구조면 오탐 가능 → run_id 집계 방식이 더 안정적

---

## 7) terminate 직전: 네트워크 볼륨 “안전 종료” 체크

Network Volume 쓰는 경우 terminate 전 체크 포인트:

1. **파일 쓰기 중인지 확인**

   * 모델 다운로드/압축 해제 중이면 terminate하면 볼륨이 “불완전 상태”가 될 수 있음
2. 캐시/모델 다운로드는 가능하면:

   * “다운로드 완료 플래그 파일”을 둠
     예: `/runpod-volume/llm-models/Qwen/.../.complete`
   * entrypoint에서 `.complete` 없으면 “다시 다운로드/복구” 루틴 실행

**권장 종료 순서(워커 내부):**

* `rq worker` graceful shutdown 시그널 처리(SIGTERM)
* 현재 job 끝내고 종료(혹은 안전한 중단)
* 종료 직전에 `sync`(리눅스) 같은 flush를 한 번 호출(과한 건 아니고, 최소한의 방어)

---

## 8) terminate 실행 (Lambda → RunPod API terminate)

종료는 “stop”이 아니라 **terminate**라 했으니:

* Lambda가 RunPod API로 Pod terminate 호출
* 종료 후:

  * run 상태를 DB에 `COMPLETED/FAILED` 기록
  * 알림(Slack/SNS)
  * 비용/토큰/시간 집계 업로드(원하면)

---

## 9) 실패/재시도 설계 (운영에서 꼭 필요)

### 9-1) Pod 기동 실패

* 네트워크 볼륨 attach 실패 / 이미지 pull 실패 / Redis 연결 실패
* 대응:

  * Lambda에서 “Pod 상태가 ready 안 됨”이면 **terminate 후 재시도**
  * 재시도 횟수 제한 + 알림

### 9-2) 워커 실행 중 실패(LLM 429, OOM 등)

* RQ 자체 retry 정책 + DLQ(FailedJobRegistry)
* run_id 집계에서 fail 카운트가 올라가도:

  * “fail이 0이 아니면 전체 run은 FAILED” 같은 정책 가능
  * 또는 “일부 실패 허용” 정책 가능

### 9-3) terminate 누락(돈 새는 문제)

* Lambda가 중간에 죽거나 타임아웃 나면 Pod이 남을 수 있음
  **대책(필수):**
* 별도 “청소(cleanup) 스케줄러”를 10~30분마다 돌려서

  * `worker:<pod_id>:ready` 하트비트가 끊긴 Pod
  * `run:{run_id}`가 이미 완료인데 살아있는 Pod
  * 생성 후 최대 TTL 초과 Pod
    를 찾아 terminate

---

## 10) 네트워크 볼륨을 쓰는 이유를 이 전략에 딱 맞게 정리하면

* **terminate로 Pod은 매번 새로 뜨지만**
* **모델/캐시를 네트워크 볼륨에 두면**

  * 매번 이미지에 모델을 포함시키지 않아도 되고
  * 매번 HF 다운로드를 반복하지 않아도 됨
  * “start-terminate” 전략이 비용 면에서 성립함

---

## 부록: run_id 키·entrypoint·Lambda 의사코드

### run_id / Redis 키 네이밍 (레포와 동일하게 사용)

- `run:{run_id}:total` — 이번 run의 job 개수 (현재 enqueue 1회당 1로 설정)
- `run:{run_id}:done` — 성공 완료된 job 수 (워커에서 INCR)
- `run:{run_id}:fail` — 최종 실패한 job 수 (RQ on_failure 콜백에서 INCR)
- `worker:{pod_id}:ready` — 워커 기동 시 1, TTL 120초 (레포 `rq_worker.py`에서 설정)

Lambda는 **완료 조건**: `GET /api/v1/batch/run/{run_id}/status` 의 `completed === true` (즉 `done + fail >= total`).

### entrypoint.sh 예시 (외부 구현: Pod 이미지에 포함 후 커맨드로 지정)

```bash
#!/bin/bash
set -e
VOLUME_MOUNT="${VOLUME_MOUNT:-/runpod-volume}"

# 1) 볼륨 마운트 체크
if [ ! -d "$VOLUME_MOUNT" ] || [ ! -w "$VOLUME_MOUNT" ]; then
  echo "Volume not mounted or not writable: $VOLUME_MOUNT"
  exit 1
fi

# 2) 모델/캐시 디렉터리 (필요 시 .complete 플래그로 다운로드 스킵)
# MODEL_DIR="${VOLUME_MOUNT}/llm-models/Qwen/..."
# [ -f "${MODEL_DIR}/.complete" ] || { 다운로드; touch "${MODEL_DIR}/.complete"; }

# 3) RQ 워커 실행 (RUNPOD_POD_ID 있으면 ready 키 설정됨)
exec python scripts/rq_worker.py
```

→ **이 스크립트를 이미지에 넣고, RunPod Pod 생성 시 커맨드를 `./entrypoint.sh` 로 지정하는 것은 외부(이미지 빌드·RunPod 설정)에서 수행.**

### Lambda 오케스트레이터 의사코드 (전부 외부 구현)

1. **Payload**: EventBridge에서 `run_id`, `job_type`, `restaurants`(또는 scope) 전달.
2. **Pod 생성**: RunPod API로 Pod 생성(Network Volume attach, env: `REDIS_HOST`, `RQ_QUEUE_NAME`, `RUNPOD_POD_ID`, `RUN_ID` 등, command: `./entrypoint.sh`). → **외부 구현.**
3. **Ready 대기**: Redis에서 `worker:{pod_id}:ready` 존재할 때까지 폴링(최대 1~3분). → **외부 구현.** (레포는 키만 설정)
4. **Enqueue**: `POST /api/v1/batch/enqueue` with `run_id`, `job_type`, `restaurants`. 응답 `job_id`, `run_id` 저장. → **레포 API 사용.**
5. **완료 대기**: `GET /api/v1/batch/run/{run_id}/status` 폴링, `completed === true` 되면 다음 단계. → **레포 API 사용.**
6. **Terminate**: RunPod API로 해당 Pod terminate. → **외부 구현.**
7. **Cleanup(별도 스케줄)**: 10~30분마다, ready 만료·run 완료 후 미종료 Pod·최대 TTL 초과 Pod 조회 후 terminate. → **외부 구현.**
