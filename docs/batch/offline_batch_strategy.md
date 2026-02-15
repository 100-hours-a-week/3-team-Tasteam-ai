# 오프라인 배치 전략 (적용 완료)

배치 실행 주체 분리, 작업 큐·재시도·DLQ, 결과 버전 관리를 반영한 아키텍처.  
**RQ 방식만** 사용 (batch-runner 제거됨).

---

## 1. 배치 실행 주체를 API에서 분리

### 1.1 RQ 워커 (큐 소비자)

`BATCH_USE_QUEUE=true` 시 API가 배치 작업을 **큐에 넣고**, RQ 워커가 소비.

| 구성요소 | 설명 |
|----------|------|
| `src/queue_tasks.py` | RQ job (run_sentiment/summary/comparison_batch_job, run_all_batch_job) |
| `scripts/rq_worker.py` | RQ 워커 진입점 |
| `src/api/routers/batch.py` | POST /api/v1/batch/enqueue, GET /api/v1/batch/status/{job_id} |
| `scripts/trigger_offline_batch.py` | 오프라인 트리거 (cron/EventBridge에서 호출) |
| `docker-compose.yml` | batch-worker 서비스 (Redis 의존) |

**job_type:** `sentiment` | `summary` | `comparison` | `all` (all = sentiment→summary→comparison 순차)

**API enqueue 예:**
```bash
curl -X POST http://localhost:8001/api/v1/batch/enqueue \
  -H "Content-Type: application/json" \
  -d '{"job_type":"all","restaurants":[{"restaurant_id":1},{"restaurant_id":2}],"limit":10}'
# → {"job_id":"xxx","job_type":"all","queue":"batch"}
```

**환경 변수:**
- `BATCH_USE_QUEUE=true` : enqueue 활성화
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` : Redis 연결
- `BATCH_JOB_MAX_RETRIES=3` : 작업 단위 재시도 횟수

---

## 2. 작업 큐 + 재시도/백오프 + DLQ

| 레이어 | 구현 |
|--------|------|
| **LLM 호출 단위** | `llm_utils.py` : `MAX_RETRIES=3` + 지수 백오프(`2^attempt`), OpenAI 429 시 RunPod 폴백 |
| **배치 작업 단위** | RQ `retry=BATCH_JOB_MAX_RETRIES` → 실패 시 재시도 |
| **DLQ** | RQ `FailedJobRegistry` (실패한 job 자동 등록) |

**DLQ 조회 (RQ):**
```python
from redis import Redis
from rq import Queue
from rq.registry import FailedJobRegistry

conn = Redis.from_url("redis://localhost:6379/0")
queue = Queue("batch", connection=conn)
registry = FailedJobRegistry(queue=queue)
for job_id in registry.get_job_ids():
    job = queue.fetch_job(job_id)
    print(job.exc_info)
```

---

## 3. 결과 저장 & 버전 관리

### 3.1 키 구조

**`restaurant_id + analysis_type + model_version + prompt_version + created_at`**

| 필드 | 설명 |
|------|------|
| `restaurant_id` | 레스토랑 ID |
| `analysis_type` | sentiment \| summary \| comparison |
| `model_version` | LLM 모델 (예: Qwen/Qwen2.5-7B-Instruct) |
| `prompt_version` | 프롬프트 버전 (A/B 비교용, 기본 `v1`) |
| `created_at` | 실행 시각 |

### 3.2 스키마 확장

**analysis_metrics** (기존 + prompt_version):
- `prompt_version TEXT` 컬럼 추가 (마이그레이션으로 기존 DB 호환)

**analysis_results** (신규):
- 버전별 결과 저장
- 컬럼: `restaurant_id`, `analysis_type`, `model_version`, `prompt_version`, `result_payload`, `error_count`, `created_at`

### 3.3 Config

- `PROMPT_VERSION` : 프롬프트 버전 (기본 `v1`)
- `collect_metrics()` : `prompt_version` 인자 추가, `insert_metric`에 전달
- `should_skip_analysis()` : `model_version`, `prompt_version` 인자 추가 → 버전별 SKIP
- `get_last_success_at()` : 동일 버전 필터 지원

### 3.4 활용

| 목적 | 방법 |
|------|------|
| **재실행 안전** | 같은 (model_version, prompt_version)로 SKIP 가능 |
| **부분 실패 재개** | `analysis_results`에서 완료된 restaurant_id 제외 후 재실행 |
| **A/B 비교** | `prompt_version`별로 결과 저장 → `analysis_results` 또는 메트릭으로 비교 |

---

## 4. 요약

| 권장 사항 | 반영 내용 |
|-----------|-----------|
| **배치 실행 주체 분리** | RQ 워커, `/api/v1/batch/enqueue`, trigger_offline_batch.py |
| **작업 큐 + 재시도 + DLQ** | RQ 큐, `BATCH_JOB_MAX_RETRIES`, FailedJobRegistry(DLQ) |
| **결과 저장 & 버전 관리** | prompt_version, analysis_results, 버전별 SKIP |

---

## 5. 관련 파일

```
scripts/trigger_offline_batch.py   # 오프라인 트리거 (cron 호출용)
scripts/run_all_restaurants_api.py # ingestion (--upload-only), 동기 배치
scripts/run_single_restaurant.py   # 단일 레스토랑 재현 (디버깅)
scripts/rq_worker.py               # RQ 워커
src/queue_tasks.py                 # RQ job (run_*_batch_job, run_all_batch_job)
src/api/routers/batch.py           # enqueue / status API
src/metrics_db.py                  # prompt_version, analysis_results
src/config.py                      # PROMPT_VERSION, BATCH_USE_QUEUE, _BatchConfig
docker-compose.yml                 # batch-worker 서비스
```

---

## 6. 외부 구현 필요 항목

| 항목 | 담당 | 설명 |
|------|------|------|
| **오프라인 스케줄** | 인프라/운영 | EventBridge/Lambda/cron에서 trigger_offline_batch.py 또는 enqueue API 호출 |
| **ingestion 스케줄** | 인프라/운영 | `run_all_restaurants_api.py --upload-only` 또는 vector/upload API 호출 |
| **레스토랑 목록 소스** | 운영 | trigger 입력 JSON 생성 (DB/파일/API) |

상세: `offline_batch_processing.md` 참고.
---

대체로 **삭제해도 무방**해. 다만 “무방”의 전제는 **batch_runner가 담당하던 ‘오프라인용 기능’들이 다른 경로로 대체됐을 때**야. 지금 네 표 기준으로 보면, batch_runner는 이미 RQ 쪽으로 역할이 거의 넘어갔고, 남는 가치는 “오프라인 운영 편의 기능” 정도뿐이야.

아래 체크만 통과하면 **삭제(또는 archive)** 해도 깔끔해.

---

## ✅ 삭제해도 되는 조건 체크리스트

### 1) 오프라인 배치 트리거가 따로 있다

* EventBridge/Lambda(또는 크론) → “enqueue”를 수행해서 전체 음식점 job을 큐에 넣는다
  ✅ 있으면 batch_runner 필요 없음(기존 CLI 역할 대체)

### 2) “all” (sentiment+summary+comparison) 실행이 워크플로우로 대체됐다

batch_runner가 `all`을 한 번에 돌리던 걸,
RQ에서는 보통 2가지로 대체해.

* **(A) 오케스트레이터(부모 job)**: `job_type=all`을 받아서 자식 job 3개 enqueue
* **(B) Step Functions/상태 머신**: sentiment→summary→comparison 순서 보장

✅ 둘 중 하나 있으면 batch_runner의 “all” 가치가 사라져.

### 3) 벡터 업로드(전처리/인덱싱) 경로가 따로 있다

너 말대로 RQ는 “이미 업로드된 DB 가정”이면,
그 업로드를 담당하는 **별도 파이프라인**이 있어야 해.

예:

* ingestion 배치(별도 스케줄)
* 리뷰 업데이트 시점에 실시간/준실시간 업로드
* 관리용 `POST /admin/vector/reindex` 같은 엔드포인트

✅ 업로드 경로가 있으면 batch_runner 제거 가능.
❌ 없으면 batch_runner는 “마지막 안전망” 역할이라 당장 삭제는 위험.

### 4) 로컬/수동 재현(디버깅) 방법이 남아 있다

batch_runner는 종종 “한 번에 재현”이 쉬운 도구였거든.
이걸 RQ로 대체하려면 최소한:

* `POST /admin/analysis/{id}:run` (온디맨드 실행)
* 또는 `python -m worker.tasks analyze_restaurant --restaurant-id ...` 같은 **개발용 실행 entry**
  ✅ 이게 있으면 CLI 필요성이 줄어듦.

---

## 결론: 네 현재 상태 기준 추천

* **운영(프로덕션) 관점**: batch_runner는 거의 역할이 없음 → **삭제/아카이브 OK**
* **개발/디버깅 관점**: 완전 삭제보단

  * `scripts/legacy/batch_runner.py`로 **archive**
  * “관리자 온디맨드 실행 API”와 “업로드 파이프라인”이 안정화되면 완전 삭제
    이게 사고가 제일 적어.

---

## 실무적으로 내가 추천하는 “대체 구성” (batch_runner 삭제 후)

1. **offline_enqueuer** (스케줄러가 실행)

   * 전체 음식점 목록 조회
   * run_id 생성
   * job_type별로 enqueue(또는 all 부모 job enqueue)

2. **RQ worker**

   * job_type 하나씩 처리(sentiment/summary/comparison)

3. **관리자 API**

   * 재실행/부분 재처리(특정 restaurant, 특정 job_type)
   * 상태 조회(run_id 기준)

4. **벡터 업로드 전용 파이프라인**

   * RQ와 분리(인덱싱은 별도 책임)

---

### 한 줄 판단

**“벡터 업로드/리인덱싱 경로”와 “all 실행 오케스트레이션”만 RQ/관리자 API로 대체되어 있으면 batch_runner는 삭제해도 무방**해.

원하면, 네가 말한 현재 구조를 기준으로 **`job_type=all`을 부모 job으로 구현할지(자식 enqueue) vs Step Functions로 갈지** 중에서, 너 프로젝트 규모에 맞는 쪽을 딱 골라줄게.

