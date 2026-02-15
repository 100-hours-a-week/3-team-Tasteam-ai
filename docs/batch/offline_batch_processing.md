# 오프라인 배치 처리 방식

전체 음식점 대상으로 새벽/스케줄에 돌리는 오프라인 배치의 현재 구현.  
RQ 방식만 사용, 결과는 반환만 (DB 미저장).

---

## 1. 구현된 항목

| 항목 | 구현 | 설명 |
|------|------|------|
| **job_type=all** | ✅ | sentiment → summary → comparison 순차 실행 (run_all_batch_job) |
| **트리거 스크립트** | ✅ | `scripts/trigger_offline_batch.py` (cron/EventBridge에서 호출) |
| **ingestion 업로드** | ✅ | `run_all_restaurants_api.py --upload-only` (벡터 업로드만 수행) |
| **단일 레스토랑 재현** | ✅ | `scripts/run_single_restaurant.py` (로컬 디버깅용) |

---

## 2. 외부에서 구현해야 할 항목

| 항목 | 담당 | 설명 |
|------|------|------|
| **오프라인 스케줄** | 인프라/운영 | EventBridge/Lambda/cron에서 `trigger_offline_batch.py` 또는 enqueue API 호출 |
| **ingestion 스케줄** | 인프라/운영 | 리뷰 수집 후 `run_all_restaurants_api.py --upload-only` 실행 (또는 vector/upload API 호출) |
| **레스토랑 목록 소스** | 운영 | `trigger_offline_batch.py -i` 입력 JSON 생성 (DB/파일/API 등) |

---

## 3. 실행 흐름

```
[외부: EventBridge/cron]
    ↓
scripts/trigger_offline_batch.py -i restaurants.json -t all -b http://api:8001
    ↓
POST /api/v1/batch/enqueue (job_type=all|sentiment|summary|comparison)
    ↓
Redis 큐 → RQ 워커 → queue_tasks.run_*_batch_job
    ↓
GET /api/v1/batch/status/{job_id} → result (meta + results)
```

---

## 4. 사용 방법

### 4.1 트리거 (cron에서 호출)

```bash
# 전체 레스토랑, job_type=all
python scripts/trigger_offline_batch.py -i data/restaurants.json -t all --base-url http://localhost:8001

# 타입별, 레스토랑 5개만 (테스트)
python scripts/trigger_offline_batch.py -i data.json -t summary --limit 5
```

**cron 예시:**
```
0 3 * * * cd /app && python scripts/trigger_offline_batch.py -i /data/restaurants.json -t all -b http://api:8001
```

### 4.2 ingestion (벡터 업로드만)

```bash
python scripts/run_all_restaurants_api.py -i tasteam_app_data.json --upload-only --base-url http://localhost:8001
```

### 4.3 단일 레스토랑 재현 (로컬 디버깅)

```bash
python scripts/run_single_restaurant.py -r 1 --base-url http://localhost:8001
```

### 4.4 enqueue API 직접 호출

```bash
curl -X POST http://localhost:8001/api/v1/batch/enqueue \
  -H "Content-Type: application/json" \
  -d '{"job_type":"all","restaurants":[{"restaurant_id":1},{"restaurant_id":2}],"limit":10}'
```

---

## 5. job_type

| job_type | 설명 |
|----------|------|
| `sentiment` | 감성 분석만 |
| `summary` | 요약만 |
| `comparison` | 비교만 |
| `all` | sentiment → summary → comparison 순차 (batch_runner all 역할) |

---

## 6. 결과 형식

```json
{
  "meta": {
    "run_id": "offline-20250213-1430-a1b2c3d4",
    "trigger_type": "OFFLINE_SCHEDULED",
    "analysis_scope": "RESTAURANT",
    "type": "all",
    "restaurant_count": 5,
    "schedule_time": "2025-02-13T14:30:00.123456"
  },
  "results": [
    {
      "restaurant_id": 1,
      "summary": { ... },
      "sentiment": { ... },
      "comparison": { ... },
      "errors": {}
    }
  ]
}
```

---

## 7. 관련 파일

```
scripts/trigger_offline_batch.py   # 오프라인 트리거 (cron 호출용)
scripts/run_all_restaurants_api.py # ingestion (--upload-only), 동기 배치
scripts/run_single_restaurant.py   # 단일 레스토랑 재현 (디버깅)
scripts/rq_worker.py               # RQ 워커
src/queue_tasks.py                 # RQ job (run_*_batch_job, run_all_batch_job)
src/api/routers/batch.py           # POST /batch/enqueue, GET /batch/status/{job_id}
```
