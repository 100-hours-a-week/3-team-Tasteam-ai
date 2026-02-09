이 문서는 **부하테스트·배치 실행** 방식을 목적별로 구분하고, 사용할 명령어와 옵션을 정리한 것입니다.  
각각 **목적**을 이렇게 구분해서 보면 됩니다.

---

### 1. `run_all_restaurants_api.py`에 640k JSON 넣고 돌리기

**목적:** **“대량 데이터를 한 번 끝까지 처리할 수 있는지”** 보는 것.

- 64만 리뷰 / 100개 레스토랑 같은 **실제 규모 데이터**를 **한 사이클**에 다 돌림.
- 요약·감성·비교 API를 **실제 배치 작업**처럼 호출해서, **처리 가능 여부·소요 시간·에러 여부**를 확인.
- “처리량(req/s)”보다는 **“이 데이터를 다 처리하는 데 걸리는 시간 / 제대로 끝나는지”**가 관심사.

→ **대규모 배치 처리 검증 / E2E 실험.**

---

### 2. 기존 load_test (640k 안 씀)

**목적:** **“이 API가 초당 몇 건 견디는지, 지연은 어떤지”**를 **작은 payload**로 재는 것.

- 작은 테스트 데이터(첫 레스토랑 리뷰 몇 개)로 **같은 요청을 여러 번** 보냄.
- **처리량(req/s), P50/P95/P99 지연, 성공률**을 측정.
- payload가 작고 단순해서, **서버의 “순수 처리 능력”**을 비교하기 좋음.

→ **처리량·지연 측정 (작은 요청 기준).**

---

### 3. 640k를 써서 load_test 돌리기

**목적:** **“실제에 가까운 큰 payload로 요청을 반복했을 때의 처리량·지연”**을 보는 것.

- 640k JSON에서 **일부만 잘라서**(예: 10개 레스토랑, 레스토랑당 최대 100개 리뷰) **그걸로** load_test를 돌림.
- 요청 하나당 크기가 커지므로, **2번보다 req/s는 낮고, 지연은 높게** 나오는 게 정상.
- “작은 요청 기준 성능(2번)”과 “큰 요청 기준 성능(3번)”을 **같은 지표(처리량·지연)**로 비교할 수 있음.

→ **처리량·지연 측정 (큰/실제에 가까운 요청 기준).**

---

### 한 줄로 정리

| 실행 방식 | 목적 |
|-----------|------|
| **1. run_all_restaurants_api + 640k** | “이 64만 건을 한 번에 다 돌릴 수 있나?” → **대규모 배치 처리 검증** |
| **2. 기존 load_test** | “작은 요청으로 초당 몇 건, 지연은?” → **처리량·지연 (소형 payload)** |
| **3. load_test + 640k** | “큰 요청으로 초당 몇 건, 지연은?” → **처리량·지연 (대형 payload)** |

그래서 **목적**은:

- **1** = “한 번에 큰 일을 다 할 수 있는지”
- **2·3** = “같은 종류의 일을 반복했을 때 얼마나 빠르고 많이 처리하는지”  
  (2는 작은 일, 3은 큰 일로 측정)  
이렇게 나누어 생각하면 됩니다.

---

## 설명

### test_all_task.py 부하테스트 옵션

| 옵션 | 설명 |
|------|------|
| `--load-test` | 부하테스트 모드로 실행 (감성·요약·비교 배치 API를 반복 호출). |
| `--total-requests` | 총 요청 수 (기본값 예: 100). |
| `--concurrent-users` | 동시에 보내는 요청 수(동시 사용자 수). |
| `--ramp-up` | 부하를 점진적으로 올리는 시간(초). 0이면 바로 전부 시작. |
| `--load-test-data` | 부하테스트용 대형 JSON 경로. 없으면 내장 작은 테스트 데이터 사용. **지정 시 부하테스트 전에 벡터 DB 업로드 자동 수행** (요약/비교 API 검색용). |
| `--load-test-max-reviews-per-restaurant` | 감성 배치 시 레스토랑당 최대 리뷰 수 (기본 100). `--load-test-data` 사용 시만 적용. |
| `--load-test-ports` | 여러 포트에 동시 부하테스트. 예: `8001 8002 8003` → 포트별 처리량·지연 수집. 업로드도 각 포트에 수행. |
| `--load-test-scenario` | 요청 순서 시나리오 파일. `--load-test-data`와 함께 사용 시, 요청마다 시나리오의 restaurant_id 순서대로 payload 생성. |
| `--no-load-test-upload` | `--load-test-data` 지정 시에도 벡터 업로드 생략 (이미 업로드된 경우). |
| `--load-test-upload-timeout` | 부하테스트 벡터 업로드 요청 타임아웃(초). 대용량 JSON 시 조정 (기본 3600). |
| `--save-results` | 부하테스트 결과를 저장할 JSON 파일 경로. 클라이언트 측 통계(지연·처리량·성공률)와, `--prometheus-url` 지정 시 해당 구간 Prometheus 메트릭을 포함. |
| `--prometheus-url` | 부하테스트 구간의 Prometheus 메트릭을 조회해 `--save-results` JSON의 `prometheus_metrics`에 포함 (예: `http://localhost:9090`). 비우면 생략. |
| `--abort-after-connection-errors` | 연속 N회 연결 실패(서버 다운 등) 시 부하테스트 조기 종료. 그때까지의 부분 결과를 저장 (기본값: 10, 0이면 비활성화). |
| `--save-container-logs` | 부하테스트 중 앱 다운(조기 종료) 발생 시 `docker logs`로 old_sync/new_sync/new_async 컨테이너 로그를 지정 디렉터리에 저장 (예: `--save-container-logs .` → `logs_rampup_old_sync.txt`, `logs_rampup_new_sync.txt`, `logs_rampup_new_async.txt`). 컨테이너 접두사는 환경변수 `DOCKER_COMPOSE_PROJECT`(기본 `tasteam-new-async`) 사용. |

### 시나리오 파일 (scenario.txt)

- **형식**: 한 줄에 `restaurant_id` 하나(정수). Zipf 분포로 생성하면 인기 있는 ID가 더 자주 반복됨 (80-20 부하 시뮬레이션).
- **생성**: `convert_kr3_tsv.py`에서 `--output-scenario`, `--scenario-requests`, `--zipf-alpha`로 JSON과 함께 생성. **입력은 kr3.tsv**이며, scenario.txt는 **출력**으로 생성됨.
- **사용 조건**: `--load-test-scenario`에 넣을 JSON은 **반드시** `--load-test-data`로 주는 JSON과 **같은 데이터**(같은 레스토랑 집합)여야 함. 시나리오에 나오는 restaurant_id가 해당 JSON에 없으면 해당 요청은 fallback payload로 처리됨.
- **동작**: i번째 요청에서 감성은 시나리오 i번째 ID 한 레스토랑만, 요약·비교는 i번째·(i+1)번째 ID 두 레스토랑으로 payload가 만들어짐.

### 부하테스트에서 나오는 지표

- **처리량(req/s)**: 초당 성공한 요청 수.
- **P50 / P95 / P99**: 응답 시간의 50%/95%/99% 백분위(지연).
- **성공률**: 200 응답 비율. 4xx/5xx는 실패로 집계됨.

### run_all_restaurants_api.py 옵션

- `-i`: 입력 JSON (레스토랑·리뷰 데이터).
- `-o`: 결과 저장 경로.
- `--ports`: 여러 포트에 동일 업로드·요약·감성·비교를 병렬로 수행. 결과는 `results_by_port` 형태로 한 JSON에 저장.
- `--no-upload`: 이미 업로드된 상태일 때 업로드 단계 생략.

---

## 명령어

### test_all_task.py 부하테스트

```bash
# 기본 (단일 포트, 작은 payload) — 처리량/지연 측정
python test_all_task.py --load-test --total-requests 500 --concurrent-users 10 --ramp-up 20

# 결과 저장
python test_all_task.py --load-test --total-requests 500 --concurrent-users 5 --ramp-up 20 --save-results load_test_baseline_results.json

# 640k JSON 사용 (대형 payload). 지정한 JSON이 부하테스트 전에 벡터 DB에 자동 업로드됨.
python test_all_task.py --load-test --load-test-data real_service_simul_review_data_640k.json --total-requests 50 --concurrent-users 5 --save-results load_test_640k.json

# 업로드 생략 (이미 업로드된 경우)
python test_all_task.py --load-test --load-test-data tasteam_app_kr3_640k_even.json --no-load-test-upload --total-requests 100

# 레스토랑당 리뷰 200개까지 (감성 배치)
python test_all_task.py --load-test --load-test-data real_service_simul_review_data_640k.json --load-test-max-reviews-per-restaurant 200 --total-requests 30

# 여러 포트(8001, 8002, 8003)에 동시 부하테스트 — 포트별 처리량/지연 수집
python test_all_task.py --load-test --load-test-ports 8001 8002 8003 --total-requests 50 --concurrent-users 5 --save-results load_test_multi_port.json

# 시나리오 부하테스트 (Zipf 요청 순서) — convert_kr3_tsv.py로 생성한 scenario.txt 사용
# 1) 시나리오 생성 (동일 JSON과 함께 생성해 두면 됨)
python scripts/convert_kr3_tsv.py --input data/kr3.tsv --output tasteam_app_all_review_data.json \
  --power-law --restaurants 100 \
  --output-scenario scenario.txt --scenario-requests 20000

# 2) 부하테스트 시 시나리오 적용 (요청마다 scenario의 restaurant_id 순서대로 payload 사용)
python test_all_task.py --load-test --load-test-data tasteam_app_all_review_data.json \
  --load-test-scenario scenario.txt --total-requests 500 --concurrent-users 10 --save-results load_test_scenario.json
```

- **시나리오**: `scenario.txt`는 한 줄에 `restaurant_id` 하나씩 (Zipf 분포로 인기 있는 ID가 더 자주 나옴). `--load-test-data`로 준 JSON에 있는 레스토랑만 사용 가능. 감성/요약/비교 모두 요청 인덱스에 따라 시나리오 순서대로 payload가 바뀜.

### run_all_restaurants_api.py (배치 처리)

```bash
# 단일 포트 (기본 8001)
python scripts/run_all_restaurants_api.py -i real_service_simul_review_data_640k.json -o results.json

# 여러 포트에 동시 전송 (동일 작업을 8001, 8002, 8003에 각각 수행)
python scripts/run_all_restaurants_api.py -i real_service_simul_review_data_640k.json -o results.json --ports 8001 8002 8003

# 업로드 생략 (이미 업로드된 경우)
python scripts/run_all_restaurants_api.py -i data.json -o results.json --no-upload --ports 8001 8002 8003
```

구분	전체 레스토랑 수	리뷰(레스토랑당)
Load test 감성/요약/비교	10개 고정	기본 100개 (옵션으로 변경 가능)

구분 전체 레스토랑 수  배치 레스토랑 개수(리뷰수는 데이터 따라 감)
run_all_restaurants_api	전체 (또는 --limit)	기본 100개, 한 번에 10개 레스토랑씩 배치

load test(시나리오 사용 x) -> 상위 10개 이용 배치 요청 처리
load test(시나리오 사용 o) -> 데이터에서 시나리오 따라 10개씩 묶어 배치 요청 처리

experiment 1: 통제된(workload controlled) 조건에서 아키텍처 차이를 정량 비교 (리뷰 분포 균등, 요청 균등)
Experiment 2: 실서비스 분포(skew + long-tail + burst)에서 tail/실패율 방어력 비교 (리뷰 분포 비균등, 요청 비균등)

```bash
# 목적: old/new/async 순수 아키텍처 비교
# 분포/쏠림 변수를 줄이고
# old/new/async의 구조 차이(큐/락/IO)가 얼마나 드러나는지 보는 실험.
# 요청 대상(키)의 분포가 균등/통제됨
# 리뷰 분포 균등
# 사전 선정 10개 레스토랑을 균등 라운드로빈
# Experiment 1: ramp-up included (20s)
# “원인 분리용”이라서 둘 다 통제
# “통제된(workload controlled) 조건에서 아키텍처 차이를 정량 비교”
python test_all_task.py --load-test --load-test-ports 8001 8002 8003 \
  --load-test-data tasteam_app_kr3_640k_even.json \
  --total-requests 500 --concurrent-users 10 --ramp-up 20 \
  --save-results bench_rampup_500.json
```

```bash
# 목적: 실서비스 쏠림/캐시 편향에서 old/new/async가 tail을 얼마나 방어하는지 보는 실험.
# 요청 분포 비균등(Zipf/hot-key)
# 리뷰 분포 비균등
# 시나리오가 “10개 묶음”이라면 버스트/팬아웃 성질까지 포함
# 그래서 여기서는 평균보다 tail(p95/p99) + 실패율이 핵심 지표.
# Experiment 2: steady-state only (no ramp-up)
# “현실 재현용”이라서 요청+리뷰 쏠림을 함께 줌
# “실서비스 분포(skew + long-tail + burst)에서 tail/실패율 방어력 비교”
python test_all_task.py --load-test --load-test-ports 8001 8002 8003 \
  --load-test-data real_service_simul_review_data_640k.json \
  --load-test-scenario scenario.txt \
  --total-requests 500 --concurrent-users 10 \
  --save-results bench_zipf_500.json
```

---

```bash
# 목적: old/new/async 오프라인 처리 성능 비교
# 100개 레스토랑에서 10개씩 배치처리 진행.
# 총 처리 시간, 처리량 비교
# "밤에 돌리는 일괄 분석(오프라인)"
python scripts/run_all_restaurants_api.py -i real_service_simul_review_data_640k.json -o results.json --ports 8001 8002 8003
```