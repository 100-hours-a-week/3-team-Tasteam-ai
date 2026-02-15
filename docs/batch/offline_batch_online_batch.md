좋아. 그럼 **run_id 설계가 딱 깔끔해진다.**
네 경우엔 “오프라인(전체 음식점)”과 “온라인(개인 유저의 리뷰 목록)”이 **스코프와 소유자가 완전히 다르기 때문**에, run_id는 공통으로 쓰되 meta에 “누가/무엇을/왜”를 명확히 남기는 게 정답이야.

---

## run_id 정의

**run_id = “분석 실행 1회(Execution)”**

* 오프라인: 새벽에 전체 음식점 대상으로 돌린 “배치 1회”
* 온라인: 특정 사용자가 버튼 눌러 시작한 “요청 1회”

즉, 둘 다 run_id가 있고, 의미는 동일(실행 1회)인데 **trigger_type이 다름**.

---

## `analysis_run`에 꼭 넣어야 하는 meta (혼합 운영 기준 최소셋)

공통(둘 다)

* `run_id` ✅
* `trigger_type` ✅: `OFFLINE_SCHEDULED` | `ONLINE_USER`
* `analysis_scope` ✅: `RESTAURANT` | `USER_REVIEWS`
* `pipeline_version`
* `model_id`, `prompt_version`
* `generated_at`(=started_at/created_at 중 하나로 통일)
* `input_snapshot_hash`
* `input_counts.review_count_used`

오프라인 전용으로 유용

* `target_set_version` 또는 `restaurant_count_targeted`
* `schedule_time`(해당 배치의 기준 시각)

온라인 전용으로 필수

* `user_id` ✅ (누가 눌렀는지)
* `request_id`(로그 코릴레이션)
* (옵션) `ui_action` = `"click_review_analysis"`

> 핵심: **온라인은 무조건 user_id가 들어가야** “이 run이 누구 요청인지”가 남음.

---

## `analysis_result`는 분리 저장을 추천 (오프라인 결과와 온라인 결과는 “소비처”가 다름)

### 오프라인 결과 (음식점 단위)

* `restaurant_id`
* `analysis_type`
* `run_id`
* `result_json`

### 온라인 결과 (유저 단위)

* `user_id`
* `analysis_type`
* `run_id`
* `result_json`

여기서 중요한 포인트:

✅ 오프라인 결과는 “음식점 상세 페이지/랭킹/의사결정”에 쓰이고
✅ 온라인 결과는 “내 리뷰 분석 화면(개인화)”에 쓰임

그래서 **같은 analysis_result 테이블에 다 때려넣기보다**

* `restaurant_analysis_result`
* `user_analysis_result`
  로 **물리적으로 분리**하는 편이 운영/권한/인덱싱/쿼리가 훨씬 편해.

(한 테이블로 가려면 `subject_type`/`subject_id` polymorphic 키가 필요한데, 실전에서 쿼리/인덱스가 지저분해지는 경우가 많아.)

---

## 오프라인/온라인에서 meta 필드 의미가 어떻게 달라지나

* `input_snapshot_hash`

  * 오프라인: “이 음식점의 이번 배치에 사용된 리뷰 집합”
  * 온라인: “이 유저가 선택한 리뷰 리스트(혹은 기간/필터) 집합”
* `review_count_used`

  * 오프라인: 음식점별 사용 리뷰 수
  * 온라인: 유저가 가진 리뷰 중 분석에 포함된 수

---

## API 엔드포인트(패턴 B 유지) 추천

### 음식점(오프라인 결과 조회)

* `GET /analysis/restaurants/{restaurant_id}?type=summary`
* `GET /admin/analysis/restaurants/{restaurant_id}?type=summary`

### 유저(온라인 결과 조회)

* `GET /analysis/users/me?type=summary` (로그인 기준)
* `GET /admin/analysis/users/{user_id}?type=summary`

### 실행 트리거(온라인)

* `POST /analysis/users/me:run`

  * 응답은 보통 `{ run_id, status: "QUEUED" }` 정도

오프라인은 EventBridge가 트리거하니까 API로 굳이 열 필요 없고, 열더라도 admin-only가 좋아.

---

## run_id 생성 규칙 (혼합 운영에서 추천)

* 오프라인: `offline-YYYYMMDD-HHMM` 같은 사람이 읽을 수 있는 값 + UUID suffix
* 온라인: `online-{user_id}-{timestamp}-{uuid}` 또는 그냥 UUID
* 중요한 건 **유일성 + 로그 추적성**
  (사람이 읽기 쉬우면 운영이 편해짐)

---

## 정리

* 오프라인/온라인 둘 다 하면 **run_id는 “실행 1회”로 공통**
* 구분은 `trigger_type`, `analysis_scope`, `user_id` 같은 meta로 한다
* 결과는 소비처가 달라서 **restaurant 결과 테이블과 user 결과 테이블을 분리**하는 걸 추천

원하면, 네가 원하는 화면/기능 기준으로

* “음식점 결과는 최신만 유지할지(덮어쓰기) vs 히스토리 쌓을지”
* “온라인 유저 결과는 버튼 누를 때마다 새 run을 쌓을지 vs 최신만 유지할지”
  이 두 가지 정책을 딱 정해주는 추천안을 줄게.

---

좋은 질문이야 👍
결론부터 말하면:

> ✅ **RQ를 쓰면 “배치 러너 스크립트” 대신 “RQ 워커 프로세스”가 실행 주체가 된다.**
> 하지만 완전히 대체라기보다는, **배치 러너의 역할이 워커로 이동하는 것**에 가깝다.

---

# 1️⃣ RQ 없이 배치 러너 스크립트 방식

## 구조

```
EventBridge
   ↓
Lambda
   ↓
ECS Task 실행
   ↓
batch_runner.py 실행
   ↓
for restaurant in all:
    LLM 호출
    DB 저장
```

### 특징

* 한 프로세스가 전체 배치를 끝까지 처리
* 실패하면 전체 중단 가능
* 재시도 직접 구현 필요
* 병렬화 직접 구현 필요

이게 전통적인 “배치 러너 스크립트 방식”이야.

---

# 2️⃣ RQ 사용하는 경우

## 구조

```
EventBridge
   ↓
enqueue script (또는 Lambda)
   ↓
Redis Queue

RQ Worker (항상 실행 중)
   ↓
job 1 처리
job 2 처리
job 3 처리 ...
```

여기서 **batch_runner.py는 사라지는 게 아니라 역할이 바뀐다.**

---

# 🔁 역할 변화 비교

| 기존 배치 러너       | RQ 방식         |
| -------------- | ------------- |
| 스크립트가 모든 대상 순회 | enqueue만 수행   |
| 스크립트가 직접 실행    | 워커가 job 단위 실행 |
| 병렬 직접 구현       | 워커 개수로 병렬 제어  |
| 재시도 직접 구현      | RQ retry 사용   |

---

# 3️⃣ RQ 방식에서 “배치 러너”는 무엇이 되나?

보통 이렇게 나뉜다:

### 🔹 1) enqueue 스크립트

예: `offline_enqueue.py`

```python
for restaurant in restaurants:
    queue.enqueue(analyze_restaurant, restaurant.id, run_id)
```

이게 기존 배치 러너의 “대상 순회” 역할만 담당.

---

### 🔹 2) worker 함수

```python
def analyze_restaurant(restaurant_id, run_id):
    # 리뷰 로드
    # 벡터서치
    # LLM 호출
    # DB 저장
```

이게 기존 배치 러너 내부 로직.

---

# 4️⃣ 왜 RQ 방식이 더 안정적인가?

### 기존 방식

* restaurant 1000개 중 500번째에서 에러 → 전체 중단

### RQ 방식

* 500번째 job 실패
* 나머지는 계속 처리
* 실패 job만 retry

---

# 5️⃣ 너 프로젝트 기준 추천

너는:

* LLM Pod 분리
* 오프라인 전체 음식점
* 온라인 사용자 버튼
* 결과 DB 저장

이 구조에서는:

> 🔥 RQ + enqueue 스크립트 + worker 조합이 가장 깔끔함

---

# 6️⃣ 그래서 정확한 답

> RQ 방식으로 배치러너 스크립트를 사용하는건가?

👉 완전히 동일한 형태는 아니고,

* 배치 러너 스크립트 = “enqueue 담당”
* 실제 실행은 = RQ worker

이렇게 역할이 분리된다.

---

# 🧠 한 단계 더 깊은 질문

너가 고민해야 할 건 이것이야:

> 오프라인 배치에서 run_id는 enqueue 시 생성할까?
> 아니면 worker가 생성할까?

이 결정이 구조를 확정짓는다.

원하면 여기까지 깔끔하게 정리해줄게.
