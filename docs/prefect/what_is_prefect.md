Workflows
Flows
Tasks
Assets
Caching
States
Runtime context
Artifacts
Task runners
Global concurrency limits
Tag-based concurrency limits
Deployments
Deployments
Schedules
Work pools
Workers
Configuration
Variables
Blocks
Settings and profiles
Prefect server
Telemetry
Automations
Events
Automations
Event triggers
Prefect Cloud
Rate limits and data retention
SLAs
Webhooks

좋아 👍
이 목록이 복잡해 보이지만, 사실은 **3개의 층(layer)**으로 나뉘어 있어.

> 1️⃣ 실행 모델 (코드 레벨)
> 2️⃣ 운영/배포 레벨
> 3️⃣ 클라우드/엔터프라이즈 기능

이 구조로 보면 갑자기 단순해진다.

---

# 🟢 1️⃣ 실행 모델 (코드 레벨)

이건 네가 실제로 Python 코드에서 만지는 것들.

### 🔹 Workflows

* Prefect에서 Flow + Task를 묶어 부르는 상위 개념

---

### 🔹 Flows

* 오케스트레이션 단위
* 실행 컨텍스트
* 상태 관리의 루트

```python
@flow
def retrain():
    ...
```

---

### 🔹 Tasks

* 실행 단위
* 재시도/상태/로그 관리 가능

```python
@task
def train():
    ...
```

---

### 🔹 States

* task/flow의 상태
* 성공, 실패, 재시도, 취소 등

---

### 🔹 Runtime context

* 현재 flow run 정보
* 실행 ID, 파라미터 등

---

### 🔹 Caching

* 같은 입력이면 다시 실행하지 않음
* ML 파이프라인에서 feature build 최적화에 유용

---

### 🔹 Task runners

* 병렬 실행 방식
* ThreadPool / ProcessPool / Dask 등

---

### 🔹 Global / Tag-based concurrency limits

* 동시에 몇 개 실행할지 제한
* GPU 1대면 동시에 train 1개만 실행 같은 설정

---

### 🔹 Artifacts

* 실행 결과 기록
* 모델 metrics 저장
* UI에 표시되는 결과물

---

# 🔵 2️⃣ 운영 / 배포 레벨

이건 “코드를 실제로 돌리게 하는 시스템” 관련.

---

### 🔹 Deployments

* Flow를 배포 가능한 객체로 정의
* 스케줄 + 파라미터 포함

---

### 🔹 Schedules

* 매일 새벽 1시 실행
* cron 기반

---

### 🔹 Work pools

* 어떤 환경에서 실행할지 정의
* Docker, Kubernetes, Local 등

---

### 🔹 Workers

* 실제로 task를 실행하는 프로세스

---

### 🔹 Configuration / Variables / Blocks

* 환경 설정 저장
* DB 연결 정보
* S3 자격증명
* Slack webhook 등

---

### 🔹 Prefect server

* 오픈소스 서버
* 실행 상태 저장
* UI 제공

---

# 🟣 3️⃣ 클라우드 / 엔터프라이즈 기능

Cloud 전용 기능들.

---

### 🔹 Prefect Cloud

* SaaS 버전

---

### 🔹 Rate limits & data retention

* 실행 기록 보관 기간
* API 제한

---

### 🔹 SLAs

* 특정 시간 내 실행 보장
* 실패 알림

---

### 🔹 Webhooks

* 외부 서비스 연동

---

### 🔹 Telemetry

* 사용 통계 수집

---

### 🔹 Events & Automations

* 특정 이벤트 발생 시 자동 작업
* 예: train 실패하면 Slack 알림

---

# 🎯 너 지금 단계에서 필요한 것

사실 지금 당장 필요한 건 딱 이것뿐이야:

* Flow
* Task
* submit / delay
* retry
* schedule
* worker

나머지는 나중 단계.

---

# 🔥 Prefect를 이렇게 생각하면 편하다

```
[코드 레벨]
Flow + Task

[운영 레벨]
Deployment + Worker + Schedule

[Cloud 레벨]
모니터링 + 알림 + SLA
```

---

# 🧠 한 줄 요약

이 목록은 복잡해 보이지만,

> Prefect는 “코드 실행 추상화 + 운영 추상화 + SaaS 관리 기능”
> 이 세 층으로 나뉜다.

---

지금 네 상태에서 제일 중요한 질문은 이거다:

👉 “내가 지금 당장 필요한 건 코드 레벨 기능인가, 운영 레벨 기능인가?”

이걸 명확히 하면 Prefect 도입 범위가 결정된다.

---

from prefect import flow, task
import httpx


@task(log_prints=True)
def get_stars(repo: str):
    url = f"https://api.github.com/repos/{repo}"
    count = httpx.get(url).json()["stargazers_count"]
    print(f"{repo} has {count} stars!")


@flow(name="GitHub Stars")
def github_stars(repos: list[str]):
    for repo in repos:
        get_stars(repo)


# run the flow!
if __name__ == "__main__":
    github_stars(["PrefectHQ/prefect"])