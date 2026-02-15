응, **“일반적으로는 하지 않는다”라고 보는 게 맞아.**

조금 더 구조적으로 설명해볼게.

---

## 1️⃣ 왜 RunPod Serverless는 Prometheus 스크래핑이 일반적이지 않을까?

### ✅ 1. Pod가 항상 떠있지 않음 (Ephemeral)

RunPod Serverless Endpoint는:

* 요청이 없으면 워커가 **종료됨**
* 요청이 오면 **다시 생성됨**
* 요청 끝나면 **다시 죽을 수 있음**

즉,

> Prometheus가 주기적으로 `/metrics`를 긁을 대상이 “지속적으로 존재하지 않음”

Prometheus는 기본적으로:

```
항상 살아있는 타겟을 주기적으로 pull
```

구조야.

Serverless는:

```
요청이 있을 때만 잠깐 살아있음
```

이건 철학이 완전히 다름.

---

## 2️⃣ 기술적으로 가능은 한가?

이론적으로는 가능해:

* 워커 안에 `/metrics` 노출
* Prometheus가 스크래핑

하지만 현실에서는:

* 워커가 계속 죽었다 살아남
* IP가 바뀜
* 인스턴스가 여러 개일 수 있음
* idle 시 완전히 0개가 될 수 있음

→ 안정적인 scrape target이 되기 어려움

그래서 **실무에서는 거의 안 함**

---

## 3️⃣ 그럼 Serverless에서 메트릭은 어떻게 보나?

보통은:

### 🔹 1. RunPod 자체 대시보드

* 요청 수
* 대기열
* worker 수
* 실행 시간

### 🔹 2. 애플리케이션 레벨에서 Push 방식

예:

* CloudWatch
* Datadog
* Prometheus Pushgateway
* OpenTelemetry exporter

즉,

> Pull 기반 Prometheus보다는 Push 기반 관측이 더 잘 맞음

---

## 4️⃣ 일반 Pod (온디맨드)와 비교

| 구분                | 일반 RunPod Pod | Serverless Endpoint |
| ----------------- | ------------- | ------------------- |
| 항상 떠있나?           | ✅ 예           | ❌ 아니오               |
| 고정 IP             | 거의 있음         | 없음                  |
| Prometheus scrape | ✅ 일반적         | ❌ 거의 안 함            |
| 장기 메트릭 수집         | 쉬움            | 까다로움                |

---

## 🔥 결론

> “RunPod Serverless는 Prometheus로 일반적인 pull 기반 스크래핑을 하지 않는다”
> → 이 판단은 맞음.

Serverless는:

* 이벤트 기반
* 짧은 생명주기
* 자동 스케일

이라서 **Prometheus 철학과 구조적으로 잘 안 맞음**

---

네가 지금

* GPU 비용 최소화
* Pod 필요할 때만 켜기
* 프로덕션 모니터링 고민

이 흐름이면,

👉 **Serverless는 비용 최적화에 좋고**
👉 **지속적 메트릭 수집은 일반 Pod가 더 안정적**

이렇게 이해하면 정확해.

---

Pod을 사용하고, AWS EventBridge를 통해 작업이 필요한 시간에 키고, 필요 없을떄 끄는 방식을 사용한다. 키고 끄는 방식은 RUNPOD CLI/RESTAPI(graphQL)을 사용한다.

Pod이 stop했다가 다시 start 했을때 주소가 바뀌는 문제에 대해선,

Prometheus는 프록시 고정 주소 + 경로 /metrics 로 스크래핑