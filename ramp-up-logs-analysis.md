좋아, **로그를 보면 원인이 아주 명확하게 갈라져**.
결론부터 말하면 👉 **“ramp-up 중 서버가 죽은 이유는 공통 원인이 아니라, 파이프라인별로 ‘다른 실패 모드’ 때문”**이야.

아래는 **old_sync / new_sync / new_async 로그를 종합한 사망 원인 분석**이야.

---

## 1️⃣ old_sync — “살아있다 죽은 게 아니라, 응답 불능(hang)”

### 로그에서 보이는 핵심 패턴

* `/metrics`는 **계속 200 OK**
* 모델 로딩, 벡터 업로드, sentiment 처리까지 **정상 진행**
* **fatal error / traceback 없음**
* 마지막까지 **프로세스는 살아 있음** 

### 의미

* FastAPI 프로세스 ❌ 크래시 안 함
* 컨테이너 ❌ exit 안 함
* 하지만 Prometheus에서는 `context deadline exceeded`

👉 **전형적인 soft-death**

* 이벤트 루프가 막힘
* CPU 바운드 작업이 async 루프를 점유
* `/metrics`는 간헐적으로 응답하나, 스케줄링이 밀림

📌 **old_sync는 “서버는 살아 있는데 말을 못 하는 상태”**

---

## 2️⃣ new_sync — “모델 로딩 중 치명적 예외 → 프로세스 크래시”

### 결정적 로그

```
ERROR - Unhandled exception: [ONNXRuntimeError] : NO_SUCHFILE
Load model ... onnx/model.onnx failed. File doesn't exist
```

이게 핵폭탄이야. 

### 의미

* `xenova/paraphrase-multilingual-mpnet-base-v2`
* ONNX 모델을 로딩하려 했는데
* **다운로드/캐시가 완료되기 전에 로딩 시도**
* → **프로세스에서 unhandled exception 발생**
* → FastAPI 프로세스 종료
* → 컨테이너 DNS에서 사라짐
* → Prometheus: `no such host`

📌 **new_sync는 “명백한 hard-death (crash)”**

---

## 3️⃣ new_async — “new_sync와 동일한 크래시 패턴”

### 로그 패턴

* HuggingFace / Xenova 모델 요청 폭증
* 동일한 ONNX 경로 접근
* 동일한 시점에서 로딩 실패 후 중단 

### 차이점

* async 구조라서:

  * ramp-up 중 **동시에 여러 요청이 모델 로딩 트리거**
  * 캐시 race condition 가능성 ↑
  * 실패 확률 ↑

📌 **new_async 역시 hard-death**
하지만:

> **구조적으로 async가 더 빨리 죽었을 가능성 높음**

---

## 4️⃣ 왜 ramp-up “9분 만에” 죽는가?

이건 **정상적인 현상**이야. 이유는 이거야 👇

### ramp-up 특징

* 초반: 요청 적음 → 모델 로딩, 캐시 준비
* 중반: 요청 증가 → CPU 100%+
* 후반:

  * old_sync → 이벤트 루프 고착
  * new_sync / new_async → **아직 안정화 안 된 모델 캐시를 동시에 접근**

즉:

> **“모델 warm-up이 끝나기 전에, ramp-up이 모델 로딩 경쟁을 일으킨 것”**

---

## 5️⃣ 파이프라인별 사망 원인 요약 (아주 중요)

| 파이프라인     | 죽음 타입      | 원인                                       |
| --------- | ---------- | ---------------------------------------- |
| old_sync  | soft-death | CPU 바운드 작업으로 이벤트 루프 고착                   |
| new_sync  | hard-death | ONNX 모델 파일 missing → unhandled exception |
| new_async | hard-death | async 동시 로딩 → 캐시 race + 동일 에러            |

---

## 6️⃣ 이게 “실험 실패”냐? ❌

아니, **이건 실험 성공이다.**

왜냐면:

* **아키텍처별 failure mode가 다름**
* sync vs async의 차이가 **성능이 아니라 안정성**에서 드러남
* ramp-up이 왜 필요한지 **정확히 증명됨**

이건 보고서에 이렇게 쓸 수 있어:

> “Ramp-up 테스트 중,
> sync 파이프라인은 응답 불능 상태로 전이되었고,
> async 파이프라인은 모델 로딩 경쟁으로 프로세스 크래시가 발생했다.”

---

## 7️⃣ 다음 액션 (추천 순서)

1️⃣ **모델 warm-up 단계를 명시적으로 분리**

* 서버 시작 시 모델 preload
* 첫 요청에서 로딩 ❌

2️⃣ **ONNX / HF 캐시 경로를 컨테이너 볼륨으로 고정**

* race condition 제거

3️⃣ **ramp-up 전에 ‘idle warm phase’ 추가**

* 1~2분 가벼운 요청만

---

### 한 줄 결론

> **컨테이너를 죽였다 살려서 생긴 문제가 아니다.
> ramp-up이 “모델 로딩 경쟁 + CPU 고착”을 정확히 찔렀고,
> 파이프라인별로 전혀 다른 방식으로 죽었다.**

원하면 다음으로:

* 🔧 **모델 warm-up 코드 패턴**
* 🚦 **ramp-up 시나리오 설계 수정안**
* 🧪 **이 결과를 실험 리포트 문장/표로 정리**

바로 이어서 가줄게.
