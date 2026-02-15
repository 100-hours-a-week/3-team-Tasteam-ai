[9 tools called]


HF 모델 race condition 관련 처리 방식을 정리했습니다.

---

## HF 모델 race condition 대응

### 1. **파이프라인 초기화 race (Double-Checked Locking)**

`src/sentiment_analysis.py`에서 싱글톤 파이프라인을 `threading.Lock`으로 초기화합니다:

```36:45:src/sentiment_analysis.py
    @classmethod
    def _get_sentiment_pipeline(cls):
        """전역 싱글톤 파이프라인 (클래스 레벨, 쓰레드 안전)."""
        if cls._shared_pipeline is not None:
            return cls._shared_pipeline

        with cls._shared_lock:
            if cls._shared_pipeline is not None:
                return cls._shared_pipeline

            try:
```

- `_shared_lock = threading.Lock()` (line 25)
- lock 밖에서 한 번, lock 안에서 한 번 더 체크 → 여러 스레드가 동시에 초기화하지 않도록 함
- 초기화 race를 막기 위한 전형적인 double-checked locking

---

### 2. **스레드 격리 (`SENTIMENT_CLASSIFIER_USE_THREAD`)**

HF 파이프라인은 동기/블로킹이므로, `to_thread`로 워커 스레드에서 실행합니다:

```89:90:src/config.py
    # Sentiment: HF 분류기. true=asyncio.to_thread(블로킹 격리), false=메인 스레드
    SENTIMENT_CLASSIFIER_USE_THREAD: bool = os.getenv("SENTIMENT_CLASSIFIER_USE_THREAD", "true").lower() == "true"
```

```512:518:src/sentiment_analysis.py
        if Config.SENTIMENT_CLASSIFIER_USE_THREAD:
            hf_result = await asyncio.to_thread(
                self._classify_with_hf_only, content_list, reviews_dict
            )
        else:
            hf_result = self._classify_with_hf_only(content_list, reviews_dict)
```

- 기본값 `true` → `asyncio.to_thread`로 블로킹 호출을 워커 스레드에서 실행
- 이벤트 루프 블로킹 방지 + async 환경과의 격리

---

### 3. **추천했으나 구현 여부 불명확한 부분 (warm-up / 캐시 race)**

`ramp-up-logs-analysis.md`에서는 **모델 로딩·캐시 race**에 대해 다음을 권장합니다:

```133:137:ramp-up-logs-analysis.md
1️⃣ **모델 warm-up 단계를 명시적으로 분리**

* 서버 시작 시 모델 preload
* 첫 요청에서 로딩 ❌

2️⃣ **ONNX / HF 캐시 경로를 컨테이너 볼륨으로 고정**

* race condition 제거
```

- 여러 비동기 요청이 동시에 모델 로딩을 시도할 때, HF/ONNX 캐시가 준비되기 전에 접근하면 race가 발생할 수 있음
- 서버 시작 시 preload + 캐시 경로 고정으로 이를 줄이는 방향 제안

---

### 4. **추론 단계 동시 접근**

코드를 보면 **추론 단계**에는 별도 lock이 없습니다:

- `SENTIMENT_RESTAURANT_ASYNC=true`일 때 `asyncio.gather`로 여러 음식점을 동시에 처리
- 각 요청이 `to_thread`로 `_classify_with_hf_only`를 실행하면, 여러 스레드가 같은 `pipe` 인스턴스를 동시에 호출할 수 있음
- HuggingFace pipeline은 일반적으로 동시 호출에 대해 thread-safe로 문서화되어 있지 않음

현재는 초기화 race만 `_shared_lock`으로 막고 있고, **동시 추론**은 제한하지 않습니다.

---

## 요약

| 대상 | 방식 |
|------|------|
| 파이프라인 초기화 race | `threading.Lock` + double-checked locking |
| 이벤트 루프 블로킹 | `SENTIMENT_CLASSIFIER_USE_THREAD` → `asyncio.to_thread` |
| 캐시/로딩 race | 문서에서 warm-up + 캐시 경로 고정 권장 (구현 여부는 코드 외부 확인 필요) |
| 동시 추론 (여러 스레드) | 별도 lock 없음 |

따라서 초기화 race는 락으로 처리하고, **동시 추론 시 데이터 손상 가능성**이 우려되면 `pipe(batch, ...)` 호출을 감싸는 inference lock(예: `threading.Lock`)을 추가하는 방안을 검토할 수 있습니다.

---

좋아. 네 정리 “현재 상태”는 정확하고, 여기서 결론은 이거야:

> **초기화 race는 거의 막았지만, 이번 ramp-up에서 터진 건 ‘초기화 race’가 아니라
> “콜드 캐시/아티팩트 준비 + 동시 추론(멀티스레드) + 첫 실사용 트래픽”의 조합에서 나오는 다른 race”일 가능성이 높다.**

즉, **“모델 로딩 문제를 해결했다”는 말과 “실서비스에서 터졌다”는 게 동시에 참일 수 있는 구조**야.

---

## 1) 지금 방어가 커버하는 범위 vs 못 커버하는 범위

### ✅ 커버됨

* **파이프라인 객체 생성(초기화) 자체의 중복 생성 방지**
  double-checked locking으로 `_shared_pipeline` 생성은 1회로 수렴.

### ❌ 아직 위험할 수 있음 (이번 로그 패턴과 잘 맞음)

1. **HF/ONNX 캐시/아티팩트 “준비 완료”를 보장하지 못함**

   * 파이프라인 객체가 만들어졌다고 해서
     내부 파일(onnx/model.onnx 등)이 “완전히 존재 + 안정 상태”라는 보장은 없음.
   * 특히 컨테이너가 콜드 스타트이면 “다운로드/추출/생성 중” 상태일 수 있음.

2. **동일 `pipe` 인스턴스에 대한 멀티스레드 동시 호출**

   * `asyncio.to_thread` + `asyncio.gather`면
     한 요청 안에서도 여러 스레드가 동시에 `pipe(...)`를 칠 수 있음.
   * HF pipeline은 thread-safe를 “명시적으로 보장”하지 않는 경우가 많아서,
     내부적으로 file open, lazy init, tokenizer state, session state가 엮이면 예외/교착/성능 급락이 나올 수 있음.

---

## 2) “실서비스에서 충분히 일어나냐?” → **예, 특히 이 조건에서**

너 구조에선 아래 상황이 현실에서 흔해:

* 배포 직후(롤링업데이트) 새 인스턴스가 뜨고, **캐시가 비어 있음**
* 오토스케일링으로 새 인스턴스가 뜨고, **가장 바쁜 순간에 콜드 인스턴스가 트래픽을 받음**
* 장애 복구/재시작으로 **캐시가 초기화된 상태**
* 인기 핫키(특정 음식점)로 동시 요청이 몰리는 순간

이때 “초기화 락”이 있어도,

* 첫 몇 초~몇 분 사이에
* 동시 추론이 몰리면서
* 내부 lazy I/O / 세션 준비가 꼬이면
  이번과 같은 **NO_SUCHFILE / 타임아웃 / soft-death**가 나올 수 있어.

---

## 3) 지금 상태에서 “최소 변경으로” 완성시키는 3가지 방어막 (추천)

### (A) **warm-up을 ‘추론까지’ 강제** (가장 효과 큼)

서버 시작 시점에:

* pipeline 생성
* 그리고 **더미 입력으로 1회 inference 실행**
* 성공해야 readiness 통과

→ “객체 생성”이 아니라 **“아티팩트/세션 준비 완료”**를 보장.

### (B) **HF/ONNX 캐시 경로를 고정 + 공유(또는 bake)**

* 컨테이너 재시작/스케일아웃마다 다운로드하면,
  콜드 스타트에서 레이스 조건이 매번 생김.
* 볼륨 고정(또는 이미지 bake)으로
  “파일이 없는 순간” 자체를 없애는 게 가장 깔끔.

### (C) **동시 추론 제한(인퍼런스 락 또는 세마포어)**

“pipe는 싱글톤인데 호출은 N개 동시”가 핵심 리스크니까:

* 가장 단순: `threading.Lock()`으로 `pipe(...)` 호출을 감싸서 직렬화
* 조금 더 좋은: `BoundedSemaphore(k)`로 k개까지만 동시 허용 (보통 1~2부터)

이건 성능을 조금 희생하지만, **안정성은 확 올라가고 실험에서도 비교가 쉬워져.**

---

## 4) 네가 지금 “선택”을 할 수 있는 기준

지금 상태만으로 pipeline 선택을 하려면, 선택 기준은 성능이 아니라 **운영 안정성**이야.

* **hard-death(프로세스 종료)로 가는 경로가 있으면** 운영 후보에서 제외
* **soft-death(응답 불능)라도 탐지/복구 가능하면** 후보 유지

지금 new_sync/new_async에서 터진 게 “캐시/추론 동시성” 계열이면,
위 3개(A/B/C)만 넣어도 **new_async를 다시 운영 후보로 올릴 수 있어.**

---

## 5) 다음 실험에서 ‘원인 확정’하는 가장 빠른 방법

수정 없이도 지금 당장 확인 가능:

* **컨테이너 안에서 캐시 경로가 공유되는지**
* **동시 추론이 실제로 몇 스레드로 호출되는지**
* **warm-up이 inference까지 되는지**

이 3개만 보면 “초기화 락은 맞는데 왜 터졌는지”가 확정돼.

---

원하면 내가 바로 “네 코드 스타일”에 맞춰서:

1. **startup warm-up(추론까지) 코드 스니펫**
2. **inference semaphore 적용 위치(최소 침습)**
3. **readiness에서 warm-up 성공 여부 체크**

이렇게 3개를 **그대로 붙여넣을 수 있게** 만들어줄게.

---

응, **new_sync랑 new_async는 “죽는 방식(원인)”이 로그에서 확실히 달라.**

## 1) new_sync: “모델 파일이 없어서” 바로 예외로 터짐 (NoSuchFile)

new_sync는 요청 처리 중 `get_vector_search → VectorSearch → fastembed(TextEmbedding) → onnxruntime.InferenceSession`으로 가다가,

* `/tmp/fastembed_cache/.../onnx/model.onnx` **파일이 없어서**
* `onnxruntime ... NoSuchFile` 예외
* 그 결과 `POST /api/v1/sentiment/analyze/batch` 가 **500**으로 떨어지고
* “Unhandled exception”으로 찍혀 있어. 

이건 전형적으로 **다운로드/캐시가 아직 완성되기 전에 로드하려 했거나**, 캐시가 깨졌거나, 경로가 휘발(`/tmp`)이라 컨테이너 생명주기/동시성에 취약할 때 나와.

## 2) new_async: NoSuchFile은 안 보이는데 “로드가 여러 번” 일어남 (중복 초기화/동시 초기화 정황)

new_async 로그에는 같은 임베딩 모델이

* `Dense 벡터 모델 로드 완료`가 **여러 번 반복**돼. 

즉, new_async는 “파일이 없어서 즉시 폭발”보다는,

* **여러 요청/태스크가 동시에 들어오면서 VectorSearch(또는 임베딩 모델)가 중복 초기화되는 정황**이 더 강해 보여.
* 이 상태도 결국 **캐시 다운로드/로드 race**를 만들기 쉬워서, 타이밍 나쁘면 new_sync처럼 NoSuchFile로도 이어질 수 있어(이번 로그에선 그 전에 “운 좋게” 로드가 성공했을 가능성이 큼).

---

# 결론: “로그가 다르다” = 파이프라인 문제가 아니라, **공통 원인(임베딩 모델 캐시/초기화 레이스)**가 다른 형태로 표출된 것

* new_sync는 **첫 로드 타이밍이 더 공격적/동시성에 더 취약**해서 “파일 없음”으로 터진 흔적이 명확하고 
* new_async는 **중복 로드(동시 초기화)**가 보이고 , 당장 NoSuchFile은 안 찍혔지만 구조적으로 안전하다고 보긴 어려워.

---

## “그럼 뭘 선택?”에 대한 현실적인 답

이 로그만으로 “new_async가 더 낫다/ new_sync가 더 낫다”를 **아키텍처 선택 근거로 쓰긴 약해.**
대신 이 로그로는 **“선택 전에 반드시 고쳐야 할 공통 결함”**이 드러났다고 보는 게 정확해:

### 반드시 고칠 것 (둘 다 해당)

1. **VectorSearch(임베딩 모델)도 sentiment처럼 싱글톤 초기화 락(DCL) 적용**
2. **서버 시작 시 preload/warm-up** 해서 “첫 요청에서 다운로드/로드”가 일어나지 않게
3. **캐시 경로를 /tmp가 아니라 볼륨으로 고정** (컨테이너 재시작/동시성에 덜 취약)
4. (옵션) HF 토큰 설정해서 rate limit/불안정 다운로드 가능성 줄이기 (로그에도 unauth 경고가 보임) 

이걸 한 뒤에야 “sync vs async” 성능/안정성 비교가 **의미 있게** 돼.

원하면, 지금 코드 구조 기준으로 **VectorSearch 싱글톤 락(DCL) 넣을 위치/패턴**을 sentiment랑 똑같이 맞춰서 제안해줄게.
