좋아.
**RunPod Pod(온디맨드 GPU Pod)**을 **CLI / REST API로 생성(start)·중지(stop)·삭제(terminate)** 하는 방법을 정리해줄게.

---

# 1️⃣ REST API 방식 (GraphQL 기반)

RunPod는 **GraphQL API**를 사용해.
엔드포인트:

```
https://api.runpod.io/graphql
```

요청 시 헤더:

```http
Authorization: Bearer <RUNPOD_API_KEY>
Content-Type: application/json
```

---

## ✅ 1. Pod 생성 (Start)

### 🔹 예시: GPU Pod 생성

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { podFindAndDeployOnDemand( input: { name: \"my-llm-pod\", imageName: \"vllm/vllm-openai:v0.11.0\", gpuTypeId: \"NVIDIA A100-SXM4-80GB\", cloudType: SECURE, containerDiskInGb: 50, volumeInGb: 100, ports: \"8000/http\" } ) { id name imageName desiredStatus } }"
  }'
```

### 주요 파라미터

| 필드                  | 설명         |
| ------------------- | ---------- |
| `imageName`         | Docker 이미지 |
| `gpuTypeId`         | GPU 타입     |
| `containerDiskInGb` | 컨테이너 디스크   |
| `volumeInGb`        | 네트워크 볼륨    |
| `ports`             | 노출 포트      |

---

## ✅ 2. Pod 중지 (Stop)

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { podStop(input: { podId: \"POD_ID\" }) { id desiredStatus } }"
  }'
```

`desiredStatus: STOPPED` 로 바뀜

---

## ✅ 3. Pod 재시작 (Start existing)

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { podResume(input: { podId: \"POD_ID\" }) { id desiredStatus } }"
  }'
```

---

## ✅ 4. Pod 완전 삭제 (Terminate)

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation { podTerminate(input: { podId: \"POD_ID\" }) }"
  }'
```

이건 **리소스 완전 삭제 + 과금 종료**

---

## ✅ 5. Pod 목록 조회

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ myself { pods { id name desiredStatus runtime { uptimeInSeconds } } } }"
  }'
```

---

# 2️⃣ RunPod CLI 방식

### 🔹 설치

```bash
pip install runpod
```

### 🔹 로그인

```bash
runpod config set apiKey YOUR_API_KEY
```

---

## Pod 목록 확인

```bash
runpod pods list
```

---

## Pod 중지

```bash
runpod pods stop POD_ID
```

---

## Pod 시작

```bash
runpod pods start POD_ID
```

---

## Pod 삭제

```bash
runpod pods delete POD_ID
```

---

# 🔥 실전에서 많이 쓰는 구조

너처럼 AWS + RunPod 연동하는 경우라면 보통:

```
EventBridge → Lambda → RunPod GraphQL 호출
```

Lambda에서:

* 요청 많아지면 → podResume
* N분 idle → podStop
* 배치 작업 끝나면 → podTerminate

이런 식으로 자동화함.

---

# ⚠️ Stop vs Terminate 차이

| 동작        | 과금        | 볼륨 유지  |
| --------- | --------- | ------ |
| stop      | GPU 과금 중단 | 유지     |
| terminate | 완전 삭제     | 볼륨만 남김 |

LLM 모델을 네트워크 볼륨에 두면
보통 **stop 방식이 재시작 속도 면에서 유리**

---

# 💡 너 프로젝트 관점에서 정리

네가 고민 중인 구조라면:

* 실시간 LLM → `stop/resume`
* 배치 전용 GPU → `deploy → 작업 → terminate`
* last_request_at 기반 idle shutdown → `podStop`

이게 제일 합리적.

---

원하면:

* ✅ Python으로 Lambda에서 호출하는 예제 코드
* ✅ idle 자동 종료 watchdog 예제
* ✅ FastAPI에서 GPU 자동 스케일 제어 코드

중 어떤 걸 더 자세히 볼래?
