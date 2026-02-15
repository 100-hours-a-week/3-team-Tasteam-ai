# RunPod Serverless 로그 트러블슈팅 요약

`runpod_serverless_logs` 디렉터리에서 수집한 RunPod Serverless 엔드포인트 사용 시 트러블슈팅을 한 문서로 정리한 요약입니다. 개별 상세는 각 `runpod_serverless_failN.log` 및 상위 RunPod 문서를 참고하세요.

---

## 1. 로그 파일 구분

| 유형 | 예시 | 내용 |
|------|------|------|
| **앱 서버 로그** | `runpod_serverless_fail1.log` ~ `fail8.log` | 로컬 tasteam API 서버가 **RunPod Serverless** 엔드포인트로 LLM 요청 시 기록한 로그. (벡터/Sentiment warm-up, Serverless 호출·실패·재시도 등) |
| **엔드포인트** | 로그 내 `vLLM RunPod Serverless: 엔드포인트 g09uegksn7h7ed` | 당시 사용한 Serverless 엔드포인트 ID. 런마다 동일하거나 바뀔 수 있음. |

---

## 2. 트러블슈팅 타임라인 (요약)

| 런 | 시점 | 엔드포인트 | 비고 |
|----|------|------------|------|
| **fail1** | 2026-02-13 13:50 | g09uegksn7h7ed | Serverless 사용 시작 |
| **fail2** | 13:55 | g09uegksn7h7ed | Sentiment 재판정·Summary·Comparison JSON 파싱 실패 다수 |
| **fail3** | 14:03 | g09uegksn7h7ed | |
| **fail4** | 14:08 | g09uegksn7h7ed | |
| **fail5** | 14:13 | g09uegksn7h7ed | |
| **fail6** | 14:28 | g09uegksn7h7ed | |
| **fail7** | 14:43 | 2mpd5y6lvccfk1 | 엔드포인트 전환 |
| **fail8** | 15:03 | 2mpd5y6lvccfk1 | |

상세 타임스탬프·에러 메시지는 각 `../runpod_serverless_logs/runpod_serverless_failN.log` 에서 확인.

---

## 3. RunPod Serverless 관련 공통 이슈

### 3.1 Serverless 특성 (Ephemeral)

- **요청 없으면 워커 종료** → Cold start, 첫 요청 지연·타임아웃 가능.
- **요청 처리 중 워커 종료** → 500 등 비정상 응답.
- **Prometheus 스크래핑** → 타겟이 항상 존재하지 않아 일반적인 pull 방식에 불리. (상세: `docs/runpod/why_dont_use_runpod_serverless.md`)

### 3.2 500 / 연결 실패

- Serverless 워커 내부(vLLM)가 요청 처리 중 죽거나, cold start 지연으로 클라이언트 타임아웃이 나는 경우.
- **조치**: 재시도(클라이언트 쪽), 요청 크기·동시성 완화, 또는 **Pod** 전환 검토. (현재 프로젝트는 Pod 사용, Serverless 미사용.)

### 3.3 엔드포인트 전환

- fail1~6 은 `g09uegksn7h7ed`, fail7~8 은 `2mpd5y6lvccfk1` 사용. 설정·엔드포인트 재생성 등으로 ID가 바뀐 경우.

### 3.4 LLM 응답 JSON 파싱 실패 (Sentiment 재판정 / Summary / Comparison)

- **증상**: `Expecting property name enclosed in double quotes`, `Unterminated string` 등으로 JSON 파싱 실패. API는 200이지만 1차 분류 결과 사용·요약/비교 fallback 발생.
- **원인**: Serverless vLLM이 마크다운 래핑·불완전 응답을 반환하거나, 응답이 잘려서 유효하지 않은 JSON이 됨. (Pod 쪽에서 적용한 대응: 마크다운 제거·재시도·토큰 cap 등은 `runpod_pod/trouble_shooting/TROUBLESHOOTING_SUMMARY.md` 3.4 참고.)
- **조치**: 동일한 방어 로직(파싱 전 정규화·재시도)을 Serverless 호출 경로에도 적용하거나, 안정성 우선 시 **Pod** 사용 권장.

---

## 4. 참고 문서

| 문서 | 설명 |
|------|------|
| [runpod_serverless_fail1.log](../runpod_serverless_logs/runpod_serverless_fail1.log) ~ fail8.log | 런별 앱 서버 로그 (원문) |
| [why_dont_use_runpod_serverless.md](../../../../runpod/why_dont_use_runpod_serverless.md) | Serverless 미사용 이유, Prometheus·Ephemeral 정리 |
| [what_runpod_network_volume.md](../../../../runpod/what_runpod_network_volume.md) | Network Volume (Serverless에서의 제약 일부 언급) |

RunPod **Pod** 트러블슈팅 요약: [runpod_pod/trouble_shooting/TROUBLESHOOTING_SUMMARY.md](../../runpod_pod/trouble_shooting/TROUBLESHOOTING_SUMMARY.md)
