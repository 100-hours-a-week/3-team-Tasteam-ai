# RunPod Pod 로그 트러블슈팅 요약

`runpod_pod_logs` 디렉터리에서 진행한 트러블슈팅을 하나의 문서로 정리한 요약입니다. 개별 상세는 각 `*_trouble.md` 및 `260214_*.md`를 참고하세요.

---

## 1. 로그 파일 구분

| 유형 | 예시 | 내용 |
|------|------|------|
| **앱 로그** | `0747_app_log.log`, `0723_app_server_log.log` | 로컬 tasteam API 서버 (벡터/Sentiment/요약/비교, RunPod 호출) |
| **컨테이너 로그** | `0747_container_log.log`, `0657_container_log.log` | RunPod Pod 내 vLLM 서버 (`/v1/chat/completions` 등) |
| **트러블 문서** | `0533_trouble.md` ~ `0747_trouble.md` | 해당 런 기준 분석·이슈·조치 |
| **일자별** | `260214_trouble.md`, `260214_network_volume_trouble.md` | 2026-02-14 전반, 네트워크 볼륨/Stop 정책 |

---

## 2. 트러블슈팅 타임라인 (요약)

| 런/문서 | 시점 | 상태 | 주요 이슈 |
|---------|------|------|------------|
| **0533** | 2026-02-14 17:33 | 트러블 | Sentiment/Summary max_tokens 400, Summary 입력 4731 초과, KeyError 'service' |
| **0647** | 18:46 | 트러블 | Sentiment 948 / Summary 540 → 입력 토큰 추정이 실제보다 작음 |
| **0657** | 18:57 | 트러블 | Sentiment 배치 JSON 파싱 실패 1건 (62개 리뷰, restaurant_id=4) |
| **0705** | 19:06 | 트러블 | 동일 JSON 파싱 실패, 재시도 조건(len≤3000) 검토 |
| **0723** | 19:23 | 트러블 | 동일 JSON 파싱 실패; **감성 1차 점수** 62개 전부 0.002로 나오는 문제 발견 |
| **0747** | 19:47 | **No trouble** | 전 구간 200 OK, Sentiment 0.3~0.8 구간 적용 확인 |
| **260214** | (일자) | 트러블 | RunPod **모델 ID 404** (qwen/qwen2.5 vs 경로 기반 ID 불일치), Spark PYTHON_VERSION_MISMATCH |
| **260214_network_volume** | (참고) | 문서화 | Pod Stop 없음(Terminate만), Network Volume 비용/운영 정책 |

---

## 3. 이슈별 정리 및 조치

### 3.1 RunPod vLLM 모델 ID 404 (260214)

- **증상**: `POST /v1/chat/completions` → **404**, `The model 'qwen/qwen2.5-7B-Instruct' does not exist.`
- **원인**: vLLM은 **경로**로 모델 로드(`/workspace/llm-models/Qwen/Qwen2.5-7B-Instruct`). 클라이언트는 `model: "qwen/qwen2.5-7B-Instruct"` 로 요청 → ID 불일치.
- **조치**: RunPod에서 `GET /v1/models` 로 실제 모델 `id` 확인 후, tasteam 설정(Config/LLM 호출 시 model 이름)을 그 ID와 동일하게 변경. (예: 경로 문자열 사용)

---

### 3.2 Sentiment / Summary max_tokens 400 (0533, 0647)

- **증상**: vLLM이 `max_tokens too large: 948` (또는 540 등) 로 400 반환. 입력 3205면 허용 출력 891인데 948 전송 등.
- **원인**: 입력 토큰 추정이 **실제보다 작게** 나와서, cap이 서버 허용값(4096 − input − 여유)보다 크게 설정됨.
- **조치**:
  - **llm_utils**: `_estimate_input_tokens` 를 더 보수적으로 (예: `(total*2)//3` → `(total*3)//4`). cap은 `min(절대상한, 4096 − 추정입력 − 256)`.
  - **summary_pipeline**: payload가 4096을 넘지 않도록 토큰 추정 보수화·잘라내기 강화.
  - Sentiment 쪽 cap이 1 미만이면 최대 512 등으로 제한해 400 방지.

---

### 3.3 Summary 입력 4731 초과·KeyError 'service' (0533)

- **증상**: 요청 입력이 4731 토큰으로 4096 초과 → 400; 또는 LLM 응답에 `service` 키 없음 → `out["service"]` 접근 시 KeyError, 배치 500.
- **조치**:
  - `_payload_within_context_limit` / 토큰 추정으로 4096 이하로 잘라내기.
  - `out.setdefault(cat, {"summary": "", "bullets": [], "evidence": []})` 등으로 키 없어도 안전 처리.

---

### 3.4 Sentiment: 근본 원인과 파생 현상

Sentiment 관련해서 보였던 현상은 **하나의 근본 원인(3.5)** 에서 나온 것이다.

- **근본 원인(1번)**  
  감성 1차 분류에서 **긍정 점수를 잘못 읽음**(3.5). 인덱스 1을 긍정으로 가정했는데, 파이프라인 출력 순서에 따라 부정 점수(≈0.002)를 읽게 됨 → 모든 리뷰가 threshold를 넘지 못한 것처럼 처리됨.

- **파생 1: 전부 재판정(3.6처럼 보인 현상)**  
  “0.8 초과만 positive, 나머지는 전부 재판정” 로직이었고, 점수를 잘못 읽어서 **실제로는 전부** 재판정 대상이 됨. 구간 로직이 없어서가 아니라, **점수 오류** 때문에 62개 전부가 재판정으로 간 것.

- **파생 2: JSON 파싱 실패(3.4)**  
  restaurant 1,2,3,5는 각 20개씩, 4는 62개를 **전부** 재판정으로 보냄. 20개 단위는 입력+출력 **총 토큰이 한도 안**이라 LLM이 끝까지 응답하고 올바른 JSON 배열을 반환 → 파싱 성공. **62개만** 한 번에 보내서 총 토큰이 커지고, 응답이 잘리거나 형식이 깨짐 → 파싱 실패.  
  즉, “다른 레스토랑은 threshold를 넘긴 게 많아서”가 아니라, **전부 재판정으로 보냈지만** 20개는 토큰 한도 안이라 잘 나온 것이고, 62개만 한도 초과로 LLM이 온전히 분류·응답하지 못한 것이다.

- **3번(JSON 파싱 강화)에 대해**  
  1번(긍정 점수 레이블 기준 수정)을 먼저 충족했다면, 62개 전부 재판정으로 가지 않아 재판정 요청·응답 크기가 작아지고, JSON 파싱 실패 자체가 거의 나오지 않았을 가능성이 크다. 따라서 **3번(마크다운 제거·재시도 등)은 1번을 고친 뒤에는 “필수”로 취할 이유가 없었던 조치**이다. 다만 LLM이 가끔 비표준 응답을 줄 수 있으므로, 방어적으로 3번 조치는 유지하는 것을 권장한다. 적용한 기술적 조치(마크다운 제거·파싱 실패 시 1회 재시도 등)는 아래 **4. 적용한 코드/설정 변경 요약** 참고.

---

### 3.5 감성 1차 분류 – 긍정 점수 반대로 나옴 (0723) **[근본 원인]**

- **증상**: test_data_sample restaurant_id=4 기준 **62개 전부** positive_score ≈ 0.002, 전부 LLM 재판정 대상으로 나옴. (실제 리뷰는 대부분 긍정.)
- **원인**: HF 파이프라인 출력 순서가 **점수 순** 등으로 바뀌는 경우, `out[1]`이 긍정이 아니라 **부정** 점수일 수 있음. 즉, **인덱스 1을 긍정으로 가정**한 것이 잘못됨.
- **조치**:
  - **레이블 문자열로 긍정 점수 선택**: 출력 리스트에서 `_map_label_to_binary(item["label"]) == "positive"` 인 항목의 `score` 를 사용. (`sentiment_analysis.py` 및 `scripts/test_sentiment_threshold.py` 동일 적용.)

---

### 3.6 LLM 재판정 대상 구간 (0723 → 0747) **[3.5의 파생; 구간 자체는 유효한 개선]**

- **당시 현상**: positive_score ≤ 0.8 인 **전부**를 LLM 재판정 대상으로 넣어서, 62개가 모두 재판정으로 감. (원인은 3.5로, 점수를 잘못 읽어 전부 “미달”로 처리된 것.)
- **조치**: 구간 기준 적용.
  - **positive_score > 0.8** → positive, 재판정 없음.
  - **0.3 ≤ positive_score < 0.8** → 1차 negative, **이 구간만** LLM 재판정 대상에 추가.
  - **positive_score < 0.3** → 확정 negative, 재판정 대상에 **넣지 않음**.
- **결과**: 0747에서 restaurant_id=4는 62개 중 **4개만** 재판정, 나머지는 positive 또는 확정 negative로 처리.

---

### 3.7 RunPod Network Volume / Stop (260214_network_volume_trouble.md)

- **내용**: Network Volume을 붙인 Pod은 **Stop 불가, Terminate만 가능**. Stop/Start vs Terminate+재배포·Network Volume 비용($0.07/GB·월 등) 정리. 데이터 보존·GPU 갈아끼우기에는 Network Volume이 유리하다는 설명.

---

## 4. 적용한 코드/설정 변경 요약

| 구분 | 파일 | 변경 요약 |
|------|------|------------|
| 모델 ID | Config / LLM 호출부 | RunPod `GET /v1/models` 의 `id`와 동일한 값 사용 |
| 입력 토큰 추정 | `llm_utils.py` | `(total*3)//4` 등 보수적 추정, cap 계산 시 context 기반 상한 적용 |
| Summary payload | `summary_pipeline.py` | 4096 초과 방지, KeyError 방지(`setdefault` 등) |
| Sentiment JSON | `sentiment_analysis.py` | `_parse_sentiment_json`, 마크다운 제거, 파싱 실패 시 1회 재시도(길이 제한 제거) |
| 긍정 점수 선택 | `sentiment_analysis.py` | `out[1]` 대신 레이블이 "positive"인 항목의 score 사용 |
| 재판정 구간 | `sentiment_analysis.py` | 0.3 미만 확정 negative 제외, 0.3~0.8만 LLM 재판정 |
| 테스트 스크립트 | `scripts/test_sentiment_threshold.py` | `--from-test-data 4`, 레이블 기반 점수, 구간별 개수 출력 |

---

## 5. 참고 문서 목록

| 문서 | 설명 |
|------|------|
| `0533_trouble.md` | Sentiment/Summary 400, 4731, KeyError 'service' |
| `0647_trouble.md` | max_tokens 948/540, 입력 추정 보수화 |
| `0657_trouble.md` | Sentiment JSON 파싱 실패, _parse_sentiment_json·재시도 도입 |
| `0705_trouble.md` | 동일 JSON 실패, 재시도 조건(len 제한 제거) |
| `0723_trouble.md` | JSON 실패 유지, 감성 점수 레이블 순서·재판정 구간 수정 |
| `0747_trouble.md` | **No trouble** – 전 구간 정상, 0.3~0.8 구간 적용 확인 |
| `260214_trouble.md` | RunPod 모델 ID 404, Spark 버전 불일치 |
| `260214_network_volume_trouble.md` | Network Volume·Stop/Terminate·비용 정책 |

위 문서들에 상세 로그 인용·조치 내역이 있으므로, 특정 런이나 이슈는 해당 파일을 참고하면 됩니다.
