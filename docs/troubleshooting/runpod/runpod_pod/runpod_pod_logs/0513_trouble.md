[3 tools called]


앱 서버 로그는 확인했습니다. Pod 컨테이너 로그는 앞부분과 에러만 확인합니다.


[2 tools called]


0513 로그만 기준으로 정리했습니다.

---

# RunPod Pod 로그 분석 (0513)

## 1. 파일

| 파일 | 내용 |
|------|------|
| **0513_app_server_log.log** | 로컬 앱 서버(tasteam) 로그, 2026-02-14 17:12~17:14 |
| **0513_pod_container_log.log** | RunPod 컨테이너(vLLM) 로그, 기동~API 서버 준비 구간 |
| **0513_** | 없음(또는 빈 파일) |

---

## 2. 앱 서버 로그 (0513_app_server_log.log)

### 정상 동작

- 앱 기동, 벡터/임베딩/Sentiment warm-up 성공  
- RunPod Pod 연결: `http://213.173.108.70:17517/v1`  
- 벡터 업로드 142개, 레스토랑 5개 대표 벡터 생성  
- Sentiment: 일부 레스토랑(1, 5 등) LLM 재판정 200 OK  
- Comparison: restaurant_id 4, 5 요청 200 OK  
- `/api/v1/llm/comparison`, `/api/v1/llm/comparison/batch`, `/api/v1/vector/search/similar` 200 OK  

### 문제 (RunPod vLLM 400 Bad Request)

**1) Sentiment 배치 – max_tokens 2048 초과**

- `max_tokens' or 'max_completion_tokens' is too large: 2048`  
- 입력 토큰 2321 / 3076 / 3205 등 → 허용 출력(4096−input)보다 2048이 커서 400  
- 일부 레스토랑(2, 3, 4 등)에서 **LLM 재판정 실패 → 1차 분류 결과 사용** 로그 반복  

**2) 요약(Summary) – max_tokens 1500 / 입력 초과**

- `max_tokens ... too large: 1500` (input 2878 → 1500 > 4096−2878)  
- `your request has 4731 input tokens` (입력만으로 4096 초과)  
- **LLM 호출 실패** / **LLM 비동기 호출 실패** 로 요약 실패  
- `/api/v1/llm/summarize`, `/api/v1/llm/summarize/batch` 는 200 OK 이지만, 내부적으로 요약 LLM은 실패한 상태  

---

## 3. Pod 컨테이너 로그 (0513_pod_container_log.log)

- vLLM 0.11.0 기동, Qwen2.5-7B-Instruct 로드(약 15초), max_model_len=4096  
- API 서버 `http://0.0.0.0:8000` 기동 완료  
- 로그 상 ERROR/404/500 없음 → Pod 쪽은 정상 응답 중, 400은 **요청 조건(토큰 한도)** 때문에 반환된 것으로 보임  

---

## 4. 결론 및 이미 적용된 대응

0513 로그는 **이전에 논의했던 RunPod 400 이슈와 동일**합니다.

- **원인**  
  - Sentiment: 요청 `max_tokens=2048` 이 (4096 − input)보다 커서 400  
  - Summary: `max_tokens=1500` 초과 또는 **입력 4731토큰**으로 4096 초과  

이미 코드에는 아래가 반영된 상태입니다.

1. **llm_utils**  
   - RunPod 호출 시 `_cap_max_tokens_for_context()`로 **max_tokens를 (4096 − 추정 input − 64) 이하로 제한**  
2. **summary_pipeline**  
   - `_payload_within_context_limit()`로 **입력이 4096 한도 넘으면 리뷰 개수 줄여 payload 축소**  
   - evidence는 payload 길이에 맞춰 슬라이스해 인덱스 정렬 유지  

따라서 **0513 시점 로그는 위 수정 이전 실행**으로 보는 것이 맞고, 같은 플로우를 **현재 코드**로 다시 돌리면 Sentiment/Summary 400은 줄어들거나 사라져야 합니다.  
추가로 0513 이후에 수집한 새 로그가 있으면, 그 파일 이름/경로를 알려주시면 그 기준으로 다시 분석해 드리겠습니다.