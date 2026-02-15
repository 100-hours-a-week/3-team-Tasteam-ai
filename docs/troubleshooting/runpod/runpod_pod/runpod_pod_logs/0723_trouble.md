0723 로그 기준으로 정리했습니다.

---

# RunPod Pod 로그 분석 (0723)

## 1. 파일

| 파일 | 내용 |
|------|------|
| **0723_app_erver_log.log** | 로컬 앱 (2026-02-14 19:23~19:25) ※ 파일명 typo: app_erver |
| **0723_container_log.log** | RunPod vLLM 컨테이너 |

---

## 2. 앱 서버 로그 요약

### 정상

- 기동, 벡터/임베딩/Sentiment warm-up, RunPod Pod 연결
- 벡터 업로드 142개, 레스토랑 5개
- Sentiment: 단일 200 OK. 배치 RunPod 요청 5건 모두 **200 OK**
- 요약: 단일 200 OK, 배치 2건 모두 200 OK
- Comparison 단일/배치 200 OK
- Pod 컨테이너: **400 에러 없음** (전부 200 OK)

### 이슈 1건: Sentiment 배치 – JSON 파싱 실패 (0657/0705와 동일)

- **로그**: `LLM 재판정 실패: JSON 파싱 불가(비표준 형식). 1차 분류 결과를 사용합니다.`
- **시점**: 배치 Sentiment **마지막** 요청 직후 (62개 리뷰, restaurant_id: 4)
- **의미**: RunPod 200 OK였지만 응답이 기대한 JSON 배열로 파싱되지 않음 → 1차 분류만 사용.
- **참고**: 재시도 조건 완화(len(raw) 제한 제거) 후에도 동일 실패가 나온다면, 1회 수정 요청 재시도까지 한 뒤 **수정 응답도 파싱 실패**한 경우이거나, LLM이 62개 분량에서 비표준/잘린 JSON을 계속 내보내는 케이스로 보임.

---

## 3. Pod 컨테이너 로그

- `/v1/chat/completions` 요청 모두 **200 OK**
- max_tokens / context length 관련 에러 없음.

---

## 4. 결론

| 구분 | 상태 | 비고 |
|------|------|------|
| **Sentiment 400** | 미발생 | (total*3)//4 추정 유지 |
| **Summary 400** | 미발생 | 동일 |
| **Sentiment JSON 파싱** | 1건 실패 | 62개 리뷰(restaurant_id=4) 한 건. 0705와 동일 패턴. |

정리하면, 0723에서도 **max_tokens/context 400은 없고**, Sentiment 배치에서 **62개 리뷰(restaurant_id=4) 한 건만** JSON 파싱 실패가 재현된 상태입니다. test_data_sample 기준 62개는 restaurant_id=4만 해당하므로, 해당 레스토랑에 대한 LLM 재판정 출력 형식/품질을 추가로 보강하거나(프롬프트·후처리), 62개를 두 번에 나눠 요청하는 방식 등을 검토할 수 있습니다.

---

(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python scripts/test_sentiment_threshold.py --from-test-data 4
test_data_sample에서 restaurant_id=4 리뷰 62개 로드 (/Users/js/tasteam-new-async/data/test_data_sample.json)
/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loading weights: 100%|█| 201/201 [00:00<00:0

=== 1차 분류 요약 (threshold=0.8) ===
총 리뷰: 62
positive(재판정 제외): 0
LLM 재판정 대상: 62

--- positive_score 구간별 개수 ---
  [0.0, 0.3): 62개  (확정 negative(재판정 불필요 가정))
  [0.3, 0.5): 0개  (LLM 재판정 대상)
  [0.5, 0.8): 0개  (LLM 재판정 대상)
  [0.8, 1.0): 0개  (positive)

--- 리뷰별 점수 (앞 20개 + 마지막 5개) ---
  0: 0.002 negative [LLM]  판교에서 맛있고 접근성도 좋아요 하이볼 너무 맛있어요 ㅎㅎ
  1: 0.002 negative [LLM]  스테이크동을 먹었는데 제가 그 동안 먹었던 덮밥 중에 원탑이었습니다👍🏻 …
  2: 0.002 negative [LLM]  혼밥만 먹다가 직장분들이랑 점심 같이 먹었는데 진짜 모든 메뉴 짱짱 다들…
  3: 0.002 negative [LLM]  정말 맛있네요. 너무 맛나서 2일 연속 찾았어요. 신선하고 맛있고 가성비…
  4: 0.002 negative [LLM]  N번째 방문중인 최애 맛집! 회사 근처라 점심때나 퇴근후에 일주일에 한번…
  5: 0.002 negative [LLM]  음식이 맛있고 배경음악도 좋아요.
  6: 0.002 negative [LLM]  회사 사람들하고 점심 먹으러 왔는데 음식도 맛있고 직원분들이 너무 친절하…
  7: 0.003 negative [LLM]  감사합니다
  8: 0.002 negative [LLM]  완전 가성비 맛집!! 연어도 넘 부드럽고 스테이크도 디게 부드러워서 진짜…
  9: 0.002 negative [LLM]  맛있어서 배달도 자주 시켜먹었는데 배달에 없는 카레 메뉴를 먹기위해서 방…
  10: 0.002 negative [LLM]  둘다 너무 맛있고, 특히 사케동이 연어가 너무 신선하고 부드러워서 먹기 …
  11: 0.002 negative [LLM]  가격도 너무 괜찮고 국이 진짜 맛있어요! 약간 매콤하면서 맛과 간이 한국…
  12: 0.002 negative [LLM]  사케동, 연어뱃살동의 연어가 신선하고 맛있습니다. 고로케 바삭하고 고소해…
  13: 0.002 negative [LLM]  감사합니다.
  14: 0.002 negative [LLM]  꼭 가라아케 드ㅛㅔ요 짐짜 마싰음요
  15: 0.002 negative [LLM]  덮밥맛집으로 오늘 먹은 3개의 메뉴를 적극적으로 추천 드립니다. 모든 고…
  16: 0.002 negative [LLM]  맛있는 다양한 종류의 덮밥 그리고 서브 메뉴로 주문한 오징어튀김까지 🦑 …
  17: 0.002 negative [LLM]  연어 정말 신선하고 부드럽고 맛있어요!!👍👍 너무 맛있게 먹어서 기분이 …
  18: 0.002 negative [LLM]  항정간장덮밥? 먹었는데 맛있었어요
  19: 0.002 negative [LLM]  사케동이랑 돈카츠 카레 먹었는데 양이 정말 푸짐하고 음식 하나하나의 퀄리…
  ...
  57: 0.002 negative [LLM]  맛있어요. 페이코 식권 해주세요 :)
  58: 0.002 negative [LLM]  스테이크동이 맛있어요
  59: 0.002 negative [LLM]  스테이크동 맛있어요
  60: 0.013 negative [LLM]  김치가있어서 개운하네요
  61: 0.001 negative [LLM]  음

--- 참고: 0.3 미만을 확정 negative로 두면 ---
  확정 negative(재판정 제외): 62개
  애매 구간만 LLM 재판정: 0개
(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % 

---

[[{'label': 'negative', 'score': 0.0021763721015304327},
  {'label': 'positive', 'score': 0.9978235960006714}]]

이게 이 모델의 출력이야

---

출력 순서가 점수 기준으로 바뀔 수 있어, 인덱스가 아니라 **레이블이 'positive'인 항목의 score**를 쓰도록 스크립트를 수정하겠습니다.


[2 tools called]


수정 반영했습니다. 이제 `out[1]` 대신 **`label`이 긍정으로 매핑되는 항목**의 `score`를 쓰도록 바꿨습니다. (`_map_label_to_binary(lab) == "positive"`인 항목)

파이프라인이 점수 순으로 `[positive, negative]`처럼 순서를 바꿔도, 항상 긍정 점수를 쓰므로 `python scripts/test_sentiment_threshold.py --from-test-data 4`를 다시 실행하면 긍정 리뷰는 0.99 근처, 구간별 개수도 기대한 대로 나와야 합니다.

---

(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % python scripts/test_sentiment_threshold.py --from-test-data 4
test_data_sample에서 restaurant_id=4 리뷰 62개 로드 (/Users/js/tasteam-new-async/data/test_data_sample.json)
/Users/js/miniconda3/envs/env_ai/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Loading weights: 100%|█| 201/201 [00:00<00:0

=== 1차 분류 요약 (threshold=0.8) ===
총 리뷰: 62
positive(재판정 제외): 58
LLM 재판정 대상: 4

--- positive_score 구간별 개수 ---
  [0.0, 0.3): 4개  (확정 negative(재판정 불필요 가정))
  [0.3, 0.5): 0개  (LLM 재판정 대상)
  [0.5, 0.8): 0개  (LLM 재판정 대상)
  [0.8, 1.0): 58개  (positive)

--- 리뷰별 점수 (앞 20개 + 마지막 5개) ---
  0: 0.998 positive  판교에서 맛있고 접근성도 좋아요 하이볼 너무 맛있어요 ㅎㅎ
  1: 0.998 positive  스테이크동을 먹었는데 제가 그 동안 먹었던 덮밥 중에 원탑이었습니다👍🏻 …
  2: 0.998 positive  혼밥만 먹다가 직장분들이랑 점심 같이 먹었는데 진짜 모든 메뉴 짱짱 다들…
  3: 0.998 positive  정말 맛있네요. 너무 맛나서 2일 연속 찾았어요. 신선하고 맛있고 가성비…
  4: 0.998 positive  N번째 방문중인 최애 맛집! 회사 근처라 점심때나 퇴근후에 일주일에 한번…
  5: 0.998 positive  음식이 맛있고 배경음악도 좋아요.
  6: 0.998 positive  회사 사람들하고 점심 먹으러 왔는데 음식도 맛있고 직원분들이 너무 친절하…
  7: 0.997 positive  감사합니다
  8: 0.998 positive  완전 가성비 맛집!! 연어도 넘 부드럽고 스테이크도 디게 부드러워서 진짜…
  9: 0.998 positive  맛있어서 배달도 자주 시켜먹었는데 배달에 없는 카레 메뉴를 먹기위해서 방…
  10: 0.998 positive  둘다 너무 맛있고, 특히 사케동이 연어가 너무 신선하고 부드러워서 먹기 …
  11: 0.998 positive  가격도 너무 괜찮고 국이 진짜 맛있어요! 약간 매콤하면서 맛과 간이 한국…
  12: 0.998 positive  사케동, 연어뱃살동의 연어가 신선하고 맛있습니다. 고로케 바삭하고 고소해…
  13: 0.998 positive  감사합니다.
  14: 0.998 positive  꼭 가라아케 드ㅛㅔ요 짐짜 마싰음요
  15: 0.998 positive  덮밥맛집으로 오늘 먹은 3개의 메뉴를 적극적으로 추천 드립니다. 모든 고…
  16: 0.998 positive  맛있는 다양한 종류의 덮밥 그리고 서브 메뉴로 주문한 오징어튀김까지 🦑 …
  17: 0.998 positive  연어 정말 신선하고 부드럽고 맛있어요!!👍👍 너무 맛있게 먹어서 기분이 …
  18: 0.998 positive  항정간장덮밥? 먹었는데 맛있었어요
  19: 0.998 positive  사케동이랑 돈카츠 카레 먹었는데 양이 정말 푸짐하고 음식 하나하나의 퀄리…
  ...
  57: 0.998 positive  맛있어요. 페이코 식권 해주세요 :)
  58: 0.998 positive  스테이크동이 맛있어요
  59: 0.998 positive  스테이크동 맛있어요
  60: 0.987 positive  김치가있어서 개운하네요
  61: 0.001 negative [LLM]  음

--- 참고: 0.3 미만을 확정 negative로 두면 ---
  확정 negative(재판정 제외): 4개
  애매 구간만 LLM 재판정: 0개
(env_ai) js@jinsoos-MacBook-Pro tasteam-new-async % 

---

0.3 미만은 확정 negative로 두고, 0.3~0.8 구간만 LLM 재판정 대상으로 넣도록 수정합니다.


[3 tools called]


반영했습니다. 기준은 이렇게 적용돼 있어요.

- **positive_score > 0.8** → positive, 재판정 없음  
- **0.3 ≤ positive_score < 0.8** → 1차는 negative, **LLM 재판정 대상에만 추가**  
- **positive_score < 0.3** → 확정 negative, 재판정 대상에 **넣지 않음**

그래서 test_data_sample restaurant_id=4 기준이면 62개 중 58개는 positive, 4개는 확정 negative로 처리되고, **LLM 재판정은 0건**으로 나갈 거예요.