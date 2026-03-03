# EasyDistill 학습 방식 vs 본 프로젝트 태스크 적합성

본 프로젝트(tasteam-new-async)에서 사용하는 태스크별로, EasyDistill이 제공하는 학습 방식 중 어떤 것이 적합한지 정리한 문서입니다.

---

## 1. 본 프로젝트의 주요 태스크

| 태스크 | 설명 | 사용 모델/방식 |
|--------|------|----------------|
| **요약(Summary)** | 리뷰 → 카테고리별(service/price/food) JSON 요약 생성 | LLM 생성 (Qwen2.5-7B-Instruct / OpenAI) |
| **감성 분석(Sentiment)** | 리뷰 → 긍정/부정/중립 분류 | 분류 모델 (Dilwolf/Kakao_app-kr_sentiment) |
| **벡터 검색** | 리뷰 임베딩·하이브리드 검색 | 임베딩 모델 (paraphrase-multilingual-mpnet-base-v2 등) |
| **비교(Comparison)** | Kiwi + lift 기반 비교 | 규칙/통계 (LLM 생성 아님) |
| **DeepFM (추천/CTR)** | CTR 예측·추천용 DeepFM 학습 (Criteo 형식) | PyTorch DeepFM (FM + DNN, 생성형 LLM 아님) |

---

## 2. EasyDistill에서 제공하는 학습 방식(job_type) 요약

EasyDistill은 `easydistill --config <config.json>` 실행 시 config 내 `job_type`에 따라 서로 다른 파이프라인을 실행합니다.

### 2.1 텍스트 KD (Knowledge Distillation)

| job_type | 설명 | Teacher | Student 학습 |
|----------|------|---------|--------------|
| **kd_black_box_local** | Teacher를 로컬 vLLM으로 두고 응답 생성 → 학생 SFT | 로컬 vLLM (예: Qwen2.5-7B) | 전체 파인튜닝 (QLoRA 미지원) |
| **kd_black_box_api** | Teacher를 API로 두고 응답 생성 → 학생 SFT | API (예: OpenAI) | 동일 |
| **kd_black_box_train_only** | 이미 생성된 labeled 데이터로 학생만 SFT | 없음 | 동일 |
| **kd_white_box** | Teacher logits로 학생 학습 (동일 구조/접근 필요) | 로컬 모델 (logits 출력) | 동일 + logits KD |

- **데이터**: `instruction_path`(입력 instruction 목록), `labeled_path`(teacher 응답 포함), `template`(Jinja2 채팅 템플릿).
- **학습**: `kd/train.py` — Hugging Face `SFTTrainer`, **PEFT/QLoRA 없음**.

### 2.2 멀티모달 KD (MMKD)

| job_type | 설명 |
|----------|------|
| **mmkd_black_box_local** / **mmkd_black_box_api** / **mmkd_white_box** | 이미지+텍스트 멀티모달 모델 증류 (VLM → 소형 VLM). |
| **mmkd_rl_grounding** | 멀티모달 RL 그라운딩 (PEFT 사용). |

- 본 프로젝트는 **텍스트 전용** 요약/감성만 사용하므로 MMKD는 이미지 입력이 필요한 경우에만 해당.

### 2.3 에이전트 / 특수 KD

| job_type | 설명 |
|----------|------|
| **agentkd_local** | 에이전트 궤적(도구 호출 등) 증류. |
| **speckd_local** | 특수 스펙 KD (Deepspeed 기반). |

- 요약은 “입력 리뷰 → 출력 JSON” 단일 호출 구조라, 에이전트 궤적/도구 호출과는 태스크가 다름.

### 2.4 데이터 합성 (Synthesis)

| job_type 예시 | 설명 |
|----------------|------|
| **instruction_expansion_api** / **instruction_expansion_batch** | instruction 확장. |
| **instruction_refinement_*** | instruction 정제. |
| **instruction_response_extraction_api** / **_batch** | instruction–response 추출. |
| **cot_generation_api** / **cot_generation_batch** | CoT 생성. |
| **cot_long2short_*** / **cot_short2long_*** | CoT 길이 변환. |

- KD **학습**이 아니라 **데이터 생성** 단계. 요약용 (instruction, response) 쌍을 대량 만들 때 보조적으로 활용 가능.

### 2.5 순위/정책 학습

| job_type | 설명 |
|----------|------|
| **rank_*** | DPO 등 랭킹 기반 학습. |
| **rl_ppo** / **rl_grpo** | PPO/GRPO 강화학습. |
| **rl_reward_api** / **rl_reward_local** | 보상 추론. |

- Black-box SFT 이후 정렬(alignment) 강화용. 요약 품질을 reward로 개선할 때만 고려.

---

## 3. 본 프로젝트 태스크별 적합 여부

### 3.1 요약(Summary) — **적합: kd_black_box_***

| 항목 | 내용 |
|------|------|
| **적합한 방식** | **kd_black_box_local**, **kd_black_box_api**, **kd_black_box_train_only** |
| **이유** | 요약은 “instruction(리뷰 JSON) → response(요약 JSON)” 생성 태스크로, EasyDistill의 텍스트 Black-box KD와 동일한 형태. Teacher(현재 Qwen2.5-7B 또는 API) 출력으로 labeled 데이터를 만든 뒤, 소형 학생 모델(Qwen2.5-0.5B 등)을 SFT하면 됨. |
| **사용 시 유의** | • `instruction_path`: 요약 파이프라인에 넣는 리뷰 payload(또는 그 참조)를 instruction으로 구성. <br>• `template`: 현재 `summary_pipeline`의 시스템 프롬프트 + 사용자 메시지 형식을 Qwen 채팅 템플릿으로 맞춤. <br>• 학생 학습에 QLoRA를 쓰려면 `kd/train.py` 수정 또는 별도 QLoRA 학습 스크립트로 `labeled_path`만 소비하는 방식 필요. |

### 3.2 감성 분석(Sentiment) — **부적합**

| 항목 | 내용 |
|------|------|
| **적합한 방식** | 없음 (EasyDistill 메인 job_type으로는 부적합) |
| **이유** | 감성 분석은 **분류 모델**(Dilwolf/Kakao_app-kr_sentiment)을 사용하는 것이며, “instruction → long-form 응답” 생성이 아님. EasyDistill의 KD 파이프라인은 **생성형 LLM → 생성형 LLM** 증류에 맞춰져 있음. |
| **대안** | 분류기 전용 증류(teacher logits → 소형 분류기)는 별도 구현 또는 다른 도구 사용. |

### 3.3 벡터 검색(임베딩) — **부적합**

| 항목 | 내용 |
|------|------|
| **적합한 방식** | 없음 |
| **이유** | 임베딩 모델 학습/증류는 **문장 임베딩 공간**을 다루는 태스크로, EasyDistill의 causal LM 생성 KD와 목적·데이터 형식이 다름. |

### 3.4 비교(Comparison) — **부적합**

| 항목 | 내용 |
|------|------|
| **적합한 방식** | 없음 |
| **이유** | Kiwi + lift 기반 규칙/통계 처리이며, LLM 생성 증류와 무관함. |

### 3.5 DeepFM (추천/CTR) — **부적합**

| 항목 | 내용 |
|------|------|
| **적합한 방식** | 없음 (EasyDistill 메인 job_type으로는 부적합) |
| **이유** | DeepFM은 **CTR 예측·추천**용으로, **이산/연속 피처 → 스칼라 예측(클릭률 등)** 구조의 PyTorch 모델(FM + DNN)이다. **생성형 LLM이 아니며**, instruction → long-form 텍스트 응답을 다루지 않는다. EasyDistill의 모든 KD 파이프라인은 **LLM(teacher) → LLM(student)** 증류를 전제로 하므로, 태스크·데이터 형식·학습 목표가 전혀 다르다. |
| **본 프로젝트 내 학습** | DeepFM 학습은 `deepfm_training/`에서 전처리 + PyTorch 학습으로 수행하며, Prefect 파이프라인(`training_flow.py`)으로 자동화되어 있다. 증류가 필요하면 추천 도메인 전용 소형 모델로의 지식 증류는 별도 설계·구현이 필요하다. |

---

## 4. 요약 표: “우리 태스크 → EasyDistill 방식”

| 본 프로젝트 태스크 | 적합한 EasyDistill 방식 | 비고 |
|-------------------|-------------------------|------|
| **요약(Summary)** | **kd_black_box_local**, **kd_black_box_api**, **kd_black_box_train_only** | Teacher=현재 LLM, Student=소형 LLM, 데이터 형식만 맞추면 됨. QLoRA는 kd/train 수정 또는 별도 스크립트. |
| 감성 분석 | 없음 | 분류 모델; 생성 KD와 무관. |
| 벡터 검색 | 없음 | 임베딩 태스크. |
| 비교 | 없음 | 비생성 태스크. |
| **DeepFM (추천/CTR)** | 없음 | CTR/추천용 DeepFM; LLM 아님. 학습은 `deepfm_training/` + Prefect로 별도 운영. |

---

## 5. 참고

- EasyDistill 메인 문서: `easydistill/README.md`
- 본 프로젝트 요약 파이프라인: `src/summary_pipeline.py`, `src/api/routers/llm.py`
- DeepFM 학습 파이프라인: `deepfm_training/`, `docs/prefect/deepfm_training_pipeline.md`
- KD + Prefect + QLoRA 조합: `docs/kd/kd_qlora_prefect.md` (해당 문서가 있다면)
