응, **좋아.** 다만 “무조건 Prefect”가 아니라 **Prefect로 감싸면 가치가 생기는 지점**이 명확해.

네 케이스(4040 샘플 + OpenAI 라벨 일부 + self-hosted 라벨 + QLoRA 학습)는 **파이프라인 단계가 여러 개**라서 Prefect가 딱 맞는 편이야.

---

## Prefect로 감싸면 좋은 이유 (너한테 직접 이득)

### 1) 재현성

* “이번엔 윈도우 W=30/S=15”
* “이번엔 골드 500개”
* “이번엔 teacher 모델 변경”
  이런 변경이 생길 때 **같은 절차로 다시 돌리기**가 쉬워짐.

### 2) 실패 복구

* OpenAI 라벨링 중간에 rate limit/timeout
* self-hosted worker 죽음
* 학습 중 OOM
  이런 게 나와도 **task 단위 재시도/재개**가 가능.

### 3) 비용 통제

* “OpenAI 라벨링은 최대 N개까지만”
* “실패율 높으면 중단”
  같은 정책을 파이프라인에 넣기 쉬움.

### 4) 포트폴리오 가치

면접에서 강하게 말할 수 있는 포인트:

* “데이터→라벨→검증→학습→평가”를 **하나의 워크플로우로 운영 가능**
* MLOps/운영형 백엔드 역량 강조 가능

---

## 반대로, Prefect가 과한 경우

* 딱 한 번만 실험하고 끝
* 데이터/라벨/학습을 수동으로 한 번만 돌릴 거면
* 이미 bash 스크립트로 충분히 관리 가능하면

이땐 오버엔지니어링일 수 있어.

하지만 너는 이미 **반복 실험**을 하고 있고(샘플 생성, split, 평가…),
추가로 라벨링/학습까지 들어가면 반복성이 더 커져서 Prefect 쪽이 유리해.

---

## “어디까지” Prefect로 감싸는 게 적절하냐 (추천 범위)

모든 걸 다 감싸지 말고, **변경/실패가 잦은 구간만** 감싸는 게 베스트.

### ✅ 추천 플로우 구성 (최소 but 강력)

1. **build_dataset_flow**

* 식당 단위 split
* 윈도우/샘플 생성
* train/val/test 저장 + 버전 태깅

2. **labeling_flow**

* OpenAI 골드 라벨링 (예: 500개 cap)
* self-hosted teacher 라벨링 (나머지)
* 품질 필터링 + dedup
* 라벨 데이터셋 버전 저장

3. **train_student_flow**

* QLoRA 학습 실행
* artifact 저장(모델/어댑터)
* 학습 메타 기록

4. **evaluate_flow**

* val/test 평가
* 샘플링 휴먼 평가 리스트 뽑기
* 리포트 생성

> 이 4개만 있어도 “운영 가능한 파이프라인”이 된다.

---

## Prefect 안 쓰고도 되는 부분

* 모델/학습 코드 내부 로직
* 프롬프트 템플릿 자체

이건 그냥 코드로 두고, Prefect는 “오케스트레이션”만.

---

## 결론

너가 지금 하려는 건:

* 데이터 생성
* 일부 OpenAI 라벨링
* self-hosted 대량 라벨링
* QLoRA 학습
* 평가

이라서,

> ✅ Prefect로 감싸는 게 충분히 좋은 선택이고, 포트폴리오에도 플러스다.
> 다만 “최소 3~4개 flow” 수준으로 얇게 감싸라.

---

## 반영: Flow 이름 / 입력·출력 / 저장 경로 / 메타데이터

**데이터·라벨·학습 수치와 실행 순서는 [distill_strategy.md](distill_strategy.md)를 따른다.** (식당 단위 split, OpenAI 골드 300~800, self-hosted 2000~2500 + 품질 필터, 학습 데이터 2500~3200, val/test OpenAI 라벨 등.)

현재 구조(데이터 증강 → 라벨링 → QLoRA 학습 → 평가) 기준으로 4개 flow의 구체 스펙을 아래처럼 잡았다. 구현은 `scripts/distill_flows.py`에서 Prefect flow로 정의되어 있다.

### 1. build_dataset_flow

| 항목 | 내용 |
|------|------|
| **입력** | `input_path`(원본 리뷰 JSON), `out_dir`(출력 루트), `window_configs`(예: [(30,15),(50,25),(20,10)]), `add_full_restaurant`, `train_ratio`, `val_ratio`, `test_ratio`, `seed` |
| **출력** | `dataset_version`(예: 타임스탬프 또는 run id), `train_path`, `val_path`, `test_path`, `stats_path` |
| **저장 경로** | `{out_dir}/datasets/{dataset_version}/` 아래 `train.json`, `val.json`, `test.json`, `stats.json` |
| **메타** | `stats.json`에 n_reviews, n_restaurants, n_train/n_val/n_test, window_configs, split 비율 |

### 2. labeling_flow

| 항목 | 내용 |
|------|------|
| **입력** | `train_path`(또는 samples JSON), `openai_cap`(골드 라벨 개수 상한, **전략: 300~800**, 기본 500), `teacher_config`(self-hosted teacher 설정), `output_labeled_dir` |
| **출력** | `labeled_version`, `labeled_path`(EasyDistill 형식 instruction + output) |
| **저장 경로** | `{output_labeled_dir}/labeled/{labeled_version}/train_labeled.json` (+ 품질 필터/dedup 메타) |
| **메타** | openai_count, self_hosted_count, filtered_count, dedup_count |
| **전략 반영** | Train에서 OpenAI로 골드만 생성; 나머지는 self-hosted teacher → **품질 필터** 필수: JSON 파싱 성공, 길이 범위, 금지 표현, 근거 포함 정책, 입력 근거 재사용률 휴리스틱. 무필터 teacher 라벨은 사용하지 않음. |

### 3. train_student_flow

| 항목 | 내용 |
|------|------|
| **입력** | `labeled_path`, `student_model`, `output_dir`, `qlora_config`(r, alpha, target_modules 등), `gold_oversample_ratio`(선택, **전략: 골드 20~30% 비중**) |
| **출력** | `adapter_path`, `training_meta_path` |
| **저장 경로** | `{output_dir}/runs/{run_id}/` 아래 adapter, `training_meta.json` |
| **메타** | steps, loss curve 요약, GPU 메모리, 소요 시간 |
| **전략 반영** | 학습 데이터: 골드(OpenAI) 300~800 + self-hosted 필터 후 2000~2500 → **총 2500~3200**. 골드는 oversample로 비중 20~30% 유지 가능. |

### 4. evaluate_flow

| 항목 | 내용 |
|------|------|
| **입력** | `val_path`, `test_path`, `adapter_path`(또는 학생 모델 경로), `output_dir` |
| **출력** | `report_path`, `human_eval_sample_path`(휴먼 평가용 샘플 리스트) |
| **저장 경로** | `{output_dir}/eval/{run_id}/report.json`, `human_eval_samples.json` |
| **메타** | val/test 메트릭, 샘플링된 review_id 목록 |
| **전략 반영** | **Val**: 300~500 샘플, **Test**: 400~600 샘플. 가능하면 val/test 라벨은 OpenAI로 생성(평가 기준 고정). 옵션: 휴먼 검증 50~100개 + 자동지표. |

---

### 실행 순서 (distill_strategy.md §9)

1. 4040 샘플 확정 (build_dataset_flow)
2. 식당 단위 split 고정 (동일 flow에서 수행)
3. Train에서 500개 뽑아 OpenAI로 골드 생성 (labeling_flow, openai_cap=500)
4. 나머지 Train은 self-hosted teacher 라벨링 + 품질 필터 (labeling_flow)
5. student QLoRA 학습 (train_student_flow)
6. Val/Test는 OpenAI 라벨로 평가 + 샘플 수동 평가 50개 (evaluate_flow)

---

# 1️⃣ self-hosted teacher 라벨 vs OpenAI 골드 라벨 vs OpenAI 평가 라벨

이 셋은 **역할이 다르다.**

---

## ✅ ① OpenAI 골드 라벨 (Train 일부, 500개 cap)

**목적:**
→ “고품질 기준 정답” 만들기

**사용 위치:**
→ Train 데이터 일부 (예: 500개)

**역할:**

* Student가 “따라야 할 기준 스타일”
* teacher 품질 기준점
* self-hosted teacher 품질 검증 기준

이건 학습에 들어가는 “고품질 supervised 라벨”.

---

## ✅ ② Self-hosted teacher 라벨 (Train 나머지)

**목적:**
→ 대량 확장 (cheap scaling)

**사용 위치:**
→ Train의 나머지 2k~3k

**역할:**

* Student 학습용 데이터
* 단, 노이즈 있음

이건 “약한 라벨”이야.

---

## ✅ ③ OpenAI 라벨 (Val/Test 평가용)

**목적:**
→ Student 성능 자동 비교

**사용 위치:**
→ Val/Test

**역할:**

* ROUGE/BERTScore 기준
* GPT-as-judge 기준
* Teacher 모사 정도 측정

이건 학습에 안 쓰이고 **평가용**이야.

---

# 🔥 핵심 차이

| 구분                  | Train    | Val/Test |
| ------------------- | -------- | -------- |
| OpenAI 골드           | ✔ 학습에 사용 | ❌        |
| Self-hosted teacher | ✔ 학습에 사용 | ❌        |
| OpenAI 평가 라벨        | ❌        | ✔ 평가용    |

OpenAI 골드와 OpenAI 평가 라벨은 “같은 모델”일 수는 있지만
**역할이 다르다.**

---

# 2️⃣ 품질 필터란 무엇인가?

Self-hosted teacher 라벨은 노이즈가 있다.
그래서 그냥 다 쓰면 student가 이상한 걸 배울 수 있다.

품질 필터는:

> “학습에 쓰기 전에 걸러내는 단계”

---

## 🎯 현실적인 품질 필터 예시 (요약 태스크 기준)

### ① JSON 구조 검증

* 필수 필드 있는지
* category key 누락 없는지

→ 실패 시 버림

---

### ② 길이 필터

* 너무 짧음 (예: 1문장)
* 너무 김 (token 폭주)

→ 일정 범위만 허용

---

### ③ 근거 기반성 휴리스틱

* 입력 리뷰 단어 재사용 비율 < 일정 threshold면 제거
* hallucination 의심 문구 필터

예:

* “전통 있는 30년 맛집입니다”
  (입력에 없음)

---

### ④ OpenAI 골드와 샘플 비교 (선택)

Train 중 일부에 대해:

* self-hosted 요약 vs OpenAI 요약
* BERTScore 낮으면 해당 모델 설정 조정

---

### ⑤ 반복/붕괴 감지

* 모든 요약이 동일 패턴
* 카테고리 비어 있음
* price 항상 “언급이 적어요”

---

# 3️⃣ 왜 품질 필터가 중요하냐?

Distill 구조에서:

```
Teacher → Student
```

인데 teacher가 나쁘면 student는 더 나빠진다.

특히 QLoRA는:

> 작은 파라미터 업데이트로
> 분포를 강하게 따라간다

그래서 노이즈가 치명적일 수 있다.

---

# 4️⃣ 요약 구조 정리

```
Train:
  500 → OpenAI Gold (고품질)
  2500 → Self-hosted + 품질 필터

Val/Test:
  OpenAI 라벨 생성
  + 50개 Human Eval
```

---

# 5️⃣ 한 줄로 요약

* OpenAI 골드 = 학습용 고품질 기준
* Self-hosted teacher = 대량 확장용 (필터 필수)
* OpenAI 평가 라벨 = 자동 평가 기준
* 품질 필터 = 노이즈 제거 장치

---

> ROUGE / BERTScore / GPT-judge는 **학습에 쓰지 않고**,
> **Val/Test에서만 사용한다.**

---

# 1️⃣ 학습(Train) 단계에서는 무엇을 쓰는가?

학습에서는 이런 구조다:

```
입력 X → Student 출력
정답 Y (OpenAI gold 또는 teacher label)
→ Cross-Entropy Loss 계산
→ 파라미터 업데이트
```

즉, 전통적인 supervised fine-tuning:

> Teacher summary = y_true
> Student summary = y_pred
> → token-level loss로 학습

여기에는 ROUGE/BERTScore가 안 들어간다.

왜냐하면:

* ROUGE는 미분 불가능
* BERTScore도 일반 학습 루프에 안 들어감
* GPT-judge는 더더욱 불가능

---

# 2️⃣ 그럼 ROUGE/BERTScore는 언제 쓰나?

이건 전부 **평가 지표**다.

```
Val/Test:
Student summary vs Reference summary
→ ROUGE
→ BERTScore
→ GPT-as-judge
```

즉:

* 모델 업데이트 안 함
* gradient 없음
* 성능 측정만

---

# 3️⃣ 왜 학습에 안 쓰는가?

학습 손실 함수는:

```
L = - Σ log P(token | previous_tokens, input)
```

이 구조가 LLM fine-tuning의 기본이야.

ROUGE는:

* n-gram overlap
* discrete metric

미분이 안 돼서 직접 학습에 못 쓴다.

---

# 4️⃣ 예외는 있나?

있긴 있다.

### 🔹 RLHF / Reinforcement Learning

* ROUGE를 reward로 써서 PPO 학습
* GPT-judge를 reward model로 써서 fine-tune

하지만:

> 이건 훨씬 복잡한 단계
> QLoRA 기본 구조에는 해당 안 됨

너 현재 설계는 SFT (Supervised Fine-Tuning)다.

---

# 5️⃣ 네 파이프라인에서 정리하면

## Train

```
OpenAI gold / self-hosted teacher summary
→ Student가 그대로 맞추도록 학습
```

## Val/Test

```
Student output
vs
OpenAI reference

→ ROUGE
→ BERTScore
→ GPT-as-judge
```

그리고

```
+ Human evaluation 50개
```

---

# 6️⃣ 핵심 요약

| 구간         | 무엇을 쓰는가                            |
| ---------- | ---------------------------------- |
| Train      | Cross-entropy (teacher summary 기반) |
| Val/Test   | ROUGE / BERTScore / GPT-judge      |
| Human eval | 별도 수동 평가                           |

---

# 🎯 한 줄 정리

> ROUGE/BERTScore/GPT-judge는 학습에 쓰지 않는다.
> 오직 Val/Test 평가 지표다.

---

## Val/Test 라벨 정리

### OpenAI 평가 라벨

Student output vs OpenAI reference → ROUGE / BERTScore / GPT-as-judge (자동 평가용)

### 휴먼 평가 라벨 스키마 (50~100개)

evaluate_flow에서 `human_eval_samples.json`으로 내보내는 샘플에 대해, 휴먼이 아래 필드로 채운 결과를 `human_labels.json`에 저장:

| 필드 | 타입 | 설명 |
|------|------|------|
| sample_id | str | 샘플 ID |
| model_name | str | 평가 대상 모델명 |
| relevance | 1~5 | 관련성 |
| faithfulness | 1~5 | 근거 충실도 |
| structure_consistency | 1~5 | 구조 일관성 |
| hallucination | 0/1 | 환각 여부 (0=없음, 1=있음) |
| overall_score | 1~5 | 종합 점수 |
| comment | str | (선택) 코멘트 |

---

### 실행

사전 요구: `pip install prefect`

```bash
# build_dataset_flow만 실행 (데이터 증강 → train/val/test 저장)
python scripts/distill_flows.py build_dataset

# 출력 경로 지정
python scripts/distill_flows.py build_dataset --input tasteam_app_all_review_data.json --out-dir distill_pipeline_output

# 4개 flow 순차 실행 (build_dataset 완전 구현, labeling/train/evaluate는 스텁)
python scripts/distill_flows.py all --out-dir distill_pipeline_output
```

자세한 파라미터는 `scripts/distill_flows.py --help` 및 각 flow docstring 참고.

---


`scripts/distill_flows.py`는 **요약 KD(Knowledge Distillation) 파이프라인을 Prefect로 돌리는 오케스트레이션 스크립트**입니다.

---

## 역할 요약

**`docs/easydistill/distill_by_prefect.md`와 `distill_strategy.md`에 정의된 4단계 파이프라인을 Prefect flow/task로 실행**합니다.

---

## 4개 Flow 구성

| Flow | 역할 | 구현 상태 |
|------|------|-----------|
| **build_dataset_flow** | 식당 단위 split, 슬라이딩 윈도우 샘플 생성, train/val/test 저장 + 버전 태깅 | ✅ **구현됨** — `scripts/data_augmentation.py`를 subprocess로 호출 |
| **labeling_flow** | OpenAI 골드 + self-hosted teacher 라벨링, 품질 필터/dedup | ⚠️ **스텁** — placeholder `train_labeled.json`만 생성 |
| **train_student_flow** | QLoRA SFT로 student 학습, adapter 저장, 메타 기록 | ⚠️ **스텁** — placeholder `training_meta.json`만 생성 |
| **evaluate_flow** | val/test 평가(ROUGE/BERTScore/GPT-judge), 휴먼 평가 샘플 생성 | ⚠️ **스텁** — placeholder `report.json`, `human_eval_samples.json`만 생성 |

---

## 실행 가능한 동작

- **`build_dataset`**: `data_augmentation.py`를 호출해 실제 데이터 증강과 train/val/test 분할 수행.
- **`all`**: 4개 flow를 순서대로 실행하지만, labeling/train/evaluate는 스텁이라 placeholder 파일만 생김.

---

## 정리

- **오케스트레이션**: Prefect flow/task로 4단계 순서, 입출력 경로, 버전 태깅 관리.
- **전략 반영**: `distill_strategy.md`, `distill_by_prefect.md`에 맞춘 상수·주석 포함.
- **현재 실제 동작**: `build_dataset_flow`만 완전 구현되어 있고, 나머지 3개는 구조와 인터페이스만 정의된 스텁입니다.

---


각 스텁을 구현하는 기본 방향만 정리해볼게요.

---

## 1. labeling_flow / labeling_task

**역할**: Train 샘플에 대해 OpenAI 골드 + self-hosted teacher 라벨링, 품질 필터, `labeled_path` 생성.

**진행 순서**:
1. `train_path` JSON 로드 (samples)
2. Train에서 `openai_cap`개만 샘플링 → OpenAI API로 요약 생성 (현재 `summary_pipeline` 프롬프트/형식 유지)
3. 나머지 샘플은 self-hosted teacher(예: vLLM Qwen2.5-7B)로 요약 생성
4. 품질 필터 적용: JSON 구조 검증, 길이 필터, 근거 재사용 비율, 반복/붕괴 감지 등
5. `labeled_path`에 EasyDistill 형식(instruction + output)으로 저장

**참고**: EasyDistill `kd/infer.py`가 teacher 라벨 생성 패턴을 보여주고, `summary_pipeline`의 instruction/출력 형식을 재사용하면 됨.

---

## 2. train_student_flow / train_student_task

**역할**: 라벨된 데이터로 QLoRA SFT 실행, adapter 저장.

**진행 순서**:
1. `labeled_path`를 EasyDistill/트레이너용 포맷으로 로드
2. 골드 샘플 oversample (예: 20~30% 비중)
3. `bitsandbytes` 4bit + `peft` LoRA로 student 모델 로드
4. `SFTTrainer` 또는 `Trainer`로 SFT (config에 `report_to="wandb"` 가능)
5. adapter를 `output_dir/runs/{run_id}/`에 저장
6. 학습 메타를 `training_meta.json`에 기록

**참고**: `easydistill/kd/train.py`의 데이터 로딩/템플릿 패턴을 따르되, 모델 로딩만 `BitsAndBytesConfig` + `get_peft_model(LoraConfig(...))`로 교체. 별도 `scripts/train_qlora.py`를 만들고, `train_student_task`에서 subprocess로 호출하는 방식이 관리하기 쉬움.

---

## 3. evaluate_flow / evaluate_task

**역할**: val/test에서 Student 출력 vs OpenAI 레퍼런스 비교, 자동 지표·휴먼 평가 샘플 생성.

**진행 순서**:
1. val/test에 대해 OpenAI로 레퍼런스 요약 생성 (또는 미리 생성해둔 labeled val/test 사용)
2. `adapter_path`에서 학습된 student 로드 → val/test 샘플에 대해 추론
3. ROUGE, BERTScore, (선택) GPT-as-judge 계산
4. 리포트를 `report.json`에 저장
5. 휴먼 평가용 50~100개 샘플 ID를 뽑아 `human_eval_samples.json`에 저장 (입력/모델 출력/레퍼런스 포함하면 휴먼 평가에 편함)

**참고**: `scripts/evaluate_summary.py`가 있으면 평가 로직을 재사용 가능. ROUGE/BERTScore는 `rouge-score`, `bert-score` 패키지로 계산.

---

## 구현 전략 요약

| 스텁 | 구현 방식 |
|------|-----------|
| **labeling** | 별도 스크립트(예: `scripts/label_for_distill.py`) + task에서 subprocess 호출, 또는 task 내부에 라벨링 로직 직접 작성 |
| **train_student** | 별도 `scripts/train_qlora.py` 작성 후 subprocess 호출 권장 (의존성·환경 분리, 디버깅 용이) |
| **evaluate** | 기존 `evaluate_summary.py` 활용 또는 `scripts/eval_distill.py` 신규 작성 후 subprocess 호출 |

**실행 순서**  
labeling → train_student → evaluate 순으로 의존성이 있으므로, 각각 단독 실행·테스트가 가능하도록 하고, `distill_flows.py`에서는 subprocess나 함수 호출로 묶어서 사용하는 방식이 관리에 유리합니다.