
정리하면 아래와 같습니다.

---

## 결론: **요약(Summary) 등 LLM 생성 태스크에는 가능, 나머지 태스크는 프레임워크 범위 밖**

EasyDistill은 **LLM → LLM 지식 증류**(teacher/student 모두 생성형 LLM)에 맞춰져 있어서, **현재 프로젝트의 “LLM으로 응답을 생성하는” 부분에만** 그대로 맞습니다.

---

## 1. 프로젝트 태스크 vs EasyDistill

| 프로젝트 태스크 | 모델/역할 | EasyDistill 적용 |
|-----------------|-----------|------------------|
| **요약 파이프라인** | LLM (Qwen2.5-7B / OpenAI): 리뷰 → JSON 요약 생성 | ✅ **가능** (black-box KD로 소형 학생 모델 학습) |
| **LLM 완성/채팅** | 동일 LLM | ✅ **가능** (동일 방식) |
| **감성 분석** | 분류 모델 (Dilwolf/Kakao_app-kr_sentiment) | ❌ 생성형 LLM 아님 → EasyDistill job 타입과 다름 |
| **벡터 검색** | 임베딩 모델 | ❌ 임베딩/검색 태스크 → EasyDistill 범위 아님 |
| **비교(Kiwi/lift)** | 규칙/통계 | ❌ LLM 아님 |

즉, **“지금 프로젝트 task에 맞게 distillation”**을 **요약(및 일반 LLM 생성)에만** 쓰는 것은 가능하고, 감성/검색/비교는 EasyDistill 설계와 다릅니다.

---

## 2. 요약 태스크에 EasyDistill 쓰는 방법 (가능한 흐름)

- **Teacher**: 현재 쓰는 LLM (예: vLLM Qwen2.5-7B-Instruct 또는 동일 모델).
- **Student**: 더 작은 LLM (예: Qwen2.5-0.5B-Instruct, 1.5B 등).
- **데이터**:  
  - `instruction_path` (예: `train.json`): 각 샘플이 **한 개의 “user 입력”** = 지금 요약 파이프라인에 넣는 **리뷰 JSON** (서비스/가격/음식 리스트).  
  - `template`: 지금 `summary_pipeline`의 **시스템 프롬프트 + user 메시지**를 Qwen 채팅 형식으로 만드는 Jinja2 템플릿.
- **라벨**: Teacher가 위 입력에 대해 생성한 **JSON 요약** (service/price/food summary, bullets, evidence, overall_summary).  
  → EasyDistill은 `kd_black_box_local` 기준으로 `instruction_path`만 주면 teacher(vLLM)가 추론해 `labeled_path`를 만듦.
- **실행**:  
  - `job_type`: `kd_black_box_local` (teacher를 로컬 vLLM으로 돌릴 때) 또는, teacher 출력을 미리 만들어 두면 `kd_black_box_train_only`로 학생만 학습 가능.  
  - config에 `dataset.instruction_path`, `dataset.labeled_path`, `dataset.template`, `models.teacher`, `models.student`, `training.*` 지정.

즉, **요약 태스크를 “instruction = 리뷰 JSON, response = 요약 JSON” 형태로 정의하면**, 현재 프로젝트 task에 맞게 EasyDistill로 distillation 진행하는 것이 **가능**합니다.

---

## 3. 구체적으로 맞춰야 할 것

1. **데이터 형식**  
   - EasyDistill `kd/infer.py`는 `instruction_path` JSON에서 각 항목의 `instruction` 필드를 “user content”로 사용합니다.  
   - 따라서 `train.json`을 `[{"instruction": "<현재 summary_pipeline에 넣는 리뷰 JSON>"}, ...]` 형태로 맞추면 됩니다. (시스템 프롬프트는 template에서 처리하도록 하면 됨.)

2. **템플릿**  
   - `dataset.template`에 Qwen용 chat template을 두고,  
     - 시스템: 현재 `summary_pipeline`의 instructions 문자열.  
     - 사용자: `instruction` 필드 내용.  
   - 그러면 지금 서빙하는 것과 동일한 입력/출력 형식을 유지할 수 있습니다.

3. **Teacher**  
   - `kd_black_box_local`을 쓰면 teacher는 반드시 **로컬 vLLM**으로 올라가는 모델이어야 합니다.  
   - 지금 RunPod vLLM으로 Qwen2.5-7B를 쓰고 있다면, 그와 동일한 모델 경로/이름을 `models.teacher`에 두면 됩니다.  
   - Teacher를 API(OpenAI 등)로만 쓰고 싶다면: EasyDistill의 `kd_black_box_api` 등 지원 여부를 확인한 뒤, 지원되면 API로 라벨을 만들고, 아니면 **미리 API로 `labeled_path`를 생성해 두고** `kd_black_box_train_only`로 학생만 학습하는 방식이 됩니다.

4. **학생 모델**  
   - `models.student`에 Qwen2.5-0.5B-Instruct 등 원하는 소형 모델을 지정하면 됩니다.

---

## 4. 감성 분석은?

- 감성은 **분류 모델**이라, EasyDistill이 제공하는 “instruction → long-form response” KD 파이프라인과는 다릅니다.  
- 같은 걸 하려면 **분류기 전용 증류**(teacher 분류기 logits → 작은 분류기 학습) 등 별도 구현이 필요하고, **EasyDistill로 현재 프로젝트의 감성 태스크를 그대로 distillation하는 것은 설계상 맞지 않습니다.**

---

**요약**:  
- **가능한 것**: 요약(및 일반 LLM 생성) 태스크를, 지금 쓰는 프롬프트/입출력에 맞춰 EasyDistill의 black-box KD로 teacher(현재 LLM) → student(작은 LLM) 증류.  
- **불가능/부적합**: 감성·벡터검색·비교 등 비-생성 태스크는 EasyDistill의 job 타입과 다르므로, “현재 프로젝트 task에 맞게 EasyDistill로만 distillation”은 **요약(및 LLM 생성) 쪽에만** 해당합니다.