## 문서 목적

이 문서는 `ml/deepfm_pipeline/` 기준으로 **현재 구현되어 있는 AI 서버(DeepFM) 파이프라인의 실제 동작 방식**을 요약한다.

> **변경(2026-03-17)**: 과거에는 `ml/deepfm_pipeline/api`에 **Admin FastAPI**(예: `/admin/deepfm/*`)가 있었으나, 현재 레포에서는 **DeepFM API는 삭제**되었고 배치/워크플로(스크립트·Prefect flow)만 유지된다.

- **서비스 간 계약(Contract)**: `docs/service_extraction/service_constract.md`
- **설계(Design)**: `docs/design/deepfm/deepfm_design.md`
- **구현 현황(Design 대비)**: `docs/design/deepfm/implementation_status.md`

> 핵심: 서비스 계약은 “API 서버가 raw를 S3에 적재 → AI 서버가 raw 기반 feature 생성/학습/추론 → 추천 결과만 S3 저장”을 정의한다.  
> 하지만 **현재 코드 구현**은 “추론 입력을 raw로 직접 받는 구조”가 아니라, **전처리된 후보 feature vector(CSV) + 메타(CSV)** 를 입력으로 받아 추천 결과 CSV를 만든다.

---

## 1) 전체 흐름(현재 구현)

### 1-1. 입력(현재 구현의 실질 입력)

현재 배치 추론(추천 생성)의 입력은 다음 2가지 파일이다.

- **후보 feature vector CSV**: 각 행이 한 후보(user–restaurant 등)에 대응하는 숫자 벡터
  - 파일 예시/설명: `docs/design/api/ML_API_DTO.md`의 “스코어링 입력: 후보 CSV”
  - 요구사항: **컬럼 수 = 해당 run의 `feature_sizes.txt`의 필드 수**
- **메타 CSV(선택)**: 후보별 식별자/컨텍스트를 같이 실어 주입
  - 컬럼: `user_id`, `anonymous_id`, `restaurant_id`, `context_snapshot`
  - 현재 구현은 호환을 위해 `member_id`만 있으면 `user_id`로 간주

> 즉, “raw(events/restaurants/menus)”는 서비스 계약상 AI 서버의 upstream이지만,  
> 이 레포 내부의 배치 추천 생성 코드(`utils/score_batch.py`)는 **raw를 직접 읽지 않는다**.

### 1-2. 모델(버전) 산출물

학습 실행 결과(run 디렉터리)에는 보통 아래 파일들이 존재한다.

- `model.pt`: PyTorch 모델 가중치
- `feature_sizes.txt`: feature field 크기 정보(전처리/인덱싱 결과)
- `pipeline_version.txt`: 파이프라인 버전 문자열
- `run_manifest.json`: run 메타(생성 시각, metrics 등)

코드 기준:
- 학습 플로우: `ml/deepfm_pipeline/training_flow.py`
- 배치 추론 로더: `ml/deepfm_pipeline/utils/score_batch.py`의 `load_run()`

### 1-3. 출력(서비스 계약 준수 출력)

AI 서버는 추천 결과를 **서비스 계약 경로**로 S3에 저장하고, 완료 마커를 생성한다.

- **Bucket**: `tasteam-{env}-analytics`
- **Key 규칙**:

```
recommendations/
  pipeline_version=VERSION/
    dt=YYYY-MM-DD/
      part-00001.csv
      _SUCCESS
```

- **추천 결과 CSV 스키마(헤더)**:
  - `user_id`, `anonymous_id`, `restaurant_id`
  - `score`, `rank`
  - `context_snapshot`
  - `pipeline_version`, `generated_at`, `expires_at` (TTL 기본 24h)

추천 결과 생성 코어:
- `ml/deepfm_pipeline/utils/score_batch.py`의 `run()`

S3 업로드(+ `_SUCCESS`)는 현재 배치 스크립트가 담당:
- `ml/deepfm_pipeline/scripts/score_batch_to_s3.py`

---

## 2) 실행 주체(트리거 방식)

### 2-1. “폴링 기반” 운영 전제

서비스 계약상 추천 결과의 DB 적재는 **API 서버가 S3를 polling** 해서 수행한다.

- AI 서버: 추천 결과 파일 + `_SUCCESS` 생성
- API 서버: `(pipeline_version, dt)` 탐색 → `_SUCCESS` 확인 → CSV import → DB 적재

관련 계약:
- `docs/service_extraction/service_constract.md`의 §5~§6

### 2-2. 현재 구현에서의 추천 생성 트리거(HTTP 아님)

현재 레포에서는 추천 생성(배치 추론)을 **HTTP 엔드포인트로 트리거하지 않는다.**  
추천 생성 실행은 배치 잡/스크립트로 수행한다.

- 스크립트: `ml/deepfm_pipeline/scripts/score_batch_to_s3.py`

예시(개념):

```bash
python ml/deepfm_pipeline/scripts/score_batch_to_s3.py \
  --pipeline-version deepfm-1.0.20260227120000 \
  --candidates-path /path/to/candidates.csv \
  --meta-path /path/to/candidates_meta.csv \
  --env dev \
  --dt 2026-03-10
```

> 실제 운영에서는 이 스크립트를 cron/Prefect/워크플로 엔진 등으로 스케줄링한다.

---

## 3) Admin API로 남아있는 기능(현재)

> **Deprecated (2026-03-17)**: 이 섹션에서 설명하던 **Admin API는 삭제됨**.

현재 레포에서 “운영 트리거”는 HTTP가 아니라 다음 방식으로 수행한다.

- **학습**: `ml/deepfm_pipeline/training_flow.py` 실행(로컬/Pefect 배포/스케줄러)
- **배치 추론(추천 생성)**: `ml/deepfm_pipeline/scripts/score_batch_to_s3.py` 실행(크론/Prefect/워크플로 엔진)

---

## 4) 계약 대비 갭(현재 구현 관점에서 중요한 차이)

### 4-1. 추론 입력이 “raw”가 아니라 “feature vector 후보”인 점

`service_constract.md`는 raw(events/restaurants/menus) 적재 계약을 정의하지만,
현재 구현의 추천 생성 코어(`utils/score_batch.py`)는 아래를 전제로 한다.

- upstream에서 이미 후보를 만들고
- 후보를 **모델 입력 형태(feature vector)** 로 전처리한 뒤
- 그 결과를 `candidates.csv`로 전달

따라서 “raw → feature 생성” 단계는 이 레포에서 완결되지 않으며,
외부 ETL/피처 파이프라인(또는 향후 구현)에서 수행되어야 한다.

### 4-2. user_id / member_id 혼용

서비스 계약의 추천 결과 스키마는 `user_id`를 사용한다.  
현재 구현의 추천 결과 CSV도 `user_id`를 출력한다.

다만 일부 문서/데이터에서 `member_id`가 섞일 수 있어,
메타 CSV 입력에서는 호환을 제공한다(구현 상세는 `utils/score_batch.py` 참고).

---

## 5) 관련 파일 맵(현재 구현 기준)

- **추천 결과 생성(코어)**: `ml/deepfm_pipeline/utils/score_batch.py`
- **S3 업로드 + `_SUCCESS` 생성(배치)**: `ml/deepfm_pipeline/scripts/score_batch_to_s3.py`
- **학습 플로우(Prefect)**: `ml/deepfm_pipeline/training_flow.py`
- **전처리(학습 데이터 생성)**: `ml/deepfm_pipeline/utils/dataPreprocess.py`
- **평가(NDCG/Recall/AUC)**: `ml/deepfm_pipeline/utils/evaluate.py`
- ~~Admin API~~: (삭제됨, 2026-03-17)

