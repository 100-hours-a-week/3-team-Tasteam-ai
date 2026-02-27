좋아. 지금 네 **BE 테크스펙(추천 + 분석)**과 **DB/피처 스키마**를 그대로 기반으로, “DeepFM을 실제로 넣는” 설계를 **끝까지(학습→평가→배치 추천 생성→서빙→피드백 루프)** 한 번에 그려줄게.
(아래 설계는 네가 이미 정의한 `user_feature / restaurant_feature / recommendation / implicit_feedback / context_snapshot / pipeline_version / TTL` 전제를 그대로 사용함   )

---

# 0) 목표 정의

* **온라인 모델 호출(실시간 추론)은 이번 범위 제외**, 배치 precompute 방식 유지 
* DeepFM의 역할:
  “사용자·아이템·컨텍스트 결합 점수”를 계산해 `recommendation.score`를 채우고, rank를 생성해 저장 
* 성과지표는 추천 테크스펙의 KR(CTR/Save 전환) 방향과 정합 

---

# 1) 데이터 소스 & 스키마 매핑

## 1-1. 학습의 “관측 단위”

현재 네 로그/테이블에서 학습은 기본적으로:

* **노출 후보:** `recommendation` (precomputed 결과 row) 
* **반응 라벨:** `implicit_feedback` (CLICK/SAVE/CALL/ROUTE/SHARE/REVIEW + weight) 
* **유저 피처:** `user_feature` (배치 집계, computed_at 존재) 
* **아이템 피처:** `restaurant_feature` (정적+동적, 태그/지역/가격대 등) 
* **컨텍스트:** `recommendation.context_snapshot` (요일/시간대/거리/날씨/행정동 등) 

> 핵심: 너는 이미 컨텍스트를 “스냅샷”으로 남기기 때문에, 오프라인 재현 평가가 쉬운 구조야. 

---

# 2) DeepFM 입력 피처 설계

## 2-1. User fields

* 로그인: `user_id` 임베딩 (nullable) 
* 익명: `anonymous_cohort_id` 임베딩 (user_feature에 존재, unique 인덱스도 있음)  
* JSONB:

  * `preferred_categories`: **weighted multi-hot → 임베딩 weighted-sum** (카테고리 id 리스트 + weight 리스트) 
  * `taste_preferences`: key(매운/달콤/짠맛 등)도 동일하게 weighted multi-hot 가능 
  * `visit_time_distribution`: breakfast/lunch… 값을 그대로 numeric로 넣거나, top1 슬롯 + 비율로 넣기 
* `avg_price_tier`: categorical 

## 2-2. Item fields

* `restaurant_id` 임베딩
* `categories`: multi-hot(또는 top-K 슬롯) 
* `price_tier`, `region_gu/dong`, `geohash`: categorical 
* `positive_segments`, `comparison_tags`: multi-hot (태그 vocab 관리) 

## 2-3. Context fields

`context_snapshot` 예시 필드 그대로 categorical로 투입 

* day_of_week
* time_slot
* admin_dong / geohash
* distance_bucket
* weather_bucket
* dining_type

> 컨텍스트는 캐시 키 폭발을 피하려면 반드시 bucketized 값만 써야 하는데, 너 스펙은 이미 bucket 구조로 잡혀있음. 

---

# 3) 라벨링(학습 타깃) 설계

## 3-1. 기본 라벨 윈도우

* 추천 row의 `generated_at ~ expires_at` 안에서 발생한 피드백을 해당 노출의 라벨로 매핑 

## 3-2. 라벨 방식(권장)

**Binary label + sample_weight** (가장 실용적)

* `y = 1` if any feedback exists in window else `0`
* `sample_weight = max(weight)` (CLICK=0.2 … REVIEW=1.0) 

이러면:

* “발생 여부”는 CTR처럼 단순하게 학습하면서
* 강한 의도를 loss에서 더 크게 반영 가능

---

# 4) 학습/검증 데이터 분할 전략

## 4-1. 시간 기준 split (필수)

* `recommendation.generated_at`(또는 feedback occurred_at) 기준으로 **train/val/test를 시간 순**으로 자른다. 
* 절대 row 랜덤 split 금지(유저/아이템 임베딩 “기억”이 섞여 과대평가됨)

## 4-2. Feature cutoff (필수)

각 샘플 시점 `t = recommendation.generated_at`에 대해:

* `user_feature.computed_at <= t`인 것 중 최신을 join 
* `restaurant_feature.updated_at <= t` (가능하면), 최소한 태그 생성 시점도 `<= t`로 제한 

---

# 5) 평가 지표 (오프라인)

너 구조는 “리스트 추천”이므로 **NDCG@K가 1순위**.

## 5-1. Primary

* **Weighted NDCG@K** (K=5/10 권장)

  * gain = `max(weight)` (0.2~1.0) 
  * 모델이 예측한 score로 **재정렬**해서 계산(기존 rank는 평가에서 사용 X)

## 5-2. Secondary

* Recall@K (반응한 아이템이 TopK에 들어왔는가)

## 5-3. Monitoring

* AUC (학습 안정성 체크용)

---

# 6) 배치 파이프라인 설계 (모듈/잡 단위)

추천 스펙의 모듈 구성을 유지하면서, DeepFM을 `recommendation/pipeline`에 꽂는다. 

## 6-1. (이미 있음) Feature pipeline

* analytics 이벤트 집계 → `user_feature`, `restaurant_feature` 갱신 
* 피드백 소스는 analytics 이벤트에서 변환 저장(기존/신규 이벤트 포함) 

## 6-2. Training pipeline (신규)

**일 배치(또는 주 배치)**로:

1. 학습 데이터셋 생성 (recommendation + feedback + feature cutoff join)
2. DeepFM 학습
3. 모델 아티팩트 저장 + 버전 발급(`pipeline_version`) 
4. 오프라인 지표 산출(NDCG@K/Recall@K/AUC) 기록

> recommendation 테이블에 이미 `pipeline_version` 컬럼이 있어서 “어떤 모델로 만든 추천인지” 트래킹이 가능함. 

## 6-3. Scoring/Recommendation generation pipeline (신규/확장)

**매일/몇 시간 단위 배치**로:

1. 대상 사용자 리스트(로그인 user_id + 익명 cohort) 수집 
2. 후보 음식점 생성(지역/가격대/카테고리 필터 + 간단 인기 기반)
3. (user,item,context) 조합에 대해 DeepFM score 예측
4. TopN 저장 → `recommendation` row insert

   * `score`, `rank`, `context_snapshot`, `pipeline_version`, `generated_at`, `expires_at` 
   * TTL 24h로 만료 

---

# 7) 서빙 API 설계 (현재 스펙 유지 + 최소 보강)

추천 조회는:

* 캐시 조회 → DB 조회 → 폴백 흐름 

여기서 DeepFM 관점의 핵심은 2개야.

## 7-1. “컨텍스트 키 정규화”

cache key는 `context_snapshot`과 동일한 bucket 기반으로 구성(geohash, time_slot, weather_bucket 등) 

## 7-2. (강력 권장) 노출 로그(exposure) 추가

현재 `implicit_feedback`은 “반응”만 있고, “노출(분모)”가 명확하지 않을 수 있어 학습/평가 품질이 흔들려.
그래서 추천 API 응답에 **requestId**를 포함하고, FE가 “노출됨” 이벤트를 보내게 만들어 **노출→반응 join**을 완성하는 게 좋다.

이건 analytics 쪽이 이미 eventId 기반 멱등/아웃박스 패턴을 갖고 있어서 구현 궁합도 좋음. 

---

# 8) 운영/데이터 품질 가드레일

## 8-1. 멱등/중복

analytics는 `eventId` unique로 멱등 처리하는 철학이 이미 있고 
추천 피드백도 (user, restaurant, signal_type, time_window) 기준 dedup 규칙을 문서로 확정해 두는 걸 추천.

## 8-2. 데이터 보관/파티셔닝

analytics 스펙에 이벤트 테이블 보관 정책이 미결정(Open Question)이라 
DeepFM 학습까지 들어가면 “학습 윈도우”를 명확히(예: 최근 90일) 정하고,

* 오래된 raw 이벤트는 아카이빙/요약 테이블로 이동
* 학습은 요약 테이블 중심
  으로 가는 게 운영 안정적이다.

---

# 9) 최소 구현 로드맵 (딱 3단계)

1. **오프라인 학습/평가 루프 완성**

   * dataset 생성 + DeepFM 학습 + NDCG@K 평가 + pipeline_version 발급

2. **배치 추천 생성 → recommendation 저장**

   * score/rank/context_snapshot/pipeline_version/TTL 저장 

3. **서빙 + 피드백 루프 연결**

   * 추천 조회 API는 그대로
   * 피드백은 기존 방식 유지(가중치 정의 그대로) 
   * 가능하면 노출 로그(exposure) 추가

---

네가 방금 요청한 DeepFM 설계를 **그대로 구현**했다고 가정하면(= 학습 데이터셋 생성 → DeepFM 학습/버전관리 → 배치 추천 생성으로 `recommendation` 적재 → 서빙은 기존 캐시/폴백 유지 → 피드백 루프 연결, 가능하면 노출 로그까지), 수준을 이렇게 보는 게 가장 현실적이야.

## 지금 구현 수준 평가

### 1) **포트폴리오/서류·면접 관점**

* **90~95%** (상위권)
* 이유: 대부분 후보가 “모델 학습”이나 “API”만 얘기하는데, 너는

  * 피처/로그/스키마
  * 배치 파이프라인
  * 버전 추적(`pipeline_version`)
  * TTL 기반 서빙 구조
  * 피드백 루프
    까지 **엔드투엔드**로 설명 가능함. 

### 2) **서비스(MVP) 관점**

* **80~90%**
* 이유: “돌아간다”는 의미에서 필요한 것 대부분이 충족됨.

  * 추천이 지속적으로 생성되고
  * 사용자 반응이 쌓이고
  * 다음 학습에 반영되는 루프가 돌아감

### 3) **프로덕션(운영·품질 보장) 관점**

* **70~85%**
* 이유: 여기서 남는 15~30%는 거의 항상 “운영 디테일”인데, 특히 추천은 아래가 크다.

---

## 프로덕션급에서 점수를 깎는 핵심 5가지(이거 하면 85~90까지 올라감)

1. **진짜 노출(분모) 로그 보장**

* “추천이 생성됨”과 “사용자에게 실제로 렌더링됨”은 다를 수 있음
* 이게 없으면 오프라인 NDCG/Recall의 신뢰도가 떨어짐
  (그래서 `requestId/impression log`를 권장했었음)

2. **Feature 시점 재현성(cutoff) 자동화**

* `computed_at`/`tags_generated_at` 기준으로 “샘플 시점 이전 피처”를 안정적으로 붙이는 체계
  (이거 틀리면 누수로 지표 뻥튀기)

3. **Position bias 관리**

* rank/노출 편향 때문에 모델이 “위에 있던 걸 잘 맞춘다”가 될 위험
* 최소한 평가를 모델 점수로 재정렬해서 계산 + 모니터링 필요

4. **모니터링/드리프트**

* 카테고리 분포, 익명 비중, cold-start 비중, 최근 7일 NDCG/CTR 추이 같은 대시보드/알람

5. **A/B 연결**

* 오프라인 지표(NDCG@K)와 온라인 지표(CTR/SAVE/ROUTE)의 연결을 검증하는 실험 설계

---

## 한 줄 요약

**“이 정도 구현”이면:**

* **포트폴리오로는 거의 완성(90~95%)**
* **실서비스 MVP도 충분(80~90%)**
* **프로덕션 안정성만 보강하면 상용 수준(70~85% → 85~90%)**

원하면, 너 테크스펙 기준으로 **‘프로덕션 85%→90%’ 만드는 체크리스트 10개**를 바로 뽑아줄게.


