응, **달아두는 게 확실히 좋다.** 다만 “무조건 전부”가 아니라, **Prefect 파이프라인의 신뢰성/재현성에 직접 기여하는 범위**로 붙이는 게 베스트야.

## 1) W&B Artifacts: 사실상 필수에 가깝다

Prefect로 파이프라인이 돌아가면, 운영/면접 관점에서 제일 먼저 나오는 질문이 이거거든:

* “그 run에서 만든 모델/데이터가 정확히 뭐였지?”
* “나중에 동일 결과 재현 가능해?”
* “추천 결과가 나빠졌을 때 어떤 버전이 문제였지?”

Artifacts가 있으면 한 방에 정리돼.

### Artifacts로 남기면 좋은 것 (너 파이프라인에 딱 맞음)

* **train/val/test split 메타**(시간 범위, cutoff 기준, seed)
* **feature vocab**(카테고리/태그/코호트 id 매핑)
* **train dataset snapshot**(가능하면 full 말고 샘플+통계/해시)
* **model checkpoint**(+ config)
* **evaluation report**(NDCG@K/Recall@K/AUC)
* **scoring output**(예: score_batch.py 결과 CSV)와 `pipeline_version` 매핑

> 특히 너는 `pipeline_version`을 DB에 저장하도록 설계했으니까,
> `pipeline_version ↔ wandb artifact version` 연결만 해두면 운영 추적성이 급상승해.

## 2) W&B Sweep: “좋긴 한데” 조건부로 추천

Sweep을 붙이면 좋은 이유:

* DeepFM/LoRA와 달리 DeepFM도 **lr, embedding dim, dropout, l2, batch size** 같은 하이퍼파라미터에 성능이 꽤 흔들림
* Prefect로 실험을 자동화해두면 “재현 가능한 최적화 루프”가 됨

하지만 주의점도 있음:

* 추천은 split/시간 누수에 취약해서, Sweep이 **validation set에 과적합**할 수 있음
* 비용/시간(자원)이 늘어남

### Sweep을 붙일 때의 “안전한 운영 방식”(추천)

* Sweep은 **고정된 time-based val 구간**에서만 수행
* 상위 2~3개 설정만 뽑아서

  * seed 3회 반복 또는
  * rolling backtest 2~3구간으로 재검증
* 최종은 test 1회

즉, Sweep 자체를 목적화하지 말고 **후보 탐색용**으로 쓰면 깔끔해.

## 3) Prefect에 어떻게 “예쁘게” 붙이나 (구성만)

Prefect Flow 기준으로는 보통 이렇게 레이어링하면 가장 납득이 잘 돼:

* `build_dataset` task
  → dataset stats + split metadata를 **artifact로 로깅**
* `train_model` task
  → checkpoint/model card를 **artifact로 로깅**
* `evaluate` task
  → metrics를 W&B run에 로깅 + report artifact
* `score_batch` task
  → scoring CSV를 artifact로 로깅
* `load_to_db` task
  → `pipeline_version`와 W&B run/artifact id를 DB에 함께 기록(또는 로그로 남김)

## 결론

* **Artifacts:** 달아두는 게 거의 “필수급” (재현성/버전추적/디버깅/포트폴리오 전부 이득)
* **Sweep:** 달아두면 좋지만, **검증 과적합 방지 프로토콜**(seed 반복/rolling 재검증)까지 같이 설계하면 더 좋음

원하면, 네 Prefect flow 구조(태스크 이름만 대충) 기준으로 **W&B artifact 네이밍 규칙 + pipeline_version 매핑 규칙**을 딱 정해줄까?
