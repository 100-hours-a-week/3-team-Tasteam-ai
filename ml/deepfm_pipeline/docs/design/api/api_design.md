네 설계 기준으로 “필요한” DeepFM 관련 API 최소 셋
1) (필수) 학습 트리거

POST /admin/deepfm/train

Prefect 학습 플로우 실행 트리거

산출물: 모델 artifact + pipeline_version

이건 네가 말한 /admin/deepfm/train 그대로 OK.

2) (필수) 배치 스코어링/추천 생성 트리거

POST /admin/deepfm/score-batch (또는 /admin/recommendation/pipeline/run)

너 스펙에서 “추천 생성: 피처 조합 → 스코어 계산 → recommendation 저장”이 핵심 흐름이니까 

recommendation_techspec


학습과 별개로 추천 생성 잡을 실행할 수 있어야 해.

입력: pipeline_version, 대상 유저 범위/시간대, TTL(기본 24h) 

recommendation_techspec

출력: recommendation INSERT 완료 (score/rank/context_snapshot/pipeline_version/generated_at/expires_at) 

recommendation_techspec

3) (선택이지만 강추) 모델/버전 조회 & 활성화

GET /admin/deepfm/models
POST /admin/deepfm/activate

왜냐면 추천 응답에도 pipelineVersion을 내려주도록 돼 있고, 

recommendation_techspec


recommendation.pipeline_version 컬럼도 필수라서 

recommendation_techspec


어떤 모델이 현재 “서빙용 파이프라인 버전”인지 관리 포인트가 생겨.