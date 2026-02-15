# 문서 인덱스

문서를 카테고리별 디렉터리로 정리했습니다. 같은 주제는 같은 폴더에 묶여 있습니다.

| 디렉터리 | 내용 |
|----------|------|
| **architecture/** | 아키텍처 개요, 배포 구조, MSA·모놀리식 비교 |
| **api/** | API DTO·엔드포인트 명세 |
| **changelog/** | 최근 변경 이력 |
| **runpod/** | RunPod Pod/Serverless, GPU 플랫폼, Lambda·배치 연동, 네트워크 볼륨 |
| **batch/** | 오프라인 배치 전략·사용법, RQ 워커, trigger_offline_batch |
| **spark/** | Spark 마이크로서비스, recall seeds, Summary 시드 |
| **troubleshooting/** | 트러블슈팅 로그·요약, RunPod Pod/Serverless 로그, Postman·HF 등 이슈 |
| **operations/** | 부하 테스트, 운영 관련 |
| **analysis/** | 파이프라인 분석, 실험·참고 문서 |

## 주요 문서 빠른 링크

- [아키텍처 전체](architecture/ARCHITECTURE_OVERVIEW.md)
- [API DTO](api/API_DTO.md)
- [최근 변경사항](changelog/CHANGES_RECENT.md)
- [Spark 서비스](spark/SPARK_SERVICE.md)
- [RunPod Pod + Lambda 배치](runpod/lambda_runpod_pod.md)
- [오프라인 배치 전략](batch/offline_batch_strategy.md)

## 기타

- **etc_md/** — 프로젝트 루트의 기타 참고 문서(요약 파이프라인, 메트릭, Qdrant, vLLM 등). 카테고리별로 아직 docs 하위로 옮기지 않은 문서들.
- **hybrid_search/**, **model_selection/**, **recommend_system/** — 각 기능별 문서는 해당 디렉터리에 유지.
