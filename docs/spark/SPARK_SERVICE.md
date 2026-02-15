# Spark 마이크로서비스(MSA) (Comparison 전용)

Comparison 파이프라인에서 **전체 평균 비율**·**recall seeds** 계산 시 Spark를 사용합니다.  
**모놀리식(Monolithic)** 방식이라면 API/워커 프로세스 안에 pyspark·JVM을 함께 로드해야 하므로 기동·메모리 부담이 커집니다.  
이 프로젝트는 이 부분을 **마이크로서비스(MSA)** 로 분리해, 메인 API/워커에서는 JVM을 로드하지 않고 HTTP로만 Spark 서비스를 호출합니다.

## 구성

- **Spark 서비스**: `scripts/spark_service.py` — FastAPI 앱, pyspark 로드, 다음 엔드포인트만 제공.
- **메인 앱/워커**: `SPARK_SERVICE_URL` 환경 변수 설정 시 로컬 Spark 대신 해당 URL로 HTTP 요청.

## Spark 서비스 실행

```bash
# 기본 포트 8002
python scripts/spark_service.py

# 포트 지정
SPARK_SERVICE_PORT=9000 python scripts/spark_service.py
```

필요: `pyspark`, `fastapi`, `uvicorn`, `httpx` (서비스·메인 앱 모두).

## 환경 변수 (메인 앱/워커)

| 변수 | 설명 |
|------|------|
| `SPARK_SERVICE_URL` | Spark 서비스 base URL (예: `http://localhost:8002`). 설정 시 전체 평균·recall seeds를 이 서비스로 요청. |

미설정 시: 기존과 동일하게 **로컬 Spark** 사용(가능한 경우) 또는 **Python(Kiwi) 폴백**.

## 엔드포인트 (Spark 서비스)

| 메서드 | 경로 | body | 응답 |
|--------|------|------|------|
| POST | `/all-average-from-file` | `{"path": "/abs/path", "project_root": null}` | `{"service": float, "price": float}` |
| POST | `/recall-seeds-from-file` | `{"path": "/abs/path", "project_root": null}` | `{"service": [[phrase, count], ...], "price": ..., "food": ..., "other": ...}` |
| GET | `/health` | - | `{"status": "ok"}` |

파일 경로는 **Spark 서비스 프로세스가 읽을 수 있는 경로**여야 합니다(공유 볼륨 또는 서비스 내부 경로).

## Config 추가 사항

- `Config.SPARK_SERVICE_URL`: `src/config.py` 의 `_SparkConfig` 에 추가됨.
- `Config.RECALL_SEEDS_SPARK_THRESHOLD`: Summary용 **음식점별 recall seed** 계산 시, 리뷰 수가 이 값 미만이면 로컬 Python(Kiwi), 이상이면 Spark 사용. 기본 2000. 상세: [SUMMARY_RECALL_SEEDS.md](SUMMARY_RECALL_SEEDS.md).
