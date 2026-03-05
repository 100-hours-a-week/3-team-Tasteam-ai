# Option B1: comparison_lib 패키지 분리 플랜

## 목표
Spark 서비스가 `src`를 import하지 않도록 하여, Spark 이미지에서 qdrant-client 등 불필요 의존성 제거.

## 1. comparison_lib 패키지 생성 (프로젝트 루트)

| 파일 | 설명 |
|------|------|
| `comparison_lib/__init__.py` | comparison_pipeline·json_parse_utils 공개 API 노출 |
| `comparison_lib/config.py` | Spark/비율용 최소 설정만 (SPARK_SERVICE_URL, DISABLE_SPARK, RECALL_SEEDS_SPARK_THRESHOLD, os.getenv) |
| `comparison_lib/json_parse_utils.py` | `src/json_parse_utils.py` 전체 복사 |
| `comparison_lib/data/stopwords-ko.txt` | `src/data/stopwords-ko.txt` 복사 (recall_seeds 불용어) |
| `comparison_lib/comparison_pipeline.py` | `src/comparison_pipeline.py` 복사 후 `.json_parse_utils`, `.config` 사용 및 Config → comparison_lib.config 로 치환 |

## 2. comparison_pipeline 내부 변경 (comparison_lib 버전)

- `from .json_parse_utils import parse_json_relaxed` 유지
- `from src.config import Config` 제거 → `from . import config as _config` 또는 `from .config import ...` 후 `_get_spark_service_url()`, `_spark_disabled()`, `_comparison_spark_threshold()` 등에서 `getattr(Config, ...)` 대신 `_config.SPARK_SERVICE_URL` 등 사용
- `_load_stopwords_for_recall`: `Path(__file__).resolve().parent` 기준이므로 `comparison_lib/data/stopwords-ko.txt` 참조하도록 유지 (파일만 복사)

## 3. src 쪽 re-export (기존 호환)

- `src/json_parse_utils.py`: `from comparison_lib.json_parse_utils import *` (+ `__all__`) 로 대체
- `src/comparison_pipeline.py`: `from comparison_lib.comparison_pipeline import *` (+ `__all__`) 로 대체  
  → 기존 `from src.comparison_pipeline import ...`, `from .comparison_pipeline import ...` 유지

## 4. Spark 서비스 import 변경

- `servjces/spark/main.py`: `from src.comparison_pipeline import (...)` → `from comparison_lib.comparison_pipeline import (...)`  
- `from src.config import Config` 제거 (미사용)

## 5. Docker / 의존성

- `Dockerfile.spark-service`: `COPY comparison_lib /app/comparison_lib` 추가. `scripts/spark_service.py`가 프로젝트 루트를 path에 넣으므로 `comparison_lib` 자동 인식.
- `requirements-spark.txt`: `qdrant-client` 제거 (Spark 서비스가 src를 타지 않으므로 불필요).

## 6. 검증

- 메인 앱: `from src.comparison_pipeline import ...` / `from .comparison_pipeline import ...` 동작 확인
- Spark 서비스: `from comparison_lib.comparison_pipeline import ...` 만 사용, 컨테이너 기동 및 /health, /all-average-from-reviews 등 호출 확인
- `src` 내 다른 모듈의 `from .json_parse_utils import ...` 동작 확인 (re-export 경유)

---

## 완료 사항 (B1 적용 후)

- `comparison_lib/` 패키지 생성: `config.py`, `json_parse_utils.py`, `comparison_pipeline.py`, `data/stopwords-ko.txt`
- `src/comparison_pipeline.py`, `src/json_parse_utils.py` → comparison_lib re-export으로 교체
- `servjces/spark/main.py` → `comparison_lib.comparison_pipeline` 직접 import, `src`·`Config` 제거
- `Dockerfile.spark-service` → `COPY comparison_lib` 추가, `COPY src` 제거 (Spark 이미지에 src 없음, qdrant-client 불필요)
