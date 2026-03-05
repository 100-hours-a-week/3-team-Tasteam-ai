
두 가지를 나누어 보면 아래와 같습니다.

---

## 1. Option A: `src/__init__.py`에서 `vector_search`를 필요할 때만 import (lazy import)

**방식**: `src/__init__.py`에서 `from .vector_search import ...` 를 제거하고,  
`vector_search` 관련 이름은 **첫 사용 시점에만** import 하도록 바꿈 (함수/프로퍼티로 한 번 더 감싸기 등).

| 장점 | 단점 |
|------|------|
| Spark 서비스는 `from src.comparison_pipeline import ...` 만 쓰므로 `vector_search`가 로드되지 않음 → Spark 이미지에서 qdrant-client 제거 가능 | `from src import get_vector_search` 같은 기존 사용처가 있으면, 같은 이름을 lazy로 노출해 줘야 해서 `__init__.py` 수정이 필요함 |
| 한 곳(`src/__init__.py`)만 고치면 됨, 구조 변경은 작음 | Lazy import는 “처음 쓰는 시점”에 import가 일어나서, 그때 에러 나면 스택이 덜 직관적일 수 있음 |
| 메인 앱·테스트 등은 기존처럼 `from src import ...` 로 쓸 수 있음 (lazy로 같은 API 유지 시) | `__init__.py`에 로직이 늘어남 (lazy 래퍼/함수 등) |

---

## 2. Option B: Spark 서비스가 `comparison_pipeline`만 직접 import 해서 `src/__init__.py`를 안 타게 하기

**중요**: Python에서는 `from src.comparison_pipeline import ...` 를 하면 **반드시** `src` 패키지가 먼저 로드되고, 그때 `src/__init__.py`가 실행됩니다.  
그래서 “같은 `src` 패키지 안에서 comparison_pipeline만 골라서 import 하되 __init__.py는 실행 안 하기”는 **불가능**합니다.

그래서 Option B를 하려면 **구조를 바꿔서**, Spark 서비스가 **`src`를 아예 import 하지 않게** 만들어야 합니다.

- **B1**: `comparison_pipeline`(와 그 의존)을 `src` 밖으로 빼서 별도 패키지로 두기  
  (예: `comparison_lib` 같은 공용 패키지).  
  Spark 서비스는 `from comparison_lib.comparison_pipeline import ...` 만 쓰고, `src`는 import 안 함.
- **B2**: Spark 전용 진입 모듈을 두고, 그 안에서만 `comparison_pipeline`에 필요한 경로를 `sys.path`에 넣어서 `src` 패키지 없이 `comparison_pipeline` 단일 모듈처럼 로드하는 방식.  
  (이 경우 `comparison_pipeline`의 `from .json_parse_utils import ...` 같은 상대 import 때문에 구조가 얽히고, 유지보수가 더러워질 수 있음.)

실제로 현실적인 “Option B”는 **B1처럼 comparison_pipeline을 공용 패키지로 분리**하는 것입니다.

| 장점 | 단점 |
|------|------|
| Spark 서비스는 `src`를 전혀 안 타므로 qdrant, vector_search 등과 완전히 분리됨 | 디렉터리/패키지 구조 변경이 큼 (`comparison_lib` 생성, 기존 `from src.comparison_pipeline` 사용처 전부 수정) |
| “Spark가 쓰는 코드”와 “메인 앱 전용 코드” 경계가 명확해짐 | `comparison_pipeline`이 쓰는 `config`, `json_parse_utils` 등도 공용 패키지로 옮기거나 의존성 정리가 필요함 |
| `src/__init__.py`를 건드리지 않아도 됨 | 한 번에 수정할 파일이 많고, 테스트/import 경로 전반 점검 필요 |

---

## 3. 요약 비교

| 기준 | Option A (lazy import) | Option B (구조 분리) |
|------|------------------------|----------------------|
| 변경 범위 | `src/__init__.py` + (필요 시) 사용처 소량 | 패키지 구조 + 여러 파일 import 경로 변경 |
| Spark 이미지에서 qdrant 제거 | 가능 | 가능 |
| 리스크 | 기존 `from src import ...` 계약 유지만 잘 하면 됨 | 리팩터 범위가 커서 회귀 가능성 있음 |
| 유지보수 | `__init__.py`에 lazy 로직이 남음 | 경계가 분리돼서 나중에 Spark 전용 의존성 관리가 쉬움 |

**실용적인 선택**:  
- **당장 부담 적게** 가려면 **Option A (lazy import)** 가 수정 범위가 작고, Spark만 “딱 스파크 요청만 받게” 두는 목적에는 충분합니다.  
- **장기적으로** Spark/메인 앱 경계와 의존성을 깔끔히 나누고 싶다면 **Option B (comparison_pipeline 공용 패키지 분리)** 를 고려할 수 있습니다.