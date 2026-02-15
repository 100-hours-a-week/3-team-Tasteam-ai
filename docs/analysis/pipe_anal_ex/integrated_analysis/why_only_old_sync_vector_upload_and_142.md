[13 tools called]


세 가지 질문에 대해 정리하면 다음과 같습니다.

---

## 1. old_sync만 벡터 업로드를 진행한 이유

**원인: 임베딩 모델 백엔드 차이**

- **old_sync** (`tasteam-old-sync`): `sentence-transformers`(PyTorch) 사용 → `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` 로드 성공 후 업로드 처리
- **new_sync / new_async** (`tasteam-new-async`): `xenova`(ONNX/fastembed) 사용 → `xenova/paraphrase-multilingual-mpnet-base-v2` ONNX 로드 시 실패

new_sync 로그:

```
ONNXRuntimeError: Load model from .../onnx/model.onnx failed. File doesn't exist
```

즉, 업로드 요청은 세 포트 모두 도달했지만, new_sync/new_async는 임베딩 모델 로딩 단계에서 크래시해서 업로드가 처리되지 않은 것입니다.

---

## 2. tasteam_app_kr3_640k_even.json 사용 시에도 142개만 업로드된 이유

**벡터 업로드와 부하 테스트가 다른 스크립트/데이터를 사용합니다.**

- **부하 테스트** (`test_all_task.py --load-test`): `--load-test-data tasteam_app_kr3_640k_even.json` → 감성/요약/비교 API 호출에 사용
- **벡터 업로드**: `test_all_task.py --load-test`에서는 수행하지 않음. 업로드는 `run_all_restaurants_api.py` 또는 별도 초기화 단계에서 실행

업로드용 데이터는 코드에서 `test_data_sample.json` 기반으로 구성됩니다:

```280:285:test_all_task.py
    # 기본: test_data_sample.json 파일에서 테스트 데이터 로드
    ...
    test_data_path = project_root / "data" / "test_data_sample.json"
```

`test_data_sample.json`은 142개 리뷰, 5개 레스토랑 규모로 되어 있어, 업로드 단계에서는 이 데이터가 사용되고, 640k JSON은 부하 테스트용 API 페이로드에만 사용됩니다.

---

## 3. 3개 포트에 동시 요청 시 new_sync/new_async에서 벡터 업로드가 진행되지 않은 이유

**업로드 요청은 세 포트 모두 수신했지만, new_sync/new_async는 업로드 처리 전에 크래시합니다.**

| 포트 | 이미지 | 임베딩 백엔드 | 결과 |
|------|--------|----------------|------|
| 8001 (old_sync) | tasteam-old-sync | sentence-transformers (PyTorch) | 정상 로드 후 업로드 완료 |
| 8002 (new_sync) | tasteam-new-async | xenova (ONNX) | ONNX 파일 부재로 크래시 → 업로드 미완료 |
| 8003 (new_async) | tasteam-new-async | xenova (ONNX) | 동일 크래시 → 업로드 미완료 |

타임라인은 다음과 같습니다.

- **15:24:17**: old_sync에서 벡터 업로드 처리 시작
- **15:24:27**: new_sync/new_async에서 xenova 모델 로딩 시도 → ONNX 경로에서 크래시
- **15:24:30**: `ONNXRuntimeError: model.onnx failed. File doesn't exist` 발생

즉, 3개 포트 모두 업로드 요청을 받았지만, new_sync/new_async는 임베딩 모델 로딩 단계에서 실패해 업로드 처리까지 진행하지 못한 상황입니다.

---

## 요약

1. **old_sync만 업로드**: sentence-transformers만 정상 로드되었고, new_sync/new_async는 xenova ONNX 로딩 중 크래시.
2. **142개만 업로드**: 업로드 단계는 `test_data_sample.json`(142개 리뷰, 5개 레스토랑) 기반이며, 640k JSON은 부하 테스트용 API 페이로드용.
3. **3개 포트 요청 시**: 업로드 요청은 세 포트 모두 도달했으나, new_sync/new_async는 임베딩 모델 로딩 단계에서 크래시하여 업로드를 완료하지 못함.

---
---

[13 tools called]


세 가지 질문에 대해 정리하면 다음과 같습니다.

---

## 1. old_sync만 벡터 업로드를 진행한 이유

**원인: 임베딩 모델 백엔드 차이**

- **old_sync** (`tasteam-old-sync`): `sentence-transformers`(PyTorch) 사용 → `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` 로드 성공 후 업로드 처리
- **new_sync / new_async** (`tasteam-new-async`): `xenova`(ONNX/fastembed) 사용 → `xenova/paraphrase-multilingual-mpnet-base-v2` ONNX 로드 시 실패

new_sync 로그:

```
ONNXRuntimeError: Load model from .../onnx/model.onnx failed. File doesn't exist
```

즉, 업로드 요청은 세 포트 모두 도달했지만, new_sync/new_async는 임베딩 모델 로딩 단계에서 크래시해서 업로드가 처리되지 않은 것입니다.

---

## 2. tasteam_app_kr3_640k_even.json 사용 시에도 142개만 업로드된 이유

**벡터 업로드와 부하 테스트가 다른 스크립트/데이터를 사용합니다.**

- **부하 테스트** (`test_all_task.py --load-test`): `--load-test-data tasteam_app_kr3_640k_even.json` → 감성/요약/비교 API 호출에 사용
- **벡터 업로드**: `test_all_task.py --load-test`에서는 수행하지 않음. 업로드는 `run_all_restaurants_api.py` 또는 별도 초기화 단계에서 실행

업로드용 데이터는 코드에서 `test_data_sample.json` 기반으로 구성됩니다:

```280:285:test_all_task.py
    # 기본: test_data_sample.json 파일에서 테스트 데이터 로드
    ...
    test_data_path = project_root / "data" / "test_data_sample.json"
```

`test_data_sample.json`은 142개 리뷰, 5개 레스토랑 규모로 되어 있어, 업로드 단계에서는 이 데이터가 사용되고, 640k JSON은 부하 테스트용 API 페이로드에만 사용됩니다.

---

## 3. 3개 포트에 동시 요청 시 new_sync/new_async에서 벡터 업로드가 진행되지 않은 이유

**업로드 요청은 세 포트 모두 수신했지만, new_sync/new_async는 업로드 처리 전에 크래시합니다.**

| 포트 | 이미지 | 임베딩 백엔드 | 결과 |
|------|--------|----------------|------|
| 8001 (old_sync) | tasteam-old-sync | sentence-transformers (PyTorch) | 정상 로드 후 업로드 완료 |
| 8002 (new_sync) | tasteam-new-async | xenova (ONNX) | ONNX 파일 부재로 크래시 → 업로드 미완료 |
| 8003 (new_async) | tasteam-new-async | xenova (ONNX) | 동일 크래시 → 업로드 미완료 |

타임라인은 다음과 같습니다.

- **15:24:17**: old_sync에서 벡터 업로드 처리 시작
- **15:24:27**: new_sync/new_async에서 xenova 모델 로딩 시도 → ONNX 경로에서 크래시
- **15:24:30**: `ONNXRuntimeError: model.onnx failed. File doesn't exist` 발생

즉, 3개 포트 모두 업로드 요청을 받았지만, new_sync/new_async는 임베딩 모델 로딩 단계에서 실패해 업로드 처리까지 진행하지 못한 상황입니다.

---

## 요약

1. **old_sync만 업로드**: sentence-transformers만 정상 로드되었고, new_sync/new_async는 xenova ONNX 로딩 중 크래시.
2. **142개만 업로드**: 업로드 단계는 `test_data_sample.json`(142개 리뷰, 5개 레스토랑) 기반이며, 640k JSON은 부하 테스트용 API 페이로드용.
3. **3개 포트 요청 시**: 업로드 요청은 세 포트 모두 도달했으나, new_sync/new_async는 임베딩 모델 로딩 단계에서 크래시하여 업로드를 완료하지 못함.