가능해. **old_sync에서 그 “NoSuchFile / 중복 로드”가 안 보인 건, 파이프라인 품질 차이라기보다 “임베딩 로딩 방식/캐시 위치/초기화 타이밍”이 달라서** 그래.

## 1) old_sync는 fastembed(ONNX, /tmp 캐시) 경로가 아님

old_sync 로그를 보면 임베딩이 **sentence-transformers(PyTorch)** 로 로드돼.
`SentenceTransformer: Load pretrained ...`가 뜨고 HF에서 safetensors 등을 받는 전형적인 ST 로딩 흐름이야. 

반면 new_sync는 **fastembed + ONNXRuntime** 로 ` /tmp/fastembed_cache/.../onnx/model.onnx` 를 열다가 **파일이 없어서 즉시 터짐**. 
→ 즉, old_sync는 애초에 “/tmp/fastembed_cache의 onnx 파일”을 로드하지 않으니 **그 형태의 NoSuchFile이 발생할 조건이 없음**.

## 2) old_sync는 “먼저 한 번 크게 워밍업”이 된 상태였음

old_sync는 `/api/v1/vector/upload` 같은 작업이 먼저 수행되면서

* 리뷰 142개 배치 인코딩
* 포인트 생성/업로드
  까지 진행돼서, 임베딩 모델이 **초기에 한 번 제대로 로드+사용**된 흔적이 있어. 
  이러면 ramp-up 시점엔 “첫 로드/첫 다운로드”가 아니라서 **레이스가 훨씬 줄어**.

## 3) old_sync는 구조적으로 동시성이 낮아 “레이스가 덜 터지는” 효과도 있음

네가 정리한 것처럼 old_sync는 일부 경로(특히 summarize)가 이벤트 루프를 블로킹할 수 있어서, 결과적으로 요청이 “자연스럽게 직렬화”되는 면이 있어(좋은 건 아니지만).
동시 초기화/동시 다운로드 레이스는 **동시성이 높을수록 잘 터지니까**, old_sync에서는 그 조건이 덜 만들어졌을 가능성이 커.

---

### 요약하면

* **new_sync/new_async의 문제는 “fastembed(ONNX) + /tmp 캐시 + 동시 초기화” 조합에서 폭발**한 것(그래서 `/tmp/fastembed_cache/...model.onnx` NoSuchFile) 
* **old_sync는 (1) 다른 백엔드(sentence-transformers), (2) 먼저 벡터 업로드로 워밍업, (3) 상대적으로 낮은 동시성**이라 그 타입의 문제가 안 드러난 것 

원하면, “old_sync도 sentence-transformers인데 왜 new_*는 fastembed로 바뀌었는지(설정/분기)”를 같이 정리해서 **같은 백엔드로 맞춘 뒤에 sync vs async를 공정 비교하는 실험 설계**까지 바로 잡아줄게.
