
다운로드된 adapter로 할 수 있는 일은 크게 두 가지입니다.

---

## 1. **Merge for serving** (실서비스용 모델 만들기)

adapter를 베이스 모델과 합쳐서 **한 번에 로드 가능한 풀 모델**로 만듭니다. 이걸 해야 API/추론에 쓸 수 있습니다.

```bash
python scripts/distill_flows.py merge_for_serving \
  --adapter-path distill_pipeline_output/artifacts/7568nmcv/adapter \
  --out-dir distill_pipeline_output
```

- `--adapter-path`: 지금 받은 adapter 폴더 경로 (예: `artifacts/7568nmcv/adapter`)
- `--out-dir`: merge 결과를 둘 디렉터리 (기본 `distill_pipeline_output`)

실행이 끝나면 터미널에 **`LLM_MODEL=...`** 형태로 merged 모델 경로가 출력됩니다. 이 경로를 API/추론 설정에 넣으면 됩니다.

---

## 2. **Evaluate** (선택, ROUGE/BERTScore 등)

지표를 보고 싶다면 evaluate를 돌리면 됩니다.  
다만 앞에서 **peft/transformers 버전** 문제로 실패했으므로, 먼저 `pip install "transformers>=4.46"` 등으로 맞춘 뒤에 해야 합니다.

```bash
python scripts/distill_flows.py evaluate \
  --adapter-path distill_pipeline_output/artifacts/7568nmcv/adapter \
  --val-labeled-path distill_pipeline_output/labeled/20260226_051037/val_labeled.json \
  --test-labeled-path distill_pipeline_output/labeled/20260226_051037/test_labeled.json \
  --out-dir distill_pipeline_output
```

---

**정리**:  
- **지금 당장 할 일**: 1번 `merge_for_serving` 한 번 돌리고, 출력된 `LLM_MODEL=...` 경로를 API/서빙 설정에 넣어서 사용하면 됩니다.  
- 평가 지표가 필요하면 transformers/peft 버전 맞춘 다음 2번 `evaluate`를 실행하면 됩니다.