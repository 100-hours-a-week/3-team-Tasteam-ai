transformers 5.0.0이 설치돼 있는데도 HybridCache를 못 찾는 상황입니다. transformers 5.x에서 export 위치가 바뀌었을 수 있어 확인 중입니다.
transformers 5.0.0이 설치돼 있는데, peft가 from transformers import HybridCache로 가져오려다 실패하는 상황입니다.
transformers 5.0에서 내부 구조가 바뀌면서 HybridCache가 최상위 __init__.py에서 export되지 않았을 가능성이 큽니다.
해결 방법
방법 1: transformers를 4.x로 내리기 (권장)
지금 peft는 transformers 4.x 기준으로 HybridCache를 쓰는 것 같으니, 4.x로 맞추는 게 가장 빠릅니다.
pip install "transformers>=4.46,<5"
그 다음 evaluate 다시 실행:
python scripts/distill_flows.py evaluate \  --adapter-path distill_pipeline_output/artifacts/7568nmcv/adapter \  --val-labeled-path distill_pipeline_output/labeled/20260226_051037/val_labeled.json \  --test-labeled-path distill_pipeline_output/labeled/20260226_051037/test_labeled.json \  --out-dir distill_pipeline_output
방법 2: peft 올리기
transformers 5.0을 지원하는 최신 peft가 있다면:
pip install -U peft
그래도 HybridCache 에러가 나면, 결국 방법 1(transformers 4.x) 이 필요합니다.
정리:
원인: transformers 5.0에서 HybridCache가 상위 __init__에 없어서, peft의 from transformers import HybridCache가 실패함.
권장: pip install "transformers>=4.46,<5" 로 4.x로 낮춘 뒤 evaluate 재실행.